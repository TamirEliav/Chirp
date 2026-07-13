"""ThresholdRecorder — sample-accurate multi-event threshold capture.

c18 (#15): rewritten to walk samples sample-by-sample so `min_cross`,
`hold`, pre-trigger lookback, and post-trigger tail are all
sample-accurate rather than chunk-quantized. Existing tests continue
to pass because they feed uniform chunks at chunk-aligned parameters,
so transitions still land on chunk boundaries.

Mask-input model:
  - If `trigger_mask` (np.ndarray[bool]) is supplied, the state machine
    walks it directly. Callers that want full sample accuracy compute
    a per-sample boolean upstream (e.g. `np.abs(filtered) >= thr` ANDed
    with a chunk-level spectral gate) and pass it in.
  - Else if `should_trigger` (bool) is supplied (c12 / #16 path), the
    mask is broadcast chunk-uniform from that bool — preserves the
    current spectral-gate semantics.
  - Else the legacy chunk-uniform `trigger_peak >= threshold` compare
    is used.

Either way the state machine itself walks samples, so the parameter
semantics are uniform regardless of how the mask was produced.
"""

import collections
import datetime
import time

import numpy as np

from chirp.constants import CHUNK_FRAMES, SAMPLE_RATE
from chirp.recording import writer as _writer


class ThresholdRecorder:
    """Sample-accurate multi-event threshold recorder.

    Parameter semantics (all in seconds, converted to samples internally):
      - min_cross  — number of consecutive above-threshold samples required to start an event.
      - pre_trig   — samples kept before the first above-threshold sample of the qualifying run.
      - hold       — number of consecutive below-threshold samples after the last above sample
                     that marks the end of the event (hold=0 → ends on the very first below sample).
      - post_trig  — samples kept after `last_above + 1`, appended as tail to the WAV.
      - max_rec    — hard cap on the number of samples in a single saved WAV; overflowing
                     events are force-split.

    Multiple events can be active simultaneously: a new burst can begin while an older
    event is still draining its post-trigger tail, and the two are saved as separate WAVs.
    """

    def __init__(self, streaming: bool = False):
        self._was_enabled     = False
        self._pre_trig_deque  = collections.deque(maxlen=1)
        self._pre_trig_maxlen = 1
        self._above_streak    = 0   # consecutive above-threshold samples (carries across chunks)
        self._active_events: list = []
        # #23 / c13: monotonic + wall anchor pair, established lazily
        # on the first chunk after enable. Wall-clock onsets are derived
        # by adding a monotonic delta — immune to NTP/DST jumps.
        self._mono_anchor: float | None = None
        self._wall_anchor: datetime.datetime | None = None
        # Streaming mode (H1): when True, events append their committed
        # samples incrementally to a StreamingWavWriter as they arrive,
        # keeping only a bounded pending tail in RAM (≤ hold + post_trig
        # + one chunk) instead of buffering up to ``max_rec`` seconds.
        # Default False so the buffered path — and every test that
        # monkeypatches ``_start_flush`` to capture full buffers —
        # behaves exactly as before. RecordingEntity opts in.
        self._streaming = bool(streaming)
        # Min-total-crossing filter: accumulated above-threshold samples
        # an event must reach or its file is discarded at finalize time
        # (0 = disabled). Updated from the ``min_total_cross_sec`` param
        # on every process_chunk call so flush_all sees the last value.
        self._min_total_samps = 0

    def _streaming_active(self) -> bool:
        """True when this recorder should stream events to disk.

        Streaming is bypassed whenever ``_start_flush`` has been
        replaced on the class (the standard test seam): those tests
        assert on the full in-memory buffer handed to their capture
        stub, which only exists on the buffered path.
        """
        if not self._streaming:
            return False
        return type(self).__dict__.get('_start_flush') is _ORIG_START_FLUSH

    # ── Vectorized run-length helper ──────────────────────────────────────

    @staticmethod
    def _consecutive_counts(flags: np.ndarray, carry: int) -> np.ndarray:
        """Running count of consecutive ``True`` values ending at each
        position, resetting to 0 on ``False``.

        ``carry`` is added to the *leading* run only (positions before the
        first ``False``), so this reproduces a sample-by-sample counter
        that already stood at ``carry`` when this block began. Used to
        vectorize the above-threshold streak and the silent-hold counter
        that the state machine previously walked one sample at a time.
        """
        n = flags.shape[0]
        if n == 0:
            return np.empty(0, dtype=np.int64)
        idx = np.arange(n)
        # Index of the most recent reset (False) at or before each i, or
        # -1 if no False has been seen yet on this block.
        last_reset = np.maximum.accumulate(np.where(flags, -1, idx))
        counts = idx - last_reset
        if carry:
            counts = counts + np.where(last_reset == -1, carry, 0)
        return counts

    # ── Mask construction ────────────────────────────────────────────────

    @staticmethod
    def _build_mask(chunk: np.ndarray, *, threshold: float, trigger_peak: float,
                    should_trigger: bool | None,
                    trigger_mask: np.ndarray | None) -> np.ndarray:
        n = len(chunk)
        if trigger_mask is not None:
            m = np.asarray(trigger_mask, dtype=bool)
            if m.shape[0] != n:
                raise ValueError(
                    f"trigger_mask length {m.shape[0]} != chunk length {n}")
            return m
        if should_trigger is not None:
            return np.full(n, bool(should_trigger), dtype=bool)
        return np.full(n, trigger_peak >= threshold, dtype=bool)

    # ── Main entry point ─────────────────────────────────────────────────

    def process_chunk(self, chunk: np.ndarray, *,
                      trigger_peak: float,
                      threshold: float, min_cross_sec: float, hold_sec: float,
                      post_trig_sec: float,
                      max_rec_sec: float, pre_trig_sec: float,
                      output_dir: str, enabled: bool,
                      min_total_cross_sec: float = 0.0,
                      filename_prefix: str = '', filename_suffix: str = '',
                      sample_rate: int = SAMPLE_RATE,
                      should_trigger: bool | None = None,
                      trigger_mask: np.ndarray | None = None,
                      filename_stream: str = '',
                      global_chunk_end: int | None = None,
                      chunk_end_wall: datetime.datetime | None = None) -> dict:
        """Drive the state machine with one audio chunk.

        See module docstring for the mask-input model.

        Returns a report dict used by the entity to paint detect/record
        indicator strips (#32):
          * ``detect_mask`` — per-sample bool array (raw threshold mask
            after masking rules, length = ``len(chunk)``).
          * ``active_spans`` — list of ``(g_start, g_end)`` in absolute
            global-sample coordinates for every event still open after
            this chunk. ``g_end`` is clipped to ``global_chunk_end``.
          * ``flushed_spans`` — list of ``(g_start, g_end)`` for events
            finalised during this chunk (disable-flush, post-trigger
            tail complete, or force-split on ``max_rec``).

        When ``global_chunk_end`` is None (legacy callers), the spans
        lists are always empty but ``detect_mask`` is still populated.
        """
        n = len(chunk)
        flushed_spans: list[tuple[int, int]] = []
        active_spans: list[tuple[int, int]]  = []
        # M5: capture-time wall clock of this chunk's last sample,
        # supplied by the entity from its sample-clock anchor. When
        # present, event onsets are derived from it instead of "now" —
        # a backlogged ingest thread no longer shifts every filename
        # timestamp late by the backlog.
        self._chunk_end_wall = chunk_end_wall

        # Min-total-crossing filter threshold, in samples. Updated before
        # any flush path (including the disable-flush below) can run.
        self._min_total_samps = max(0, int(min_total_cross_sec * sample_rate))

        # ── Resize pre-trigger rolling buffer (in chunks) ─────────────────
        # Holds enough history to cover (pre_trig + min_cross) of lookback.
        needed = max(1, int((pre_trig_sec + min_cross_sec) * sample_rate / CHUNK_FRAMES) + 1)
        if needed != self._pre_trig_maxlen:
            old = list(self._pre_trig_deque)
            self._pre_trig_deque  = collections.deque(old[-needed:], maxlen=needed)
            self._pre_trig_maxlen = needed

        # Always feed the rolling lookback buffer with the current chunk.
        self._pre_trig_deque.append(chunk.copy())

        # ── Enable/disable transitions ────────────────────────────────────
        if self._was_enabled and not enabled:
            for ev in self._active_events:
                flushed_spans.append(self._span_for_flush(ev))
                self._flush_event(ev, output_dir, filename_prefix,
                                  filename_suffix, sample_rate, filename_stream)
            self._active_events = []
            self._above_streak  = 0
            self._mono_anchor   = None
            self._wall_anchor   = None
        self._was_enabled = enabled

        if not enabled:
            # Still return the raw mask so the detect strip keeps
            # rendering even while recording is disabled. No new events
            # can open, so active_spans is empty.
            mask = self._build_mask(chunk, threshold=threshold,
                                    trigger_peak=trigger_peak,
                                    should_trigger=should_trigger,
                                    trigger_mask=trigger_mask)
            return {
                'detect_mask':   mask.copy(),
                'active_spans':  [],
                'flushed_spans': [s for s in flushed_spans if s[0] >= 0],
            }

        # Lazily anchor the monotonic + wall clocks (#23 / c13).
        if self._mono_anchor is None:
            self._mono_anchor = time.monotonic()
            self._wall_anchor = datetime.datetime.now()

        # ── Parameter conversions (everything in samples) ─────────────────
        min_cross_samps = int(min_cross_sec * sample_rate)
        hold_samps      = int(hold_sec      * sample_rate)
        post_trig_samps = max(0, int(post_trig_sec * sample_rate))
        pre_trig_samps  = max(0, int(pre_trig_sec  * sample_rate))
        max_samps       = max(1, int(max_rec_sec   * sample_rate))

        mask = self._build_mask(chunk, threshold=threshold,
                                trigger_peak=trigger_peak,
                                should_trigger=should_trigger,
                                trigger_mask=trigger_mask)

        # ── Pre-walk: detect event opening (sample-accurate) ─────────────
        # Vectorized equivalent of a per-sample above-threshold streak
        # counter carrying ``self._above_streak`` across chunk
        # boundaries. ``trigger_pos`` is the first sample at which the
        # streak reaches ``need`` (only when no event is currently open).
        has_open = any(not ev['ended'] for ev in self._active_events)
        need = max(1, min_cross_samps)
        streak_counts = self._consecutive_counts(mask, self._above_streak)
        self._above_streak = int(streak_counts[-1]) if n else self._above_streak
        trigger_pos: int | None = None
        if not has_open:
            qual = np.nonzero(streak_counts >= need)[0]
            if qual.size:
                trigger_pos = int(qual[0])

        # ── Open new event if triggered ──────────────────────────────────
        just_created = None
        if trigger_pos is not None:
            just_created = self._open_event(
                trigger_pos, n, need, pre_trig_samps, sample_rate,
                global_chunk_end=global_chunk_end,
                initial_above=int(streak_counts[trigger_pos]))
            self._active_events.append(just_created)

        # ── Walk every active event through this chunk ───────────────────
        threshold_silent = max(1, hold_samps)
        to_remove = []
        # #57: Continuations spawned by force-split during this chunk are
        # collected here and appended after the loop — we cannot mutate
        # ``self._active_events`` while iterating it. Each continuation
        # has already had the leftover samples of the boundary chunk
        # walked through it for silent_samps accounting (see force-split
        # branch below).
        to_add: list[dict] = []
        for ev in self._active_events:
            if ev is just_created:
                # Just opened; walk samples after trigger_pos. The trigger
                # sample itself is above (so silent_samps stays 0) and is
                # already accounted for in last_above_kept.
                walk_start = trigger_pos + 1
            else:
                # Append this chunk to the event and walk it from sample 0.
                ev['buf'].append(chunk.copy())
                ev['samples_kept'] += n
                walk_start = 0

            if not ev['ended']:
                # Vectorized hold walk over mask[walk_start:n]: a silent
                # counter carrying ``ev['silent_samps']`` ends the event
                # at the first sample where it reaches ``threshold_silent``;
                # ``last_above_kept`` tracks the most recent above sample.
                sil0 = ev['silent_samps']
                chunk_kept_start = ev['samples_kept'] - n  # event-coord index of sample 0 of this chunk
                sub = mask[walk_start:]
                nb = sub.shape[0]
                if nb:
                    idxb = np.arange(nb)
                    last_above_acc = np.maximum.accumulate(
                        np.where(sub, idxb, -1))
                    sil_run = self._consecutive_counts(~sub, sil0)
                    ends = np.nonzero(sil_run >= threshold_silent)[0]
                    if ends.size:
                        end_b = int(ends[0])
                        ev['silent_samps'] = int(sil_run[end_b])
                        lt = int(last_above_acc[end_b])
                        if lt != -1:
                            ev['last_above_kept'] = chunk_kept_start + walk_start + lt
                        ev['ended'] = True
                        ev['target_kept'] = ev['last_above_kept'] + 1 + post_trig_samps
                        # Min-total-crossing accounting: above samples up
                        # to (and including) the sample that ended the
                        # event. Above samples in the post-trigger tail
                        # belong to whatever NEW event they may open.
                        ev['above_total'] = (ev.get('above_total', 0)
                                             + int(np.count_nonzero(sub[:end_b + 1])))
                    else:
                        ev['silent_samps'] = int(sil_run[-1])
                        lt = int(last_above_acc[-1])
                        if lt != -1:
                            ev['last_above_kept'] = chunk_kept_start + walk_start + lt
                        ev['above_total'] = (ev.get('above_total', 0)
                                             + int(np.count_nonzero(sub)))

            # Flush when the post-trigger tail is fully captured.
            if ev['ended'] and ev['samples_kept'] >= ev['target_kept']:
                flushed_spans.append(self._span_for_flush(ev))
                self._flush_event(ev, output_dir, filename_prefix,
                                  filename_suffix, sample_rate, filename_stream)
                to_remove.append(ev)
                continue

            # Force-split events longer than max_rec.
            #
            # #57: Pre-fix the first half was kept verbatim
            # (``target_kept = samples_kept``) and a fresh event had to
            # re-qualify through ``min_cross`` from scratch — up to
            # ``min_cross_sec`` of audio was silently lost between the
            # two WAVs. Post-fix the first half is pinned to exactly
            # ``max_samps`` for a clean sample-accurate boundary and a
            # butt-joined continuation event opens immediately at that
            # boundary (no min_cross gate, no pre-trigger lookback).
            # Both halves are tagged with ``split_index`` (internal
            # bookkeeping); each half's filename carries its own
            # onset timestamp, so the series is contiguous by time.
            if ev['samples_kept'] >= max_samps:
                overshoot = ev['samples_kept'] - max_samps
                ev['target_kept'] = max_samps
                # First-half part-index defaults to 1; on a 3-way split
                # the second-half (already split_index=2) keeps its
                # value when IT trips the next force-split.
                ev['split_index'] = ev.get('split_index') or 1
                flushed_spans.append(self._span_for_flush(ev))
                self._flush_event(ev, output_dir, filename_prefix,
                                  filename_suffix, sample_rate,
                                  filename_stream, splitting=True)
                to_remove.append(ev)

                cont = self._open_continuation(
                    ev, chunk, mask, n, overshoot,
                    post_trig_samps=post_trig_samps,
                    threshold_silent=threshold_silent,
                    sample_rate=sample_rate,
                    global_chunk_end=global_chunk_end,
                )
                to_add.append(cont)

        for ev in to_remove:
            if ev in self._active_events:
                self._active_events.remove(ev)
        # #57: append continuations spawned during the iteration. They
        # are already partially walked through the leftover samples of
        # the boundary chunk, so they participate in active_spans
        # reporting below and are walked further by the next chunk.
        for ev in to_add:
            self._active_events.append(ev)

        # H1: stream each open event's newly-committed region to disk so
        # RAM holds only the pending tail (samples that may yet be
        # trimmed by the hold walk or handed to a continuation).
        for ev in self._active_events:
            if ev.get('stream'):
                wm = self._stream_watermark(ev, post_trig_samps, max_samps)
                self._stream_commit(ev, wm, output_dir, filename_prefix,
                                    filename_suffix, filename_stream)

        # ── Build span report for still-open events ──────────────────────
        if global_chunk_end is not None:
            for ev in self._active_events:
                gs = ev.get('global_start')
                if gs is None:
                    continue
                # The event currently spans [global_start, global_start +
                # samples_kept). Clip to global_chunk_end defensively —
                # should be equal after chunk append, but keep the
                # invariant explicit.
                ge = min(gs + ev['samples_kept'], global_chunk_end)
                if ge > gs:
                    active_spans.append((gs, ge))

        return {
            'detect_mask':   mask.copy(),
            'active_spans':  active_spans,
            'flushed_spans': [s for s in flushed_spans if s[0] >= 0],
        }

    # ── Helpers ──────────────────────────────────────────────────────────

    def _open_event(self, trigger_pos: int, n: int, need: int,
                    pre_trig_samps: int, sample_rate: int,
                    *, global_chunk_end: int | None = None,
                    initial_above: int | None = None) -> dict:
        """Build a new event dict at the given trigger sample.

        `trigger_pos` is the sample index in the current chunk where the
        qualifying above-threshold streak first hit `need`. The event
        starts `pre_trig_samps` samples before the first above sample
        of that streak.
        """
        # Snapshot of the rolling lookback buffer (already includes the
        # current chunk because process_chunk appended it before this).
        buf_init = [c.copy() for c in self._pre_trig_deque]
        total_pre = sum(len(c) for c in buf_init)

        # Position of the trigger sample in buf_init coordinates.
        trigger_abs = total_pre - n + trigger_pos
        # First above-threshold sample of the qualifying run.
        first_above = trigger_abs - (need - 1)
        # Event start = pre_trig_samps before that (clamped to deque start).
        event_start = max(0, first_above - pre_trig_samps)

        # Drop fully-skipped chunks from the front so start_offset always
        # falls inside buf_init[0].
        while buf_init and event_start >= len(buf_init[0]):
            event_start -= len(buf_init[0])
            buf_init.pop(0)
        start_offset = event_start

        samples_kept = sum(len(c) for c in buf_init) - start_offset
        # last_above_kept: event-coord index of the sample at trigger_pos
        # in the current chunk (which is buf_init[-1]).
        last_above_kept = samples_kept - n + trigger_pos

        # Onset = wall time of the current chunk's END minus the
        # duration of the kept audio (so the timestamp points at the
        # start of the saved WAV, not at the trigger sample).
        #
        # M5: prefer the entity-supplied capture-time clock — it is
        # anchored to the sample counter at acquisition start, so an
        # ingest backlog cannot shift onsets late. Fall back to the
        # monotonic-anchor "now" for legacy callers that don't pass
        # ``chunk_end_wall``.
        chunk_end = getattr(self, '_chunk_end_wall', None)
        if chunk_end is None:
            mono_delta = time.monotonic() - self._mono_anchor
            chunk_end = self._wall_anchor + datetime.timedelta(
                seconds=mono_delta)
        onset = chunk_end - datetime.timedelta(
            seconds=samples_kept / sample_rate)

        # #32: absolute start of the event in global-sample coordinates.
        # Kept alongside `samples_kept` so span reporting stays correct
        # even across flush-on-disable (no chunk appended) and force-split
        # (target_kept replaced) paths. When the caller doesn't supply
        # global_chunk_end (legacy tests), global_start is None and the
        # event simply isn't reported in the span lists.
        global_start = (None if global_chunk_end is None
                        else global_chunk_end - samples_kept)

        return {
            'buf':             buf_init,
            'start_offset':    start_offset,
            'samples_kept':    samples_kept,
            'last_above_kept': last_above_kept,
            'silent_samps':    0,
            'ended':           False,
            'target_kept':     None,
            'onset_time':      onset,
            'global_start':    global_start,
            # #46: pin the sample rate at event-open time so a
            # mid-event session-SR change can't mislabel the WAV
            # header. ``_flush_event`` uses this instead of the
            # caller-supplied ``sample_rate`` — belt-and-suspenders
            # for the case where the per-entity flush on SR change
            # (#45 / PR 2) somehow misses an event.
            'sample_rate':     sample_rate,
            # H1 streaming state. The decision is pinned at open time so
            # a mid-event monkeypatch or flag flip can't strand an event
            # halfway between the two storage models.
            'stream':          self._streaming_active(),
            'writer':          None,   # StreamingWavWriter, lazy
            'written':         0,      # kept-coord samples already on disk
            'stream_failed':   False,
            # Accumulated above-threshold samples of this event — the
            # qualifying streak that opened it (may span chunks via the
            # carry) plus whatever the per-chunk walks add. Compared to
            # ``_min_total_samps`` at finalize time.
            'above_total':     int(initial_above
                                   if initial_above is not None else need),
        }

    def _open_continuation(self, parent_ev: dict, chunk: np.ndarray,
                           mask: np.ndarray, n: int, overshoot: int, *,
                           post_trig_samps: int,
                           threshold_silent: int,
                           sample_rate: int,
                           global_chunk_end: int | None) -> dict:
        """#57: Open a butt-joined continuation event after a force-split.

        The parent event has just been flushed with exactly
        ``max_samps`` samples; this continuation owns sample
        ``max_samps`` onwards. There is no pre-trigger lookback (the
        boundary is exact, not a re-qualification) and no
        ``min_cross`` gate (we know the signal was above threshold at
        the boundary because that's how we got here).

        ``overshoot`` = ``parent_ev['samples_kept'] - max_samps`` —
        the number of samples after the boundary that already arrived
        in the boundary chunk. Those samples become the continuation's
        initial buffer and are walked here for ``silent_samps``
        accounting (they may already trip the hold timer if the signal
        went silent right at / after the boundary).
        """
        # Position in the chunk where the continuation begins. Clamped
        # defensively for the pathological case where pre_trig +
        # lookback caused the parent's samples_kept to exceed max_samps
        # at open time (boundary in the lookback, before the chunk).
        cont_chunk_pos = max(0, min(n, n - overshoot))
        boundary_overshoot = n - cont_chunk_pos

        if boundary_overshoot > 0:
            buf_init = [chunk[cont_chunk_pos:].copy()]
        else:
            buf_init = []
        samples_kept = boundary_overshoot

        # Walk the leftover samples for hold accounting. last_above_kept
        # defaults to ``samples_kept - 1`` (the boundary sample is by
        # definition above-threshold) when the chunk has any leftover.
        # When the boundary lands exactly at chunk end (overshoot == 0)
        # we initialise to -1 — the next chunk's walk will set it
        # correctly the first time it sees an above sample.
        # Vectorized leftover walk (carry-in silent counter = 0). Mirrors
        # the per-event hold walk; ``last_above_kept`` is in the
        # continuation's buffer coordinates (offset by cont_chunk_pos).
        sil = 0
        last_above_kept = samples_kept - 1 if samples_kept > 0 else -1
        ended = False
        target_kept: int | None = None
        leftover_above = 0
        sub = mask[cont_chunk_pos:]
        nb = sub.shape[0]
        if nb:
            idxb = np.arange(nb)
            last_above_acc = np.maximum.accumulate(np.where(sub, idxb, -1))
            sil_run = self._consecutive_counts(~sub, 0)
            ends = np.nonzero(sil_run >= threshold_silent)[0]
            if ends.size:
                end_b = int(ends[0])
                lt = int(last_above_acc[end_b])
                if lt != -1:
                    last_above_kept = lt
                ended = True
                target_kept = last_above_kept + 1 + post_trig_samps
                sil = int(sil_run[end_b])
                leftover_above = int(np.count_nonzero(sub[:end_b + 1]))
            else:
                lt = int(last_above_acc[-1])
                if lt != -1:
                    last_above_kept = lt
                sil = int(sil_run[-1])
                leftover_above = int(np.count_nonzero(sub))

        # Continuation's onset is parent's onset + parent's duration so
        # the timestamps reflect the actual capture time of the
        # continuation's first sample (not the parent's start).
        parent_kept = parent_ev['target_kept']
        parent_sr = parent_ev.get('sample_rate', sample_rate)
        onset = (parent_ev['onset_time']
                 + datetime.timedelta(seconds=parent_kept / parent_sr))

        # Continuation's global_start is right after parent's end so
        # the active_spans / flushed_spans reports stay contiguous.
        pgs = parent_ev.get('global_start')
        if pgs is None or global_chunk_end is None:
            global_start = None
        else:
            global_start = pgs + parent_kept

        return {
            'buf':             buf_init,
            'start_offset':    0,
            'samples_kept':    samples_kept,
            'last_above_kept': last_above_kept,
            'silent_samps':    sil,
            'ended':           ended,
            'target_kept':     target_kept,
            'onset_time':      onset,
            # Continuations inherit the parent's pinned sample rate —
            # the original capture rate (#46) — so a mid-event SR change
            # doesn't relabel part2 with the new rate while part1 still
            # carries the old one.
            'global_start':    global_start,
            'sample_rate':     parent_sr,
            'split_index':     (parent_ev.get('split_index') or 1) + 1,
            # H1: continuations inherit the parent's storage model.
            'stream':          bool(parent_ev.get('stream')),
            'writer':          None,
            'written':         0,
            'stream_failed':   False,
            # The min-total-crossing criterion is a per-EVENT measure:
            # a continuation carries the parent's accumulated crossing
            # so the final part of a long split event is judged on the
            # whole event, never deleted for a quiet tail. (The parent's
            # walk may already have counted some of the leftover samples
            # — the small over-count only makes deletion less likely.)
            'above_total':     (int(parent_ev.get('above_total', 0))
                                + leftover_above),
        }

    # ── H1: streaming storage helpers ────────────────────────────────────

    @staticmethod
    def _stream_watermark(ev: dict, post_trig_samps: int,
                          max_samps: int) -> int:
        """Kept-coord index up to which samples are *guaranteed* to be in
        this event's WAV, whatever happens next.

        Samples past ``last_above + 1 + post_trig`` may still be trimmed
        (event ends on hold), samples past ``max_samps`` belong to the
        force-split continuation, and samples that haven't arrived yet
        obviously can't be written — the watermark is the min of all
        three. Monotonic: ``last_above_kept`` only grows and the other
        two bounds are fixed per chunk."""
        if ev['ended'] and ev['target_kept'] is not None:
            t = ev['target_kept']
        else:
            t = ev['last_above_kept'] + 1 + post_trig_samps
        return max(0, min(t, max_samps, ev['samples_kept']))

    def _stream_commit(self, ev: dict, upto: int, output_dir: str,
                       filename_prefix: str, filename_suffix: str,
                       filename_stream: str) -> None:
        """Append kept-coord samples ``[written, upto)`` to the event's
        StreamingWavWriter (created lazily) and prune them from the
        pending buffer. Any I/O failure aborts streaming for this event
        and is surfaced through the writer-pool error stats — it must
        never propagate into the trigger walk."""
        if not ev.get('stream') or ev.get('stream_failed'):
            return
        need = upto - ev['written']
        if need <= 0:
            return
        buf = ev['buf']
        try:
            if ev['writer'] is None:
                if not buf:
                    return
                channels = 2 if buf[0].ndim == 2 else 1
                ev['writer'] = _writer.StreamingWavWriter(
                    output_dir,
                    prefix=filename_prefix, suffix=filename_suffix,
                    sample_rate=ev.get('sample_rate', SAMPLE_RATE),
                    onset_time=ev['onset_time'], channels=channels,
                    filename_stream=filename_stream)
            w = ev['writer']
            skip = ev['start_offset']
            while need > 0 and buf:
                c = buf[0]
                avail = len(c) - skip
                take = min(avail, need)
                w.append(c[skip:skip + take])
                ev['written'] += take
                need -= take
                if take == avail:
                    buf.pop(0)
                    skip = 0
                else:
                    skip += take
            ev['start_offset'] = skip
        except Exception as exc:
            ev['stream_failed'] = True
            wtr = ev.get('writer')
            ev['writer'] = None
            if wtr is not None:
                try:
                    wtr.abort()
                except Exception:
                    pass
            _writer.note_stream_error(exc, stream=filename_stream,
                                      out_dir=output_dir)

    def _stream_finalize(self, ev: dict, output_dir: str,
                         filename_prefix: str, eff_suffix: str,
                         filename_stream: str) -> bool:
        """Flush a streamed event: append the remaining committed range,
        retarget the final filename (the day-subfolder output dir may
        have rolled over mid-event), and hand the publish (fsync +
        rename) to the writer pool. Returns True when the event was
        handled on the streaming path (i.e. the buffered fallback must
        not run)."""
        if not ev.get('stream'):
            return False
        if ev.get('stream_failed'):
            # Earlier samples are gone with the aborted tmp — a partial
            # buffered flush would produce a WAV that silently starts
            # mid-event, worse than no file. The failure was already
            # counted + logged at abort time.
            return True
        target = (ev['target_kept'] if ev['target_kept'] is not None
                  else ev['samples_kept'])
        self._stream_commit(ev, target, output_dir, filename_prefix,
                            eff_suffix, filename_stream)
        if ev.get('stream_failed'):
            return True
        w = ev.get('writer')
        if w is None:
            # Zero-length event (nothing ever committed) — nothing to
            # publish.
            return True
        try:
            w.retarget(output_dir, prefix=filename_prefix,
                       suffix=eff_suffix, onset_time=ev['onset_time'],
                       filename_stream=filename_stream)
            # Publish (fsync + atomic rename) on the writer pool, NOT
            # here: this runs on the ingest thread, and fsyncing a
            # multi-MB part file synchronously at a force-split
            # boundary stalls audio consumption mid-event — the
            # capture pipeline glitches under that stall and corrupts
            # the start of the next part. The tmp already holds every
            # committed sample, so deferring the close loses nothing.
            _writer.submit_close(w, filename_stream=filename_stream,
                                 out_dir=output_dir)
        except Exception as exc:
            try:
                w.abort()
            except Exception:
                pass
            _writer.note_stream_error(exc, stream=filename_stream,
                                      out_dir=output_dir)
        ev['writer'] = None
        return True

    @staticmethod
    def _span_for_flush(ev: dict) -> tuple[int, int]:
        """Return ``(g_start, g_end)`` for an event about to be flushed.

        Mirrors the trimming logic in ``_trim_event`` — uses
        ``target_kept`` when set, falls back to ``samples_kept`` for the
        disable-path flush. Returns ``(-1, -1)`` when the event has no
        global_start (legacy tests that don't pass ``global_chunk_end``).
        """
        gs = ev.get('global_start')
        if gs is None:
            return (-1, -1)
        target = ev['target_kept'] if ev['target_kept'] is not None else ev['samples_kept']
        return (gs, gs + target)

    @staticmethod
    def _trim_event(ev: dict) -> list:
        """Slice ev['buf'] to [start_offset, start_offset + target_kept).

        If target_kept is None (force-split path that didn't go through
        the ended branch), keep all samples after start_offset.
        """
        target = ev['target_kept'] if ev['target_kept'] is not None else ev['samples_kept']
        out = []
        skip = ev['start_offset']
        remaining = target
        for c in ev['buf']:
            if skip >= len(c):
                skip -= len(c)
                continue
            seg = c[skip:] if skip > 0 else c
            skip = 0
            if len(seg) > remaining:
                seg = seg[:remaining]
            out.append(seg)
            remaining -= len(seg)
            if remaining <= 0:
                break
        return out

    @staticmethod
    def _discard_event(ev: dict, filename_stream: str,
                       min_samps: int) -> None:
        """Drop an event that failed the min-total-crossing criterion.

        Buffered events simply release their RAM; streamed events abort
        their StreamingWavWriter, which deletes the in-progress ``.tmp``
        file — nothing is ever published under the canonical name.
        """
        sr = ev.get('sample_rate', SAMPLE_RATE) or SAMPLE_RATE
        w = ev.get('writer')
        ev['writer'] = None
        if w is not None:
            try:
                w.abort()
            except Exception:
                pass
        ev['buf'] = []
        tag = f' [{filename_stream}]' if filename_stream else ''
        print(f'[REC] discarded event{tag}: total threshold crossing '
              f'{ev.get("above_total", 0) / sr:.3f}s < minimum '
              f'{min_samps / sr:.3f}s')

    def _flush_event(self, ev: dict, output_dir: str,
                     filename_prefix: str, filename_suffix: str,
                     sample_rate: int, filename_stream: str,
                     *, splitting: bool = False) -> None:
        # Min-total-crossing filter: discard the whole event if its
        # accumulated above-threshold duration never reached the minimum.
        # Assessed only at FINAL flushes — a force-split flush
        # (``splitting=True``) happens mid-event while the total is
        # still growing, and the continuation inherits the running
        # total, so the criterion is applied once, on the event's last
        # file. Events without the counter (legacy callers) pass.
        min_samps = self._min_total_samps
        if (min_samps > 0 and not splitting
                and ev.get('above_total', min_samps) < min_samps):
            self._discard_event(ev, filename_stream, min_samps)
            return
        # Force-split halves used to carry a ``partNN`` filename token;
        # that is gone — every file (split or not) now shares one name
        # format, and each part's onset_time is its own capture time
        # (part2 = part1 onset + part1 duration, see
        # ``_open_continuation``), so the series stays unambiguous by
        # timestamp alone. ``split_index`` is still tracked internally.
        eff_suffix = filename_suffix
        # H1: streamed events publish their incrementally-written file;
        # buffered events go through the writer pool as before.
        if self._stream_finalize(ev, output_dir, filename_prefix,
                                 eff_suffix, filename_stream):
            return
        trimmed = self._trim_event(ev)
        # #46: prefer the SR pinned at event-open time. The caller's
        # ``sample_rate`` is the *current* session rate — if the user
        # changed it mid-event the WAV header must still reflect the
        # rate at which the samples were captured, otherwise the file
        # plays back at the wrong speed.
        ev_sr = ev.get('sample_rate', sample_rate)
        self._start_flush(trimmed, output_dir, filename_prefix,
                          eff_suffix, sample_rate=ev_sr,
                          onset_time=ev['onset_time'],
                          filename_stream=filename_stream)

    @property
    def is_recording(self) -> bool:
        return any(not ev['ended'] for ev in self._active_events)

    def flush_all(self, output_dir: str,
                  filename_prefix: str = '', filename_suffix: str = '',
                  sample_rate: int = SAMPLE_RATE,
                  filename_stream: str = '',
                  reason: str = '') -> int:
        """Flush every active event regardless of state (#17 / c16)."""
        n = len(self._active_events)
        if n and reason:
            print(f'[REC] flush_all ({reason}): {n} event(s) pending')
        for ev in self._active_events:
            self._flush_event(ev, output_dir, filename_prefix,
                              filename_suffix, sample_rate, filename_stream)
        self._active_events = []
        self._above_streak  = 0
        self._was_enabled   = False
        self._mono_anchor   = None
        self._wall_anchor   = None
        return n

    @staticmethod
    def _start_flush(buf_snapshot: list, output_dir: str,
                     prefix: str = '', suffix: str = '', sample_rate: int = SAMPLE_RATE,
                     onset_time=None, filename_stream: str = ''):
        """Submit a finished event's buffer to the WAV writer pool.

        Kept as a staticmethod on the class so the existing test
        monkeypatch (`monkeypatch.setattr(ThresholdRecorder,
        "_start_flush", ...)`) continues to take precedence over the
        real writer.
        """
        _writer.start_flush_thread(
            buf_snapshot, output_dir, prefix, suffix,
            sample_rate=sample_rate, onset_time=onset_time,
            filename_stream=filename_stream,
        )


# H1: sentinel used by ``ThresholdRecorder._streaming_active`` to detect
# the test seam — when a test monkeypatches ``_start_flush`` on the
# class, the class-dict entry no longer is this object and streaming
# steps aside so the stub receives full buffers.
_ORIG_START_FLUSH = ThresholdRecorder.__dict__['_start_flush']
