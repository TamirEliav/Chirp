"""ThresholdRecorder — multi-event threshold capture state machine.

Extracted from the monolith in the Phase 1 refactor (plan: c04). The
implementation is intentionally preserved *verbatim* from the monolith
so this commit introduces zero behavior change — the accompanying test
suite pins current semantics (including the chunk-quantization quirks
flagged by issue #15) so later commits can rewrite it safely.

Known quirks pinned by the tests:
  - `min_cross` and `hold` operate at chunk granularity (#15).
  - Flushed WAVs are trimmed to chunk boundaries (#15).
  - Event onset uses wall-clock `datetime.now()` (#23).
  - `_start_flush` launches a daemon thread that may be killed at
    interpreter shutdown before the WAV is fully written (#17).

c12 will add a `should_trigger` parameter to decouple the trigger
decision from `trigger_peak`; c13 will switch to a monotonic clock;
c16 will replace the daemon-thread flusher with a proper writer pool;
c18 will rewrite this state machine to be sample-accurate.
"""

import collections
import datetime

import numpy as np

from chirp.constants import CHUNK_FRAMES, SAMPLE_RATE
from chirp.recording import writer as _writer


class ThresholdRecorder:
    """
    Multi-event threshold recorder.

    Parameter semantics:
      - min_cross  — consecutive time above threshold required to start an event.
      - pre_trig   — audio kept before the trigger point (rolling lookback buffer).
      - hold       — silence duration after the last above-threshold sample that marks
                     the end of an event (hold=0 → event ends on the first below chunk).
      - post_trig  — audio kept after the event-end point, appended as tail to the WAV.
      - max_rec    — hard cap on the length of a single saved WAV; events longer are split.

    Multiple events can be active simultaneously: if a new above-threshold burst starts
    while an older event is still collecting its post-trigger tail, it begins a new event
    and the two are saved as separate WAV files whose audio windows overlap in time.
    """

    def __init__(self):
        self._was_enabled     = False
        self._pre_trig_deque  = collections.deque(maxlen=1)
        self._pre_trig_maxlen = 1
        self._above_streak    = 0   # consecutive above-threshold samples
        self._active_events: list = []

    def process_chunk(self, chunk: np.ndarray, *,
                      trigger_peak: float,
                      threshold: float, min_cross_sec: float, hold_sec: float,
                      post_trig_sec: float,
                      max_rec_sec: float, pre_trig_sec: float,
                      output_dir: str, enabled: bool,
                      filename_prefix: str = '', filename_suffix: str = '',
                      sample_rate: int = SAMPLE_RATE):

        # ── Resize pre-trigger rolling buffer ─────────────────────────────
        needed = max(1, int((pre_trig_sec + min_cross_sec) * sample_rate / CHUNK_FRAMES) + 1)
        if needed != self._pre_trig_maxlen:
            old = list(self._pre_trig_deque)
            self._pre_trig_deque  = collections.deque(old[-needed:], maxlen=needed)
            self._pre_trig_maxlen = needed

        # Always feed the rolling lookback buffer
        self._pre_trig_deque.append(chunk.copy())

        # ── Enable/disable transitions ────────────────────────────────────
        if self._was_enabled and not enabled:
            for ev in self._active_events:
                self._start_flush(ev['buf'], output_dir, filename_prefix,
                                  filename_suffix, sample_rate=sample_rate,
                                  onset_time=ev['onset_time'])
            self._active_events = []
            self._above_streak  = 0
        self._was_enabled = enabled

        if not enabled:
            return

        # ── Parameter conversions ─────────────────────────────────────────
        min_cross_samps  = int(min_cross_sec * sample_rate)
        hold_samps       = int(hold_sec      * sample_rate)
        post_trig_chunks = max(0, int(post_trig_sec * sample_rate / CHUNK_FRAMES))
        max_chunks       = max(1, int(max_rec_sec * sample_rate / CHUNK_FRAMES))

        above = trigger_peak >= threshold
        if above:
            self._above_streak += len(chunk)
        else:
            self._above_streak = 0

        # ── Start a new event if we have sustained above-threshold and
        #    no currently-open (non-ended) event is already capturing it ──
        has_open = any(not ev['ended'] for ev in self._active_events)
        just_created = None
        if (not has_open) and above and self._above_streak >= min_cross_samps:
            buf_init = [c.copy() for c in self._pre_trig_deque]
            n_init   = sum(len(c) for c in buf_init)
            onset    = datetime.datetime.now() - datetime.timedelta(
                seconds=n_init / sample_rate)
            just_created = {
                'buf':            buf_init,
                'ended':          False,
                'silent_samps':   0,
                'last_above_idx': len(buf_init) - 1,
                'tail_remaining': 0,
                'onset_time':     onset,
            }
            self._active_events.append(just_created)

        # ── Update every active event with this chunk ─────────────────────
        to_remove = []
        for ev in self._active_events:
            if ev is not just_created:
                ev['buf'].append(chunk.copy())
                if not ev['ended']:
                    if above:
                        ev['silent_samps']   = 0
                        ev['last_above_idx'] = len(ev['buf']) - 1
                    else:
                        ev['silent_samps'] += len(chunk)
                        if ev['silent_samps'] >= hold_samps:
                            ev['ended'] = True
                            chunks_since_last = len(ev['buf']) - 1 - ev['last_above_idx']
                            ev['tail_remaining'] = max(0, post_trig_chunks - chunks_since_last)
                else:
                    ev['tail_remaining'] -= 1

            # Flush when the post-trigger tail is fully captured
            if ev['ended'] and ev['tail_remaining'] <= 0:
                trim_end = min(ev['last_above_idx'] + 1 + post_trig_chunks, len(ev['buf']))
                self._start_flush(ev['buf'][:trim_end], output_dir, filename_prefix,
                                  filename_suffix, sample_rate=sample_rate,
                                  onset_time=ev['onset_time'])
                to_remove.append(ev)
                continue

            # Force-split events longer than max_rec
            if len(ev['buf']) >= max_chunks:
                self._start_flush(ev['buf'], output_dir, filename_prefix,
                                  filename_suffix, sample_rate=sample_rate,
                                  onset_time=ev['onset_time'])
                to_remove.append(ev)

        for ev in to_remove:
            if ev in self._active_events:
                self._active_events.remove(ev)

    @property
    def is_recording(self) -> bool:
        return any(not ev['ended'] for ev in self._active_events)

    @staticmethod
    def _start_flush(buf_snapshot: list, output_dir: str,
                     prefix: str = '', suffix: str = '', sample_rate: int = SAMPLE_RATE,
                     onset_time=None):
        """Legacy daemon-thread launcher — delegates to
        `chirp.recording.writer.start_flush_thread`. Kept as a
        staticmethod on the class so the existing test monkeypatch
        (`monkeypatch.setattr(ThresholdRecorder, "_start_flush", ...)`)
        continues to take precedence.
        """
        _writer.start_flush_thread(
            buf_snapshot, output_dir, prefix, suffix,
            sample_rate=sample_rate, onset_time=onset_time,
        )
