"""RecordingEntity — per-stream data model and chunk ingestion.

Extracted from the monolith in the Phase 1 refactor (plan: c06). Owns
the audio pipeline (capture → filter → FFT → entropy → trigger) and
all the ring buffers that back the display. No Qt widgets live here —
the UI layer pulls data out via attribute access.

Post Phase 2/3 status:
  - c10: FFT overlap reset on stream start/stop.
  - c11: saturation measured from raw (pre-filter) signal.
  - c12: should_trigger computed upstream, decoupled from trigger_peak.
  - c14: ring-buffer cursors derived from a single sample counter.
  - c19: display and analysis FFT accumulators decoupled.
  - c21: ingest_chunk runs on a dedicated background thread.
"""

import datetime
import os
import threading

import numpy as np
import sounddevice as sd

import time

from chirp.audio import AudioCapture, WavFileCapture
from chirp.audio.clock import DisciplinedClock
from chirp.audio.devices import find_device_by_name, host_api_name
from chirp.audio.ringbuffer import AudioRing
from chirp.error_log import log as _err_log
from chirp.constants import (
    CHUNK_FRAMES,
    RING_SECONDS,
    DEFAULT_FREQ_HI,
    DEFAULT_FREQ_LO,
    DEFAULT_HOLD,
    DEFAULT_MAX_REC,
    DEFAULT_MIN_CROSS,
    DEFAULT_MIN_TOTAL_CROSS,
    DEFAULT_POST_TRIG,
    DEFAULT_PRE_TRIG,
    DEFAULT_THRESHOLD,
    DISPLAY_SECONDS,
    N_DISPLAY_ROWS,
    RECORDINGS_DIR,
    SAMPLE_RATE,
    SPEC_DB_MAX,
    SPEC_DB_MIN,
    SPECTROGRAM_NPERSEG,
)
from chirp.dsp import BandpassFilter, SpectrogramAccumulator
from chirp.dsp import analytic_envelope as _envelope
from chirp.dsp import normalized_spectral_entropy as _spectral_entropy
from chirp.recording.trigger import ThresholdRecorder
from chirp.recording import writer as _wav_writer


# ── Zero-run detector ──────────────────────────────────────────────────────
# Minimum length (seconds) of an exact-zero run in the raw captured
# signal before it is counted as inserted digital silence. A live analog
# input's noise floor guarantees nonzero LSBs, so exact-zero runs of a
# millisecond are not natural audio. Field-observed insertions are
# 2–8 ms; 1 ms catches them all with headroom against chance zeros.
# (A digital input — ADAT / S/PDIF — carrying true digital silence can
# legitimately trip this; the badge is informational and clearable.)
ZERO_RUN_MIN_SEC = 0.001


# ── Sidecar clock log ──────────────────────────────────────────────────────
# One CSV row per stream per ``clock_log_interval_sec`` (default 60 s):
# the capture-ring sample index, the derived capture time (and which
# clock derived it), and the raw wall clock. The audit trail for
# multi-week runs — any filename timestamp can be verified, or
# corrected, offline against this, whatever happens to the in-app
# clocks. ~80 bytes/min/stream ≈ 10 MB over 90 days.
CLOCK_LOG_FILENAME = 'chirp_clock_log.csv'
CLOCK_LOG_MAX_BYTES = 64 * 1024 * 1024   # hard cap; appends stop past this
_CLOCK_LOG_HEADER = ('utc_iso,stream,ring_sample_index,'
                     'derived_epoch,derived_source,wall_epoch\n')
# Serialises appends across writer-pool workers: two streams that share
# an output dir share one log file.
_clock_log_lock = threading.Lock()


def _append_clock_log_row(out_dir: str, row: str) -> None:
    """Append one audit row (runs on a writer-pool worker, never on the
    ingest thread). Creates the file with a header line; silently stops
    appending once ``CLOCK_LOG_MAX_BYTES`` is reached."""
    path = os.path.join(out_dir, CLOCK_LOG_FILENAME)
    with _clock_log_lock:
        os.makedirs(out_dir, exist_ok=True)
        exists = os.path.exists(path)
        if exists and os.path.getsize(path) > CLOCK_LOG_MAX_BYTES:
            return
        with open(path, 'a', encoding='utf-8', newline='') as f:
            if not exists:
                f.write(_CLOCK_LOG_HEADER)
            f.write(row)


class RecordingEntity:

    SUPPORTED_RATES = (8000, 16000, 22050, 44100, 48000, 96000)
    SUPPORTED_DISPLAY_SECONDS = (5.0, 10.0, 15.0, 20.0, 30.0, 60.0)

    def __init__(self, name: str = 'Recording 1', device_id=None, sample_rate=SAMPLE_RATE,
                 display_seconds=DISPLAY_SECONDS):
        self.name = name
        self.sample_rate = sample_rate
        self.display_seconds = float(display_seconds)

        # Derived sizes — n_cols is authoritative; total_samples derived from it to keep sync
        self._n_cols        = max(1, int(self.display_seconds * self.sample_rate / CHUNK_FRAMES))
        self._total_samples = self._n_cols * CHUNK_FRAMES

        # Device / channel
        self.device_id    = device_id
        self.channel_mode = 'Mono'
        self.trigger_mode = 'Average'

        # Input source: 'device' (live sounddevice input) or 'wav_file'
        # (feed a WAV through the pipeline for reproducible testing).
        self.input_source  = 'device'
        self.wav_file_path: str | None = None
        self.wav_loop      = True

        # #7: shared audio monitor (set by ChirpWindow via set_monitor).
        # Kept on the entity so every rebuilt capture (device change,
        # sample-rate change, WAV switch) gets re-wired automatically.
        self._monitor = None

        # Audio pipeline. ``_make_capture`` (re)creates ``self.ring``
        # sized to the current sample rate + channel count, then wires a
        # capture that writes into it. The DSP ingest thread is the ring's
        # sole consumer.
        self.ring       = None
        self.capture    = self._make_capture(channels=1)
        self.spec_acc   = SpectrogramAccumulator()
        self.spec_acc_r = SpectrogramAccumulator()
        # H1: streaming mode — events append to a StreamingWavWriter as
        # audio arrives instead of buffering up to max_rec in RAM.
        # (Streaming steps aside automatically in tests that monkeypatch
        # ThresholdRecorder._start_flush.)
        self.recorder   = ThresholdRecorder(streaming=True)
        self.bpf        = BandpassFilter(sample_rate=self.sample_rate)
        self.bpf_r      = BandpassFilter(sample_rate=self.sample_rate)

        # Analysis FFT params — default to display params. When they
        # differ from the display FFT, a separate accumulator is used
        # for spectral entropy / trigger (#12 / c19).
        self.analysis_nperseg = SPECTROGRAM_NPERSEG
        self.analysis_window  = 'hann'
        self._analysis_acc: SpectrogramAccumulator | None = None
        self._analysis_acc_r: SpectrogramAccumulator | None = None

        # Per-stream enable switch: a disabled stream stays fully
        # configured but is skipped by the bulk transport actions
        # (Start All Acq / Start All Rec) — the individual per-stream
        # buttons still work for deliberate one-off use. Persisted.
        self.stream_enabled = True

        # Per-stream parameter lock: when True the UI disables editing of
        # this stream's configuration (trigger, band filter, spectral
        # trigger, output, reference date, input device) to prevent
        # accidental changes. Display params and the audio monitor stay
        # editable, and transport (start/stop acq/rec) still works.
        # Unlocking is guarded by a confirmation naming the stream, so one
        # user can't silently unlock another's stream. Persisted.
        self.params_locked = False

        # Recognition color — a per-stream accent shown wherever the
        # stream appears (sidebar item, view-mode tile header). Empty
        # means "unassigned"; the UI fills it from the default palette by
        # list position (see chirp.constants.default_stream_color). A
        # user-picked color is stored verbatim and persisted.
        self.color = ''

        # Trigger params
        self.threshold     = DEFAULT_THRESHOLD
        self.min_cross_sec = DEFAULT_MIN_CROSS
        # Min total crossing: an event whose ACCUMULATED above-threshold
        # duration stays below this is discarded at finalize time
        # instead of being written/published (0 = keep everything).
        self.min_total_cross_sec = DEFAULT_MIN_TOTAL_CROSS
        self.hold_sec      = DEFAULT_HOLD
        self.pre_trig_sec  = DEFAULT_PRE_TRIG
        self.post_trig_sec = DEFAULT_POST_TRIG
        self.max_rec_sec   = DEFAULT_MAX_REC
        self.freq_filter_enabled = False
        self.freq_lo       = DEFAULT_FREQ_LO
        self.freq_hi       = DEFAULT_FREQ_HI
        # 2a: 'Triggered' (threshold state machine) or 'Continuous'
        # (record everything while REC is on; a new file every
        # max_rec_sec via the force-split machinery). Persisted.
        self.rec_mode      = 'Triggered'
        # 2b: force-trigger toggle. While True the detection mask is
        # all-True (manual segment); toggling off requests an immediate
        # flush, consumed on the ingest thread.
        self.force_rec_active     = False
        self._force_stop_requested = False

        # Spectral trigger params
        self.spectral_trigger_mode = 'Amplitude Only'  # 'Amplitude Only', 'Spectral Only', 'Amp AND Spectral', 'Amp OR Spectral'
        self.spectral_threshold    = 0.5                # entropy threshold (trigger when below)
        self.spectral_entropy      = 1.0                # current entropy value (display only)
        self.spectral_entropy_r    = 1.0                # right channel entropy (stereo)
        # 2c: entropy debounce — the spectral gate turns ON only after
        # entropy has stayed below the threshold continuously for this
        # long (seconds, chunk granularity); OFF immediately on a chunk
        # above threshold. 0 = instantaneous (legacy behavior).
        self.entropy_min_cross_sec = 0.0
        self._entropy_below_sec    = 0.0   # running consecutive-below time

        # Spectrogram display
        self.spec_nperseg = SPECTROGRAM_NPERSEG
        self.spec_window  = 'hann'
        self.freq_scale   = 'Mel'
        self.gain_db      = 0.0
        self.db_floor     = SPEC_DB_MIN
        self.db_ceil      = SPEC_DB_MAX
        self.display_freq_lo = 0.0
        self.display_freq_hi = float(self.sample_rate // 2)

        # Output
        self.output_dir = RECORDINGS_DIR
        self.filename_prefix = ''
        self.filename_suffix = ''
        self.ref_date = None  # datetime.date or None; when set, files go into day-subfolder
        self.dph_folder_prefix = ''  # optional prefix for day subfolder name
        # #50: ``output_dir`` validation stamp. The UI probes the path
        # at every user-visible entry point (browse / text edit /
        # config load) and writes the result here; the sidebar reads
        # these to surface the error without waiting for the writer
        # worker to fail. Default ``True`` so entities constructed in
        # tests (which may never go through the UI) aren't flagged.
        self.output_dir_valid = True
        self.output_dir_error: str | None = None

        # Ring buffers
        self.n_freq_bins    = SPECTROGRAM_NPERSEG // 2 + 1
        self.amp_buffer     = np.zeros(self._total_samples, dtype=np.float32)
        self.amp_buffer_r   = np.zeros(self._total_samples, dtype=np.float32)
        self.abs_amp_buffer   = np.zeros(self._total_samples, dtype=np.float32)
        self.abs_amp_buffer_r = np.zeros(self._total_samples, dtype=np.float32)
        self.spec_buffer  = np.full(
            (self.n_freq_bins, self._n_cols), SPEC_DB_MIN, dtype=np.float32)
        self.spec_buffer_r = np.full(
            (self.n_freq_bins, self._n_cols), SPEC_DB_MIN, dtype=np.float32)
        self.entropy_buffer = np.ones(self._n_cols, dtype=np.float32)
        # #32: per-sample indicator buffers. ``detect_mask_buffer`` is the
        # raw threshold mask (regardless of min_cross / hold gating) so
        # the display can show where the signal crossed the threshold at
        # all; ``record_mask_buffer`` is True wherever a sample was (or
        # will be) written to a saved WAV — including pre-trigger history
        # retroactively marked when an event opens, and the post-trigger
        # tail as it fills.
        self.detect_mask_buffer = np.zeros(self._total_samples, dtype=bool)
        self.record_mask_buffer = np.zeros(self._total_samples, dtype=bool)
        # True wherever a finished event was DISCARDED by the min-total-
        # crossing filter (no WAV written) — drives the red overlay on the
        # record strip so a dropped event is visually distinct from a
        # saved one.
        self.discard_mask_buffer = np.zeros(self._total_samples, dtype=bool)
        # Single cumulative sample counter — both ring-buffer cursors
        # are derived from it so they cannot drift apart when chunk
        # size differs from CHUNK_FRAMES (#20 / c14).
        self._samples_total = 0
        self.write_head = 0
        self.col_head   = 0

        # Display state
        self.saturated  = False   # True when current chunk contains clipped audio
        # #28: latched "has ever saturated since last reset" flag —
        # sticky indicator so brief clips that happen while the user
        # isn't watching are still surfaced after the fact. Cleared
        # explicitly via clear_saturation_flag().
        self.saturated_ever = False
        # #44: surface ingestion thread errors. The old _ingest_loop
        # swallowed every exception into traceback.print_exc() — in a
        # GUI build nobody sees stdout, and a recurring DSP error would
        # silently starve the display without any user-visible signal.
        # These counters back the sidebar `!` error badge.
        self.ingest_error_count       = 0      # transient per-tick counter
        self.ingest_error_count_total = 0      # session-wide monotonic
        self.has_ever_ingest_errored  = False
        self.last_ingest_error: str | None = None  # short str(exc) for tooltip
        # Zero-run detector: exact-zero runs >= ZERO_RUN_MIN_SEC in the
        # raw captured signal. A live analog input never produces exact
        # 0.0 floats (the ADC noise floor guarantees nonzero LSBs), so
        # such runs are digital silence inserted upstream — observed in
        # the field as periodic 2–8 ms zero fills latched into one
        # endpoint's Windows capture session (split-stereo L/R streams
        # saw time-locked identical runs; PortAudio raised NO status
        # flags, so signal-level detection is the only in-app tell).
        # Cleared by restarting acquisition on ALL streams sharing the
        # endpoint. Counters mirror the capture drop-stat contract.
        # Only live-device captures are scanned — WAV playback pads
        # legit zeros (loop seam) and may contain true silence.
        self.zero_run_count       = 0      # transient per-tick
        self.zero_run_count_total = 0      # session-wide monotonic
        self.has_ever_zero_run    = False
        self.zero_run_longest     = 0      # samples, session max
        self._zero_carry          = 0      # trailing zero run in progress
        self._zero_carry_counted  = False  # carry already counted
        self.amp_ylim   = 1.05    # amplitude y-axis max (persists across mode switches)
        # Amplitude-plot Y scale: 'linear' (raw envelope, 0..amp_ylim)
        # or 'log' (20*log10, AMP_DB_MIN..AMP_DB_MAX). User-toggled via
        # right-click on the amp plot. Default is log because the
        # envelope ranges over many orders of magnitude — pure tones at
        # -40 dB are invisible on a 0..1 linear scale.
        self.amp_scale  = 'log'
        self.display_mode = 'Spectrogram'  # 'Spectrogram', 'Waveform', or 'Both'

        # Runtime
        self.acq_running = False
        self.rec_enabled = False
        self._ingest_stop = threading.Event()
        self._ingest_thread: threading.Thread | None = None
        # #53: serialise DSP-state mutation against the ingest thread.
        # ``ingest_chunk`` holds this for the duration of one chunk;
        # every rebuild path (buffer resize, FFT-param change, filter
        # reset, capture swap) acquires it so we never tear down state
        # the ingest thread is concurrently reading from. Rebuild
        # paths that tear down the ingest thread first (change_device,
        # change_sample_rate, use_wav_file, stop_acq) still take the
        # lock as belt-and-suspenders.
        self._dsp_lock = threading.Lock()
        # #53: if ``stop_acq`` / ``_stop_ingest_and_flush`` times out
        # waiting for the ingest thread, latch the failure so
        # ``start_acq`` refuses to spawn a second ingest thread
        # (silent double-spawn was the original bug). Cleared by
        # ``clear_error_flag()``.
        self._ingest_join_failed = False
        # M3: latched once when check_ingest_alive finds the ingest
        # thread dead while acq_running claims otherwise.
        self._ingest_dead_latched = False
        # M5: wall-clock ↔ sample-counter anchor, stamped at start_acq
        # (the ring is empty then, so "now" is the capture time of the
        # next sample). Used to derive the capture-time wall clock of
        # each ingested chunk for the recorder's onset timestamps.
        self._wall_anchor_time: datetime.datetime | None = None
        self._wall_anchor_samples = 0
        # Sidecar clock-log cadence (see module docs above). The 0.0
        # deadline means the first chunk of a session logs immediately;
        # ``start_acq`` re-zeroes it so every session opens with a row.
        self.clock_log_interval_sec = 60.0
        self._clock_log_next_t = 0.0
        # M8: running max of the per-chunk trigger ENVELOPE peak — the
        # exact signal ``ingest_chunk`` compares against the threshold.
        # Auto-calibrate polls this via ``consume_env_peak`` so the
        # threshold is derived from the same statistic the trigger
        # uses, not from |filtered| sampled at the UI tick rate.
        self._env_peak_acc = -1.0
        # TODO#1 (RDP): capture-stall watchdog state. ``device_name_hint``
        # / ``device_hostapi_hint`` are stamped whenever a live-device
        # capture opens, so recovery can re-resolve the device by NAME —
        # PortAudio indices shift when Windows audio endpoints churn.
        self.device_name_hint    = ''
        self.device_hostapi_hint = ''
        self.capture_stalled     = False   # latched by check_capture_stalled
        self._stall_wt           = -1      # last observed ring write_total
        self._stall_wt_t         = 0.0     # monotonic time of last advance
        self.recovery_count      = 0       # successful auto-reconnects

        # Freq mapping
        self.freq_map_idx_floor = None
        self.freq_map_frac      = None
        self.display_freqs      = None
        self.rebuild_freq_mapping()

    # ── Display reset ────────────────────────────────────────────────────

    def reset_display(self):
        """Clear display buffer CONTENTS only.

        Must never touch ``_samples_total`` or the ring cursors: the
        sample counter is the timestamp clock — ``chunk_end_wall`` (and
        every WAV filename onset) is derived as ``_wall_anchor_time +
        (_samples_total - _wall_anchor_samples) / sample_rate``. Zeroing
        it mid-acquisition snapped all subsequent filename timestamps
        back to the start_acq anchor (observed in the field as a
        consistent ~1-day-backwards jump after a Start Acq click on an
        already-running session). The cursors are re-derived from the
        counter on every ingest, so leaving them alone stays coherent.
        """
        self.amp_buffer[:]    = 0.0
        self.amp_buffer_r[:]  = 0.0
        self.spec_buffer[:]   = SPEC_DB_MIN
        self.spec_buffer_r[:] = SPEC_DB_MIN
        self.entropy_buffer[:] = 1.0
        self.detect_mask_buffer[:] = False
        self.record_mask_buffer[:] = False
        self.discard_mask_buffer[:] = False

    # ── #45: safe teardown helpers ─────────────────────────────────────

    def _effective_output_dir(self) -> str:
        """Return the output directory the next WAV would land in —
        includes the ``ref_date`` day-subfolder when that's configured.
        Centralised so teardown flushes and ``ingest_chunk`` compute
        the same path.

        #51: ``dph_folder_prefix`` is user-editable — sanitize it so a
        prefix of ``../../escape`` can't walk outside ``output_dir``.

        The day-subfolder is ``<prefix>_<days>`` — the prefix and the
        day number are always joined by a single underscore separator
        (added here, not typed by the user). ``_sanitize_token`` already
        trims any trailing underscore, so a prefix of ``day`` or ``day_``
        both yield ``day_0``, ``day_1``, …; an empty prefix yields the
        bare day number (``0``, ``1``, …).
        """
        out_dir = self.output_dir
        if self.ref_date is not None:
            from chirp.recording.writer import _sanitize_token
            days = (datetime.date.today() - self.ref_date).days
            prefix_s = _sanitize_token(self.dph_folder_prefix)
            sub = f'{prefix_s}_{days}' if prefix_s else f'{days}'
            out_dir = os.path.join(out_dir, sub)
        return out_dir

    def _flush_active_events(self, reason: str = '') -> int:
        """#45: flush every still-open event to disk via the recorder's
        ``flush_all``. Returns the number of events flushed. Safe to
        call when the recorder has no active events (returns 0).

        Callers must have already stopped the ingest thread — otherwise
        concurrent mutation of ``_active_events`` could race the flush.
        """
        rec = getattr(self, 'recorder', None)
        if rec is None:
            return 0
        try:
            return rec.flush_all(
                output_dir       = self._effective_output_dir(),
                filename_prefix  = self.filename_prefix,
                filename_suffix  = self.filename_suffix,
                sample_rate      = self.sample_rate,
                filename_stream  = self.name,
                reason           = reason,
            )
        except Exception:
            # A flush failure must not prevent the rest of teardown —
            # the writer pool's own error counter surfaces the
            # failure through the sidebar `!` badge (#44).
            import traceback
            traceback.print_exc()
            return 0

    def _stop_ingest_and_flush(self, reason: str = '') -> None:
        """#45: stop the ingest thread, drain any queued chunks, and
        flush still-open trigger events. Idempotent — safe to call
        from any teardown path (``stop_acq``, ``change_device``,
        ``change_sample_rate``, ``use_wav_file``, ``close``, or
        ``_load_settings`` during its per-entity loop).

        Must be called *before* closing the capture / rebuilding
        buffers. Order matters: stop ingestion → drain queue → flush
        trigger → rebuild. Reordering risks either a concurrent
        ``_active_events`` mutation or dropping the last chunk of
        audio that held the post-trigger tail.
        """
        if self._ingest_thread is not None:
            self._ingest_stop.set()
            # #53: the old 2-second join let a stuck ingest thread
            # silently survive — a subsequent start_acq would then
            # spawn a *second* thread draining the same queue. Use a
            # longer join window (ingest_chunk is bounded by the
            # chunk duration + filter / FFT / trigger runtime, all
            # well under 10 s) and if the thread still hasn't
            # exited, LATCH the failure so start_acq refuses to
            # double-spawn.
            self._ingest_thread.join(timeout=10.0)
            if self._ingest_thread.is_alive():
                self._ingest_join_failed   = True
                self.has_ever_ingest_errored = True
                self.last_ingest_error = (
                    'ingest thread failed to stop within 10s — '
                    'acquisition locked until app restart')
                print(f'[Chirp] {self.name}: ingest thread stuck; '
                      f'not respawning to avoid double-ingest')
                # Keep the (stuck) reference so start_acq's guard
                # fires. It's a daemon thread so interpreter exit
                # will still kill it.
            else:
                self._ingest_thread = None
        # Discard ring stragglers the stopped ingest thread didn't consume
        # so a later restart doesn't replay stale audio.
        self.ring.drain_unread()
        self._flush_active_events(reason=reason)

    # ── #28 / #29: sticky session flags ───────────────────────────────────

    def clear_saturation_flag(self) -> None:
        """Clear the sticky ``saturated_ever`` flag (#28).

        The transient ``saturated`` flag is unaffected — it will turn
        back on immediately if the next chunk is still clipping.
        """
        self.saturated_ever = False

    def clear_drop_flag(self) -> None:
        """Clear the sticky dropped-audio stats on the attached capture
        (#29). Safe to call whether or not the capture exposes the
        ``reset_drop_stats`` method (legacy / test captures may not).
        """
        cap = getattr(self, 'capture', None)
        if cap is not None and hasattr(cap, 'reset_drop_stats'):
            cap.reset_drop_stats()

    def clear_error_flag(self) -> None:
        """Clear the sticky error stats (#44).

        Resets both the entity's ingest error counters and any
        capture-layer error stats (``os_drop_count_total`` /
        ``has_ever_os_dropped``, ``open_error``). The writer-pool
        error stats are global and cleared separately by the window.
        """
        self.ingest_error_count       = 0
        self.ingest_error_count_total = 0
        self.has_ever_ingest_errored  = False
        self.last_ingest_error        = None
        self.zero_run_count           = 0
        self.zero_run_count_total     = 0
        self.has_ever_zero_run        = False
        self.zero_run_longest         = 0
        # #53: also clear the stuck-ingest-thread latch so the user
        # can try acquisition again after restarting the device.
        self._ingest_join_failed      = False
        self._ingest_dead_latched     = False
        cap = getattr(self, 'capture', None)
        if cap is not None and hasattr(cap, 'reset_error_stats'):
            cap.reset_error_stats()

    def consume_ingest_error_count(self) -> int:
        """Return & clear the transient ingest-error counter. Polled
        once per UI tick so the sidebar can flash on new errors."""
        n = self.ingest_error_count
        self.ingest_error_count = 0
        return n

    def consume_zero_run_count(self) -> int:
        """Return & clear the transient zero-run counter. Polled once
        per UI tick. Emits a throttled ``zero_run`` log line here (off
        the ingest thread's hot path is fine, but the UI tick keeps
        logging cadence uniform with the other capture-stat consumes).
        """
        n = self.zero_run_count
        self.zero_run_count = 0
        if n:
            ms = self.zero_run_longest / max(1, self.sample_rate) * 1000.0
            _err_log('zero_run', self.name,
                     f'inserted-silence detected: {n} exact-zero run(s) '
                     f'>= {ZERO_RUN_MIN_SEC*1000:.0f} ms in captured audio '
                     f'(longest {ms:.1f} ms, cumulative='
                     f'{self.zero_run_count_total}) — restart acquisition '
                     f'on ALL streams of this input device to clear')
        return n

    def _detect_zero_runs(self, raw_chunk: np.ndarray) -> None:
        """Count exact-zero runs >= ``ZERO_RUN_MIN_SEC`` in the raw
        chunk (all captured channels simultaneously zero). Runs are
        counted once, at the moment they reach the threshold — a run
        spanning many chunks (via the ``_zero_carry`` carry-over) still
        counts exactly once. Live-device captures only.
        """
        if self.input_source != 'device':
            return
        if raw_chunk.ndim == 2:
            z = np.all(raw_chunk == 0, axis=1)
        else:
            z = (raw_chunk == 0)
        n = z.shape[0]
        if n == 0:
            return
        if not z.any():
            self._zero_carry = 0
            self._zero_carry_counted = False
            return
        min_run = max(2, int(ZERO_RUN_MIN_SEC * self.sample_rate))
        d = np.diff(np.concatenate(([0], z.view(np.int8), [0])))
        starts = np.nonzero(d == 1)[0]
        ends   = np.nonzero(d == -1)[0]
        new_runs = 0
        for i in range(len(starts)):
            length = int(ends[i] - starts[i])
            if i == 0 and starts[0] == 0 and self._zero_carry > 0:
                total   = self._zero_carry + length
                counted = self._zero_carry_counted
            else:
                total   = length
                counted = False
            if total > self.zero_run_longest:
                self.zero_run_longest = total
            if total >= min_run and not counted:
                new_runs += 1
                counted = True
            if ends[i] == n:
                # Run touches chunk end — carry it (and whether it was
                # already counted) into the next chunk.
                self._zero_carry = total
                self._zero_carry_counted = counted
        if ends[-1] != n:
            self._zero_carry = 0
            self._zero_carry_counted = False
        if new_runs:
            self.zero_run_count       += new_runs
            self.zero_run_count_total += new_runs
            self.has_ever_zero_run     = True

    def consume_env_peak(self):
        """M8: return the max trigger-envelope peak observed since the
        last call and reset the accumulator; None when no chunk has
        been ingested since. Called by the auto-calibrate timer — every
        chunk contributes, unlike the old 100 ms sampling of
        ``abs_amp_buffer`` which missed most chunks and measured
        |filtered| instead of the envelope the trigger compares."""
        p = self._env_peak_acc
        self._env_peak_acc = -1.0
        return p if p >= 0.0 else None

    # ── TODO#1 (RDP): capture-stall watchdog ──────────────────────────

    #: Seconds without a single new frame from the PortAudio callback
    #: before the capture is declared dead. Callbacks fire every ~23 ms
    #: (1024 frames @ 44.1 kHz), so anything past a couple of seconds is
    #: far beyond a scheduler hiccup. 5 s (was 2 s) rides out the
    #: transient endpoint churn a remote-desktop attach (AnyDesk / RDP)
    #: causes — tearing down a stream that would have resumed by itself
    #: is far more expensive than reacting three seconds later.
    CAPTURE_STALL_SECONDS = 5.0

    def check_capture_stalled(self) -> bool:
        """Return True when acquisition claims to run on a live device
        but the PortAudio callback has stopped delivering frames (the
        RDP connect/disconnect signature — Windows tore down or
        re-routed the audio endpoint). Latches ``capture_stalled`` and
        the sticky error badge on first detection. Polled per UI tick.
        """
        if (not self.acq_running or self.input_source != 'device'
                or self.ring is None):
            self._stall_wt = -1
            return self.capture_stalled
        wt = self.ring.write_total
        now = time.monotonic()
        if wt != self._stall_wt:
            self._stall_wt = wt
            self._stall_wt_t = now
            # Frames resumed on their own (transient churn — e.g. a
            # remote-desktop attach briefly pausing the endpoint):
            # un-latch so the recovery worker doesn't tear down a
            # stream that is healthy again.
            if self.capture_stalled:
                self.capture_stalled = False
                self.last_ingest_error = (
                    'audio device resumed on its own — no reconnect needed')
                print(f'[Chirp] {self.name}: capture resumed on its own; '
                      f'recovery cancelled')
                _err_log('capture_dead', self.name,
                         'frames resumed before reconnect — transient stall')
            return self.capture_stalled
        if (not self.capture_stalled
                and now - self._stall_wt_t >= self.CAPTURE_STALL_SECONDS):
            self.capture_stalled = True
            self.has_ever_ingest_errored = True
            self.last_ingest_error = (
                'audio device stopped delivering samples (device lost / '
                'RDP session change?) — attempting reconnect')
            print(f'[Chirp] {self.name}: capture stalled — no frames for '
                  f'{self.CAPTURE_STALL_SECONDS:.0f}s; starting recovery')
            _err_log('capture_dead', self.name,
                     f'no frames for {self.CAPTURE_STALL_SECONDS:.0f}s — '
                     f'device lost (RDP session change?); auto-reconnect '
                     f'engaged')
        return self.capture_stalled

    def attempt_capture_recovery(self) -> bool:
        """One auto-reconnect attempt for a stalled live-device capture
        (throttled by the caller). Flushes in-flight events, closes the
        dead stream, re-resolves the device BY NAME, reopens, and
        restarts acquisition. ``rec_enabled`` is preserved so recording
        resumes automatically. Returns True on success."""
        if not self.capture_stalled:
            return True
        was_rec = self.rec_enabled
        # Tear down the dead pipeline. acq_running must go False so
        # start_acq will rearm (and so the watchdog stays quiet while
        # we work).
        self.acq_running = False
        self._stop_ingest_and_flush(reason='capture_recovery')
        # A stalled device stalls every stream sharing its endpoint, so
        # retire the whole shared stream — otherwise the replacement
        # captures would re-attach to the dead session still held open
        # by the sibling streams that haven't recovered yet.
        try:
            if hasattr(self.capture, 'mark_stream_dead'):
                self.capture.mark_stream_dead()
        except Exception:
            pass
        try:
            self.capture.close()
        except Exception:
            pass
        # Re-resolve by name — indices shift when endpoints churn.
        dev_id = self.device_id
        if self.device_name_hint:
            resolved, _warn = find_device_by_name(
                self.device_name_hint, self.device_hostapi_hint)
            if resolved is not None:
                dev_id = resolved
        self.device_id = dev_id
        need_ch = 2 if self.channel_mode != 'Mono' else 1
        self.capture = self._make_capture(channels=need_ch)
        if not self.capture.valid:
            return False   # stays stalled; caller retries later
        self.rec_enabled = was_rec
        self.start_acq()
        self.capture_stalled = False
        self._stall_wt = -1
        self.recovery_count += 1
        self.last_ingest_error = (
            f'audio device reconnected (auto-recovery #{self.recovery_count}'
            f') — samples during the outage were lost')
        print(f'[Chirp] {self.name}: capture recovered '
              f'(#{self.recovery_count})')
        _err_log('capture_dead', self.name,
                 f'device reconnected (auto-recovery #{self.recovery_count})')
        return True

    def check_ingest_alive(self) -> None:
        """M3: latch a distinct error if acquisition claims to be
        running but the ingest thread is dead (a BaseException escaped
        the per-chunk guard in ``_ingest_loop``). Without this, the
        capture ring silently overruns and the only symptom is a drop
        badge with a misleading "reduce streams" tooltip. Polled from
        the UI tick; latches once per death."""
        t = self._ingest_thread
        if (self.acq_running and t is not None and not t.is_alive()
                and not self._ingest_stop.is_set()
                and not getattr(self, '_ingest_dead_latched', False)):
            self._ingest_dead_latched = True
            self.has_ever_ingest_errored = True
            self.ingest_error_count_total += 1
            self.last_ingest_error = (
                'ingest thread died — audio is NOT being processed; '
                'stop and restart acquisition')
            print(f'[Chirp] {self.name}: ingest thread died unexpectedly')
            _err_log('ingest', self.name, 'ingest thread died unexpectedly '
                     '(BaseException escaped the per-chunk guard)')

    # ── Freq mapping ──────────────────────────────────────────────────────

    def rebuild_freq_mapping(self):
        n_src = self.n_freq_bins
        n_dst = N_DISPLAY_ROWS
        freqs_src = np.linspace(0, self.sample_rate / 2, n_src)
        f_lo = max(0.0, self.display_freq_lo)
        f_hi = min(float(self.sample_rate / 2), self.display_freq_hi)
        if f_hi <= f_lo:
            f_hi = float(self.sample_rate / 2)
        scale = self.freq_scale
        if scale == 'Log':
            f_min = max(f_lo, freqs_src[1], 20.0)
            dst_freqs = np.logspace(np.log10(f_min), np.log10(f_hi), n_dst)
        elif scale == 'Mel':
            mel_lo = 2595.0 * np.log10(1.0 + max(f_lo, 20.0) / 700.0)
            mel_hi = 2595.0 * np.log10(1.0 + f_hi / 700.0)
            mels   = np.linspace(mel_lo, mel_hi, n_dst)
            dst_freqs = 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
        else:
            dst_freqs = np.linspace(f_lo, f_hi, n_dst)
        frac_idx = np.interp(dst_freqs, freqs_src, np.arange(n_src))
        self.freq_map_idx_floor = np.floor(frac_idx).astype(int).clip(0, n_src - 2)
        self.freq_map_frac      = (frac_idx - self.freq_map_idx_floor).astype(np.float32)
        self.display_freqs      = dst_freqs

    def resample_spec(self, spec_buffer: np.ndarray) -> np.ndarray:
        fl = self.freq_map_idx_floor
        fr = self.freq_map_frac
        out = spec_buffer[fl] * (1.0 - fr)[:, None] + spec_buffer[fl + 1] * fr[:, None]
        out += self.gain_db
        return out

    # ── FFT param change ──────────────────────────────────────────────────

    def change_fft_params(self, nperseg: int, window: str):
        # #53: serialise against the ingest thread — this mutates
        # spec_acc, spec_buffer, n_freq_bins which ingest_chunk reads.
        with self._dsp_lock:
            self.spec_nperseg = nperseg
            self.spec_window  = window
            # Fresh accumulators start un-primed — c10 / #14.
            self.spec_acc   = SpectrogramAccumulator(nperseg, window)
            self.spec_acc_r = SpectrogramAccumulator(nperseg, window)
            self.n_freq_bins  = nperseg // 2 + 1
            self.spec_buffer  = np.full(
                (self.n_freq_bins, self._n_cols), SPEC_DB_MIN, dtype=np.float32)
            self.spec_buffer_r = np.full(
                (self.n_freq_bins, self._n_cols), SPEC_DB_MIN, dtype=np.float32)
            self.rebuild_freq_mapping()
            # Rebuild analysis split when display params change (#12 / c19).
            self._rebuild_analysis_split()

    def change_analysis_fft_params(self, nperseg: int, window: str):
        """Change the analysis FFT parameters independently of display (#12 / c19).

        When (nperseg, window) matches the display FFT, the analysis
        path reuses the display accumulator (zero overhead). Otherwise
        a dedicated analysis accumulator is created.
        """
        # #53: serialise against the ingest thread — this mutates
        # _analysis_acc which ingest_chunk reads via the analysis_acc
        # property.
        with self._dsp_lock:
            self.analysis_nperseg = nperseg
            self.analysis_window  = window
            self._rebuild_analysis_split()

    def _rebuild_analysis_split(self):
        """Create or destroy the dedicated analysis accumulator.

        Called whenever display or analysis FFT params change. When both
        sets match, the analysis path reuses `spec_acc` / `spec_acc_r`
        (shared mode). When they differ, private accumulators are created
        so entropy computation runs at its own resolution.
        """
        if (self.analysis_nperseg == self.spec_nperseg
                and self.analysis_window == self.spec_window):
            self._analysis_acc   = None
            self._analysis_acc_r = None
        else:
            self._analysis_acc   = SpectrogramAccumulator(
                self.analysis_nperseg, self.analysis_window)
            self._analysis_acc_r = SpectrogramAccumulator(
                self.analysis_nperseg, self.analysis_window)

    @property
    def analysis_acc(self) -> SpectrogramAccumulator:
        """Return the accumulator used for spectral entropy / trigger."""
        return self._analysis_acc if self._analysis_acc is not None else self.spec_acc

    @property
    def analysis_acc_r(self) -> SpectrogramAccumulator:
        """Return the right-channel analysis accumulator."""
        return self._analysis_acc_r if self._analysis_acc_r is not None else self.spec_acc_r

    # ── Capture factory ───────────────────────────────────────────────────

    def _make_capture(self, channels: int):
        """Return a capture object matching the current ``input_source``.

        Mirrors the ``AudioCapture`` contract so the rest of the
        pipeline doesn't care whether samples come from a live device
        or a WAV file.
        """
        # (Re)create the capture ring sized to the current sample rate and
        # channel count. A fresh ring per capture rebuild guarantees no
        # stale audio or mismatched channel width survives a device / SR /
        # source switch.
        cap_frames = max(CHUNK_FRAMES * 8, int(RING_SECONDS * self.sample_rate))
        self.ring = AudioRing(cap_frames, channels=channels)
        # Disciplined timestamp clock, born with the ring so its sample
        # coordinates are the ring's absolute frame counts. Only the
        # live-device capture feeds it observations; WAV playback keeps
        # the coarse start_acq-anchor fallback (its pacing is synthetic).
        self.clock = DisciplinedClock(self.sample_rate)
        self._clock_steps_seen = 0
        if self.input_source == 'wav_file' and self.wav_file_path:
            cap = WavFileCapture(self.ring, self.wav_file_path,
                                 channels=channels, loop=self.wav_loop,
                                 name=self.name)
        else:
            cap = AudioCapture(self.ring, device=self.device_id,
                               channels=channels, samplerate=self.sample_rate,
                               name=self.name, clock=self.clock)
            # TODO#1: remember the device NAME so the RDP-recovery
            # watchdog can re-resolve it after an endpoint churn
            # (PortAudio indices are not stable across churn).
            try:
                if self.device_id is not None:
                    info = sd.query_devices(self.device_id)
                    self.device_name_hint    = info.get('name', '') or ''
                    self.device_hostapi_hint = host_api_name(info)
            except Exception:
                pass
        # Re-wire the monitor on every new capture so a device / SR /
        # WAV-file switch doesn't silently drop the loopback (#7).
        if self._monitor is not None:
            try:
                cap.set_monitor(self._monitor, id(self), self._monitor_channel())
            except Exception:
                pass
        return cap

    def _monitor_channel(self):
        """Which capture channel the monitor loopback should play,
        honoring ``channel_mode`` so the audio matches the spectrogram /
        amplitude plots: 1 = right, 0 = left / mono, None = both
        (stereo). Mirrors the display/record selection in
        ``process_chunk`` — without it a 'Right' stream feeds both
        columns and the mono monitor averages the left (a *different*
        stream when two streams split one stereo input)."""
        if self.channel_mode == 'Right':
            return 1
        if self.channel_mode == 'Stereo':
            return None
        return 0

    def set_monitor(self, monitor) -> None:
        """Attach (or detach with ``None``) the shared AudioMonitor.

        The monitor gates by ``source_id == id(entity)`` so a stream
        only reaches the output when this entity has been selected as
        the monitor source via ``monitor.set_source(id(entity))``.
        """
        self._monitor = monitor
        cap = self.capture
        if cap is not None:
            try:
                cap.set_monitor(
                    monitor,
                    id(self) if monitor is not None else None,
                    self._monitor_channel(),
                )
            except Exception:
                pass

    # ── Device change ─────────────────────────────────────────────────────

    def change_device(self, device_id, channels):
        was_running = self.acq_running
        if was_running:
            self.capture.pause()
            self.acq_running = False
        # #45: flush in-flight events before tearing down the capture.
        # Preserves rec_enabled so the caller's recording state
        # survives the swap.
        self._stop_ingest_and_flush(reason='change_device')
        self.capture.close()
        # Defensive: discard any frames the callback wrote into the ring
        # between pause and close (the helper already drained it).
        self.ring.drain_unread()
        self.device_id = device_id
        self.input_source = 'device'
        self.capture = self._make_capture(channels=channels)
        if not self.capture.valid:
            self.acq_running = False
            return False
        if was_running:
            # Re-spin the ingest thread via start_acq so the new
            # capture is consumed.
            self.start_acq()
        return True

    def use_wav_file(self, path: str, loop: bool = True) -> tuple[bool, str | None]:
        """Switch the input source to a WAV file.

        Reads the file's sample rate and channel count; if the rate
        differs from the current session rate, the whole pipeline is
        rebuilt at the file's rate (same path as a live SR change).

        Returns ``(ok, warning)``. ``ok`` is False when the file could
        not be opened; a warning string is returned when the session
        sample rate had to change to match the file.
        """
        was_running = self.acq_running
        if was_running:
            self.capture.pause()
            self.acq_running = False
        # #45: flush in-flight events before swapping the input source.
        self._stop_ingest_and_flush(reason='use_wav_file')
        self.capture.close()
        self.ring.drain_unread()

        self.input_source  = 'wav_file'
        self.wav_file_path = path
        self.wav_loop      = loop

        need_ch = 2 if self.channel_mode != 'Mono' else 1
        probe = self._make_capture(channels=need_ch)
        if not probe.valid:
            # #49: do NOT fall back to the live device. The user
            # explicitly requested WAV replay; silently switching to
            # the default microphone produces "wrong source"
            # data-corruption — a researcher analyses what they think
            # is the canned clip but is actually mic hiss. Keep the
            # invalid capture in place — its ``open_error`` is
            # surfaced via the sidebar's `!` badge through the
            # existing ``_update_error_sticky`` plumbing.
            self.capture = probe  # invalid, but its open_error is set
            err = f"Could not open WAV file: {path}"
            self.last_ingest_error = (probe.open_error or err)[:200]
            self.has_ever_ingest_errored = True
            return False, err

        warning = None
        # #54: WAV has more channels than the session is configured to
        # use → silently truncated. Surface it so the user knows.
        if probe.file_channels > need_ch:
            warning = (f"WAV file has {probe.file_channels} channels but "
                       f"session is configured for {need_ch} — "
                       f"channels {need_ch + 1}–{probe.file_channels} "
                       f"will be ignored")
            # Latch a sticky warning on the entity so the sidebar can
            # surface it past the initial load modal.
            self.last_ingest_error = warning[:200]
            self.has_ever_ingest_errored = True

        file_sr = probe.file_sample_rate
        if file_sr and file_sr != self.sample_rate and file_sr in self.SUPPORTED_RATES:
            probe.close()
            # change_sample_rate will call _make_capture again with the
            # new rate, reusing the WAV source we just configured.
            self.change_sample_rate(file_sr)
            sr_warning = (f"Session sample rate changed to {file_sr} Hz to "
                          f"match WAV file")
            warning = f"{warning}; {sr_warning}" if warning else sr_warning
        elif file_sr and file_sr != self.sample_rate:
            sr_warning = (f"WAV file sample rate ({file_sr} Hz) is not a "
                          f"supported session rate — resampling is not "
                          f"performed; timing will be off")
            warning = f"{warning}; {sr_warning}" if warning else sr_warning
            self.capture = probe
        else:
            self.capture = probe

        if was_running and self.capture.valid:
            # start_acq re-spins the ingest thread we stopped above.
            self.start_acq()
        return True, warning

    # ── Sample rate change ──────────────────────────────────────────────

    def change_sample_rate(self, new_rate: int):
        if new_rate == self.sample_rate:
            return
        was_running = self.acq_running
        if was_running:
            self.capture.pause()
            self.acq_running = False
        # #45: flush in-flight events at the OLD sample rate — WAV
        # playback would otherwise resume at the new rate with stale
        # samples appended to an open event and render a garbled file.
        self._stop_ingest_and_flush(reason='change_sample_rate')
        self.capture.close()
        self.ring.drain_unread()

        # TODO#7: display range follows the Nyquist. If the user had the
        # high limit AT the old Nyquist (the default full-range view),
        # follow the new Nyquist in both directions; a user-narrowed
        # range is preserved and only clamped down when it would exceed
        # the new Nyquist.
        old_nyq = float(self.sample_rate // 2)
        new_nyq = float(new_rate // 2)
        was_full_range = self.display_freq_hi >= old_nyq - 1.0

        self.sample_rate = new_rate
        self._n_cols        = max(1, int(self.display_seconds * new_rate / CHUNK_FRAMES))
        self._total_samples = self._n_cols * CHUNK_FRAMES
        if was_full_range:
            self.display_freq_hi = new_nyq
        else:
            self.display_freq_hi = min(self.display_freq_hi, new_nyq)

        # Rebuild buffers
        self.amp_buffer       = np.zeros(self._total_samples, dtype=np.float32)
        self.amp_buffer_r     = np.zeros(self._total_samples, dtype=np.float32)
        self.abs_amp_buffer   = np.zeros(self._total_samples, dtype=np.float32)
        self.abs_amp_buffer_r = np.zeros(self._total_samples, dtype=np.float32)
        self.spec_buffer  = np.full(
            (self.n_freq_bins, self._n_cols), SPEC_DB_MIN, dtype=np.float32)
        self.spec_buffer_r = np.full(
            (self.n_freq_bins, self._n_cols), SPEC_DB_MIN, dtype=np.float32)
        self.entropy_buffer = np.ones(self._n_cols, dtype=np.float32)
        self.detect_mask_buffer = np.zeros(self._total_samples, dtype=bool)
        self.record_mask_buffer = np.zeros(self._total_samples, dtype=bool)
        self.discard_mask_buffer = np.zeros(self._total_samples, dtype=bool)
        self._samples_total = 0
        self.write_head = 0
        self.col_head   = 0
        # The counter reset above is legitimate here (sample counts at
        # the old rate are dimensionally wrong at the new one), but the
        # wall anchor must die with it — a stale anchor paired with a
        # zeroed counter is exactly the backwards-timestamp bug. start_acq
        # (called below when was_running, or later by the user) stamps a
        # fresh pair before any chunk is ingested.
        self._wall_anchor_time = None
        self._wall_anchor_samples = 0

        # Rebuild filters and capture
        self.bpf   = BandpassFilter(sample_rate=new_rate)
        self.bpf_r = BandpassFilter(sample_rate=new_rate)
        # TODO#7: reset the FFT accumulators — their 4096-sample overlap
        # still holds OLD-rate audio. Without this, the first columns
        # after an SR change mix old samples into the new-rate FFT: the
        # spectrogram briefly shows the previous signal at wrongly
        # scaled frequencies, and the spectral trigger evaluates that
        # garbage as if primed.
        self.spec_acc.reset()
        self.spec_acc_r.reset()
        if self._analysis_acc is not None:
            self._analysis_acc.reset()
        if self._analysis_acc_r is not None:
            self._analysis_acc_r.reset()
        need_ch = 2 if self.channel_mode != 'Mono' else 1
        self.capture = self._make_capture(channels=need_ch)
        self.rebuild_freq_mapping()

        if was_running and self.capture.valid:
            # start_acq re-spins the ingest thread we stopped above.
            self.start_acq()

    # ── Display buffer change ──────────────────────────────────────────────

    def change_display_seconds(self, new_secs: float):
        if new_secs == self.display_seconds:
            return
        # #53: serialise buffer reallocation against the ingest
        # thread. Without the lock, ingest_chunk can mid-write a
        # half-freed buffer and crash with a shape mismatch.
        with self._dsp_lock:
            self.display_seconds = float(new_secs)
            self._n_cols        = max(1, int(self.display_seconds * self.sample_rate / CHUNK_FRAMES))
            self._total_samples = self._n_cols * CHUNK_FRAMES

            # Rebuild buffers
            self.amp_buffer       = np.zeros(self._total_samples, dtype=np.float32)
            self.amp_buffer_r     = np.zeros(self._total_samples, dtype=np.float32)
            self.abs_amp_buffer   = np.zeros(self._total_samples, dtype=np.float32)
            self.abs_amp_buffer_r = np.zeros(self._total_samples, dtype=np.float32)
            self.spec_buffer  = np.full(
                (self.n_freq_bins, self._n_cols), SPEC_DB_MIN, dtype=np.float32)
            self.spec_buffer_r = np.full(
                (self.n_freq_bins, self._n_cols), SPEC_DB_MIN, dtype=np.float32)
            self.entropy_buffer = np.ones(self._n_cols, dtype=np.float32)
            self.detect_mask_buffer = np.zeros(self._total_samples, dtype=bool)
            self.record_mask_buffer = np.zeros(self._total_samples, dtype=bool)
            self.discard_mask_buffer = np.zeros(self._total_samples, dtype=bool)
            # Preserve the sample clock — it anchors filename timestamps
            # (see reset_display). Re-derive the cursors for the NEW
            # ring geometry with the same modulo rule ingest_chunk uses,
            # so readers between now and the next chunk stay in bounds.
            self.write_head = self._samples_total % self._total_samples
            self.col_head   = (self._samples_total // CHUNK_FRAMES) % self._n_cols

    # ── Transport ─────────────────────────────────────────────────────────

    def _latch_start_failure(self, detail: str | None = None) -> None:
        """Surface a failed acquisition start on the sidebar `!` badge
        and in chirp_errors.log. Before this, a device that refused to
        (re)open made Start Acq silently do nothing — no badge, no log,
        no dialog (the classic symptom: Stop Acq inside an RDP session,
        then Start Acq is dead until the device is reselected)."""
        msg = (detail or getattr(self.capture, 'open_error', None)
               or 'unknown error')
        self.has_ever_ingest_errored = True
        self.ingest_error_count_total += 1
        self.last_ingest_error = f'could not start acquisition — {msg}'
        print(f'[Chirp] {self.name}: could not start acquisition — {msg}')
        _err_log('open', self.name, f'could not start acquisition — {msg}')

    def _reopen_capture(self):
        """Close the current capture and open a fresh one for the
        current device / channel mode. Returns the new capture.

        The shared stream is marked dead first: this path exists for
        the case where the existing PortAudio stream is unusable (a
        WDM-KS pin that refuses to restart, a churned endpoint), so the
        replacement capture must renegotiate a *new* stream rather than
        re-attach to the broken one still held by sibling streams.
        """
        try:
            if hasattr(self.capture, 'mark_stream_dead'):
                self.capture.mark_stream_dead()
        except Exception:
            pass
        try:
            self.capture.close()
        except Exception:
            pass
        need_ch = 2 if self.channel_mode != 'Mono' else 1
        self.capture = self._make_capture(channels=need_ch)
        return self.capture

    def start_acq(self):
        if self.acq_running:
            return
        # #53: refuse to start if a prior ingest thread got stuck
        # and a stop attempt couldn't join it. Without this guard
        # we would silently double-spawn — both threads draining
        # the same queue, ring-buffer writes racing, chunks
        # arbitrarily split between two pipelines.
        if (self._ingest_thread is not None
                and self._ingest_thread.is_alive()):
            self._ingest_join_failed   = True
            self.has_ever_ingest_errored = True
            self.last_ingest_error = (
                'cannot start acquisition — previous ingest thread '
                'is still alive (restart the app)')
            print(f'[Chirp] {self.name}: refusing to double-spawn '
                  f'ingest thread; prior thread still alive')
            return
        # L6 (RDP / WDM-KS): stop_acq CLOSES a live-device capture, so
        # a restart opens a fresh stream here. Resuming a paused stream
        # is not reliable: a stopped-but-open WDM-KS pin that idled
        # through an RDP session switch (endpoint churn, device power
        # management) can refuse to restart or restart as a zombie that
        # never delivers frames — a fresh open renegotiates the pin
        # from scratch. Callers that just built a valid capture
        # (change_device, attempt_capture_recovery, change_sample_rate)
        # skip the rebuild and go straight to resume below.
        if self.input_source == 'device' and not self.capture.valid:
            self._reopen_capture()
        if not self.capture.valid:
            self._latch_start_failure()
            return
        # Clear stale overlap so the first few FFT columns after a
        # restart don't mix zero-padding into the spectrum (#14).
        self.spec_acc.reset()
        self.spec_acc_r.reset()
        if self._analysis_acc is not None:
            self._analysis_acc.reset()
        if self._analysis_acc_r is not None:
            self._analysis_acc_r.reset()
        try:
            self.capture.resume()
        except Exception as exc:
            # A previously-paused stream can refuse to restart after
            # audio-endpoint churn. For live devices, fall back to one
            # full close + reopen before giving up.
            detail = f'{type(exc).__name__}: {exc}'
            if self.input_source != 'device':
                self._latch_start_failure(detail)
                return
            if not self._reopen_capture().valid:
                self._latch_start_failure()
                return
            try:
                self.capture.resume()
            except Exception as exc2:
                self._latch_start_failure(f'{type(exc2).__name__}: {exc2}')
                return
        self.acq_running = True
        # M5: anchor the capture wall clock to the sample counter.
        # The ring was drained at the last stop, so the next sample
        # ingested was captured essentially "now". Aware-UTC so a
        # DST transition mid-run can't skew the filename's local
        # token — conversion to local time happens once, at
        # filename-composition time (writer._compose_filename).
        self._wall_anchor_time = datetime.datetime.now(
            datetime.timezone.utc)
        self._wall_anchor_samples = self._samples_total
        # Open every acquisition session with a clock-log row.
        self._clock_log_next_t = 0.0
        # Start ingestion thread (#19 / c21).
        self._ingest_dead_latched = False
        self._ingest_stop.clear()
        t = threading.Thread(target=self._ingest_loop,
                             name=f'chirp-ingest-{self.name}',
                             daemon=True)
        self._ingest_thread = t
        t.start()

    def stop_acq(self):
        if self.acq_running:
            self.capture.pause()
            self.acq_running = False
            self.rec_enabled = False
            # Stop ingestion thread + flush in-flight events (#19 / c21,
            # #45). ``_stop_ingest_and_flush`` joins the ingest thread,
            # drains the queue, and calls ``recorder.flush_all`` so any
            # event still mid-recording or mid-tail lands on disk.
            self._stop_ingest_and_flush(reason='stop_acq')
            # L6 (RDP / WDM-KS): release the device while idle. A
            # paused-but-open stream still holds the device (exclusively
            # so under WDM-KS), and its kernel state can be invalidated
            # by RDP session churn or power management while idle — a
            # later stream.start() then fails on the stale handle.
            # start_acq opens a fresh stream instead of resuming.
            if self.input_source == 'device':
                try:
                    self.capture.close()
                except Exception:
                    pass
            self.bpf.reset()
            self.bpf_r.reset()
            self.spec_acc.reset()
            self.spec_acc_r.reset()
            if self._analysis_acc is not None:
                self._analysis_acc.reset()
            if self._analysis_acc_r is not None:
                self._analysis_acc_r.reset()

    def _ingest_loop(self):
        """Background ingestion thread — drains the capture ring and
        calls `ingest_chunk` per CHUNK_FRAMES slice until stop is signaled.

        Moved off the Qt main thread in c21 (#19) so the GUI event loop
        isn't blocked by DSP / FFT / trigger processing. Reads only whole
        ``CHUNK_FRAMES`` blocks (leaving any partial tail for the next
        wake) so the spectrogram / trigger pipeline keeps seeing exactly
        chunk-sized input, as it did with the old queue.
        """
        ring = self.ring
        while not self._ingest_stop.is_set():
            avail = ring.available
            if avail < CHUNK_FRAMES:
                # Nothing whole to process yet — wait briefly, but wake
                # immediately on stop.
                if self._ingest_stop.wait(0.005):
                    break
                continue
            n_whole = (avail // CHUNK_FRAMES) * CHUNK_FRAMES
            start_abs, block = ring.read(n_whole)
            for off in range(0, n_whole, CHUNK_FRAMES):
                if self._ingest_stop.is_set():
                    break
                chunk = block[off:off + CHUNK_FRAMES]
                try:
                    # abs_end: this chunk's last-sample index in ring
                    # coordinates — the disciplined clock's timebase.
                    # read() clamps start_abs forward over evicted
                    # (overrun) regions, so drops shift abs_end rather
                    # than silently freezing the timestamp clock.
                    self.ingest_chunk(chunk,
                                      abs_end=start_abs + off + CHUNK_FRAMES)
                except Exception as exc:
                    # #44: don't let a processing error crash the ingest
                    # thread — bump counters so the sidebar can surface a
                    # sticky `!` badge, preserve the message for the
                    # tooltip, and log a traceback for post-mortem. The
                    # display stalls briefly but recovers on the next chunk.
                    self.ingest_error_count       += 1
                    self.ingest_error_count_total += 1
                    self.has_ever_ingest_errored   = True
                    # Keep the message short — tooltips truncate and the
                    # full traceback is on stderr anyway.
                    self.last_ingest_error = f'{type(exc).__name__}: {exc}'[:200]
                    import traceback
                    traceback.print_exc()
                    _err_log('ingest', self.name,
                             f'{type(exc).__name__}: {exc} | '
                             f'{traceback.format_exc(limit=3)}')

    def start_rec(self):
        if not self.acq_running:
            self.start_acq()
        self.rec_enabled = True

    def stop_rec(self):
        self.rec_enabled = False
        # 2b: a manual segment can't outlive REC — the disable-flush in
        # the recorder closes it on the next chunk.
        self.force_rec_active = False

    def set_force_trigger(self, active: bool) -> None:
        """2b: toggle the manual force-trigger segment.

        ON: the detection mask becomes all-True from the next chunk —
        an event opens immediately (with pre-trigger lookback) and Max
        Rec splitting applies as usual. OFF: requests an immediate
        flush, consumed on the ingest thread — the segment ends with no
        hold / post-trigger tail. Only meaningful while REC is enabled.
        """
        active = bool(active)
        if active == self.force_rec_active:
            return
        if active:
            self.force_rec_active = True
            self._force_stop_requested = False
        else:
            self.force_rec_active = False
            self._force_stop_requested = True

    # ── Chunk ingestion ───────────────────────────────────────────────────

    def ingest_chunk(self, raw_chunk: np.ndarray, abs_end: int | None = None):
        """#53: public entry point — holds the DSP lock for the
        duration of one chunk so buffer / filter / accumulator
        mutations from UI-thread rebuild paths cannot race the
        ingestion pipeline. The lock is per-chunk (not per-session) so
        rebuild paths get a turn between chunks.

        ``abs_end`` is the chunk's last-sample index in capture-ring
        coordinates (passed by ``_ingest_loop``); it keys the
        disciplined timestamp clock. Callers without ring coordinates
        (tests, direct feeds) omit it and get the coarse anchor clock.
        """
        with self._dsp_lock:
            self._ingest_chunk_locked(raw_chunk, abs_end)

    def _ingest_chunk_locked(self, raw_chunk: np.ndarray,
                             abs_end: int | None = None):
        # 2b: force-trigger toggled OFF — flush the manual segment
        # immediately (no hold / post-trigger tail). Consumed here, on
        # the ingest thread, so the recorder is never touched from the
        # GUI thread.
        if self._force_stop_requested:
            self._force_stop_requested = False
            self._flush_active_events(reason='force_trigger_stop')

        # Inserted-silence (zero-run) detection on the raw chunk, before
        # any channel selection — the upstream insertion zeroes every
        # channel of the endpoint simultaneously.
        self._detect_zero_runs(raw_chunk)

        mode = self.channel_mode
        if raw_chunk.ndim == 2:
            left  = raw_chunk[:, 0]
            right = raw_chunk[:, 1]
        else:
            left = right = raw_chunk

        if mode == 'Right':
            display = right
            record  = right
        elif mode == 'Stereo':
            display = left
            record  = raw_chunk
        else:  # Mono or Left
            display = left
            record  = left

        n   = len(left)
        # #20 / c14: derive both ring-buffer cursors from a single sample
        # clock so they cannot drift apart when chunk size != CHUNK_FRAMES.
        # Also assert the chunk fits — a single oversize chunk would
        # otherwise smear across the buffer multiple times.
        if n > self._total_samples:
            raise ValueError(
                f"chunk of {n} samples exceeds buffer capacity "
                f"{self._total_samples}")
        self.write_head = self._samples_total % self._total_samples
        self.col_head   = (self._samples_total // CHUNK_FRAMES) % self._n_cols
        end = self.write_head + n

        # Display spectrogram — always uses unfiltered signal.
        db_col, lin_mag = self.spec_acc.compute_column(display)
        self.spec_buffer[:, self.col_head] = db_col
        if mode == 'Stereo':
            db_col_r, lin_mag_r = self.spec_acc_r.compute_column(right)
            self.spec_buffer_r[:, self.col_head] = db_col_r

        # Analysis FFT — when analysis params differ from display, a
        # separate accumulator produces its own magnitude spectrum for
        # spectral entropy computation (#12 / c19).
        if self._analysis_acc is not None:
            _, lin_mag = self._analysis_acc.compute_column(display)
            if mode == 'Stereo':
                _, lin_mag_r = self._analysis_acc_r.compute_column(right)

        # Saturation must be measured on the *raw* (pre-filter) signal:
        # the bandpass attenuates clipped peaks and would otherwise hide
        # genuine input clipping (#18, c11).
        if mode == 'Stereo':
            raw_peak = max(float(np.max(np.abs(left))),
                           float(np.max(np.abs(right))))
        else:
            raw_peak = float(np.max(np.abs(display)))
        self.saturated = raw_peak >= 0.99
        if self.saturated:
            # #28: latch the session-wide flag. Cleared only on
            # explicit user request via clear_saturation_flag().
            self.saturated_ever = True

        # Trigger peak + filtered signal for amplitude display
        freq_on = self.freq_filter_enabled
        lo, hi = self.freq_lo, self.freq_hi
        if mode == 'Stereo':
            if freq_on:
                filt_l, peak_l = self.bpf.filter_chunk(left, lo, hi)
                filt_r, peak_r = self.bpf_r.filter_chunk(right, lo, hi)
            else:
                filt_l, filt_r = left, right
                peak_l = float(np.max(np.abs(left)))
                peak_r = float(np.max(np.abs(right)))
            tm = self.trigger_mode
            if tm == 'Left Channel':
                trigger_peak = peak_l
            elif tm == 'Right Channel':
                trigger_peak = peak_r
            elif tm == 'Any Channel':
                trigger_peak = max(peak_l, peak_r)
            elif tm == 'Both Channels':
                trigger_peak = min(peak_l, peak_r)
            else:
                trigger_peak = (peak_l + peak_r) * 0.5
            amp_l, amp_r = filt_l, filt_r
        else:
            if freq_on:
                filt, trigger_peak = self.bpf.filter_chunk(display, lo, hi)
            else:
                filt = display
                trigger_peak = float(np.max(np.abs(display)))
            amp_l = filt
            amp_r = None

        # ── Spectral entropy computation ──────────────────────────────
        entropy_l = _spectral_entropy(lin_mag)
        if mode == 'Stereo':
            entropy_r = _spectral_entropy(lin_mag_r)
            self.spectral_entropy_r = entropy_r
            tm = self.trigger_mode
            if tm == 'Left Channel':
                entropy = entropy_l
            elif tm == 'Right Channel':
                entropy = entropy_r
            elif tm == 'Any Channel':
                entropy = min(entropy_l, entropy_r)   # min = most tonal
            elif tm == 'Both Channels':
                entropy = max(entropy_l, entropy_r)   # max = both must be tonal
            else:  # Average
                entropy = (entropy_l + entropy_r) * 0.5
        else:
            entropy = entropy_l
        self.spectral_entropy = entropy
        self.entropy_buffer[self.col_head] = entropy

        # ── Build per-sample trigger mask (single source of truth) ─────
        # This array is THE detection signal. It is:
        #   (1) passed to the state machine verbatim as ``trigger_mask=``
        #       so min_cross / hold / pre+post trigger are walked
        #       sample-by-sample on the same bools the UI shows.
        #   (2) written verbatim to ``detect_mask_buffer`` so the yellow
        #       "det" indicator strip is literally the detection input,
        #       not a parallel per-sample computation that can drift
        #       from what the recorder sees.
        #
        # Amplitude component: per-sample ENVELOPE ≥ threshold under
        # the active `trigger_mode` rule. Envelope = |analytic signal|
        # (Hilbert transform magnitude), which is smooth across
        # waveform zero crossings. Using |filtered signal| instead
        # (pre-fix) made narrowband signals — pure tones, bandpassed
        # bioacoustic calls — dip below threshold every half-cycle,
        # so the consecutive-above-samples streak could never reach
        # ``min_cross_samps``. See chirp/dsp/envelope.py and
        # tests/test_envelope_trigger.py for details. Spectral
        # component: the entropy trigger is chunk-level (one entropy
        # value per FFT column), so it contributes a scalar AND/OR
        # gate.
        if mode == 'Stereo':
            env_fl = _envelope(filt_l)
            env_fr = _envelope(filt_r)
            tm = self.trigger_mode
            if tm == 'Left Channel':
                filt_combined_env = env_fl
            elif tm == 'Right Channel':
                filt_combined_env = env_fr
            elif tm == 'Any Channel':
                filt_combined_env = np.maximum(env_fl, env_fr)
            elif tm == 'Both Channels':
                filt_combined_env = np.minimum(env_fl, env_fr)
            else:  # Average
                filt_combined_env = (env_fl + env_fr) * 0.5
        else:
            filt_combined_env = _envelope(filt)
        amp_mask = filt_combined_env >= self.threshold

        # M8: accumulate the envelope peak for auto-calibrate. Cheap
        # scalar max; consumed (and reset) by ``consume_env_peak``.
        env_peak = float(filt_combined_env.max()) if n else 0.0
        if env_peak > self._env_peak_acc:
            self._env_peak_acc = env_peak

        stm = self.spectral_trigger_mode
        # #14: spectral entropy is meaningless during FFT warm-up. Use
        # the analysis accumulator — it may have different params from
        # the display accumulator (#12 / c19).
        spec_primed = self.analysis_acc.primed and (
            mode != 'Stereo' or self.analysis_acc_r.primed)
        # 2c: entropy debounce — track how long entropy has stayed
        # continuously below the threshold (chunk granularity). The
        # spectral gate only turns ON once that reaches
        # ``entropy_min_cross_sec`` (0 = instantaneous, legacy) and
        # turns OFF on the first chunk back above threshold.
        if not spec_primed:
            self._entropy_below_sec = 0.0
        elif entropy < self.spectral_threshold:
            self._entropy_below_sec += n / self.sample_rate
        else:
            self._entropy_below_sec = 0.0
        if stm == 'Amplitude Only':
            trigger_mask = amp_mask
        elif not spec_primed:
            # Warm-up: drop spectral contribution. AND/Only suppress;
            # OR falls back to amplitude alone.
            if stm in ('Spectral Only', 'Amp AND Spectral'):
                trigger_mask = np.zeros(n, dtype=bool)
            else:  # 'Amp OR Spectral'
                trigger_mask = amp_mask
        else:
            spec_triggered = (
                self._entropy_below_sec > 0.0
                and self._entropy_below_sec >= self.entropy_min_cross_sec)
            if stm == 'Spectral Only':
                trigger_mask = np.full(n, spec_triggered, dtype=bool)
            elif stm == 'Amp AND Spectral':
                trigger_mask = (amp_mask
                                if spec_triggered
                                else np.zeros(n, dtype=bool))
            else:  # 'Amp OR Spectral'
                trigger_mask = (np.ones(n, dtype=bool)
                                if spec_triggered
                                else amp_mask)

        # 2a / 2b: continuous mode and the force-trigger toggle override
        # detection entirely — the mask (and therefore the detect strip
        # and the recorder walk) is all-True while active.
        continuous = (self.rec_mode == 'Continuous' and self.rec_enabled)
        if continuous or self.force_rec_active:
            trigger_mask = np.ones(n, dtype=bool)

        # Write amplitude buffers (filtered when band filter active)
        if mode == 'Stereo':
            abs_l = np.abs(amp_l)
            abs_r = np.abs(amp_r)
            if end <= self._total_samples:
                self.amp_buffer    [self.write_head:end] = amp_l
                self.amp_buffer_r  [self.write_head:end] = amp_r
                self.abs_amp_buffer  [self.write_head:end] = abs_l
                self.abs_amp_buffer_r[self.write_head:end] = abs_r
            else:
                split = self._total_samples - self.write_head
                wrap  = end % self._total_samples
                self.amp_buffer    [self.write_head:] = amp_l[:split]
                self.amp_buffer    [:wrap]            = amp_l[split:]
                self.amp_buffer_r  [self.write_head:] = amp_r[:split]
                self.amp_buffer_r  [:wrap]            = amp_r[split:]
                self.abs_amp_buffer  [self.write_head:] = abs_l[:split]
                self.abs_amp_buffer  [:wrap]            = abs_l[split:]
                self.abs_amp_buffer_r[self.write_head:] = abs_r[:split]
                self.abs_amp_buffer_r[:wrap]            = abs_r[split:]
        else:
            abs_l = np.abs(amp_l)
            if end <= self._total_samples:
                self.amp_buffer    [self.write_head:end] = amp_l
                self.abs_amp_buffer[self.write_head:end] = abs_l
            else:
                split = self._total_samples - self.write_head
                wrap  = end % self._total_samples
                self.amp_buffer    [self.write_head:] = amp_l[:split]
                self.amp_buffer    [:wrap]            = amp_l[split:]
                self.abs_amp_buffer[self.write_head:] = abs_l[:split]
                self.abs_amp_buffer[:wrap]            = abs_l[split:]

        # #32: write the detect-strip ring buffer from the SAME array
        # the state machine will walk. One source of truth — no
        # parallel per-sample computation. Wrap-aware, mirroring the
        # amp-buffer path above.
        if end <= self._total_samples:
            self.detect_mask_buffer[self.write_head:end] = trigger_mask
        else:
            split = self._total_samples - self.write_head
            wrap  = end % self._total_samples
            self.detect_mask_buffer[self.write_head:]   = trigger_mask[:split]
            self.detect_mask_buffer[:wrap]              = trigger_mask[split:]

        # Single source of truth: advance the cumulative sample clock,
        # then re-derive both ring-buffer cursors. This guarantees they
        # stay coherent regardless of `n` (#20 / c14) and gives readers
        # the legacy "where the next sample lands" semantics.
        self._samples_total += n
        self.write_head = self._samples_total % self._total_samples
        self.col_head   = (self._samples_total // CHUNK_FRAMES) % self._n_cols

        # #51: route through ``_effective_output_dir`` so the
        # day-subfolder prefix is sanitized along the same path
        # teardown uses. Previously duplicated the join here with a
        # raw ``dph_folder_prefix``, susceptible to path-traversal.
        out_dir = self._effective_output_dir()

        # Capture-time wall clock of this chunk's last sample.
        # Primary: the disciplined clock — sample-smooth, steered onto
        # wall time by callback observations (immune to crystal drift,
        # ingest backlog, and capture holes; see chirp/audio/clock.py).
        # Steps are deferred while a recording event is open so no file
        # spans a discontinuity. Fallback (no observations yet, WAV
        # playback, tests): the M5 start_acq anchor + sample counter.
        chunk_end_wall = None
        epoch = None
        clock = self.clock
        if clock is not None and abs_end is not None:
            epoch = clock.wall_at(
                abs_end,
                allow_step=not self.recorder.has_active_events())
        if epoch is not None:
            chunk_end_wall = datetime.datetime.fromtimestamp(
                epoch, tz=datetime.timezone.utc)
            if clock.step_count != self._clock_steps_seen:
                self._clock_steps_seen = clock.step_count
                _err_log('clock_step', self.name,
                         f'timestamp clock stepped forward '
                         f'{clock.last_step_sec:.3f}s (capture hole: '
                         f'device stall / dropped samples); step '
                         f'#{clock.step_count}')
        elif self._wall_anchor_time is not None:
            chunk_end_wall = self._wall_anchor_time + datetime.timedelta(
                seconds=(self._samples_total - self._wall_anchor_samples)
                / self.sample_rate)

        # Sidecar clock log: one audit row per interval while acquiring
        # (direct test feeds never set acq_running, so they don't log).
        if self.acq_running:
            now_m = time.monotonic()
            if now_m >= self._clock_log_next_t:
                self._clock_log_next_t = (
                    now_m + max(0.0, self.clock_log_interval_sec))
                self._queue_clock_log_row(
                    abs_end, chunk_end_wall,
                    'clock' if epoch is not None else
                    ('anchor' if chunk_end_wall is not None else ''))

        # 2a / 2b: effective open-gating params. Continuous mode records
        # from the first sample after Start Rec (no qualification run,
        # no lookback); a forced segment opens immediately but keeps the
        # user's pre-trigger lookback.
        eff_min_cross = self.min_cross_sec
        eff_pre_trig  = self.pre_trig_sec
        if continuous:
            eff_min_cross = 0.0
            eff_pre_trig  = 0.0
        elif self.force_rec_active:
            eff_min_cross = 0.0

        report = self.recorder.process_chunk(
            record,
            trigger_peak  = trigger_peak,
            threshold     = self.threshold,
            min_cross_sec = eff_min_cross,
            min_total_cross_sec = self.min_total_cross_sec,
            hold_sec      = self.hold_sec,
            post_trig_sec = self.post_trig_sec,
            max_rec_sec   = self.max_rec_sec,
            pre_trig_sec  = eff_pre_trig,
            output_dir    = out_dir,
            enabled       = self.rec_enabled,
            filename_prefix = self.filename_prefix,
            filename_suffix = self.filename_suffix,
            sample_rate   = self.sample_rate,
            trigger_mask  = trigger_mask,
            filename_stream = self.name,
            global_chunk_end = self._samples_total,
            chunk_end_wall = chunk_end_wall,
        )

        # #32: paint the record_mask indicator buffer from the
        # recorder's span report. detect_mask_buffer was already
        # painted above from the SAME trigger_mask array the state
        # machine walked — 1:1 with detection, by construction.
        self._paint_record_buffer(report, n)

    # ── Sidecar clock log ─────────────────────────────────────────────────

    def _queue_clock_log_row(self, abs_end, chunk_end_wall, source) -> None:
        """Build one clock-audit CSV row and hand the append to the
        writer pool — the ingest thread never touches disk. The row
        pairs the capture-side sample index with the derived capture
        time (``source``: 'clock' = disciplined, 'anchor' = M5
        fallback, '' = none) and the raw wall clock, so any filename
        timestamp can be audited or corrected offline. Never raises."""
        try:
            from chirp.recording.writer import _sanitize_token
            wall = time.time()
            iso = datetime.datetime.fromtimestamp(
                wall, tz=datetime.timezone.utc).isoformat(
                timespec='milliseconds')
            derived = ('' if chunk_end_wall is None
                       else f'{chunk_end_wall.timestamp():.6f}')
            sample_idx = self._samples_total if abs_end is None else abs_end
            stream = _sanitize_token(self.name) or 'stream'
            row = (f'{iso},{stream},{sample_idx},{derived},{source},'
                   f'{wall:.6f}\n')
            # Base output_dir, NOT the day-subfolder: the audit trail
            # must stay one continuous file across day rollovers.
            out_dir = self.output_dir
            _wav_writer.submit_call(
                lambda: _append_clock_log_row(out_dir, row),
                filename_stream=self.name, out_dir=out_dir)
        except Exception:
            pass

    # ── #32: indicator buffer painting ───────────────────────────────────

    def _paint_record_buffer(self, report: dict, n: int) -> None:
        """Paint ``record_mask_buffer`` from the recorder's span report.

        The chunk just ingested covers global samples
        ``[global_end - n, global_end)`` — which maps to ring positions
        ``[prev_write_head, prev_write_head + n)`` (wrapping).

        ``record_mask_buffer`` is *cleared* for the chunk's range first
        (so stale True values from a previous pass over this ring
        region don't persist), then ORed with True for every span in
        ``active_spans`` + ``flushed_spans`` that overlaps the
        currently-visible ring window. This is what retroactively
        lights up the pre-trigger samples at the moment an event opens:
        their global range falls inside the ring window and the OR-in
        hits past ring positions.
        """
        if report is None:
            return

        total = self._total_samples
        g_end = self._samples_total
        g_begin_chunk = g_end - n
        ring_start = g_begin_chunk % total

        # Clear the just-written chunk's range first, in BOTH masks — as
        # fresh audio overwrites a ring region its stale record/discard
        # marks must not linger.
        end = ring_start + n
        if end <= total:
            self.record_mask_buffer[ring_start:end] = False
            self.discard_mask_buffer[ring_start:end] = False
        else:
            self.record_mask_buffer[ring_start:] = False
            self.record_mask_buffer[:end - total] = False
            self.discard_mask_buffer[ring_start:] = False
            self.discard_mask_buffer[:end - total] = False

        # Visible ring window in global-sample coords.
        ring_window_start = g_end - total

        spans = list(report.get('active_spans') or [])
        spans.extend(report.get('flushed_spans') or [])
        for g_lo, g_hi in spans:
            lo = max(int(g_lo), ring_window_start)
            hi = min(int(g_hi), g_end)
            if hi <= lo:
                continue
            self._or_range(self.record_mask_buffer, lo, hi, total)

        # Discarded events: paint them into the discard mask AND clear
        # them from the record mask — an event that streamed green while
        # open must flip fully red once the min-total-crossing filter
        # drops it, with no green sliver left over from earlier ticks.
        for g_lo, g_hi in (report.get('discarded_spans') or []):
            lo = max(int(g_lo), ring_window_start)
            hi = min(int(g_hi), g_end)
            if hi <= lo:
                continue
            self._or_range(self.discard_mask_buffer, lo, hi, total)
            self._clear_range(self.record_mask_buffer, lo, hi, total)

    @staticmethod
    def _or_range(buf: np.ndarray, g_lo: int, g_hi: int, total: int) -> None:
        """OR True into a circular buffer over global sample range
        ``[g_lo, g_hi)``. Caller guarantees the range fits in the ring.
        """
        b0 = g_lo % total
        length = g_hi - g_lo
        end = b0 + length
        if end <= total:
            buf[b0:end] = True
        else:
            buf[b0:] = True
            buf[:end - total] = True

    @staticmethod
    def _clear_range(buf: np.ndarray, g_lo: int, g_hi: int, total: int) -> None:
        """Set False into a circular buffer over global sample range
        ``[g_lo, g_hi)``. Caller guarantees the range fits in the ring.
        """
        b0 = g_lo % total
        length = g_hi - g_lo
        end = b0 + length
        if end <= total:
            buf[b0:end] = False
        else:
            buf[b0:] = False
            buf[:end - total] = False

    # ── Mini amplitude for sidebar ────────────────────────────────────────

    def get_mini_amplitude(self, n_points: int = 200) -> np.ndarray:
        buf = self.abs_amp_buffer
        if len(buf) < n_points:
            return buf
        chunk_size = len(buf) // n_points
        trimmed = buf[:chunk_size * n_points]
        return trimmed.reshape(n_points, chunk_size).max(axis=1)

    # ── Serialisation ────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialise user-configurable settings to a plain dict."""
        try:
            dev_info = sd.query_devices(self.device_id) if self.device_id is not None else None
            dev_name = dev_info['name'] if dev_info else ''
            dev_hostapi = host_api_name(dev_info) if dev_info else ''
        except Exception:
            dev_name = ''
            dev_hostapi = ''
        return {
            'name':                self.name,
            'device_name':         dev_name,
            'device_hostapi':      dev_hostapi,
            'sample_rate':         self.sample_rate,
            'display_seconds':     self.display_seconds,
            'channel_mode':        self.channel_mode,
            'trigger_mode':        self.trigger_mode,
            'threshold':           self.threshold,
            'min_cross_sec':       self.min_cross_sec,
            'min_total_cross_sec': self.min_total_cross_sec,
            'hold_sec':            self.hold_sec,
            'post_trig_sec':       self.post_trig_sec,
            'max_rec_sec':         self.max_rec_sec,
            'pre_trig_sec':        self.pre_trig_sec,
            'freq_filter_enabled': self.freq_filter_enabled,
            'freq_lo':             self.freq_lo,
            'freq_hi':             self.freq_hi,
            'spec_nperseg':        self.spec_nperseg,
            'spec_window':         self.spec_window,
            'freq_scale':          self.freq_scale,
            'gain_db':             self.gain_db,
            'db_floor':            self.db_floor,
            'db_ceil':             self.db_ceil,
            'display_freq_lo':     self.display_freq_lo,
            'display_freq_hi':     self.display_freq_hi,
            'output_dir':          self.output_dir,
            'filename_prefix':     self.filename_prefix,
            'filename_suffix':     self.filename_suffix,
            'ref_date':            self.ref_date.isoformat() if self.ref_date else None,
            'dph_folder_prefix':   self.dph_folder_prefix,
            'amp_ylim':            self.amp_ylim,
            'amp_scale':           self.amp_scale,
            'spectral_trigger_mode': self.spectral_trigger_mode,
            'spectral_threshold':    self.spectral_threshold,
            'entropy_min_cross_sec': self.entropy_min_cross_sec,
            'rec_mode':            self.rec_mode,
            'display_mode':        self.display_mode,
            'analysis_nperseg':    self.analysis_nperseg,
            'analysis_window':     self.analysis_window,
            'input_source':        self.input_source,
            'wav_file_path':       self.wav_file_path,
            'wav_loop':            self.wav_loop,
            'stream_enabled':      self.stream_enabled,
            'params_locked':       self.params_locked,
            'color':               self.color,
        }

    @classmethod
    def from_dict(cls, d: dict):
        """Create a RecordingEntity from a settings dict.
        Returns (entity, warning_msg_or_None).
        """
        # Resolve device by name (#21 / c20 — multi-strategy matcher).
        dev_name = d.get('device_name', '')
        hostapi_hint = d.get('device_hostapi', '')
        device_id, warning = find_device_by_name(dev_name, hostapi_hint)

        sr = d.get('sample_rate', SAMPLE_RATE)
        ds = d.get('display_seconds', DISPLAY_SECONDS)
        e = cls(name=d.get('name', 'Recording'), device_id=device_id,
                sample_rate=sr, display_seconds=ds)

        # Scalar attributes
        for attr in ('channel_mode', 'trigger_mode', 'threshold',
                     'min_cross_sec', 'min_total_cross_sec',
                     'hold_sec', 'post_trig_sec', 'max_rec_sec', 'pre_trig_sec',
                     'freq_filter_enabled', 'freq_lo', 'freq_hi',
                     'freq_scale', 'gain_db', 'db_floor', 'db_ceil',
                     'display_freq_lo', 'display_freq_hi',
                     'output_dir', 'filename_prefix', 'filename_suffix',
                     'dph_folder_prefix', 'amp_ylim', 'amp_scale',
                     'spectral_trigger_mode', 'spectral_threshold',
                     'entropy_min_cross_sec', 'rec_mode',
                     'display_mode',
                     'input_source', 'wav_file_path', 'wav_loop',
                     'stream_enabled', 'params_locked', 'color'):
            if attr in d:
                setattr(e, attr, d[attr])

        # Spec params: always apply, even when they happen to equal the
        # defaults. The pre-c17 shortcut "skip if defaults" was a bug —
        # if the constructor's defaults ever change, an old config file
        # would silently snap to the new defaults instead of preserving
        # the user's original intent (#22 / c17).
        nperseg = d.get('spec_nperseg', SPECTROGRAM_NPERSEG)
        window  = d.get('spec_window', 'hann')
        e.change_fft_params(nperseg, window)

        # Analysis FFT params (#12 / c19). Legacy files won't have these
        # keys, so fall back to display params for backward compat.
        a_nperseg = d.get('analysis_nperseg', nperseg)
        a_window  = d.get('analysis_window', window)
        e.change_analysis_fft_params(a_nperseg, a_window)

        # Ref date
        ref = d.get('ref_date')
        if ref:
            try:
                e.ref_date = datetime.date.fromisoformat(ref)
            except (ValueError, TypeError):
                e.ref_date = None

        # Channel mode may need stereo device
        need_ch = 2 if e.channel_mode != 'Mono' else 1
        if need_ch == 2 and device_id is not None:
            try:
                max_ch = sd.query_devices(device_id)['max_input_channels']
                if max_ch < 2:
                    e.channel_mode = 'Mono'
            except Exception:
                e.channel_mode = 'Mono'
        if need_ch == 2:
            e.change_device(device_id, 2)

        # If the saved config used a WAV file, re-open it so the capture
        # actually points at the file (the setattr loop only set the
        # attributes; __init__ already created a live-device capture).
        if e.input_source == 'wav_file' and e.wav_file_path:
            ok, wav_warning = e.use_wav_file(e.wav_file_path, loop=e.wav_loop)
            # #49: ALWAYS propagate the warning — both on failure
            # ("could not open WAV") and on success-with-caveat
            # (channel truncation, SR change). Pre-fix only the
            # ``not ok and wav_warning`` branch was forwarded, so a
            # successful WAV open with channel truncation never
            # reached the user.
            if wav_warning:
                rec_name = e.name or '<unnamed>'
                tag = f"[{rec_name}] {wav_warning}"
                warning = f"{warning}; {tag}" if warning else tag

        e.rebuild_freq_mapping()
        return e, warning

    # ── Cleanup ───────────────────────────────────────────────────────────

    def close(self):
        # #45: final flush. Covers "delete stream" and "Load settings"
        # paths that drop the entity without a prior ``stop_acq``.
        # Wrapped in try/except because a crash here would prevent the
        # capture from closing and leak a PortAudio stream.
        try:
            self._stop_ingest_and_flush(reason='close')
        except Exception:
            import traceback
            traceback.print_exc()
        try:
            self.capture.close()
        except Exception:
            pass
