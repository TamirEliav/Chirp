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
import queue
import threading

import numpy as np
import sounddevice as sd

from chirp.audio import AudioCapture, WavFileCapture
from chirp.audio.devices import find_device_by_name, host_api_name
from chirp.constants import (
    CHUNK_FRAMES,
    DEFAULT_FREQ_HI,
    DEFAULT_FREQ_LO,
    DEFAULT_HOLD,
    DEFAULT_MAX_REC,
    DEFAULT_MIN_CROSS,
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
from chirp.dsp import normalized_spectral_entropy as _spectral_entropy
from chirp.recording.trigger import ThresholdRecorder


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

        # Audio pipeline
        self.queue      = queue.Queue(maxsize=200)
        self.capture    = self._make_capture(channels=1)
        self.spec_acc   = SpectrogramAccumulator()
        self.spec_acc_r = SpectrogramAccumulator()
        self.recorder   = ThresholdRecorder()
        self.bpf        = BandpassFilter(sample_rate=self.sample_rate)
        self.bpf_r      = BandpassFilter(sample_rate=self.sample_rate)

        # Analysis FFT params — default to display params. When they
        # differ from the display FFT, a separate accumulator is used
        # for spectral entropy / trigger (#12 / c19).
        self.analysis_nperseg = SPECTROGRAM_NPERSEG
        self.analysis_window  = 'hann'
        self._analysis_acc: SpectrogramAccumulator | None = None
        self._analysis_acc_r: SpectrogramAccumulator | None = None

        # Trigger params
        self.threshold     = DEFAULT_THRESHOLD
        self.min_cross_sec = DEFAULT_MIN_CROSS
        self.hold_sec      = DEFAULT_HOLD
        self.pre_trig_sec  = DEFAULT_PRE_TRIG
        self.post_trig_sec = DEFAULT_POST_TRIG
        self.max_rec_sec   = DEFAULT_MAX_REC
        self.freq_filter_enabled = False
        self.freq_lo       = DEFAULT_FREQ_LO
        self.freq_hi       = DEFAULT_FREQ_HI

        # Spectral trigger params
        self.spectral_trigger_mode = 'Amplitude Only'  # 'Amplitude Only', 'Spectral Only', 'Amp AND Spectral', 'Amp OR Spectral'
        self.spectral_threshold    = 0.5                # entropy threshold (trigger when below)
        self.spectral_entropy      = 1.0                # current entropy value (display only)
        self.spectral_entropy_r    = 1.0                # right channel entropy (stereo)

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
        # Single cumulative sample counter — both ring-buffer cursors
        # are derived from it so they cannot drift apart when chunk
        # size differs from CHUNK_FRAMES (#20 / c14).
        self._samples_total = 0
        self.write_head = 0
        self.col_head   = 0

        # Display state
        self.saturated  = False   # True when current chunk contains clipped audio
        self.amp_ylim   = 1.05    # amplitude y-axis max (persists across mode switches)
        self.display_mode = 'Spectrogram'  # 'Spectrogram', 'Waveform', or 'Both'

        # Runtime
        self.acq_running = False
        self.rec_enabled = False
        self._ingest_stop = threading.Event()
        self._ingest_thread: threading.Thread | None = None

        # Freq mapping
        self.freq_map_idx_floor = None
        self.freq_map_frac      = None
        self.display_freqs      = None
        self.rebuild_freq_mapping()

    # ── Display reset ────────────────────────────────────────────────────

    def reset_display(self):
        """Clear display buffers and reset write heads. Does NOT affect recording/triggering."""
        self.amp_buffer[:]    = 0.0
        self.amp_buffer_r[:]  = 0.0
        self.spec_buffer[:]   = SPEC_DB_MIN
        self.spec_buffer_r[:] = SPEC_DB_MIN
        self.entropy_buffer[:] = 1.0
        self.detect_mask_buffer[:] = False
        self.record_mask_buffer[:] = False
        self._samples_total = 0
        self.write_head = 0
        self.col_head   = 0

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
        if self.input_source == 'wav_file' and self.wav_file_path:
            cap = WavFileCapture(self.queue, self.wav_file_path,
                                 channels=channels, loop=self.wav_loop)
        else:
            cap = AudioCapture(self.queue, device=self.device_id,
                               channels=channels, samplerate=self.sample_rate)
        # Re-wire the monitor on every new capture so a device / SR /
        # WAV-file switch doesn't silently drop the loopback (#7).
        if self._monitor is not None:
            try:
                cap.set_monitor(self._monitor, id(self))
            except Exception:
                pass
        return cap

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
                cap.set_monitor(monitor, id(self) if monitor is not None else None)
            except Exception:
                pass

    # ── Device change ─────────────────────────────────────────────────────

    def change_device(self, device_id, channels):
        was_running = self.acq_running
        if was_running:
            self.capture.pause()
            self.acq_running = False
        self.capture.close()
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        self.device_id = device_id
        self.input_source = 'device'
        self.capture = self._make_capture(channels=channels)
        if not self.capture.valid:
            self.acq_running = False
            return False
        if was_running:
            self.capture.resume()
            self.acq_running = True
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
        self.capture.close()
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

        self.input_source  = 'wav_file'
        self.wav_file_path = path
        self.wav_loop      = loop

        need_ch = 2 if self.channel_mode != 'Mono' else 1
        probe = self._make_capture(channels=need_ch)
        if not probe.valid:
            # Fall back to live device on failure.
            probe.close()
            self.input_source = 'device'
            self.wav_file_path = None
            self.capture = self._make_capture(channels=need_ch)
            return False, f"Could not open WAV file: {path}"

        warning = None
        file_sr = probe.file_sample_rate
        if file_sr and file_sr != self.sample_rate and file_sr in self.SUPPORTED_RATES:
            probe.close()
            # change_sample_rate will call _make_capture again with the
            # new rate, reusing the WAV source we just configured.
            self.change_sample_rate(file_sr)
            warning = (f"Session sample rate changed to {file_sr} Hz to "
                       f"match WAV file")
        elif file_sr and file_sr != self.sample_rate:
            warning = (f"WAV file sample rate ({file_sr} Hz) is not a "
                       f"supported session rate — resampling is not "
                       f"performed; timing will be off")
            self.capture = probe
        else:
            self.capture = probe

        if was_running and self.capture.valid:
            self.capture.resume()
            self.acq_running = True
        return True, warning

    # ── Sample rate change ──────────────────────────────────────────────

    def change_sample_rate(self, new_rate: int):
        if new_rate == self.sample_rate:
            return
        was_running = self.acq_running
        if was_running:
            self.capture.pause()
            self.acq_running = False
        self.capture.close()
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

        self.sample_rate = new_rate
        self._n_cols        = max(1, int(self.display_seconds * new_rate / CHUNK_FRAMES))
        self._total_samples = self._n_cols * CHUNK_FRAMES
        self.display_freq_hi = min(self.display_freq_hi, float(new_rate // 2))

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
        self._samples_total = 0
        self.write_head = 0
        self.col_head   = 0

        # Rebuild filters and capture
        self.bpf   = BandpassFilter(sample_rate=new_rate)
        self.bpf_r = BandpassFilter(sample_rate=new_rate)
        need_ch = 2 if self.channel_mode != 'Mono' else 1
        self.capture = self._make_capture(channels=need_ch)
        self.rebuild_freq_mapping()

        if was_running and self.capture.valid:
            self.capture.resume()
            self.acq_running = True

    # ── Display buffer change ──────────────────────────────────────────────

    def change_display_seconds(self, new_secs: float):
        if new_secs == self.display_seconds:
            return
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
        self._samples_total = 0
        self.write_head = 0
        self.col_head   = 0

    # ── Transport ─────────────────────────────────────────────────────────

    def start_acq(self):
        if not self.acq_running and self.capture.valid:
            # Clear stale overlap so the first few FFT columns after a
            # restart don't mix zero-padding into the spectrum (#14).
            self.spec_acc.reset()
            self.spec_acc_r.reset()
            if self._analysis_acc is not None:
                self._analysis_acc.reset()
            if self._analysis_acc_r is not None:
                self._analysis_acc_r.reset()
            self.capture.resume()
            self.acq_running = True
            # Start ingestion thread (#19 / c21).
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
            # Stop ingestion thread (#19 / c21).
            self._ingest_stop.set()
            if self._ingest_thread is not None:
                self._ingest_thread.join(timeout=2.0)
                self._ingest_thread = None
            self.bpf.reset()
            self.bpf_r.reset()
            self.spec_acc.reset()
            self.spec_acc_r.reset()
            if self._analysis_acc is not None:
                self._analysis_acc.reset()
            if self._analysis_acc_r is not None:
                self._analysis_acc_r.reset()

    def _ingest_loop(self):
        """Background ingestion thread — drains the audio queue and
        calls `ingest_chunk` continuously until stop is signaled.

        Moved off the Qt main thread in c21 (#19) so the GUI event
        loop isn't blocked by DSP / FFT / trigger processing.
        """
        while not self._ingest_stop.is_set():
            try:
                chunk = self.queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                self.ingest_chunk(chunk)
            except Exception:
                # Don't let a processing error crash the ingestion
                # thread — log and continue. The display will stall
                # briefly but recover on the next chunk.
                import traceback
                traceback.print_exc()

    def start_rec(self):
        if not self.acq_running:
            self.start_acq()
        self.rec_enabled = True

    def stop_rec(self):
        self.rec_enabled = False

    # ── Chunk ingestion ───────────────────────────────────────────────────

    def ingest_chunk(self, raw_chunk: np.ndarray):
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

        # ── Compute should_trigger (c12 / #16) ────────────────────────
        # The state machine no longer infers triggering from
        # `trigger_peak`. We compute the boolean here and pass it
        # explicitly so the spectral path doesn't have to forge fake
        # amplitude peaks. `trigger_peak` keeps its true post-filter
        # value and remains useful for the saturation indicator + UI.
        amp_above = (trigger_peak >= self.threshold)
        stm = self.spectral_trigger_mode
        # #14: spectral entropy is meaningless during FFT warm-up.
        # Use the analysis accumulator — it may have different params
        # from the display accumulator (#12 / c19).
        spec_primed = self.analysis_acc.primed and (
            mode != 'Stereo' or self.analysis_acc_r.primed)
        if stm == 'Amplitude Only':
            should_trigger = amp_above
        elif not spec_primed:
            # Warm-up: drop spectral contribution. AND/Only suppress;
            # OR falls back to amplitude alone.
            if stm in ('Spectral Only', 'Amp AND Spectral'):
                should_trigger = False
            else:  # 'Amp OR Spectral'
                should_trigger = amp_above
        else:
            spec_triggered = (entropy < self.spectral_threshold)
            if stm == 'Spectral Only':
                should_trigger = spec_triggered
            elif stm == 'Amp AND Spectral':
                should_trigger = amp_above and spec_triggered
            else:  # 'Amp OR Spectral'
                should_trigger = amp_above or spec_triggered

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

        # #32: per-sample amplitude-threshold crossing mask for the
        # detect strip. Must use the *filtered* signal (when the
        # bandpass is enabled) combined per-sample under the active
        # `trigger_mode` rule, so the strip exactly mirrors the
        # chunk-level trigger decision (`amp_above`). Previously used
        # the raw `record` signal — the strip would light up on
        # out-of-band energy the filter was suppressing, making it
        # look like the trigger ignored the bandpass. Spectral /
        # min_cross gating is still pre-stripped (pre-gating) so the
        # user can see raw amplitude crossings while tuning.
        if mode == 'Stereo':
            abs_fl = np.abs(filt_l)
            abs_fr = np.abs(filt_r)
            tm = self.trigger_mode
            if tm == 'Left Channel':
                filt_combined_abs = abs_fl
            elif tm == 'Right Channel':
                filt_combined_abs = abs_fr
            elif tm == 'Any Channel':
                filt_combined_abs = np.maximum(abs_fl, abs_fr)
            elif tm == 'Both Channels':
                filt_combined_abs = np.minimum(abs_fl, abs_fr)
            else:  # Average
                filt_combined_abs = (abs_fl + abs_fr) * 0.5
        else:
            filt_combined_abs = np.abs(filt)
        sample_detect = filt_combined_abs >= self.threshold

        # Write detect-mask ring buffer for the chunk just arrived
        # (wrap-aware, mirroring the amp-buffer path above).
        if end <= self._total_samples:
            self.detect_mask_buffer[self.write_head:end] = sample_detect
        else:
            split = self._total_samples - self.write_head
            wrap  = end % self._total_samples
            self.detect_mask_buffer[self.write_head:]   = sample_detect[:split]
            self.detect_mask_buffer[:wrap]              = sample_detect[split:]

        # Single source of truth: advance the cumulative sample clock,
        # then re-derive both ring-buffer cursors. This guarantees they
        # stay coherent regardless of `n` (#20 / c14) and gives readers
        # the legacy "where the next sample lands" semantics.
        self._samples_total += n
        self.write_head = self._samples_total % self._total_samples
        self.col_head   = (self._samples_total // CHUNK_FRAMES) % self._n_cols

        # Compute effective output dir (with day subfolder if ref_date set)
        out_dir = self.output_dir
        if self.ref_date is not None:
            days = (datetime.date.today() - self.ref_date).days
            out_dir = os.path.join(out_dir, f'{self.dph_folder_prefix}{days}')

        report = self.recorder.process_chunk(
            record,
            trigger_peak  = trigger_peak,
            threshold     = self.threshold,
            min_cross_sec = self.min_cross_sec,
            hold_sec      = self.hold_sec,
            post_trig_sec = self.post_trig_sec,
            max_rec_sec   = self.max_rec_sec,
            pre_trig_sec  = self.pre_trig_sec,
            output_dir    = out_dir,
            enabled       = self.rec_enabled,
            filename_prefix = self.filename_prefix,
            filename_suffix = self.filename_suffix,
            sample_rate   = self.sample_rate,
            should_trigger = should_trigger,
            filename_stream = self.name,
            global_chunk_end = self._samples_total,
        )

        # #32: paint the record_mask indicator buffer from the recorder's
        # span report. detect_mask_buffer was already painted above from
        # the raw per-sample threshold crossings.
        self._paint_record_buffer(report, n)

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

        # Clear the just-written chunk's range first.
        end = ring_start + n
        if end <= total:
            self.record_mask_buffer[ring_start:end] = False
        else:
            self.record_mask_buffer[ring_start:] = False
            self.record_mask_buffer[:end - total] = False

        spans = list(report.get('active_spans') or [])
        spans.extend(report.get('flushed_spans') or [])
        if not spans:
            return

        # Visible ring window in global-sample coords.
        ring_window_start = g_end - total
        for g_lo, g_hi in spans:
            lo = max(int(g_lo), ring_window_start)
            hi = min(int(g_hi), g_end)
            if hi <= lo:
                continue
            self._or_range(self.record_mask_buffer, lo, hi, total)

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
            'spectral_trigger_mode': self.spectral_trigger_mode,
            'spectral_threshold':    self.spectral_threshold,
            'display_mode':        self.display_mode,
            'analysis_nperseg':    self.analysis_nperseg,
            'analysis_window':     self.analysis_window,
            'input_source':        self.input_source,
            'wav_file_path':       self.wav_file_path,
            'wav_loop':            self.wav_loop,
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
                     'min_cross_sec', 'hold_sec', 'post_trig_sec', 'max_rec_sec', 'pre_trig_sec',
                     'freq_filter_enabled', 'freq_lo', 'freq_hi',
                     'freq_scale', 'gain_db', 'db_floor', 'db_ceil',
                     'display_freq_lo', 'display_freq_hi',
                     'output_dir', 'filename_prefix', 'filename_suffix',
                     'dph_folder_prefix', 'amp_ylim',
                     'spectral_trigger_mode', 'spectral_threshold',
                     'display_mode',
                     'input_source', 'wav_file_path', 'wav_loop'):
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
            if not ok and wav_warning:
                warning = f"{warning}; {wav_warning}" if warning else wav_warning

        e.rebuild_freq_mapping()
        return e, warning

    # ── Cleanup ───────────────────────────────────────────────────────────

    def close(self):
        try:
            self.capture.close()
        except Exception:
            pass
