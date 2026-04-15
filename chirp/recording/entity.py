"""RecordingEntity — per-stream data model and chunk ingestion.

Extracted from the monolith in the Phase 1 refactor (plan: c06). Owns
the audio pipeline (capture → filter → FFT → entropy → trigger) and
all the ring buffers that back the display. No Qt widgets live here —
the UI layer pulls data out via attribute access.

Behavior is unchanged from the monolith. Known issues pinned by the
module docstrings in `chirp.dsp` and `chirp.recording.trigger`:

  - #14: FFT accumulator overlap is not reset on start/stop → c10.
  - #16: spectral trigger is forced through `trigger_peak` → c12.
  - #18: saturation is measured post-filter → c11.
  - #20: ring-buffer cursors desync when chunk size != CHUNK_FRAMES → c14.
  - #19: `ingest_chunk` runs on the Qt main thread via `_update_plot` → c21.
"""

import datetime
import os
import queue

import numpy as np
import sounddevice as sd

from chirp.audio import AudioCapture
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

        # Audio pipeline
        self.queue      = queue.Queue(maxsize=200)
        self.capture    = AudioCapture(self.queue, device=device_id, samplerate=self.sample_rate)
        self.spec_acc   = SpectrogramAccumulator()
        self.spec_acc_r = SpectrogramAccumulator()
        self.recorder   = ThresholdRecorder()
        self.bpf        = BandpassFilter(sample_rate=self.sample_rate)
        self.bpf_r      = BandpassFilter(sample_rate=self.sample_rate)

        # Trigger params
        self.threshold     = DEFAULT_THRESHOLD
        self.min_cross_sec = DEFAULT_MIN_CROSS
        self.hold_sec      = DEFAULT_HOLD
        self.post_trig_sec = DEFAULT_POST_TRIG
        self.max_rec_sec   = DEFAULT_MAX_REC
        self.pre_trig_sec  = DEFAULT_PRE_TRIG
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
        self.write_head = 0
        self.col_head   = 0

        # Display state
        self.saturated  = False   # True when current chunk contains clipped audio
        self.amp_ylim   = 1.05    # amplitude y-axis max (persists across mode switches)
        self.display_mode = 'Spectrogram'  # 'Spectrogram', 'Waveform', or 'Both'

        # Runtime
        self.acq_running = False
        self.rec_enabled = False

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
        self.capture = AudioCapture(self.queue, device=device_id, channels=channels,
                                    samplerate=self.sample_rate)
        if not self.capture.valid:
            self.acq_running = False
            return False
        if was_running:
            self.capture.resume()
            self.acq_running = True
        return True

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
        self.write_head = 0
        self.col_head   = 0

        # Rebuild filters and capture
        self.bpf   = BandpassFilter(sample_rate=new_rate)
        self.bpf_r = BandpassFilter(sample_rate=new_rate)
        need_ch = 2 if self.channel_mode != 'Mono' else 1
        self.capture = AudioCapture(self.queue, device=self.device_id,
                                    channels=need_ch, samplerate=new_rate)
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
        self.write_head = 0
        self.col_head   = 0

    # ── Transport ─────────────────────────────────────────────────────────

    def start_acq(self):
        if not self.acq_running and self.capture.valid:
            # Clear stale overlap so the first few FFT columns after a
            # restart don't mix zero-padding into the spectrum (#14).
            self.spec_acc.reset()
            self.spec_acc_r.reset()
            self.capture.resume()
            self.acq_running = True

    def stop_acq(self):
        if self.acq_running:
            self.capture.pause()
            self.acq_running = False
            self.rec_enabled = False
            self.bpf.reset()
            self.bpf_r.reset()
            self.spec_acc.reset()
            self.spec_acc_r.reset()

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
        end = self.write_head + n

        # Spectrogram always uses unfiltered signal
        db_col, lin_mag = self.spec_acc.compute_column(display)
        self.spec_buffer[:, self.col_head] = db_col
        if mode == 'Stereo':
            db_col_r, lin_mag_r = self.spec_acc_r.compute_column(right)
            self.spec_buffer_r[:, self.col_head] = db_col_r

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
        spec_primed = self.spec_acc.primed and (
            mode != 'Stereo' or self.spec_acc_r.primed)
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

        self.write_head = end % self._total_samples
        self.col_head   = (self.col_head + 1) % self._n_cols

        # Compute effective output dir (with day subfolder if ref_date set)
        out_dir = self.output_dir
        if self.ref_date is not None:
            days = (datetime.date.today() - self.ref_date).days
            out_dir = os.path.join(out_dir, f'{self.dph_folder_prefix}{days}')

        self.recorder.process_chunk(
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
        )

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
            dev_name = sd.query_devices(self.device_id)['name'] if self.device_id is not None else ''
        except Exception:
            dev_name = ''
        return {
            'name':                self.name,
            'device_name':         dev_name,
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
        }

    @classmethod
    def from_dict(cls, d: dict):
        """Create a RecordingEntity from a settings dict.
        Returns (entity, warning_msg_or_None).
        """
        # Resolve device by name
        device_id = None
        warning = None
        dev_name = d.get('device_name', '')
        if dev_name:
            for i, info in enumerate(sd.query_devices()):
                if info['max_input_channels'] > 0 and info['name'] == dev_name:
                    device_id = i
                    break
            if device_id is None:
                # Try partial match
                for i, info in enumerate(sd.query_devices()):
                    if info['max_input_channels'] > 0 and dev_name.split('[')[0].strip() in info['name']:
                        device_id = i
                        break
            if device_id is None:
                warning = f"Device '{dev_name}' not found — using default"

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
                     'display_mode'):
            if attr in d:
                setattr(e, attr, d[attr])

        # Spec params that may need rebuild
        nperseg = d.get('spec_nperseg', SPECTROGRAM_NPERSEG)
        window  = d.get('spec_window', 'hann')
        if nperseg != SPECTROGRAM_NPERSEG or window != 'hann':
            e.change_fft_params(nperseg, window)

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

        e.rebuild_freq_mapping()
        return e, warning

    # ── Cleanup ───────────────────────────────────────────────────────────

    def close(self):
        try:
            self.capture.close()
        except Exception:
            pass
