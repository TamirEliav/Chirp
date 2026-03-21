"""
sap_rec.py — Sound Analysis & Recording  (v4 — multi-recording)
"""

import collections
import datetime
import json
import os
import queue
import sys
import threading

import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import numpy as np
import scipy.io.wavfile
import scipy.signal
import sounddevice as sd

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QPushButton, QSlider, QLabel, QLineEdit,
    QFileDialog, QFrame, QSizePolicy, QDoubleSpinBox, QComboBox, QCheckBox,
    QScrollArea, QStackedLayout, QDialog, QCalendarWidget, QMessageBox, QSpinBox,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QSize, QDate, QPointF
from PyQt5.QtGui import QFont, QPainter, QColor, QPainterPath, QPen, QPolygonF

# ── Audio ──────────────────────────────────────────────────────────────────────
SAMPLE_RATE         = 44100
CHANNELS            = 1
CHUNK_FRAMES        = 1024
DTYPE               = 'float32'

# ── Display ────────────────────────────────────────────────────────────────────
DISPLAY_SECONDS     = 10.0
SPECTROGRAM_NPERSEG = 4096
COLORMAP            = 'inferno'
ANIMATION_INTERVAL  = 50
SPEC_DB_MIN         = -100.0
SPEC_DB_MAX         = 0.0
N_DISPLAY_ROWS      = 256

# ── Recording defaults ─────────────────────────────────────────────────────────
DEFAULT_THRESHOLD   = 0.05
DEFAULT_MIN_CROSS   = 0.20
DEFAULT_HOLD        = 0.50
DEFAULT_MAX_REC     = 10.0
DEFAULT_PRE_TRIG    = 1.00
RECORDINGS_DIR      = './recordings'
DEFAULT_FREQ_LO     = 1000.0
DEFAULT_FREQ_HI     = 8000.0

# ── Catppuccin Mocha palette ───────────────────────────────────────────────────
C = {
    'base':     '#1e1e2e',
    'mantle':   '#181825',
    'surface0': '#313244',
    'surface1': '#45475a',
    'surface2': '#585b70',
    'text':     '#cdd6f4',
    'subtext':  '#a6adc8',
    'blue':     '#89b4fa',
    'green':    '#a6e3a1',
    'red':      '#f38ba8',
    'yellow':   '#f9e2af',
    'mauve':    '#cba6f7',
    'pink':     '#f5c2e7',
}

QSS = f"""
QMainWindow, QWidget {{
    background-color: {C['base']};
    color: {C['text']};
    font-family: 'Segoe UI';
    font-size: 10pt;
}}
QGroupBox {{
    border: 1px solid {C['surface1']};
    border-radius: 6px;
    margin-top: 10px;
    padding: 8px 10px 6px 10px;
    font-weight: bold;
    color: {C['blue']};
    font-size: 8pt;
    letter-spacing: 1px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 4px;
    background-color: {C['base']};
}}
QPushButton {{
    background-color: {C['surface0']};
    border: 1px solid {C['surface1']};
    border-radius: 5px;
    padding: 9px 20px;
    color: {C['text']};
    min-width: 120px;
    min-height: 36px;
    font-size: 10pt;
}}
QPushButton:hover   {{ background-color: {C['surface1']}; }}
QPushButton:pressed {{ background-color: {C['surface2']}; }}
QPushButton#btn_start_acq               {{ border-color: {C['blue']};  color: {C['blue']};  }}
QPushButton#btn_start_acq[active=true]  {{ background-color: {C['blue']};  color: {C['mantle']}; font-weight: bold; }}
QPushButton#btn_start_rec               {{ border-color: {C['green']}; color: {C['green']}; }}
QPushButton#btn_start_rec[active=true]  {{ background-color: {C['green']}; color: {C['mantle']}; font-weight: bold; }}
QPushButton#btn_stop_acq  {{ border-color: {C['red']}; color: {C['red']}; }}
QPushButton#btn_stop_rec  {{ border-color: {C['red']}; color: {C['red']}; }}
QPushButton#btn_browse    {{ min-width: 80px; }}
QPushButton#btn_small     {{ min-width: 28px; min-height: 24px; padding: 2px; font-size: 9pt; }}
QSlider::groove:horizontal {{
    height: 4px; background: {C['surface1']}; border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    background: {C['mauve']}; border-radius: 2px;
}}
QSlider::handle:horizontal {{
    width: 16px; height: 16px; margin: -6px 0;
    background: {C['mauve']}; border-radius: 8px;
    border: 2px solid {C['base']};
}}
QSlider::handle:horizontal:hover {{ background: {C['pink']}; }}
QLineEdit, QDoubleSpinBox, QComboBox {{
    background-color: {C['mantle']};
    border: 1px solid {C['surface1']};
    border-radius: 4px;
    padding: 5px 8px;
    color: {C['text']};
    selection-background-color: {C['blue']};
    min-height: 28px;
}}
QLineEdit:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {C['blue']};
}}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background-color: {C['surface0']};
    border: none;
    width: 18px;
}}
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{
    background-color: {C['surface1']};
}}
QComboBox::drop-down {{
    border: none;
    width: 24px;
    background-color: {C['surface0']};
}}
QComboBox QAbstractItemView {{
    background-color: {C['mantle']};
    border: 1px solid {C['surface1']};
    color: {C['text']};
    selection-background-color: {C['surface1']};
}}
QFrame[frameShape="4"] {{ color: {C['surface0']}; max-height: 1px; }}
QCheckBox {{ spacing: 6px; color: {C['text']}; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border: 1px solid {C['surface1']}; border-radius: 3px; background: {C['mantle']}; }}
QCheckBox::indicator:checked {{ background: {C['mauve']}; border-color: {C['mauve']}; }}
QCheckBox::indicator:hover {{ border-color: {C['blue']}; }}
QLabel#param_label {{ color: {C['subtext']}; min-width: 90px; font-size: 9pt; }}
QLabel#status_on   {{ color: {C['green']}; font-weight: bold; font-size: 10pt; }}
QLabel#status_off  {{ color: {C['surface2']};              font-size: 10pt; }}
QLabel#trig_active {{ color: {C['red']};   font-weight: bold; font-size: 10pt; }}
QLabel#trig_idle   {{ color: {C['surface2']};              font-size: 10pt; }}
QScrollArea {{ border: none; }}
"""


# ──────────────────────────────────────────────────────────────────────────────
# AudioCapture
# ──────────────────────────────────────────────────────────────────────────────
class AudioCapture:
    def __init__(self, audio_queue: queue.Queue, device=None, channels=1, samplerate=SAMPLE_RATE):
        self._queue    = audio_queue
        self._channels = channels
        self._stream   = None
        try:
            self._stream = sd.InputStream(
                samplerate=samplerate, channels=channels,
                dtype=DTYPE, blocksize=CHUNK_FRAMES,
                device=device,
                callback=self._callback,
            )
        except Exception as exc:
            print(f"[AudioCapture] Failed to open device {device}: {exc}")

    @property
    def valid(self):
        return self._stream is not None

    def _callback(self, indata, frames, time_info, status):
        try:
            if self._channels == 1:
                self._queue.put_nowait(indata[:, 0].copy())
            else:
                self._queue.put_nowait(indata[:, :2].copy())
        except queue.Full:
            pass

    def resume(self):
        if self._stream is not None:
            self._stream.start()

    def pause(self):
        if self._stream is not None:
            self._stream.stop()

    def close(self):
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:
                pass
            self._stream.close()
            self._stream = None


# ──────────────────────────────────────────────────────────────────────────────
# SpectrogramAccumulator
# ──────────────────────────────────────────────────────────────────────────────
class SpectrogramAccumulator:
    WINDOW_TYPES = ('hann', 'hamming', 'blackman', 'bartlett', 'flattop')
    FFT_SIZES    = (256, 512, 1024, 2048, 4096)

    def __init__(self, nperseg=SPECTROGRAM_NPERSEG, window='hann'):
        self._n       = nperseg
        self._overlap = np.zeros(self._n, dtype=np.float32)
        self._window  = scipy.signal.windows.get_window(window, self._n).astype(np.float32)

    def compute_column(self, chunk: np.ndarray) -> np.ndarray:
        combined    = np.concatenate([self._overlap, chunk])
        window_data = combined[-self._n:]
        self._overlap = window_data.copy()
        fft_mag = np.abs(np.fft.rfft(window_data * self._window))
        return (20.0 * np.log10(fft_mag + 1e-10)).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# BandpassFilter
# ──────────────────────────────────────────────────────────────────────────────
class BandpassFilter:
    def __init__(self, sample_rate=SAMPLE_RATE):
        self._sos    = None
        self._zi     = None
        self._params = (None, None)
        self._sample_rate = sample_rate

    def _redesign(self, low_hz: float, high_hz: float) -> bool:
        nyq = self._sample_rate * 0.5
        lo  = max(1.0, low_hz)
        hi  = min(nyq - 1.0, high_hz)
        if lo >= hi:
            self._sos = self._zi = None
            self._params = (low_hz, high_hz)
            return False
        self._sos    = scipy.signal.butter(4, [lo / nyq, hi / nyq],
                                           btype='band', output='sos')
        self._zi     = scipy.signal.sosfilt_zi(self._sos)
        self._params = (low_hz, high_hz)
        return True

    def get_peak(self, chunk: np.ndarray, low_hz: float, high_hz: float) -> float:
        _, peak = self.filter_chunk(chunk, low_hz, high_hz)
        return peak

    def filter_chunk(self, chunk: np.ndarray, low_hz: float, high_hz: float):
        """Return (filtered_signal, peak). If filter invalid, returns (chunk, peak)."""
        if (low_hz, high_hz) != self._params:
            if not self._redesign(low_hz, high_hz):
                return chunk, float(np.max(np.abs(chunk)))
        if self._sos is None:
            return chunk, float(np.max(np.abs(chunk)))
        filtered, self._zi = scipy.signal.sosfilt(self._sos, chunk, zi=self._zi)
        return filtered, float(np.max(np.abs(filtered)))

    def reset(self):
        if self._sos is not None:
            self._zi = scipy.signal.sosfilt_zi(self._sos)


# ──────────────────────────────────────────────────────────────────────────────
# ThresholdRecorder
# ──────────────────────────────────────────────────────────────────────────────
class ThresholdRecorder:
    _IDLE      = 'IDLE'
    _PENDING   = 'PENDING'
    _RECORDING = 'RECORDING'

    def __init__(self):
        self._state            = self._IDLE
        self._buf: list        = []
        self._pending_chunks: list = []
        self._pending_samples  = 0
        self._silent_count     = 0
        self._was_enabled      = False
        self._pre_trig_deque   = collections.deque(maxlen=1)
        self._pre_trig_maxlen  = 1

    def process_chunk(self, chunk: np.ndarray, *,
                      trigger_peak: float,
                      threshold: float, min_cross_sec: float, hold_sec: float,
                      max_rec_sec: float, pre_trig_sec: float,
                      output_dir: str, enabled: bool,
                      filename_prefix: str = '', filename_suffix: str = '',
                      sample_rate: int = SAMPLE_RATE):
        if self._was_enabled and not enabled:
            if self._state == self._RECORDING and self._buf:
                self._start_flush(self._buf, output_dir, filename_prefix, filename_suffix,
                                  sample_rate=sample_rate)
            self._reset()
        self._was_enabled = enabled

        needed = max(1, int(pre_trig_sec * sample_rate / CHUNK_FRAMES))
        if needed != self._pre_trig_maxlen:
            old = list(self._pre_trig_deque)
            self._pre_trig_deque  = collections.deque(old[-needed:], maxlen=needed)
            self._pre_trig_maxlen = needed

        if not enabled:
            self._pre_trig_deque.append(chunk.copy())
            return

        min_cross_samps = int(min_cross_sec * sample_rate)
        hold_samps      = int(hold_sec      * sample_rate)
        max_chunks      = max(1, int(max_rec_sec * sample_rate / CHUNK_FRAMES))

        if self._state == self._IDLE:
            if trigger_peak >= threshold:
                self._state           = self._PENDING
                self._pending_chunks  = [chunk.copy()]
                self._pending_samples = len(chunk)
            else:
                self._pre_trig_deque.append(chunk.copy())

        elif self._state == self._PENDING:
            if trigger_peak >= threshold:
                self._pending_chunks.append(chunk.copy())
                self._pending_samples += len(chunk)
                if self._pending_samples >= min_cross_samps:
                    self._state        = self._RECORDING
                    self._buf          = list(self._pre_trig_deque) + self._pending_chunks
                    self._pending_chunks  = []
                    self._pending_samples = 0
                    self._silent_count = 0
            else:
                for c in self._pending_chunks:
                    self._pre_trig_deque.append(c)
                self._pre_trig_deque.append(chunk.copy())
                self._pending_chunks  = []
                self._pending_samples = 0
                self._state = self._IDLE

        elif self._state == self._RECORDING:
            self._buf.append(chunk.copy())
            if trigger_peak < threshold:
                self._silent_count += len(chunk)
                if self._silent_count >= hold_samps:
                    self._start_flush(self._buf, output_dir, filename_prefix, filename_suffix,
                                      sample_rate=sample_rate)
                    self._reset()
            else:
                self._silent_count = 0

            if self._state == self._RECORDING and len(self._buf) > max_chunks:
                self._start_flush(self._buf, output_dir, filename_prefix, filename_suffix,
                                  sample_rate=sample_rate)
                self._buf = []

    def _reset(self):
        self._state           = self._IDLE
        self._buf             = []
        self._pending_chunks  = []
        self._pending_samples = 0
        self._silent_count    = 0
        self._pre_trig_deque.clear()

    @property
    def is_recording(self) -> bool:
        return self._state == self._RECORDING

    @staticmethod
    def _start_flush(buf_snapshot: list, output_dir: str,
                     prefix: str = '', suffix: str = '', sample_rate: int = SAMPLE_RATE):
        threading.Thread(
            target=ThresholdRecorder._write_wav,
            args=(list(buf_snapshot), output_dir, prefix, suffix, sample_rate),
            daemon=True,
        ).start()

    @staticmethod
    def _write_wav(buf_snapshot: list, output_dir: str,
                   prefix: str = '', suffix: str = '', sample_rate: int = SAMPLE_RATE):
        audio = np.concatenate(buf_snapshot)
        if audio.ndim == 1:
            audio = audio.flatten()
        pcm16 = (audio * 32767.0).clip(-32768, 32767).astype(np.int16)
        os.makedirs(output_dir, exist_ok=True)
        now   = datetime.datetime.now()
        epoch_ms = int(now.timestamp() * 1000)
        local_ts = now.strftime('%Y%m%d_%H%M%S_%f')[:-3]
        parts = [p for p in [prefix.rstrip('_'), str(epoch_ms), local_ts, suffix.lstrip('_')] if p]
        fname = '_'.join(parts) + '.wav'
        path  = os.path.join(output_dir, fname)
        scipy.io.wavfile.write(path, sample_rate, pcm16)
        n_samples = audio.shape[0]
        ch_str = 'stereo' if audio.ndim == 2 else 'mono'
        print(f'[REC] saved {path}  ({n_samples/sample_rate:.2f} s, {ch_str})')


# ──────────────────────────────────────────────────────────────────────────────
# RecordingEntity  — all per-recording state (no Qt widgets)
# ──────────────────────────────────────────────────────────────────────────────
class RecordingEntity:

    SUPPORTED_RATES = (8000, 16000, 22050, 44100, 48000, 96000)
    SUPPORTED_DISPLAY_SECONDS = (5.0, 10.0, 15.0, 20.0, 30.0, 60.0)

    def __init__(self, name: str = 'Recording 1', device_id=None, sample_rate=SAMPLE_RATE,
                 display_seconds=DISPLAY_SECONDS):
        self.name = name
        self.sample_rate = sample_rate
        self.display_seconds = float(display_seconds)

        # Derived sizes
        self._total_samples = int(self.display_seconds * self.sample_rate)
        self._n_cols        = int(self.display_seconds * self.sample_rate / CHUNK_FRAMES)

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
        self.max_rec_sec   = DEFAULT_MAX_REC
        self.pre_trig_sec  = DEFAULT_PRE_TRIG
        self.freq_filter_enabled = False
        self.freq_lo       = DEFAULT_FREQ_LO
        self.freq_hi       = DEFAULT_FREQ_HI

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
        self.n_freq_bins  = SPECTROGRAM_NPERSEG // 2 + 1
        self.amp_buffer   = np.zeros(self._total_samples, dtype=np.float32)
        self.amp_buffer_r = np.zeros(self._total_samples, dtype=np.float32)
        self.spec_buffer  = np.full(
            (self.n_freq_bins, self._n_cols), SPEC_DB_MIN, dtype=np.float32)
        self.spec_buffer_r = np.full(
            (self.n_freq_bins, self._n_cols), SPEC_DB_MIN, dtype=np.float32)
        self.write_head = 0
        self.col_head   = 0

        # Display state
        self.amp_ylim = 1.05  # amplitude y-axis max (persists across mode switches)

        # Runtime
        self.acq_running = False
        self.rec_enabled = False

        # Freq mapping
        self.freq_map_idx_floor = None
        self.freq_map_frac      = None
        self.display_freqs      = None
        self.rebuild_freq_mapping()

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
        self._total_samples = int(self.display_seconds * new_rate)
        self._n_cols = int(self.display_seconds * new_rate / CHUNK_FRAMES)
        self.display_freq_hi = min(self.display_freq_hi, float(new_rate // 2))

        # Rebuild buffers
        self.amp_buffer   = np.zeros(self._total_samples, dtype=np.float32)
        self.amp_buffer_r = np.zeros(self._total_samples, dtype=np.float32)
        self.spec_buffer  = np.full(
            (self.n_freq_bins, self._n_cols), SPEC_DB_MIN, dtype=np.float32)
        self.spec_buffer_r = np.full(
            (self.n_freq_bins, self._n_cols), SPEC_DB_MIN, dtype=np.float32)
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
        self._total_samples = int(self.display_seconds * self.sample_rate)
        self._n_cols = int(self.display_seconds * self.sample_rate / CHUNK_FRAMES)

        # Rebuild buffers
        self.amp_buffer   = np.zeros(self._total_samples, dtype=np.float32)
        self.amp_buffer_r = np.zeros(self._total_samples, dtype=np.float32)
        self.spec_buffer  = np.full(
            (self.n_freq_bins, self._n_cols), SPEC_DB_MIN, dtype=np.float32)
        self.spec_buffer_r = np.full(
            (self.n_freq_bins, self._n_cols), SPEC_DB_MIN, dtype=np.float32)
        self.write_head = 0
        self.col_head   = 0

    # ── Transport ─────────────────────────────────────────────────────────

    def start_acq(self):
        if not self.acq_running and self.capture.valid:
            self.capture.resume()
            self.acq_running = True

    def stop_acq(self):
        if self.acq_running:
            self.capture.pause()
            self.acq_running = False
            self.rec_enabled = False
            self.bpf.reset()
            self.bpf_r.reset()

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
        self.spec_buffer[:, self.col_head] = self.spec_acc.compute_column(display)
        if mode == 'Stereo':
            self.spec_buffer_r[:, self.col_head] = self.spec_acc_r.compute_column(right)

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
            if tm == 'Any Channel':
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

        # Write amplitude buffers (filtered when band filter active)
        if mode == 'Stereo':
            if end <= self._total_samples:
                self.amp_buffer  [self.write_head:end] = amp_l
                self.amp_buffer_r[self.write_head:end] = amp_r
            else:
                split = self._total_samples - self.write_head
                self.amp_buffer  [self.write_head:] = amp_l[:split]
                self.amp_buffer  [:end % self._total_samples] = amp_l[split:]
                self.amp_buffer_r[self.write_head:] = amp_r[:split]
                self.amp_buffer_r[:end % self._total_samples] = amp_r[split:]
        else:
            if end <= self._total_samples:
                self.amp_buffer[self.write_head:end] = amp_l
            else:
                split = self._total_samples - self.write_head
                self.amp_buffer[self.write_head:] = amp_l[:split]
                self.amp_buffer[:end % self._total_samples] = amp_l[split:]

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
            max_rec_sec   = self.max_rec_sec,
            pre_trig_sec  = self.pre_trig_sec,
            output_dir    = out_dir,
            enabled       = self.rec_enabled,
            filename_prefix = self.filename_prefix,
            filename_suffix = self.filename_suffix,
            sample_rate   = self.sample_rate,
        )

    # ── Mini amplitude for sidebar ────────────────────────────────────────

    def get_mini_amplitude(self, n_points: int = 200) -> np.ndarray:
        buf = np.abs(self.amp_buffer)
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
                     'min_cross_sec', 'hold_sec', 'max_rec_sec', 'pre_trig_sec',
                     'freq_filter_enabled', 'freq_lo', 'freq_hi',
                     'freq_scale', 'gain_db', 'db_floor', 'db_ceil',
                     'display_freq_lo', 'display_freq_hi',
                     'output_dir', 'filename_prefix', 'filename_suffix',
                     'dph_folder_prefix', 'amp_ylim'):
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


# ──────────────────────────────────────────────────────────────────────────────
# MiniAmplitudeWidget  — QPainter waveform preview for sidebar
# ──────────────────────────────────────────────────────────────────────────────
class MiniAmplitudeWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._data = np.zeros(200, dtype=np.float32)
        self._color = QColor(C['blue'])
        self.setFixedHeight(30)

    def set_data(self, data: np.ndarray):
        self._data = data
        self.update()

    def set_color(self, color: QColor):
        self._color = color

    def paintEvent(self, event):
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(C['mantle']))

        data = self._data
        n = len(data)
        if n < 2 or w < 2:
            p.end()
            return

        y_max = max(float(data.max()), 0.01)
        # Pre-compute all points as QPolygonF (much faster than per-point path)
        xs = np.linspace(0, w, n, dtype=np.float64)
        ys = h - (data / y_max) * h
        pts = [QPointF(0.0, float(h))]
        pts.extend(QPointF(float(xs[i]), float(ys[i])) for i in range(n))
        pts.append(QPointF(float(w), float(h)))
        poly = QPolygonF(pts)

        color = QColor(self._color)
        color.setAlpha(80)
        path = QPainterPath()
        path.addPolygon(poly)
        path.closeSubpath()
        p.fillPath(path, color)
        pen = QPen(self._color)
        pen.setWidthF(1.0)
        p.setPen(pen)
        p.drawPolyline(poly)
        p.end()


# ──────────────────────────────────────────────────────────────────────────────
# RecordingSidebarItem
# ──────────────────────────────────────────────────────────────────────────────
class RecordingSidebarItem(QWidget):
    clicked  = pyqtSignal(int)
    move_up  = pyqtSignal(int)
    move_down = pyqtSignal(int)
    delete   = pyqtSignal(int)
    renamed  = pyqtSignal(int, str)

    def __init__(self, index: int, name: str, parent=None):
        super().__init__(parent)
        self._index = index
        self._selected = False
        self._editing = False
        self._last_status = None  # cached (acq, rec, trig) to avoid redundant setStyleSheet

        self.setFixedHeight(82)
        self.setCursor(Qt.PointingHandCursor)

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(6, 4, 6, 4)
        vbox.setSpacing(2)

        # Row 1: name + status dots
        row1 = QHBoxLayout()
        row1.setSpacing(6)
        self._name_label = QLabel(name)
        self._name_label.setStyleSheet(f'color: {C["text"]}; font-weight: bold; font-size: 9pt; border: none;')
        self._name_edit = QLineEdit(name)
        self._name_edit.setStyleSheet(f'background: {C["surface0"]}; font-size: 9pt; min-height: 20px; padding: 1px 4px;')
        self._name_edit.hide()
        self._name_edit.editingFinished.connect(self._finish_edit)

        self._lbl_acq  = QLabel('\u25cf')
        self._lbl_rec  = QLabel('\u25cf')
        self._lbl_trig = QLabel('\u25cf')
        for lbl in (self._lbl_acq, self._lbl_rec, self._lbl_trig):
            lbl.setFixedWidth(12)
            lbl.setStyleSheet(f'color: {C["surface2"]}; font-size: 8pt;')

        row1.addWidget(self._name_label)
        row1.addWidget(self._name_edit)
        row1.addStretch()
        row1.addWidget(self._lbl_acq)
        row1.addWidget(self._lbl_rec)
        row1.addWidget(self._lbl_trig)
        vbox.addLayout(row1)

        # Row 2: mini amplitude
        self._mini_amp = MiniAmplitudeWidget()
        vbox.addWidget(self._mini_amp)

        # Row 3: move/delete buttons
        row3 = QHBoxLayout()
        row3.setSpacing(4)
        btn_up = QPushButton('\u25b2')
        btn_dn = QPushButton('\u25bc')
        btn_del = QPushButton('\u2715')
        for b in (btn_up, btn_dn, btn_del):
            b.setObjectName('btn_small')
            b.setFixedSize(24, 20)
        btn_up.clicked.connect(lambda: self.move_up.emit(self._index))
        btn_dn.clicked.connect(lambda: self.move_down.emit(self._index))
        btn_del.clicked.connect(lambda: self.delete.emit(self._index))
        row3.addWidget(btn_up)
        row3.addWidget(btn_dn)
        row3.addWidget(btn_del)
        row3.addStretch()
        vbox.addLayout(row3)

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, v):
        self._index = v

    def set_selected(self, sel: bool):
        self._selected = sel
        if sel:
            self.setStyleSheet(
                f'background: {C["surface0"]}; '
                f'border: 2px solid {C["blue"]}; border-radius: 6px; '
                f'border-left: 4px solid {C["blue"]};')
            self._name_label.setStyleSheet(
                f'color: {C["blue"]}; font-weight: bold; font-size: 9pt; border: none;')
        else:
            self.setStyleSheet(
                f'background: {C["mantle"]}; '
                f'border: 1px solid {C["surface1"]}; border-radius: 6px;')
            self._name_label.setStyleSheet(
                f'color: {C["text"]}; font-weight: bold; font-size: 9pt; border: none;')

    def set_name(self, name: str):
        self._name_label.setText(name)

    def update_status(self, acq: bool, rec: bool, trig: bool):
        key = (acq, rec, trig)
        if key == self._last_status:
            return
        self._last_status = key
        self._lbl_acq .setStyleSheet(f'color: {C["blue"] if acq else C["surface2"]}; font-size: 8pt;')
        self._lbl_rec .setStyleSheet(f'color: {C["green"] if rec else C["surface2"]}; font-size: 8pt;')
        self._lbl_trig.setStyleSheet(f'color: {C["red"] if trig else C["surface2"]}; font-size: 8pt;')

    def update_mini_amp(self, data: np.ndarray):
        self._mini_amp.set_data(data)

    def mouseDoubleClickEvent(self, event):
        self._start_edit()

    def mousePressEvent(self, event):
        if not self._editing:
            self.clicked.emit(self._index)

    def _start_edit(self):
        self._editing = True
        self._name_edit.setText(self._name_label.text())
        self._name_label.hide()
        self._name_edit.show()
        self._name_edit.setFocus()
        self._name_edit.selectAll()

    def _finish_edit(self):
        if not self._editing:
            return
        self._editing = False
        new_name = self._name_edit.text().strip()
        if not new_name:
            new_name = self._name_label.text()
        self._name_label.setText(new_name)
        self._name_edit.hide()
        self._name_label.show()
        self.renamed.emit(self._index, new_name)


# ──────────────────────────────────────────────────────────────────────────────
# RecordingSidebar
# ──────────────────────────────────────────────────────────────────────────────
class RecordingSidebar(QWidget):
    selection_changed = pyqtSignal(int)
    add_requested     = pyqtSignal()
    delete_requested  = pyqtSignal(int)
    move_requested    = pyqtSignal(int, int)   # (index, direction: -1 up, +1 down)
    item_renamed      = pyqtSignal(int, str)
    start_all_acq     = pyqtSignal()
    stop_all_acq      = pyqtSignal()
    start_all_rec     = pyqtSignal()
    stop_all_rec      = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(250)
        self._items: list[RecordingSidebarItem] = []

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(4, 4, 4, 4)
        vbox.setSpacing(4)

        # All-recordings transport
        all_row1 = QHBoxLayout()
        all_row1.setSpacing(4)
        btn_sa = QPushButton('Start All Acq')
        btn_xa = QPushButton('Stop All Acq')
        btn_sa.setObjectName('btn_start_acq')
        btn_xa.setObjectName('btn_stop_acq')
        for b in (btn_sa, btn_xa):
            b.setFixedHeight(28)
            b.setStyleSheet(b.styleSheet() + 'min-width: 0px; padding: 4px 8px; font-size: 9pt;')
        btn_sa.clicked.connect(self.start_all_acq.emit)
        btn_xa.clicked.connect(self.stop_all_acq.emit)
        all_row1.addWidget(btn_sa)
        all_row1.addWidget(btn_xa)
        vbox.addLayout(all_row1)

        all_row2 = QHBoxLayout()
        all_row2.setSpacing(4)
        btn_sr = QPushButton('Start All Rec')
        btn_xr = QPushButton('Stop All Rec')
        btn_sr.setObjectName('btn_start_rec')
        btn_xr.setObjectName('btn_stop_rec')
        for b in (btn_sr, btn_xr):
            b.setFixedHeight(28)
            b.setStyleSheet(b.styleSheet() + 'min-width: 0px; padding: 4px 8px; font-size: 9pt;')
        btn_sr.clicked.connect(self.start_all_rec.emit)
        btn_xr.clicked.connect(self.stop_all_rec.emit)
        all_row2.addWidget(btn_sr)
        all_row2.addWidget(btn_xr)
        vbox.addLayout(all_row2)

        btn_add = QPushButton('+  Add Recording')
        btn_add.setObjectName('btn_browse')
        btn_add.setFixedHeight(32)
        btn_add.clicked.connect(self.add_requested.emit)
        vbox.addWidget(btn_add)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll_widget = QWidget()
        self._scroll_layout = QVBoxLayout(self._scroll_widget)
        self._scroll_layout.setContentsMargins(0, 0, 0, 0)
        self._scroll_layout.setSpacing(4)
        self._scroll_layout.addStretch()
        scroll.setWidget(self._scroll_widget)
        vbox.addWidget(scroll)

    def add_item(self, name: str) -> int:
        idx = len(self._items)
        item = RecordingSidebarItem(idx, name)
        item.clicked.connect(self._on_item_clicked)
        item.move_up.connect(lambda i: self.move_requested.emit(i, -1))
        item.move_down.connect(lambda i: self.move_requested.emit(i, 1))
        item.delete.connect(self.delete_requested.emit)
        item.renamed.connect(self.item_renamed.emit)
        self._scroll_layout.insertWidget(idx, item)
        self._items.append(item)
        return idx

    def remove_item(self, idx: int):
        if 0 <= idx < len(self._items):
            item = self._items.pop(idx)
            self._scroll_layout.removeWidget(item)
            item.deleteLater()
            self._reindex()

    def select(self, idx: int):
        for i, item in enumerate(self._items):
            item.set_selected(i == idx)

    def swap_items(self, idx_a: int, idx_b: int):
        if not (0 <= idx_a < len(self._items) and 0 <= idx_b < len(self._items)):
            return
        # Swap in list
        self._items[idx_a], self._items[idx_b] = self._items[idx_b], self._items[idx_a]
        # Remove both from layout (higher index first to preserve lower)
        hi, lo = max(idx_a, idx_b), min(idx_a, idx_b)
        self._scroll_layout.removeWidget(self._items[hi])
        self._scroll_layout.removeWidget(self._items[lo])
        # Re-insert at correct positions
        self._scroll_layout.insertWidget(lo, self._items[lo])
        self._scroll_layout.insertWidget(hi, self._items[hi])
        self._reindex()

    def update_item_status(self, idx: int, acq: bool, rec: bool, trig: bool):
        if 0 <= idx < len(self._items):
            self._items[idx].update_status(acq, rec, trig)

    def update_item_amp(self, idx: int, data: np.ndarray):
        if 0 <= idx < len(self._items):
            self._items[idx].update_mini_amp(data)

    def update_item_name(self, idx: int, name: str):
        if 0 <= idx < len(self._items):
            self._items[idx].set_name(name)

    def clear_all(self):
        for item in self._items:
            self._scroll_layout.removeWidget(item)
            item.deleteLater()
        self._items.clear()

    def count(self):
        return len(self._items)

    def _reindex(self):
        for i, item in enumerate(self._items):
            item.index = i

    def _on_item_clicked(self, idx: int):
        self.selection_changed.emit(idx)


# ──────────────────────────────────────────────────────────────────────────────
# SapWindow
# ──────────────────────────────────────────────────────────────────────────────
class SapWindow(QMainWindow):

    _THR_SCALE  = 1000
    _TIME_SCALE = 100
    _DB_SCALE   = 10

    def __init__(self):
        super().__init__()

        # Entities
        self._entities: list[RecordingEntity] = []
        self._selected_idx = -1
        self._next_num = 1
        self._dragging = False

        # View mode
        self._view_mode = False
        self._vm_axes: list[dict] = []
        self._vm_n_cols = 1
        self._vm_panel_height = 300

        # Time axis (regenerated per entity's sample_rate)
        self._t_axis_key = (0, 0)  # track (sample_rate, display_seconds) for cached t_axis
        self._t_axis = np.array([], dtype=np.float32)

        self._build_figure()
        self._build_ui()
        self._connect_signals()

        # Create first recording entity
        self._add_recording()

        self._timer = QTimer(self)
        self._timer.setInterval(ANIMATION_INTERVAL)
        self._timer.timeout.connect(self._update_plot)
        self._timer.start()

        self.setWindowTitle('SAP_rec — Sound Analysis & Recording')
        self.resize(1400, 850)

    # ──────────────────────────────────────────────────────────────────────
    # matplotlib figure
    # ──────────────────────────────────────────────────────────────────────

    def _build_figure(self):
        plt.rcParams.update({
            'figure.facecolor': C['base'],
            'axes.facecolor':   C['mantle'],
            'axes.edgecolor':   C['surface1'],
            'axes.labelcolor':  C['subtext'],
            'xtick.color':      C['subtext'],
            'ytick.color':      C['subtext'],
            'xtick.labelsize':  8,
            'ytick.labelsize':  8,
            'axes.labelsize':   9,
        })
        self._fig = plt.figure()
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._bg = None
        self._canvas.mpl_connect('resize_event', self._recapture_bg)
        self._is_stereo_layout = False
        self._setup_axes(stereo=False)

    def _setup_axes(self, stereo=False):
        self._fig.clf()
        self._is_stereo_layout = stereo
        e = self._sel if self._selected_idx >= 0 else None

        db_floor = e.db_floor if e else SPEC_DB_MIN
        db_ceil  = e.db_ceil  if e else SPEC_DB_MAX
        if db_floor >= db_ceil:
            db_floor = db_ceil - 0.1
        threshold = e.threshold if e else DEFAULT_THRESHOLD
        sr = e.sample_rate if e else SAMPLE_RATE
        disp_secs = e.display_seconds if e else DISPLAY_SECONDS
        n_cols = e._n_cols if e else int(disp_secs * sr / CHUNK_FRAMES)

        if stereo:
            gs = self._fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.10)
            self._ax_spec   = self._fig.add_subplot(gs[0])
            self._ax_spec_r = self._fig.add_subplot(gs[1], sharex=self._ax_spec)
            self._ax_amp    = self._fig.add_subplot(gs[2], sharex=self._ax_spec)
        else:
            gs = self._fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.08)
            self._ax_spec   = self._fig.add_subplot(gs[0])
            self._ax_spec_r = None
            self._ax_amp    = self._fig.add_subplot(gs[1], sharex=self._ax_spec)

        self._fig.subplots_adjust(top=0.97, bottom=0.06, left=0.07, right=0.99)
        n_disp = N_DISPLAY_ROWS
        dummy  = np.full((n_disp, n_cols), db_floor, dtype=np.float32)

        self._spec_im = self._ax_spec.imshow(
            dummy, aspect='auto', origin='lower',
            extent=[0.0, disp_secs, 0, n_disp],
            vmin=db_floor, vmax=db_ceil,
            cmap=COLORMAP, interpolation='nearest',
        )
        self._ax_spec.set_ylabel('Freq (Hz) \u2014 L' if stereo else 'Frequency (Hz)')
        self._cursor_spec = self._ax_spec.axvline(x=0.0, color=C['green'], linewidth=1.0, alpha=0.7)

        if stereo:
            self._spec_im_r = self._ax_spec_r.imshow(
                dummy, aspect='auto', origin='lower',
                extent=[0.0, disp_secs, 0, n_disp],
                vmin=db_floor, vmax=db_ceil,
                cmap=COLORMAP, interpolation='nearest',
            )
            self._ax_spec_r.set_ylabel('Freq (Hz) \u2014 R')
            self._cursor_spec_r = self._ax_spec_r.axvline(x=0.0, color=C['green'], linewidth=1.0, alpha=0.7)
        else:
            self._spec_im_r = None
            self._cursor_spec_r = None

        ts = int(disp_secs * sr)
        t_axis = self._get_t_axis(sr, disp_secs)
        (self._amp_line,) = self._ax_amp.plot(
            t_axis, np.zeros(ts),
            color=C['blue'], linewidth=0.7, antialiased=False,
            label='L' if stereo else None,
        )
        if stereo:
            (self._amp_line_r,) = self._ax_amp.plot(
                t_axis, np.zeros(ts),
                color=C['pink'], linewidth=0.7, antialiased=False, label='R',
            )
            self._ax_amp.legend(loc='upper right', fontsize=8,
                                facecolor=C['mantle'], edgecolor=C['surface1'],
                                labelcolor=C['text'])
        else:
            self._amp_line_r = None

        self._ax_amp.set_xlim(0.0, disp_secs)
        self._ax_amp.set_ylim(0.0, e.amp_ylim if e else 1.05)
        self._ax_amp.set_xlabel('Time (s)')
        self._ax_amp.set_ylabel('Amplitude')

        self._threshold_line = self._ax_amp.axhline(
            y=threshold, color=C['yellow'], linewidth=1.5, linestyle=(0, (6, 3)),
        )
        self._threshold_label = self._ax_amp.text(
            0.005, threshold + 0.03, f'thr = {threshold:.3f}',
            transform=self._ax_amp.get_yaxis_transform(),
            color=C['yellow'], fontsize=8,
        )
        self._cursor_amp = self._ax_amp.axvline(x=0.0, color=C['green'], linewidth=1.0, alpha=0.7)

        if e:
            self._update_spec_yticks(e)

        # Mark animated artists for blitting
        for a in self._get_config_artists():
            a.set_animated(True)
        self._canvas.draw()
        self._bg = self._canvas.copy_from_bbox(self._fig.bbox)

    def _get_config_artists(self):
        """Return list of all animated artists in config mode."""
        arts = [self._spec_im, self._amp_line,
                self._cursor_spec, self._cursor_amp,
                self._threshold_line, self._threshold_label]
        if self._spec_im_r is not None:
            arts.extend([self._spec_im_r, self._cursor_spec_r])
        if self._amp_line_r is not None:
            arts.append(self._amp_line_r)
        return arts

    @staticmethod
    def _get_vm_artists(vm: dict):
        """Return animated artists for one view-mode cell."""
        arts = [vm['spec_im'], vm['amp_line'],
                vm['cursor_spec'], vm['cursor_amp'],
                vm['thr_line'], vm['status_text']]
        if vm['amp_line_r'] is not None:
            arts.append(vm['amp_line_r'])
        return arts

    def _recapture_bg(self, event=None):
        """Re-capture background after resize."""
        if self._view_mode:
            for vm in self._vm_axes:
                for a in self._get_vm_artists(vm):
                    a.set_animated(True)
            self._canvas.draw()
            self._bg = self._canvas.copy_from_bbox(self._fig.bbox)
        else:
            for a in self._get_config_artists():
                a.set_animated(True)
            self._canvas.draw()
            self._bg = self._canvas.copy_from_bbox(self._fig.bbox)

    @staticmethod
    def _apply_spec_yticks(ax, e: RecordingEntity):
        """Apply frequency y-tick labels to a spectrogram axis."""
        n_dst = N_DISPLAY_ROWS
        f_lo = e.display_freqs[0] if len(e.display_freqs) else 0.0
        f_hi = e.display_freqs[-1] if len(e.display_freqs) else e.sample_rate / 2
        if e.freq_scale == 'Linear':
            step = max(500, round((f_hi - f_lo) / 8 / 500) * 500) or 5000
            tick_freqs = np.arange(
                np.ceil(f_lo / step) * step,
                f_hi + 1,
                step)
        else:
            tick_freqs = np.array([50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
            tick_freqs = tick_freqs[(tick_freqs >= f_lo) & (tick_freqs <= f_hi)]
        tick_rows = np.interp(tick_freqs, e.display_freqs, np.arange(n_dst))
        labels = [f'{f/1000:.0f}k' if f >= 1000 else f'{int(f)}' for f in tick_freqs]
        ax.set_yticks(tick_rows)
        ax.set_yticklabels(labels, fontsize=7)

    def _update_spec_yticks(self, e: RecordingEntity):
        self._apply_spec_yticks(self._ax_spec, e)
        if self._ax_spec_r is not None:
            self._apply_spec_yticks(self._ax_spec_r, e)

    def _get_t_axis(self, sr: int, disp_secs: float = DISPLAY_SECONDS) -> np.ndarray:
        """Return a time axis array for the given sample rate + buffer length, cached."""
        key = (sr, disp_secs)
        if key != self._t_axis_key:
            self._t_axis_key = key
            self._t_axis = np.linspace(0.0, disp_secs,
                                       int(disp_secs * sr),
                                       endpoint=False, dtype=np.float32)
        return self._t_axis

    @property
    def _sel(self) -> RecordingEntity:
        if 0 <= self._selected_idx < len(self._entities):
            return self._entities[self._selected_idx]
        return None

    # ──────────────────────────────────────────────────────────────────────
    # Qt layout
    # ──────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        hbox = QHBoxLayout(root)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.setSpacing(0)

        # Sidebar
        self._sidebar = RecordingSidebar()
        hbox.addWidget(self._sidebar)

        # Right panel (existing layout)
        right = QWidget()
        vbox = QVBoxLayout(right)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        # Canvas inside a scroll area (scrollable in view mode)
        self._canvas_scroll = QScrollArea()
        self._canvas_scroll.setWidgetResizable(True)
        self._canvas_scroll.setWidget(self._canvas)
        self._canvas_scroll.setFrameShape(QFrame.NoFrame)
        vbox.addWidget(self._canvas_scroll, stretch=1)

        # View-mode toolbar (hidden initially)
        self._view_toolbar = self._build_view_toolbar()
        vbox.addWidget(self._view_toolbar)
        self._view_toolbar.hide()

        # Config panels — store refs for hide/show in view mode
        self._config_widgets: list[QWidget] = []
        for builder in (self._build_transport, self._build_params,
                        self._build_spec_params, self._build_settings):
            hl = self._hline()
            panel = builder()
            vbox.addWidget(hl)
            vbox.addWidget(panel)
            self._config_widgets.append(hl)
            self._config_widgets.append(panel)

        hbox.addWidget(right, stretch=1)

    def _build_view_toolbar(self) -> QWidget:
        """Thin toolbar shown only in view mode: Config button + layout controls."""
        w = QWidget()
        w.setStyleSheet(f'background-color: {C["mantle"]};')
        w.setFixedHeight(40)
        h = QHBoxLayout(w)
        h.setContentsMargins(10, 4, 10, 4)
        h.setSpacing(14)

        self._btn_config_mode = QPushButton('\u2190  Config Mode')
        self._btn_config_mode.setStyleSheet(
            f'QPushButton {{ background-color: {C["surface0"]}; color: {C["green"]}; '
            f'border: 1px solid {C["green"]}; border-radius: 5px; '
            f'padding: 5px 14px; font-weight: bold; min-width: 0px; }}'
            f'QPushButton:hover {{ background-color: {C["surface1"]}; }}'
        )
        self._btn_config_mode.clicked.connect(self._toggle_view_mode)
        h.addWidget(self._btn_config_mode)

        h.addStretch()

        lbl_c = QLabel('Columns:')
        lbl_c.setStyleSheet(f'color: {C["subtext"]}; font-size: 9pt;')
        h.addWidget(lbl_c)
        self._vm_cols_spin = QSpinBox()
        self._vm_cols_spin.setRange(1, 6)
        self._vm_cols_spin.setValue(1)
        self._vm_cols_spin.setFixedWidth(50)
        self._vm_cols_spin.setStyleSheet(
            f'QSpinBox {{ background-color: {C["surface0"]}; color: {C["text"]}; '
            f'border: 1px solid {C["surface1"]}; border-radius: 3px; padding: 2px; }}'
        )
        self._vm_cols_spin.valueChanged.connect(self._on_vm_cols_changed)
        h.addWidget(self._vm_cols_spin)

        h.addSpacing(10)

        lbl_h = QLabel('Height:')
        lbl_h.setStyleSheet(f'color: {C["subtext"]}; font-size: 9pt;')
        h.addWidget(lbl_h)
        self._vm_height_sl = QSlider(Qt.Horizontal)
        self._vm_height_sl.setRange(120, 700)
        self._vm_height_sl.setValue(self._vm_panel_height)
        self._vm_height_sl.setFixedWidth(180)
        self._vm_height_sl.valueChanged.connect(self._on_vm_height_changed)
        h.addWidget(self._vm_height_sl)
        self._vm_height_lbl = QLabel(f'{self._vm_panel_height}px')
        self._vm_height_lbl.setFixedWidth(45)
        self._vm_height_lbl.setStyleSheet(f'color: {C["subtext"]}; font-size: 9pt;')
        h.addWidget(self._vm_height_lbl)

        return w

    def _hline(self) -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.HLine)
        f.setFrameShadow(QFrame.Plain)
        return f

    # ── Transport ─────────────────────────────────────────────────────────

    def _build_transport(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet(f'background-color: {C["mantle"]};')
        h = QHBoxLayout(w)
        h.setContentsMargins(14, 10, 14, 10)
        h.setSpacing(18)

        acq_box = QGroupBox('ACQUISITION')
        acq_h   = QHBoxLayout(acq_box)
        acq_h.setSpacing(10)
        self._btn_start_acq = QPushButton('Start Acq')
        self._btn_stop_acq  = QPushButton('Stop Acq')
        self._btn_start_acq.setObjectName('btn_start_acq')
        self._btn_stop_acq .setObjectName('btn_stop_acq')
        acq_h.addWidget(self._btn_start_acq)
        acq_h.addWidget(self._btn_stop_acq)

        rec_box = QGroupBox('RECORDING')
        rec_h   = QHBoxLayout(rec_box)
        rec_h.setSpacing(10)
        self._btn_start_rec = QPushButton('Start Rec')
        self._btn_stop_rec  = QPushButton('Stop Rec')
        self._btn_start_rec.setObjectName('btn_start_rec')
        self._btn_stop_rec .setObjectName('btn_stop_rec')
        rec_h.addWidget(self._btn_start_rec)
        rec_h.addWidget(self._btn_stop_rec)

        status_box = QGroupBox('STATUS')
        status_v   = QVBoxLayout(status_box)
        status_v.setSpacing(4)
        self._lbl_acq_status  = QLabel('ACQ  \u25cf  STOPPED')
        self._lbl_rec_status  = QLabel('REC  \u25cf  STOPPED')
        self._lbl_trig_status = QLabel('TRIG \u25cf  IDLE')
        self._lbl_acq_status .setObjectName('status_off')
        self._lbl_rec_status .setObjectName('status_off')
        self._lbl_trig_status.setObjectName('trig_idle')
        mono = QFont('Consolas', 10)
        for lbl in (self._lbl_acq_status, self._lbl_rec_status, self._lbl_trig_status):
            lbl.setFont(mono)
        status_v.addWidget(self._lbl_acq_status)
        status_v.addWidget(self._lbl_rec_status)
        status_v.addWidget(self._lbl_trig_status)
        self._blink_counter = 0

        self._btn_reset = QPushButton('Reset Params')
        self._btn_reset.setObjectName('btn_browse')

        self._btn_view_mode = QPushButton('\u25a3  View Mode')
        self._btn_view_mode.setObjectName('btn_view_mode')
        self._btn_view_mode.setStyleSheet(
            f'QPushButton {{ background-color: {C["surface0"]}; color: {C["mauve"]}; '
            f'border: 1px solid {C["mauve"]}; border-radius: 5px; '
            f'padding: 6px 16px; font-weight: bold; min-width: 0px; }}'
            f'QPushButton:hover {{ background-color: {C["surface1"]}; }}'
        )

        self._btn_save = QPushButton('\U0001f4be Save')
        self._btn_load = QPushButton('\U0001f4c2 Load')
        for btn in (self._btn_save, self._btn_load):
            btn.setObjectName('btn_browse')

        h.addWidget(acq_box)
        h.addWidget(rec_box)
        h.addWidget(self._btn_reset)
        h.addWidget(self._btn_save)
        h.addWidget(self._btn_load)
        h.addWidget(self._btn_view_mode)
        h.addStretch()
        h.addWidget(status_box)
        return w

    # ── Trigger Parameters ────────────────────────────────────────────────

    def _build_params(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet(f'background-color: {C["mantle"]};')
        outer = QHBoxLayout(w)
        outer.setContentsMargins(14, 6, 14, 6)

        # Hidden threshold spinbox (synced from amplitude graph drag)
        self._sb_thr = QDoubleSpinBox()
        self._sb_thr.setRange(0.0, 1.0)
        self._sb_thr.setSingleStep(0.005)
        self._sb_thr.setDecimals(3)
        self._sb_thr.setValue(DEFAULT_THRESHOLD)
        self._sb_thr.hide()

        # Timing group
        time_box = QGroupBox('TRIGGER TIMING')
        time_g   = QGridLayout(time_box)
        time_g.setVerticalSpacing(8)
        time_g.setHorizontalSpacing(10)

        self._sb_mc = self._spinbox_row(time_g, 0, 0, 'Min Cross',
            sb_min=0.0, sb_max=60.0, sb_step=0.1, sb_dec=2,
            suffix=' s', init=DEFAULT_MIN_CROSS)
        self._sb_hold = self._spinbox_row(time_g, 0, 2, 'Hold',
            sb_min=0.0, sb_max=60.0, sb_step=0.1, sb_dec=2,
            suffix=' s', init=DEFAULT_HOLD)
        self._sb_maxr = self._spinbox_row(time_g, 1, 0, 'Max Rec',
            sb_min=1.0, sb_max=3600.0, sb_step=1.0, sb_dec=1,
            suffix=' s', init=DEFAULT_MAX_REC)
        self._sb_pre = self._spinbox_row(time_g, 1, 2, 'Pre-Trigger',
            sb_min=0.0, sb_max=60.0, sb_step=0.1, sb_dec=2,
            suffix=' s', init=DEFAULT_PRE_TRIG)

        # Freq filter group
        freq_box = QGroupBox('TRIGGER BAND FILTER')
        freq_g   = QGridLayout(freq_box)
        freq_g.setVerticalSpacing(8)
        freq_g.setHorizontalSpacing(10)

        self._chk_freq = QCheckBox('Enable')
        self._chk_freq.setChecked(False)

        self._sb_freq_lo = QDoubleSpinBox()
        self._sb_freq_lo.setRange(1.0, SAMPLE_RATE / 2 - 1)
        self._sb_freq_lo.setValue(DEFAULT_FREQ_LO)
        self._sb_freq_lo.setSingleStep(100.0)
        self._sb_freq_lo.setDecimals(0)
        self._sb_freq_lo.setSuffix(' Hz')
        self._sb_freq_lo.setFixedWidth(100)
        self._sb_freq_lo.setEnabled(False)

        self._sb_freq_hi = QDoubleSpinBox()
        self._sb_freq_hi.setRange(1.0, SAMPLE_RATE / 2 - 1)
        self._sb_freq_hi.setValue(DEFAULT_FREQ_HI)
        self._sb_freq_hi.setSingleStep(100.0)
        self._sb_freq_hi.setDecimals(0)
        self._sb_freq_hi.setSuffix(' Hz')
        self._sb_freq_hi.setFixedWidth(100)
        self._sb_freq_hi.setEnabled(False)

        self._chk_freq.toggled.connect(lambda on: (
            self._sb_freq_lo.setEnabled(on),
            self._sb_freq_hi.setEnabled(on),
        ))

        lbl_lo = QLabel('Lo')
        lbl_lo.setObjectName('param_label')
        lbl_hi = QLabel('Hi')
        lbl_hi.setObjectName('param_label')
        freq_g.addWidget(self._chk_freq,     0, 0)
        freq_g.addWidget(lbl_lo,             0, 1)
        freq_g.addWidget(self._sb_freq_lo,   0, 2)
        freq_g.addWidget(lbl_hi,             0, 3)
        freq_g.addWidget(self._sb_freq_hi,   0, 4)

        outer.addWidget(time_box, stretch=2)
        outer.addWidget(freq_box, stretch=1)
        return w

    def _spinbox_row(self, grid, row, col, label, *,
                     sb_min, sb_max, sb_step, sb_dec, suffix, init) -> QDoubleSpinBox:
        lbl = QLabel(label)
        lbl.setObjectName('param_label')
        lbl.setFixedWidth(70)
        sb = QDoubleSpinBox()
        sb.setRange(sb_min, sb_max)
        sb.setSingleStep(sb_step)
        sb.setDecimals(sb_dec)
        sb.setValue(init)
        sb.setSuffix(suffix)
        sb.setFixedWidth(100)
        sb.setAlignment(Qt.AlignRight)
        grid.addWidget(lbl, row, col)
        grid.addWidget(sb,  row, col + 1)
        return sb

    # ── Spectrogram display parameters ────────────────────────────────────

    def _build_spec_params(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet(f'background-color: {C["mantle"]};')
        outer = QHBoxLayout(w)
        outer.setContentsMargins(14, 6, 14, 6)

        box  = QGroupBox('SPECTROGRAM DISPLAY')
        grid = QGridLayout(box)
        grid.setVerticalSpacing(8)
        grid.setHorizontalSpacing(10)
        grid.setColumnStretch(1, 3)
        grid.setColumnStretch(4, 1)

        self._sl_gain, self._sb_gain = self._param_row(grid, 0, 0, 'Gain',
            sl_min=-200, sl_max=600, sl_init=0,
            sb_min=-20.0, sb_max=60.0, sb_step=1.0, sb_dec=1, suffix=' dB',
            scale=self._DB_SCALE)

        self._sl_floor, self._sb_floor = self._param_row(grid, 1, 0, 'dB Floor',
            sl_min=-1200, sl_max=0, sl_init=int(SPEC_DB_MIN * self._DB_SCALE),
            sb_min=-120.0, sb_max=0.0, sb_step=1.0, sb_dec=1, suffix=' dB',
            scale=self._DB_SCALE)

        self._sl_ceil, self._sb_ceil = self._param_row(grid, 2, 0, 'dB Ceil',
            sl_min=-1200, sl_max=0, sl_init=int(SPEC_DB_MAX * self._DB_SCALE),
            sb_min=-120.0, sb_max=0.0, sb_step=1.0, sb_dec=1, suffix=' dB',
            scale=self._DB_SCALE)

        lbl_fft = QLabel('FFT')
        lbl_fft.setObjectName('param_label')
        lbl_fft.setFixedWidth(55)
        self._combo_fft = QComboBox()
        self._combo_fft.setFixedWidth(90)
        for sz in SpectrogramAccumulator.FFT_SIZES:
            self._combo_fft.addItem(str(sz), userData=sz)
        self._combo_fft.setCurrentText(str(SPECTROGRAM_NPERSEG))

        lbl_win = QLabel('Win')
        lbl_win.setObjectName('param_label')
        lbl_win.setFixedWidth(55)
        self._combo_win = QComboBox()
        self._combo_win.setFixedWidth(90)
        for wn in SpectrogramAccumulator.WINDOW_TYPES:
            self._combo_win.addItem(wn.capitalize(), userData=wn)
        self._combo_win.setCurrentIndex(0)

        lbl_fscale = QLabel('Scale')
        lbl_fscale.setObjectName('param_label')
        lbl_fscale.setFixedWidth(55)
        self._combo_fscale = QComboBox()
        self._combo_fscale.setFixedWidth(90)
        self._combo_fscale.addItems(['Linear', 'Log', 'Mel'])
        self._combo_fscale.setCurrentText('Mel')

        lbl_dfl = QLabel('Lo')
        lbl_dfl.setObjectName('param_label')
        lbl_dfl.setFixedWidth(55)
        self._sb_disp_freq_lo = QDoubleSpinBox()
        self._sb_disp_freq_lo.setRange(0.0, SAMPLE_RATE / 2 - 1)
        self._sb_disp_freq_lo.setValue(0.0)
        self._sb_disp_freq_lo.setSingleStep(100.0)
        self._sb_disp_freq_lo.setDecimals(0)
        self._sb_disp_freq_lo.setSuffix(' Hz')
        self._sb_disp_freq_lo.setFixedWidth(90)

        lbl_dfh = QLabel('Hi')
        lbl_dfh.setObjectName('param_label')
        lbl_dfh.setFixedWidth(55)
        self._sb_disp_freq_hi = QDoubleSpinBox()
        self._sb_disp_freq_hi.setRange(1.0, SAMPLE_RATE / 2)
        self._sb_disp_freq_hi.setValue(SAMPLE_RATE / 2)
        self._sb_disp_freq_hi.setSingleStep(100.0)
        self._sb_disp_freq_hi.setDecimals(0)
        self._sb_disp_freq_hi.setSuffix(' Hz')
        self._sb_disp_freq_hi.setFixedWidth(90)

        grid.addWidget(lbl_fft,            0, 3)
        grid.addWidget(self._combo_fft,    0, 4)
        grid.addWidget(lbl_win,            1, 3)
        grid.addWidget(self._combo_win,    1, 4)
        grid.addWidget(lbl_fscale,         2, 3)
        grid.addWidget(self._combo_fscale, 2, 4)
        grid.addWidget(lbl_dfl,            3, 3)
        grid.addWidget(self._sb_disp_freq_lo,  3, 4)
        grid.addWidget(lbl_dfh,            4, 3)
        grid.addWidget(self._sb_disp_freq_hi,  4, 4)

        self._chk_shared_spec = QCheckBox('Sync across all recordings')
        grid.addWidget(self._chk_shared_spec, 5, 0, 1, 5)

        outer.addWidget(box)
        return w

    def _param_row(self, grid, row, col, label,
                   sl_min, sl_max, sl_init,
                   sb_min, sb_max, sb_step, sb_dec, suffix,
                   scale) -> tuple:
        lbl = QLabel(label)
        lbl.setObjectName('param_label')
        lbl.setFixedWidth(90)

        sl = QSlider(Qt.Horizontal)
        sl.setRange(sl_min, sl_max)
        sl.setValue(sl_init)

        sb = QDoubleSpinBox()
        sb.setRange(sb_min, sb_max)
        sb.setSingleStep(sb_step)
        sb.setDecimals(sb_dec)
        sb.setValue(sl_init / scale)
        sb.setSuffix(suffix)
        sb.setFixedWidth(100)
        sb.setAlignment(Qt.AlignRight)

        def _on_sl(v, _sb=sb, _sc=scale):
            _sb.blockSignals(True)
            _sb.setValue(v / _sc)
            _sb.blockSignals(False)

        def _on_sb(v, _sl=sl, _sc=scale):
            iv = max(_sl.minimum(), min(_sl.maximum(), int(round(v * _sc))))
            _sl.blockSignals(True)
            _sl.setValue(iv)
            _sl.blockSignals(False)

        sl.valueChanged.connect(_on_sl)
        sb.valueChanged.connect(_on_sb)

        grid.addWidget(lbl, row, col)
        grid.addWidget(sl,  row, col + 1)
        grid.addWidget(sb,  row, col + 2)
        return sl, sb

    # ── Settings ──────────────────────────────────────────────────────────

    def _build_settings(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet(f'background-color: {C["mantle"]};')
        outer = QHBoxLayout(w)
        outer.setContentsMargins(14, 6, 14, 10)
        outer.setSpacing(20)

        folder_box = QGroupBox('OUTPUT FOLDER')
        folder_h   = QHBoxLayout(folder_box)
        folder_h.setSpacing(10)
        self._folder_edit = QLineEdit(RECORDINGS_DIR)
        self._folder_edit.setPlaceholderText('Path to recordings folder...')
        btn_browse = QPushButton('Browse...')
        btn_browse.setObjectName('btn_browse')
        btn_browse.setFixedWidth(90)
        btn_browse.clicked.connect(self._on_browse)
        folder_h.addWidget(self._folder_edit, stretch=1)
        folder_h.addWidget(btn_browse)

        device_box = QGroupBox('INPUT DEVICE / CHANNEL')
        device_g   = QGridLayout(device_box)
        device_g.setVerticalSpacing(6)
        device_g.setHorizontalSpacing(10)
        self._device_combo = QComboBox()
        self._device_combo.setMinimumWidth(200)
        self._populate_device_combo()
        btn_refresh = QPushButton('Refresh')
        btn_refresh.setObjectName('btn_browse')
        btn_refresh.setFixedWidth(70)
        btn_refresh.clicked.connect(self._on_refresh_devices)
        lbl_ch = QLabel('Mode')
        lbl_ch.setObjectName('param_label')
        self._chan_combo = QComboBox()
        self._chan_combo.addItems(['Mono', 'Left', 'Right', 'Stereo'])
        self._chan_combo.setCurrentIndex(0)
        lbl_trig = QLabel('Trigger')
        lbl_trig.setObjectName('param_label')
        self._trig_combo = QComboBox()
        self._trig_combo.addItems(['Average', 'Any Channel', 'Both Channels'])
        self._trig_combo.setCurrentIndex(0)
        self._trig_combo.setEnabled(False)
        device_g.addWidget(self._device_combo, 0, 0, 1, 3)
        device_g.addWidget(btn_refresh,        0, 3)
        lbl_sr = QLabel('Rate')
        lbl_sr.setObjectName('param_label')
        self._sr_combo = QComboBox()
        for r in RecordingEntity.SUPPORTED_RATES:
            self._sr_combo.addItem(f'{r} Hz', userData=r)
        self._sr_combo.setCurrentText(f'{SAMPLE_RATE} Hz')
        self._sr_combo.setFixedWidth(90)
        lbl_buf = QLabel('Buffer')
        lbl_buf.setObjectName('param_label')
        self._buf_combo = QComboBox()
        for s in RecordingEntity.SUPPORTED_DISPLAY_SECONDS:
            label = f'{int(s)}s' if s == int(s) else f'{s}s'
            self._buf_combo.addItem(label, userData=s)
        self._buf_combo.setCurrentText(f'{int(DISPLAY_SECONDS)}s')
        self._buf_combo.setFixedWidth(70)
        device_g.addWidget(lbl_ch,             1, 0)
        device_g.addWidget(self._chan_combo,    1, 1)
        device_g.addWidget(lbl_trig,           1, 2)
        device_g.addWidget(self._trig_combo,   1, 3)
        device_g.addWidget(lbl_sr,             2, 0)
        device_g.addWidget(self._sr_combo,     2, 1)
        device_g.addWidget(lbl_buf,            2, 2)
        device_g.addWidget(self._buf_combo,    2, 3)

        fname_box = QGroupBox('FILENAME')
        fname_g   = QGridLayout(fname_box)
        fname_g.setSpacing(6)
        lbl_pfx = QLabel('Prefix')
        lbl_pfx.setObjectName('param_label')
        self._prefix_edit = QLineEdit()
        self._prefix_edit.setPlaceholderText('e.g. bird1_')
        lbl_sfx = QLabel('Suffix')
        lbl_sfx.setObjectName('param_label')
        self._suffix_edit = QLineEdit()
        self._suffix_edit.setPlaceholderText('e.g. _cage3')
        fname_g.addWidget(lbl_pfx,           0, 0)
        fname_g.addWidget(self._prefix_edit, 0, 1)
        fname_g.addWidget(lbl_sfx,           1, 0)
        fname_g.addWidget(self._suffix_edit, 1, 1)

        ref_box = QGroupBox('REFERENCE DATE')
        ref_g   = QGridLayout(ref_box)
        ref_g.setSpacing(6)
        self._chk_ref_date = QCheckBox('Days post hatch')
        self._date_line = QLineEdit(datetime.date.today().strftime('%Y-%m-%d'))
        self._date_line.setPlaceholderText('YYYY-MM-DD')
        self._date_line.setMinimumWidth(100)
        self._date_line.setEnabled(False)
        self._btn_pick_date = QPushButton('\u2026')
        self._btn_pick_date.setObjectName('btn_small')
        self._btn_pick_date.setFixedSize(28, 28)
        self._btn_pick_date.setEnabled(False)
        self._lbl_day_count = QLabel('Day: —')
        self._lbl_day_count.setObjectName('param_label')
        lbl_dph_pfx = QLabel('Folder prefix')
        lbl_dph_pfx.setObjectName('param_label')
        self._dph_prefix_edit = QLineEdit()
        self._dph_prefix_edit.setPlaceholderText('e.g. day_')
        self._dph_prefix_edit.setEnabled(False)
        self._chk_ref_date.toggled.connect(self._date_line.setEnabled)
        self._chk_ref_date.toggled.connect(self._btn_pick_date.setEnabled)
        self._chk_ref_date.toggled.connect(self._dph_prefix_edit.setEnabled)
        date_row = QHBoxLayout()
        date_row.setSpacing(4)
        date_row.addWidget(self._date_line, stretch=1)
        date_row.addWidget(self._btn_pick_date)
        date_row.addWidget(self._lbl_day_count)
        pfx_row = QHBoxLayout()
        pfx_row.setSpacing(4)
        pfx_row.addWidget(lbl_dph_pfx)
        pfx_row.addWidget(self._dph_prefix_edit)
        ref_g.addWidget(self._chk_ref_date,  0, 0)
        ref_g.addLayout(date_row,            1, 0)
        ref_g.addLayout(pfx_row,             2, 0)

        outer.addWidget(folder_box, stretch=3)
        outer.addWidget(fname_box, stretch=1)
        outer.addWidget(ref_box, stretch=1)
        outer.addWidget(device_box, stretch=2)
        return w

    # ──────────────────────────────────────────────────────────────────────
    # Signal wiring
    # ──────────────────────────────────────────────────────────────────────

    def _connect_signals(self):
        # Transport
        self._btn_start_acq.clicked.connect(self._on_start_acq)
        self._btn_stop_acq .clicked.connect(self._on_stop_acq)
        self._btn_start_rec.clicked.connect(self._on_start_rec)
        self._btn_stop_rec .clicked.connect(self._on_stop_rec)
        self._sidebar.start_all_acq.connect(self._on_start_all_acq)
        self._sidebar.stop_all_acq .connect(self._on_stop_all_acq)
        self._sidebar.start_all_rec.connect(self._on_start_all_rec)
        self._sidebar.stop_all_rec .connect(self._on_stop_all_rec)
        self._btn_reset    .clicked.connect(self._on_reset_params)
        self._btn_save     .clicked.connect(self._save_settings)
        self._btn_load     .clicked.connect(self._load_settings)
        self._btn_view_mode.clicked.connect(self._toggle_view_mode)

        # Threshold (hidden spinbox, synced from amplitude graph drag)
        self._sb_thr.valueChanged.connect(self._on_thr_spinbox)

        # Settings — write-through on change
        self._folder_edit.editingFinished.connect(self._on_folder_changed)
        self._prefix_edit.editingFinished.connect(self._on_prefix_changed)
        self._suffix_edit.editingFinished.connect(self._on_suffix_changed)
        self._chk_ref_date.toggled.connect(self._on_ref_date_toggled)
        self._date_line.editingFinished.connect(self._on_ref_date_text_changed)
        self._btn_pick_date.clicked.connect(self._on_pick_date)
        self._dph_prefix_edit.editingFinished.connect(self._on_dph_prefix_changed)
        self._device_combo.currentIndexChanged.connect(self._on_device_changed)
        self._chan_combo.currentTextChanged.connect(self._on_channel_mode_changed)
        self._trig_combo.currentTextChanged.connect(self._on_trigger_mode_changed)
        self._sr_combo.currentIndexChanged.connect(self._on_sample_rate_changed)
        self._buf_combo.currentIndexChanged.connect(self._on_display_buffer_changed)

        # Freq filter write-through
        self._chk_freq  .toggled       .connect(self._on_freq_filter_toggled)
        self._sb_freq_lo.valueChanged  .connect(self._on_freq_filter_param)
        self._sb_freq_hi.valueChanged  .connect(self._on_freq_filter_param)

        # Trigger params write-through
        self._sb_mc  .valueChanged.connect(lambda _: self._write_trigger_params())
        self._sb_hold.valueChanged.connect(lambda _: self._write_trigger_params())
        self._sb_maxr.valueChanged.connect(lambda _: self._write_trigger_params())
        self._sb_pre .valueChanged.connect(lambda _: self._write_trigger_params())

        # Spectrogram display write-through
        self._sl_gain .valueChanged.connect(lambda _: self._write_spec_params())
        self._sb_gain .valueChanged.connect(lambda _: self._write_spec_params())
        self._sl_floor.valueChanged.connect(lambda _: self._write_spec_params())
        self._sb_floor.valueChanged.connect(lambda _: self._write_spec_params())
        self._sl_ceil .valueChanged.connect(lambda _: self._write_spec_params())
        self._sb_ceil .valueChanged.connect(lambda _: self._write_spec_params())
        self._combo_fft   .currentIndexChanged.connect(self._on_fft_params_changed)
        self._combo_win   .currentIndexChanged.connect(self._on_fft_params_changed)
        self._combo_fscale.currentTextChanged .connect(self._on_freq_scale_changed)
        self._sb_disp_freq_lo.valueChanged.connect(self._on_disp_freq_changed)
        self._sb_disp_freq_hi.valueChanged.connect(self._on_disp_freq_changed)
        self._chk_shared_spec.toggled.connect(self._on_shared_spec_toggled)

        # Matplotlib events
        self._canvas.mpl_connect('button_press_event',   self._on_mpl_press)
        self._canvas.mpl_connect('motion_notify_event',  self._on_mpl_motion)
        self._canvas.mpl_connect('button_release_event', self._on_mpl_release)
        self._canvas.mpl_connect('scroll_event',         self._on_scroll)

        # Sidebar
        self._sidebar.selection_changed.connect(self._switch_selection)
        self._sidebar.add_requested.connect(self._add_recording)
        self._sidebar.delete_requested.connect(self._remove_recording)
        self._sidebar.move_requested.connect(self._move_recording)
        self._sidebar.item_renamed.connect(self._on_item_renamed)

    # ──────────────────────────────────────────────────────────────────────
    # Write-through: widgets → selected entity
    # ──────────────────────────────────────────────────────────────────────

    def _write_trigger_params(self):
        e = self._sel
        if not e:
            return
        e.min_cross_sec = self._sb_mc.value()
        e.hold_sec      = self._sb_hold.value()
        e.max_rec_sec   = self._sb_maxr.value()
        e.pre_trig_sec  = self._sb_pre.value()

    def _write_spec_params(self):
        e = self._sel
        if not e:
            return
        e.gain_db  = self._sb_gain.value()
        e.db_floor = self._sb_floor.value()
        e.db_ceil  = self._sb_ceil.value()
        if self._chk_shared_spec.isChecked():
            for ent in self._entities:
                if ent is not e:
                    ent.gain_db  = e.gain_db
                    ent.db_floor = e.db_floor
                    ent.db_ceil  = e.db_ceil

    def _on_freq_filter_toggled(self, on: bool):
        self._sb_freq_lo.setEnabled(on)
        self._sb_freq_hi.setEnabled(on)
        e = self._sel
        if e:
            e.freq_filter_enabled = on
            e.bpf.reset()
            e.bpf_r.reset()

    def _on_freq_filter_param(self, _val):
        e = self._sel
        if e:
            e.freq_lo = self._sb_freq_lo.value()
            e.freq_hi = self._sb_freq_hi.value()

    # ──────────────────────────────────────────────────────────────────────
    # Flush / Load params for selection switching
    # ──────────────────────────────────────────────────────────────────────

    def _flush_params_to_entity(self, idx: int):
        if idx < 0 or idx >= len(self._entities):
            return
        e = self._entities[idx]
        e.threshold     = self._sb_thr.value()
        e.min_cross_sec = self._sb_mc.value()
        e.hold_sec      = self._sb_hold.value()
        e.max_rec_sec   = self._sb_maxr.value()
        e.pre_trig_sec  = self._sb_pre.value()
        e.freq_filter_enabled = self._chk_freq.isChecked()
        e.freq_lo       = self._sb_freq_lo.value()
        e.freq_hi       = self._sb_freq_hi.value()
        e.gain_db       = self._sb_gain.value()
        e.db_floor      = self._sb_floor.value()
        e.db_ceil       = self._sb_ceil.value()
        e.spec_nperseg  = self._combo_fft.currentData() or SPECTROGRAM_NPERSEG
        e.spec_window   = self._combo_win.currentData() or 'hann'
        e.freq_scale    = self._combo_fscale.currentText()
        e.display_freq_lo = self._sb_disp_freq_lo.value()
        e.display_freq_hi = self._sb_disp_freq_hi.value()
        e.output_dir    = self._folder_edit.text().strip() or RECORDINGS_DIR
        e.filename_prefix = self._prefix_edit.text()
        e.filename_suffix = self._suffix_edit.text()
        if self._chk_ref_date.isChecked():
            e.ref_date = self._parse_date_text()
        else:
            e.ref_date = None
        e.dph_folder_prefix = self._dph_prefix_edit.text()
        e.channel_mode  = self._chan_combo.currentText()
        e.trigger_mode  = self._trig_combo.currentText()
        e.device_id     = self._device_combo.currentData()

    def _load_params_from_entity(self, idx: int):
        if idx < 0 or idx >= len(self._entities):
            return
        e = self._entities[idx]

        def _set(widget, val):
            widget.blockSignals(True)
            widget.setValue(val)
            widget.blockSignals(False)

        _set(self._sb_thr,  e.threshold)
        _set(self._sb_mc,   e.min_cross_sec)
        _set(self._sb_hold, e.hold_sec)
        _set(self._sb_maxr, e.max_rec_sec)
        _set(self._sb_pre,  e.pre_trig_sec)

        self._chk_freq.blockSignals(True)
        self._chk_freq.setChecked(e.freq_filter_enabled)
        self._chk_freq.blockSignals(False)
        self._sb_freq_lo.setEnabled(e.freq_filter_enabled)
        self._sb_freq_hi.setEnabled(e.freq_filter_enabled)
        _set(self._sb_freq_lo, e.freq_lo)
        _set(self._sb_freq_hi, e.freq_hi)

        _set(self._sb_gain,  e.gain_db)
        _set(self._sl_gain,  int(round(e.gain_db * self._DB_SCALE)))
        _set(self._sb_floor, e.db_floor)
        _set(self._sl_floor, int(round(e.db_floor * self._DB_SCALE)))
        _set(self._sb_ceil,  e.db_ceil)
        _set(self._sl_ceil,  int(round(e.db_ceil * self._DB_SCALE)))

        self._combo_fft.blockSignals(True)
        self._combo_fft.setCurrentText(str(e.spec_nperseg))
        self._combo_fft.blockSignals(False)
        self._combo_win.blockSignals(True)
        for i in range(self._combo_win.count()):
            if self._combo_win.itemData(i) == e.spec_window:
                self._combo_win.setCurrentIndex(i)
                break
        self._combo_win.blockSignals(False)
        self._combo_fscale.blockSignals(True)
        self._combo_fscale.setCurrentText(e.freq_scale)
        self._combo_fscale.blockSignals(False)

        self._sb_disp_freq_lo.blockSignals(True)
        self._sb_disp_freq_lo.setValue(e.display_freq_lo)
        self._sb_disp_freq_lo.blockSignals(False)
        self._sb_disp_freq_hi.blockSignals(True)
        self._sb_disp_freq_hi.setValue(e.display_freq_hi)
        self._sb_disp_freq_hi.blockSignals(False)

        self._folder_edit.setText(e.output_dir)
        self._prefix_edit.setText(e.filename_prefix)
        self._suffix_edit.setText(e.filename_suffix)

        self._chk_ref_date.blockSignals(True)
        if e.ref_date is not None:
            self._chk_ref_date.setChecked(True)
            self._date_line.setEnabled(True)
            self._btn_pick_date.setEnabled(True)
            self._dph_prefix_edit.setEnabled(True)
            self._date_line.setText(e.ref_date.strftime('%Y-%m-%d'))
            days = (datetime.date.today() - e.ref_date).days
            self._lbl_day_count.setText(f'Day: {days}')
        else:
            self._chk_ref_date.setChecked(False)
            self._date_line.setEnabled(False)
            self._btn_pick_date.setEnabled(False)
            self._dph_prefix_edit.setEnabled(False)
            self._lbl_day_count.setText('Day: —')
        self._chk_ref_date.blockSignals(False)
        self._dph_prefix_edit.setText(e.dph_folder_prefix)

        # Device combo — find by device_id
        self._device_combo.blockSignals(True)
        for i in range(self._device_combo.count()):
            if self._device_combo.itemData(i) == e.device_id:
                self._device_combo.setCurrentIndex(i)
                break
        self._device_combo.blockSignals(False)

        self._chan_combo.blockSignals(True)
        self._chan_combo.setCurrentText(e.channel_mode)
        self._chan_combo.blockSignals(False)
        self._trig_combo.blockSignals(True)
        self._trig_combo.setCurrentText(e.trigger_mode)
        self._trig_combo.blockSignals(False)
        self._trig_combo.setEnabled(e.channel_mode == 'Stereo')

        # Sample rate combo
        self._sr_combo.blockSignals(True)
        self._sr_combo.setCurrentText(f'{e.sample_rate} Hz')
        self._sr_combo.blockSignals(False)

        # Display buffer combo
        self._buf_combo.blockSignals(True)
        ds = e.display_seconds
        buf_label = f'{int(ds)}s' if ds == int(ds) else f'{ds}s'
        self._buf_combo.setCurrentText(buf_label)
        self._buf_combo.blockSignals(False)

        # Update freq range limits for this entity's sample rate
        nyq = e.sample_rate / 2
        self._sb_freq_lo.setRange(1.0, nyq - 1)
        self._sb_freq_hi.setRange(1.0, nyq - 1)
        self._sb_disp_freq_lo.setRange(0.0, nyq - 1)
        self._sb_disp_freq_hi.setRange(1.0, nyq)

        # Update threshold line
        self._sync_thr_line(e.threshold)

        # Rebuild axes if stereo layout, sample rate, or display buffer differs
        want_stereo = (e.channel_mode == 'Stereo')
        axes_changed = (e.sample_rate, e.display_seconds) != self._t_axis_key
        if want_stereo != self._is_stereo_layout or axes_changed:
            self._setup_axes(stereo=want_stereo)
        self._update_spec_yticks(e)

    # ──────────────────────────────────────────────────────────────────────
    # Selection switching
    # ──────────────────────────────────────────────────────────────────────

    def _switch_selection(self, new_idx: int):
        if new_idx == self._selected_idx:
            return
        if self._selected_idx >= 0:
            self._flush_params_to_entity(self._selected_idx)
        self._selected_idx = new_idx
        self._sidebar.select(new_idx)
        self._load_params_from_entity(new_idx)
        self._refresh_transport_ui()

    # ──────────────────────────────────────────────────────────────────────
    # Add / Remove / Move recordings
    # ──────────────────────────────────────────────────────────────────────

    def _add_recording(self):
        name = f'Recording {self._next_num}'
        self._next_num += 1
        e = RecordingEntity(name=name)
        self._entities.append(e)
        idx = self._sidebar.add_item(name)
        self._switch_selection(idx)

    def _remove_recording(self, idx: int):
        if len(self._entities) <= 1:
            return  # don't delete last
        if 0 <= idx < len(self._entities):
            e = self._entities.pop(idx)
            e.close()
            self._sidebar.remove_item(idx)
            # Re-select
            if self._selected_idx >= len(self._entities):
                self._selected_idx = len(self._entities) - 1
            elif self._selected_idx >= idx:
                self._selected_idx = max(0, self._selected_idx - 1)
            self._switch_selection(self._selected_idx)

    def _move_recording(self, idx: int, direction: int):
        new_idx = idx + direction
        if new_idx < 0 or new_idx >= len(self._entities):
            return
        # Flush current before swapping
        self._flush_params_to_entity(self._selected_idx)
        # Swap in entity list
        self._entities[idx], self._entities[new_idx] = self._entities[new_idx], self._entities[idx]
        self._sidebar.swap_items(idx, new_idx)
        # Update selection
        if self._selected_idx == idx:
            self._selected_idx = new_idx
        elif self._selected_idx == new_idx:
            self._selected_idx = idx
        self._sidebar.select(self._selected_idx)

    def _on_item_renamed(self, idx: int, name: str):
        if 0 <= idx < len(self._entities):
            self._entities[idx].name = name

    # ──────────────────────────────────────────────────────────────────────
    # Transport callbacks (operate on selected entity)
    # ──────────────────────────────────────────────────────────────────────

    def _on_start_acq(self):
        e = self._sel
        if e:
            e.start_acq()
            self._refresh_transport_ui()

    def _on_stop_acq(self):
        e = self._sel
        if e:
            e.stop_acq()
            self._refresh_transport_ui()

    def _on_start_rec(self):
        e = self._sel
        if e:
            e.start_rec()
            self._refresh_transport_ui()

    def _on_stop_rec(self):
        e = self._sel
        if e:
            e.stop_rec()
            self._refresh_transport_ui()

    def _on_start_all_acq(self):
        for e in self._entities:
            e.start_acq()
        self._refresh_transport_ui()

    def _on_stop_all_acq(self):
        for e in self._entities:
            e.stop_acq()
        self._refresh_transport_ui()

    def _on_start_all_rec(self):
        for e in self._entities:
            e.start_rec()
        self._refresh_transport_ui()

    def _on_stop_all_rec(self):
        for e in self._entities:
            e.stop_rec()
        self._refresh_transport_ui()

    def _on_reset_params(self):
        e = self._sel
        if not e:
            return
        self._sb_thr .setValue(DEFAULT_THRESHOLD)
        self._sb_mc  .setValue(DEFAULT_MIN_CROSS)
        self._sb_hold.setValue(DEFAULT_HOLD)
        self._sb_maxr.setValue(DEFAULT_MAX_REC)
        self._sb_pre .setValue(DEFAULT_PRE_TRIG)
        self._chk_freq.setChecked(False)
        self._sb_freq_lo.setValue(DEFAULT_FREQ_LO)
        self._sb_freq_hi.setValue(DEFAULT_FREQ_HI)
        self._sb_gain .setValue(0.0)
        self._sb_floor.setValue(SPEC_DB_MIN)
        self._sb_ceil .setValue(SPEC_DB_MAX)
        self._combo_fft   .setCurrentText(str(SPECTROGRAM_NPERSEG))
        self._combo_win   .setCurrentIndex(0)
        self._combo_fscale.setCurrentText('Mel')
        self._sb_disp_freq_lo.setValue(0.0)
        self._sb_disp_freq_hi.setValue(e.sample_rate / 2)
        self._folder_edit.setText(RECORDINGS_DIR)
        self._prefix_edit.clear()
        self._suffix_edit.clear()
        self._chk_ref_date.setChecked(False)
        self._date_line.setText(datetime.date.today().strftime('%Y-%m-%d'))
        self._dph_prefix_edit.clear()

    # ──────────────────────────────────────────────────────────────────────
    # Save / Load settings
    # ──────────────────────────────────────────────────────────────────────

    def _save_settings(self):
        # Flush current entity so all widget values are captured
        if self._selected_idx >= 0:
            self._flush_params_to_entity(self._selected_idx)

        data = {
            'version': 1,
            'view_mode': {
                'columns': self._vm_n_cols,
                'panel_height': self._vm_panel_height,
            },
            'recordings': [e.to_dict() for e in self._entities],
        }

        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Settings', '', 'SAP_rec Settings (*.saprec);;All Files (*)')
        if not path:
            return
        if not path.endswith('.saprec'):
            path += '.saprec'
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            QMessageBox.warning(self, 'Save Error', f'Could not save settings:\n{exc}')

    def _load_settings(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Load Settings', '', 'SAP_rec Settings (*.saprec);;All Files (*)')
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as exc:
            QMessageBox.warning(self, 'Load Error', f'Could not read file:\n{exc}')
            return

        if not isinstance(data, dict) or 'recordings' not in data:
            QMessageBox.warning(self, 'Load Error', 'Invalid settings file format.')
            return

        # Stop timer while rebuilding
        self._timer.stop()

        # Close and remove all existing entities
        for e in self._entities:
            e.stop_acq()
            e.close()
        self._entities.clear()
        self._sidebar.clear_all()
        self._selected_idx = -1

        # Restore view mode globals
        vm = data.get('view_mode', {})
        self._vm_n_cols = vm.get('columns', 1)
        self._vm_panel_height = vm.get('panel_height', 300)
        self._vm_cols_spin.blockSignals(True)
        self._vm_cols_spin.setValue(self._vm_n_cols)
        self._vm_cols_spin.blockSignals(False)
        self._vm_height_sl.blockSignals(True)
        self._vm_height_sl.setValue(self._vm_panel_height)
        self._vm_height_sl.blockSignals(False)
        self._vm_height_lbl.setText(f'{self._vm_panel_height}px')

        # Create entities from saved data
        warnings = []
        for rec_d in data['recordings']:
            ent, warn = RecordingEntity.from_dict(rec_d)
            self._entities.append(ent)
            self._sidebar.add_item(ent.name)
            if warn:
                warnings.append(warn)

        # Update next recording number
        max_num = 0
        for e in self._entities:
            # Extract trailing number from name like "Recording 3"
            parts = e.name.rsplit(' ', 1)
            if len(parts) == 2:
                try:
                    max_num = max(max_num, int(parts[1]))
                except ValueError:
                    pass
        self._next_num = max_num + 1

        # Select first entity
        if self._entities:
            self._selected_idx = 0
            self._sidebar.select(0)
            self._load_params_from_entity(0)
            e = self._entities[0]
            stereo = e.channel_mode == 'Stereo'
            self._setup_axes(stereo=stereo)
            self._update_spec_yticks(e)
            self._refresh_transport_ui()

        # If in view mode, rebuild view axes
        if self._view_mode:
            self._setup_view_mode_axes()

        self._timer.start()

        if warnings:
            QMessageBox.information(
                self, 'Device Warnings',
                'Some devices were not found:\n\n' + '\n'.join(warnings)
                + '\n\nDefault device was used instead.')

    def _refresh_transport_ui(self):
        e = self._sel
        acq = e.acq_running if e else False
        rec = e.rec_enabled if e else False

        for btn, state in ((self._btn_start_acq, acq),
                           (self._btn_start_rec, rec)):
            btn.setProperty('active', state)
            btn.style().unpolish(btn)
            btn.style().polish(btn)

        self._lbl_acq_status.setText('ACQ  \u25cf  RUNNING' if acq else 'ACQ  \u25cf  STOPPED')
        self._lbl_rec_status.setText('REC  \u25cf  RUNNING' if rec else 'REC  \u25cf  STOPPED')
        for lbl, on in ((self._lbl_acq_status, acq), (self._lbl_rec_status, rec)):
            lbl.setObjectName('status_on' if on else 'status_off')
            lbl.style().unpolish(lbl)
            lbl.style().polish(lbl)

    # ──────────────────────────────────────────────────────────────────────
    # Threshold sync
    # ──────────────────────────────────────────────────────────────────────

    def _on_thr_spinbox(self, val: float):
        e = self._sel
        if e:
            e.threshold = val
        self._sync_thr_line(val)

    def _sync_thr_line(self, val: float):
        self._threshold_line.set_ydata([val, val])
        self._threshold_label.set_y(val + 0.03)
        self._threshold_label.set_text(f'thr = {val:.3f}')
        self._canvas.draw_idle()

    def _set_thr_silent(self, val: float):
        self._sb_thr.blockSignals(True)
        self._sb_thr.setValue(val)
        self._sb_thr.blockSignals(False)

    # ──────────────────────────────────────────────────────────────────────
    # Matplotlib mouse events
    # ──────────────────────────────────────────────────────────────────────

    def _on_mpl_press(self, event):
        if self._view_mode:
            return
        e = self._sel
        if not e or event.inaxes is not self._ax_amp or event.button != 1:
            return
        _, y_disp = self._ax_amp.transData.transform((0.0, e.threshold))
        if abs(event.y - y_disp) <= 12:
            self._dragging = True
            self._timer.stop()

    def _on_mpl_motion(self, event):
        if not self._dragging or event.inaxes is not self._ax_amp:
            return
        val = float(np.clip(event.ydata, 0.0, 1.0))
        e = self._sel
        if e:
            e.threshold = val
        self._set_thr_silent(val)
        self._sync_thr_line(val)

    def _on_mpl_release(self, event):
        if self._dragging:
            self._dragging = False
            self._timer.start()

    def _on_scroll(self, event):
        if self._view_mode:
            # Zoom whichever amplitude axis the mouse is over
            for i, vm in enumerate(self._vm_axes):
                if event.inaxes is vm['ax_amp']:
                    scale = 0.85 if event.step > 0 else 1.0 / 0.85
                    _, ymax = vm['ax_amp'].get_ylim()
                    new_ymax = max(0.01, ymax * scale)
                    vm['ax_amp'].set_ylim(0.0, new_ymax)
                    if i < len(self._entities):
                        self._entities[i].amp_ylim = new_ymax
                    return
            return
        if event.inaxes is not self._ax_amp:
            return
        scale = 0.85 if event.step > 0 else 1.0 / 0.85
        _, ymax = self._ax_amp.get_ylim()
        new_ymax = max(0.01, ymax * scale)
        self._ax_amp.set_ylim(0.0, new_ymax)
        e = self._sel
        if e:
            e.amp_ylim = new_ymax

    # ──────────────────────────────────────────────────────────────────────
    # Spectrogram display callbacks
    # ──────────────────────────────────────────────────────────────────────

    def _on_disp_freq_changed(self, _val):
        e = self._sel
        if not e:
            return
        e.display_freq_lo = self._sb_disp_freq_lo.value()
        e.display_freq_hi = self._sb_disp_freq_hi.value()
        e.rebuild_freq_mapping()
        self._update_spec_yticks(e)
        if self._chk_shared_spec.isChecked():
            for ent in self._entities:
                if ent is not e:
                    ent.display_freq_lo = e.display_freq_lo
                    ent.display_freq_hi = e.display_freq_hi
                    ent.rebuild_freq_mapping()

    def _on_shared_spec_toggled(self, on: bool):
        if on:
            e = self._sel
            if not e:
                return
            for ent in self._entities:
                if ent is not e:
                    ent.gain_db  = e.gain_db
                    ent.db_floor = e.db_floor
                    ent.db_ceil  = e.db_ceil
                    ent.freq_scale = e.freq_scale
                    ent.display_freq_lo = e.display_freq_lo
                    ent.display_freq_hi = e.display_freq_hi
                    ent.rebuild_freq_mapping()
                    ent.change_fft_params(e.spec_nperseg, e.spec_window)

    def _on_freq_scale_changed(self, scale: str):
        e = self._sel
        if e:
            e.freq_scale = scale
            e.rebuild_freq_mapping()
            self._update_spec_yticks(e)
            if self._chk_shared_spec.isChecked():
                for ent in self._entities:
                    if ent is not e:
                        ent.freq_scale = scale
                        ent.rebuild_freq_mapping()

    def _on_fft_params_changed(self):
        e = self._sel
        if not e:
            return
        nperseg = self._combo_fft.currentData()
        window  = self._combo_win.currentData()
        if nperseg and window:
            e.change_fft_params(nperseg, window)
            self._update_spec_yticks(e)
            if self._chk_shared_spec.isChecked():
                for ent in self._entities:
                    if ent is not e:
                        ent.change_fft_params(nperseg, window)

    # ──────────────────────────────────────────────────────────────────────
    # Folder & device
    # ──────────────────────────────────────────────────────────────────────

    def _on_browse(self):
        e = self._sel
        start = e.output_dir if e else RECORDINGS_DIR
        chosen = QFileDialog.getExistingDirectory(self, 'Select output folder', start)
        if chosen:
            if e:
                e.output_dir = chosen
            self._folder_edit.setText(chosen)

    def _on_folder_changed(self):
        text = self._folder_edit.text().strip()
        e = self._sel
        if e:
            e.output_dir = text if text else RECORDINGS_DIR

    def _on_prefix_changed(self):
        e = self._sel
        if e:
            e.filename_prefix = self._prefix_edit.text()

    def _on_suffix_changed(self):
        e = self._sel
        if e:
            e.filename_suffix = self._suffix_edit.text()

    def _on_ref_date_toggled(self, on: bool):
        e = self._sel
        if e:
            if on:
                e.ref_date = self._parse_date_text()
                if e.ref_date:
                    days = (datetime.date.today() - e.ref_date).days
                    self._lbl_day_count.setText(f'Day: {days}')
            else:
                e.ref_date = None
                self._lbl_day_count.setText('Day: —')

    def _on_ref_date_text_changed(self):
        e = self._sel
        if e and self._chk_ref_date.isChecked():
            d = self._parse_date_text()
            if d:
                e.ref_date = d
                days = (datetime.date.today() - d).days
                self._lbl_day_count.setText(f'Day: {days}')

    def _on_pick_date(self):
        dlg = QDialog(self)
        dlg.setWindowTitle('Select reference date')
        lay = QVBoxLayout(dlg)
        cal = QCalendarWidget()
        cur = self._parse_date_text()
        if cur:
            cal.setSelectedDate(QDate(cur.year, cur.month, cur.day))
        lay.addWidget(cal)
        btn_ok = QPushButton('OK')
        btn_ok.clicked.connect(dlg.accept)
        lay.addWidget(btn_ok)
        if dlg.exec_() == QDialog.Accepted:
            qd = cal.selectedDate()
            d = datetime.date(qd.year(), qd.month(), qd.day())
            self._date_line.setText(d.strftime('%Y-%m-%d'))
            e = self._sel
            if e and self._chk_ref_date.isChecked():
                e.ref_date = d
                days = (datetime.date.today() - d).days
                self._lbl_day_count.setText(f'Day: {days}')

    def _on_dph_prefix_changed(self):
        e = self._sel
        if e:
            e.dph_folder_prefix = self._dph_prefix_edit.text()

    def _parse_date_text(self):
        text = self._date_line.text().strip()
        try:
            return datetime.datetime.strptime(text, '%Y-%m-%d').date()
        except ValueError:
            return None

    def _populate_device_combo(self, keep_current: bool = False):
        prev_name = self._device_combo.currentText() if keep_current else None
        self._device_combo.blockSignals(True)
        self._device_combo.clear()
        try:
            default_in = sd.default.device[0]
        except Exception:
            default_in = -1
        hostapis = sd.query_hostapis()
        restore_idx = 0
        default_idx = 0
        for i, d in enumerate(sd.query_devices()):
            if d['max_input_channels'] <= 0:
                continue
            # Filter out loopback / virtual devices that can't actually record
            try:
                api_name = hostapis[d['hostapi']]['name']
            except (IndexError, KeyError):
                api_name = ''
            # Skip devices from APIs that are typically not real recording inputs
            if 'Loopback' in d['name']:
                continue
            self._device_combo.addItem(f"{d['name']}  [{api_name}]", userData=i)
            idx = self._device_combo.count() - 1
            if prev_name and d['name'] in prev_name:
                restore_idx = idx
            if i == default_in:
                default_idx = idx
        self._device_combo.setCurrentIndex(restore_idx if keep_current else default_idx)
        self._device_combo.blockSignals(False)

    def _on_refresh_devices(self):
        self._populate_device_combo(keep_current=True)

    def _on_device_changed(self, _index: int):
        e = self._sel
        if not e:
            return
        device_id = self._device_combo.currentData()
        if device_id is None:
            return
        try:
            info = sd.query_devices(device_id)
            max_ch = info['max_input_channels']
        except Exception:
            max_ch = 1
        if max_ch < 2 and e.channel_mode != 'Mono':
            self._chan_combo.blockSignals(True)
            self._chan_combo.setCurrentText('Mono')
            self._chan_combo.blockSignals(False)
            e.channel_mode = 'Mono'
            self._trig_combo.setEnabled(False)
            if self._is_stereo_layout:
                self._setup_axes(stereo=False)
        self._chan_combo.setEnabled(max_ch >= 2)
        need_ch = 2 if e.channel_mode != 'Mono' else 1
        ok = e.change_device(device_id, need_ch)
        if not ok:
            QMessageBox.warning(self, 'Device Error',
                                f'Could not open device:\n{self._device_combo.currentText()}')
        self._refresh_transport_ui()

    # ──────────────────────────────────────────────────────────────────────
    # Channel mode
    # ──────────────────────────────────────────────────────────────────────

    def _on_channel_mode_changed(self, mode: str):
        e = self._sel
        if not e:
            return
        e.channel_mode = mode
        want_stereo = (mode == 'Stereo')
        is_stereo_input = (mode != 'Mono')
        self._trig_combo.setEnabled(want_stereo)

        need_ch = 2 if is_stereo_input else 1
        device_id = self._device_combo.currentData()
        if device_id is not None:
            try:
                info = sd.query_devices(device_id)
                if info['max_input_channels'] < 2 and need_ch == 2:
                    self._chan_combo.blockSignals(True)
                    self._chan_combo.setCurrentText('Mono')
                    self._chan_combo.blockSignals(False)
                    e.channel_mode = 'Mono'
                    self._trig_combo.setEnabled(False)
                    if self._is_stereo_layout:
                        self._setup_axes(stereo=False)
                    return
            except Exception:
                pass

        if want_stereo != self._is_stereo_layout:
            self._setup_axes(stereo=want_stereo)

        e.change_device(device_id, need_ch)
        if not want_stereo:
            e.amp_buffer_r[:] = 0.0
            e.spec_buffer_r[:] = SPEC_DB_MIN
        e.bpf.reset()
        e.bpf_r.reset()

    def _on_trigger_mode_changed(self, mode: str):
        e = self._sel
        if e:
            e.trigger_mode = mode

    def _on_sample_rate_changed(self, _index: int):
        e = self._sel
        if not e:
            return
        new_sr = self._sr_combo.currentData()
        if new_sr is None or new_sr == e.sample_rate:
            return
        e.change_sample_rate(new_sr)
        # Update freq range limits
        nyq = new_sr / 2
        self._sb_freq_lo.setRange(1.0, nyq - 1)
        self._sb_freq_hi.setRange(1.0, nyq - 1)
        self._sb_disp_freq_lo.setRange(0.0, nyq - 1)
        self._sb_disp_freq_hi.setRange(1.0, nyq)
        if self._sb_disp_freq_hi.value() > nyq:
            self._sb_disp_freq_hi.setValue(nyq)
        # Rebuild axes with new buffer sizes
        stereo = e.channel_mode == 'Stereo'
        self._setup_axes(stereo=stereo)
        self._update_spec_yticks(e)
        self._refresh_transport_ui()

    def _on_display_buffer_changed(self, _index: int):
        e = self._sel
        if not e:
            return
        new_secs = self._buf_combo.currentData()
        if new_secs is None or new_secs == e.display_seconds:
            return
        e.change_display_seconds(new_secs)
        # Rebuild axes with new buffer sizes
        stereo = e.channel_mode == 'Stereo'
        self._setup_axes(stereo=stereo)
        self._update_spec_yticks(e)
        self._refresh_transport_ui()

    # ──────────────────────────────────────────────────────────────────────
    # View Mode
    # ──────────────────────────────────────────────────────────────────────

    def _toggle_view_mode(self):
        self._view_mode = not self._view_mode

        if self._view_mode:
            # Save current amplitude zoom for selected entity
            e = self._sel
            if e:
                _, ymax = self._ax_amp.get_ylim()
                e.amp_ylim = ymax
            # Flush params before hiding controls
            if self._selected_idx >= 0:
                self._flush_params_to_entity(self._selected_idx)
            # Hide sidebar + config panels
            self._sidebar.hide()
            for w in self._config_widgets:
                w.hide()
            # Show view toolbar
            self._view_toolbar.show()
            # Rebuild figure for all streams
            self._setup_view_mode_axes()
        else:
            # Save view-mode zoom levels back to entities
            for i, vm in enumerate(self._vm_axes):
                if i < len(self._entities):
                    _, ymax = vm['ax_amp'].get_ylim()
                    self._entities[i].amp_ylim = ymax
            # Hide view toolbar
            self._view_toolbar.hide()
            # Show sidebar + config panels
            self._sidebar.show()
            for w in self._config_widgets:
                w.show()
            # Rebuild single-entity figure
            self._canvas.setMinimumHeight(0)
            e = self._sel
            stereo = e.channel_mode == 'Stereo' if e else False
            self._setup_axes(stereo=stereo)
            if e:
                self._load_params_from_entity(self._selected_idx)
                self._update_spec_yticks(e)

    def _setup_view_mode_axes(self):
        """Rebuild matplotlib figure with all entities in a grid layout."""
        import math
        self._fig.clf()
        self._vm_axes = []
        n = len(self._entities)
        if n == 0:
            self._canvas.draw_idle()
            return

        cols = min(self._vm_n_cols, n)
        rows = math.ceil(n / cols)

        # Set canvas height based on panel height × rows
        total_h = max(300, rows * self._vm_panel_height)
        self._canvas.setMinimumHeight(total_h)

        # Outer grid: rows × cols, each cell has inner 2 subplots (spec + amp)
        outer_gs = self._fig.add_gridspec(
            rows, cols, hspace=0.35, wspace=0.15,
            top=0.97, bottom=0.03, left=0.05, right=0.99)

        n_disp = N_DISPLAY_ROWS

        for i, e in enumerate(self._entities):
            r, c = divmod(i, cols)
            inner = outer_gs[r, c].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.08)

            ax_spec = self._fig.add_subplot(inner[0])
            ax_amp  = self._fig.add_subplot(inner[1], sharex=ax_spec)

            db_floor = min(e.db_floor, e.db_ceil - 0.1)
            dummy = np.full((n_disp, e._n_cols), db_floor, dtype=np.float32)

            e_disp = e.display_seconds
            spec_im = ax_spec.imshow(
                dummy, aspect='auto', origin='lower',
                extent=[0.0, e_disp, 0, n_disp],
                vmin=db_floor, vmax=e.db_ceil,
                cmap=COLORMAP, interpolation='nearest',
            )

            # Name as title on the left
            title_obj = ax_spec.set_title(e.name, loc='left', fontsize=9,
                                          color=C['text'], fontweight='bold', pad=3)

            # Status text on the right
            status_text = ax_spec.text(
                0.99, 1.02, '', transform=ax_spec.transAxes, fontsize=8,
                ha='right', va='bottom', fontfamily='Consolas')

            cursor_spec = ax_spec.axvline(x=0.0, color=C['green'], linewidth=1.0, alpha=0.7)
            self._apply_spec_yticks(ax_spec, e)
            ax_spec.tick_params(labelbottom=False)

            e_ts = e._total_samples
            e_t_axis = np.linspace(0.0, e_disp, e_ts, endpoint=False, dtype=np.float32)
            (amp_line,) = ax_amp.plot(
                e_t_axis, np.zeros(e_ts),
                color=C['blue'], linewidth=0.6, antialiased=False,
            )
            amp_line_r = None
            if e.channel_mode == 'Stereo':
                (amp_line_r,) = ax_amp.plot(
                    e_t_axis, np.zeros(e_ts),
                    color=C['pink'], linewidth=0.6, antialiased=False,
                )
            ax_amp.set_xlim(0.0, e_disp)
            ax_amp.set_ylim(0.0, e.amp_ylim)
            cursor_amp = ax_amp.axvline(x=0.0, color=C['green'], linewidth=1.0, alpha=0.7)

            thr_line = ax_amp.axhline(
                y=e.threshold, color=C['yellow'], linewidth=1.0,
                linestyle=(0, (6, 3)),
            )

            # Only show x-labels on bottom row
            if r < rows - 1:
                ax_amp.tick_params(labelbottom=False)
            else:
                ax_amp.set_xlabel('Time (s)', fontsize=7)

            ax_spec.set_ylabel('Freq', fontsize=7)
            ax_amp.set_ylabel('Amp', fontsize=7)

            self._vm_axes.append({
                'ax_spec': ax_spec, 'ax_amp': ax_amp,
                'spec_im': spec_im,
                'amp_line': amp_line, 'amp_line_r': amp_line_r,
                'cursor_spec': cursor_spec, 'cursor_amp': cursor_amp,
                'thr_line': thr_line,
                'title': title_obj, 'status_text': status_text,
            })

        # Set up blitting for view mode
        for vm in self._vm_axes:
            for a in self._get_vm_artists(vm):
                a.set_animated(True)
        self._canvas.draw()
        self._bg = self._canvas.copy_from_bbox(self._fig.bbox)

    def _on_vm_cols_changed(self, val):
        self._vm_n_cols = val
        if self._view_mode:
            self._setup_view_mode_axes()

    def _on_vm_height_changed(self, val):
        self._vm_panel_height = val
        self._vm_height_lbl.setText(f'{val}px')
        if self._view_mode:
            import math
            n = len(self._entities)
            cols = min(self._vm_n_cols, max(n, 1))
            rows = math.ceil(max(n, 1) / cols)
            self._canvas.setMinimumHeight(max(300, rows * val))
            self._fig.set_size_inches(
                self._fig.get_size_inches()[0],
                max(3, rows * val / self._fig.dpi))
            self._recapture_bg()

    def _update_plot_view_mode(self):
        """Update all entity displays in view mode (blitting)."""
        if not self._vm_axes or self._bg is None:
            return

        self._canvas.restore_region(self._bg)

        for i, e in enumerate(self._entities):
            if i >= len(self._vm_axes):
                break
            vm = self._vm_axes[i]
            cursor_x = (e.write_head / e.sample_rate) % e.display_seconds

            vm['spec_im'].set_data(e.resample_spec(e.spec_buffer))
            clim_lo = min(e.db_floor, e.db_ceil - 0.1)
            vm['spec_im'].set_clim(clim_lo, e.db_ceil)

            vm['amp_line'].set_ydata(np.abs(e.amp_buffer))
            if vm['amp_line_r'] is not None and e.channel_mode == 'Stereo':
                vm['amp_line_r'].set_ydata(np.abs(e.amp_buffer_r))

            vm['cursor_spec'].set_xdata([cursor_x, cursor_x])
            vm['cursor_amp'].set_xdata([cursor_x, cursor_x])
            vm['thr_line'].set_ydata([e.threshold, e.threshold])

            # Status text
            parts = []
            if e.acq_running:
                parts.append('ACQ')
            if e.rec_enabled:
                parts.append('REC')
            if e.recorder.is_recording:
                parts.append('TRIG')
            status_str = '  '.join(parts) if parts else 'STOPPED'

            vm['status_text'].set_text(status_str)
            if e.recorder.is_recording:
                vm['status_text'].set_color(C['red'])
            elif e.rec_enabled:
                vm['status_text'].set_color(C['green'])
            elif e.acq_running:
                vm['status_text'].set_color(C['blue'])
            else:
                vm['status_text'].set_color(C['surface2'])

            # Blit each animated artist
            for a in self._get_vm_artists(vm):
                a.axes.draw_artist(a)

        self._canvas.blit(self._fig.bbox)

        # Sidebar is hidden in view mode — skip expensive mini-amp updates
        # Status is still tracked so cached state stays correct
        for i, ent in enumerate(self._entities):
            self._sidebar.update_item_status(i, ent.acq_running, ent.rec_enabled,
                                             ent.recorder.is_recording)

    # ──────────────────────────────────────────────────────────────────────
    # Audio ingestion + plot refresh
    # ──────────────────────────────────────────────────────────────────────

    def _update_plot(self):
        # 1. Ingest chunks for ALL entities
        for e in self._entities:
            while True:
                try:
                    e.ingest_chunk(e.queue.get_nowait())
                except queue.Empty:
                    break

        # 2. Branch on mode
        if self._view_mode:
            self._update_plot_view_mode()
            return

        # 3. Update main display for selected entity (blitting)
        e = self._sel
        if e and self._bg is not None:
            cursor_x = (e.write_head / e.sample_rate) % e.display_seconds
            self._spec_im.set_data(e.resample_spec(e.spec_buffer))
            clim_lo = min(e.db_floor, e.db_ceil - 0.1)
            self._spec_im.set_clim(clim_lo, e.db_ceil)
            self._amp_line.set_ydata(np.abs(e.amp_buffer))
            if self._is_stereo_layout and self._spec_im_r is not None:
                self._spec_im_r.set_data(e.resample_spec(e.spec_buffer_r))
                self._spec_im_r.set_clim(clim_lo, e.db_ceil)
                self._amp_line_r.set_ydata(np.abs(e.amp_buffer_r))
                self._cursor_spec_r.set_xdata([cursor_x, cursor_x])
            self._cursor_spec.set_xdata([cursor_x, cursor_x])
            self._cursor_amp .set_xdata([cursor_x, cursor_x])

            self._canvas.restore_region(self._bg)
            for a in self._get_config_artists():
                a.axes.draw_artist(a)
            self._canvas.blit(self._fig.bbox)

        # 4. Update sidebar for ALL entities
        for i, ent in enumerate(self._entities):
            self._sidebar.update_item_status(i, ent.acq_running, ent.rec_enabled,
                                             ent.recorder.is_recording)
            self._sidebar.update_item_amp(i, ent.get_mini_amplitude())

        # 5. Update day count label if ref date active
        if e and e.ref_date is not None and self._chk_ref_date.isChecked():
            days = (datetime.date.today() - e.ref_date).days
            self._lbl_day_count.setText(f'Day: {days}')

        # 6. Trigger indicator for selected
        self._blink_counter = (self._blink_counter + 1) % 20
        if e and e.recorder.is_recording:
            blink_on = self._blink_counter < 10
            self._lbl_trig_status.setText('TRIG \u25cf  REC' if blink_on else 'TRIG \u25a0  REC')
            self._lbl_trig_status.setObjectName('trig_active')
        else:
            self._lbl_trig_status.setText('TRIG \u25cf  IDLE')
            self._lbl_trig_status.setObjectName('trig_idle')
        self._lbl_trig_status.style().unpolish(self._lbl_trig_status)
        self._lbl_trig_status.style().polish(self._lbl_trig_status)

    # ──────────────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        self._timer.stop()
        for e in self._entities:
            e.close()
        super().closeEvent(event)


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)
    win = SapWindow()
    win.showMaximized()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
