"""Top-level Qt window — ChirpWindow + main() entry point.

Extracted from the monolith in the Phase 1 refactor (plan: c08). This
is still the largest file in the project (~2700 lines) — Phase 2 and
Phase 3 fixes will chip away at it:

  - #13 (c15): bounded per-tick queue drain + drop badge
  - #17 (c16): shutdown flushes in-flight events and awaits the writer pool
  - #19 (c21): move ingest_chunk off the Qt main thread
  - #11 (c22): save-button tooltip + dirty-state indicator
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
    QGridLayout, QGroupBox, QPushButton, QLabel, QLineEdit,
    QFileDialog, QFrame, QSizePolicy, QDoubleSpinBox, QComboBox, QCheckBox,
    QScrollArea, QStackedLayout, QDialog, QCalendarWidget, QMessageBox, QSpinBox,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QSize, QDate, QPointF
from PyQt5.QtGui import QFont, QPainter, QColor, QPainterPath, QPen, QPolygonF

# Re-exports used throughout the window code. The star-import brings in
# all the module-level constants and the palette (C, QSS) so the class
# body below keeps referring to them by bare name.
from chirp import __version__
from chirp.constants import *  # noqa: F401,F403
from chirp.audio import AudioCapture, AudioMonitor  # noqa: F401
from chirp.audio.devices import list_output_devices, host_api_name
from chirp.dsp import (  # noqa: F401
    BandpassFilter,
    SpectrogramAccumulator,
    normalized_spectral_entropy as _spectral_entropy,
)
from chirp.recording.trigger import ThresholdRecorder  # noqa: F401
from chirp.recording.entity import RecordingEntity
from chirp.ui.sidebar import (
    MiniAmplitudeWidget,  # noqa: F401
    RecordingSidebar,
    RecordingSidebarItem,  # noqa: F401
)

# ──────────────────────────────────────────────────────────────────────────────
# ChirpWindow
# ──────────────────────────────────────────────────────────────────────────────
class ChirpWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        # Entities
        self._entities: list[RecordingEntity] = []
        self._selected_idx = -1
        self._next_num = 1
        self._dragging = False
        self._dragging_entropy = False
        self._current_config_path: str | None = None
        self._config_dirty = False  # #11 / c22: unsaved changes indicator

        # View mode
        self._view_mode = False
        self._vm_axes: list[dict] = []
        self._vm_n_cols = 1
        self._vm_panel_height = 300

        # #7: shared audio-monitor loopback. One output stream; each
        # RecordingEntity is wired into it in `_add_recording` and the
        # monitor itself gates on `source_id` so only the chosen stream
        # plays. Created before `_build_ui` so the UI can reference it.
        self._monitor = AudioMonitor()

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

        self._update_title()
        self.resize(1400, 850)

    # ──────────────────────────────────────────────────────────────────────
    # Dirty-state tracking (#11 / c22)
    # ──────────────────────────────────────────────────────────────────────

    def _mark_dirty(self):
        """Flag that in-memory config has changed since last save/load."""
        if not self._config_dirty:
            self._config_dirty = True
            self._update_title()
            self._update_save_tooltip()

    def _mark_clean(self):
        """Reset dirty flag after a successful save or load."""
        self._config_dirty = False
        self._update_title()
        self._update_save_tooltip()

    def _update_title(self):
        base = f'Chirp v{__version__} — Triggered Sound Recording'
        path = self._current_config_path
        if path:
            import os
            base += f'  [{os.path.basename(path)}]'
        if self._config_dirty:
            base += '  •'
        self.setWindowTitle(base)

    def _update_save_tooltip(self):
        if not hasattr(self, '_btn_save'):
            return
        path = self._current_config_path or '(no file)'
        dirty = ' (unsaved changes)' if self._config_dirty else ''
        self._btn_save.setToolTip(f'Save configuration to {path}{dirty}')

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
        dmode = e.display_mode if e else 'Spectrogram'
        self._display_mode = dmode

        show_spec = dmode in ('Spectrogram', 'Both')
        show_wave = dmode in ('Waveform', 'Both')
        show_entropy = (e.spectral_trigger_mode != 'Amplitude Only') if e else False

        # Build ordered list of subplot rows
        rows_list = []
        ratios = []
        if show_spec:
            rows_list.append('spec')
            ratios.append(2 if dmode == 'Both' else 1)
            if stereo:
                rows_list.append('spec_r')
                ratios.append(2 if dmode == 'Both' else 1)
        if show_wave:
            rows_list.append('wave')
            ratios.append(1)
            if stereo and dmode == 'Waveform':
                rows_list.append('wave_r')
                ratios.append(1)
        rows_list.append('amp')
        ratios.append(1)
        # #32: thin detect/record indicator strip under the amp axis.
        rows_list.append('events')
        ratios.append(0.25)
        if show_entropy:
            rows_list.append('entropy')
            ratios.append(1)

        gs = self._fig.add_gridspec(len(rows_list), 1, height_ratios=ratios, hspace=0.10)
        ax_rows = {key: i for i, key in enumerate(rows_list)}

        # Create the first (topmost) axis as the sharex anchor
        first_key = list(ax_rows.keys())[0]
        first_ax = self._fig.add_subplot(gs[ax_rows[first_key]])
        axes_map = {first_key: first_ax}
        for key in list(ax_rows.keys())[1:]:
            axes_map[key] = self._fig.add_subplot(gs[ax_rows[key]], sharex=first_ax)

        # Assign axes references
        self._ax_spec    = axes_map.get('spec', None)
        self._ax_spec_r  = axes_map.get('spec_r', None)
        self._ax_wave    = axes_map.get('wave', None)
        self._ax_wave_r  = axes_map.get('wave_r', None)
        self._ax_amp     = axes_map['amp']
        self._ax_events  = axes_map.get('events', None)
        self._ax_entropy = axes_map.get('entropy', None)

        self._fig.subplots_adjust(top=0.97, bottom=0.06, left=0.07, right=0.99)
        n_disp = N_DISPLAY_ROWS
        dummy  = np.full((n_disp, n_cols), db_floor, dtype=np.float32)

        # -- Spectrogram axes --
        if self._ax_spec is not None:
            self._spec_im = self._ax_spec.imshow(
                dummy, aspect='auto', origin='lower',
                extent=[0.0, disp_secs, 0, n_disp],
                vmin=db_floor, vmax=db_ceil,
                cmap=COLORMAP, interpolation='nearest',
            )
            self._ax_spec.set_ylabel('Freq (Hz) \u2014 L' if stereo else 'Frequency (Hz)')
            self._cursor_spec = self._ax_spec.axvline(x=0.0, color=C['green'], linewidth=1.0, alpha=0.7)
        else:
            self._spec_im = None
            self._cursor_spec = None

        if self._ax_spec_r is not None:
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

        # -- Waveform axes --
        ts = max(1, int(disp_secs * sr / CHUNK_FRAMES)) * CHUNK_FRAMES
        t_axis = self._get_t_axis(sr, disp_secs)

        if self._ax_wave is not None:
            amp_ylim = e.amp_ylim if e else 1.05
            if stereo and dmode == 'Both':
                # Overlaid L+R waveform
                (self._wave_line,) = self._ax_wave.plot(
                    t_axis, np.zeros(ts),
                    color=C['teal'], linewidth=0.7, antialiased=False, label='L',
                )
                (self._wave_line_r,) = self._ax_wave.plot(
                    t_axis, np.zeros(ts),
                    color=C['pink'], linewidth=0.7, antialiased=False, label='R',
                )
                self._ax_wave.legend(loc='upper right', fontsize=8,
                                     facecolor=C['mantle'], edgecolor=C['surface1'],
                                     labelcolor=C['text'])
            else:
                wave_label = 'Waveform' if not stereo else 'L'
                (self._wave_line,) = self._ax_wave.plot(
                    t_axis, np.zeros(ts),
                    color=C['teal'], linewidth=0.7, antialiased=False,
                    label=wave_label if (stereo and dmode != 'Both') else None,
                )
                if stereo and dmode != 'Both':
                    self._wave_line_r = None  # separate axis used
                elif not stereo:
                    self._wave_line_r = None
                else:
                    self._wave_line_r = None
            self._ax_wave.set_xlim(0.0, disp_secs)
            self._ax_wave.set_ylim(-amp_ylim, amp_ylim)
            self._ax_wave.set_ylabel('Wave \u2014 L' if (stereo and dmode == 'Waveform') else 'Waveform')
            self._cursor_wave = self._ax_wave.axvline(x=0.0, color=C['green'], linewidth=1.0, alpha=0.7)
        else:
            self._wave_line = None
            self._wave_line_r = None
            self._cursor_wave = None

        if self._ax_wave_r is not None:
            amp_ylim = e.amp_ylim if e else 1.05
            (self._wave_line_r,) = self._ax_wave_r.plot(
                t_axis, np.zeros(ts),
                color=C['pink'], linewidth=0.7, antialiased=False,
            )
            self._ax_wave_r.set_xlim(0.0, disp_secs)
            self._ax_wave_r.set_ylim(-amp_ylim, amp_ylim)
            self._ax_wave_r.set_ylabel('Wave \u2014 R')
            self._cursor_wave_r = self._ax_wave_r.axvline(x=0.0, color=C['green'], linewidth=1.0, alpha=0.7)
        else:
            if self._ax_wave is None:
                self._wave_line_r = self._wave_line_r if hasattr(self, '_wave_line_r') else None
            self._cursor_wave_r = None

        # -- Amplitude envelope axis --
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

        # -- #32: detect / record indicator strip --
        if self._ax_events is not None:
            # Amp no longer the bottom of its group — hide its x labels.
            self._ax_amp.tick_params(axis='x', labelbottom=False)
            # 2-row RGBA image: row 0 = detect (yellow), row 1 = record (green).
            # Initial image is fully opaque at axis facecolor — see
            # _build_events_rgba for rationale (opaque avoids blit ghosting).
            rgba0 = np.empty((2, max(1, n_cols), 4), dtype=np.float32)
            rgba0[..., 0] = 0x18 / 255.0
            rgba0[..., 1] = 0x18 / 255.0
            rgba0[..., 2] = 0x25 / 255.0
            rgba0[..., 3] = 1.0
            self._events_im = self._ax_events.imshow(
                rgba0, aspect='auto', origin='upper',
                extent=[0.0, disp_secs, 0, 2],
                interpolation='nearest',
            )
            self._ax_events.set_xlim(0.0, disp_secs)
            self._ax_events.set_ylim(0, 2)
            self._ax_events.set_yticks([0.5, 1.5])
            self._ax_events.set_yticklabels(['rec', 'det'], fontsize=7)
            self._ax_events.tick_params(axis='y', length=0, pad=2)
            self._cursor_events = self._ax_events.axvline(
                x=0.0, color=C['green'], linewidth=1.0, alpha=0.7)
            # The events strip takes the x-label when it's the last row
            # (no entropy axis below it). Otherwise entropy gets it.
            self._ax_amp.set_xlabel('')
            if self._ax_entropy is None:
                self._ax_events.set_xlabel('Time (s)')
            else:
                self._ax_events.set_xlabel('')
                self._ax_events.tick_params(axis='x', labelbottom=False)
        else:
            self._events_im = None
            self._cursor_events = None

        # -- Entropy axes --
        if self._ax_entropy is not None:
            ent_n_cols = e._n_cols if e else n_cols
            ent_t = np.linspace(0.0, disp_secs, ent_n_cols, endpoint=False, dtype=np.float32)
            ent_thr = e.spectral_threshold if e else 0.5
            (self._entropy_line,) = self._ax_entropy.plot(
                ent_t, np.ones(ent_n_cols),
                color=C['mauve'], linewidth=0.9, antialiased=False,
            )
            self._ax_entropy.set_xlim(0.0, disp_secs)
            self._ax_entropy.set_ylim(0.0, 1.05)
            self._ax_entropy.set_xlabel('Time (s)')
            self._ax_entropy.set_ylabel('Entropy')
            self._ax_amp.set_xlabel('')  # move x-label to entropy axis
            self._entropy_thr_line = self._ax_entropy.axhline(
                y=ent_thr, color=C['peach'], linewidth=1.5, linestyle=(0, (6, 3)),
            )
            self._entropy_thr_label = self._ax_entropy.text(
                0.005, ent_thr + 0.03, f'ent = {ent_thr:.3f}',
                transform=self._ax_entropy.get_yaxis_transform(),
                color=C['peach'], fontsize=8,
            )
            self._cursor_entropy = self._ax_entropy.axvline(
                x=0.0, color=C['green'], linewidth=1.0, alpha=0.7)
        else:
            self._entropy_line = None
            self._entropy_thr_line = None
            self._entropy_thr_label = None
            self._cursor_entropy = None

        if e and self._ax_spec is not None:
            self._update_spec_yticks(e)

        # Mark animated artists for blitting
        for a in self._get_config_artists():
            a.set_animated(True)
        self._canvas.draw()
        self._bg = self._canvas.copy_from_bbox(self._fig.bbox)

    def _get_config_artists(self):
        """Return list of all animated artists in config mode."""
        arts = [self._amp_line,
                self._cursor_amp,
                self._threshold_line, self._threshold_label]
        if self._spec_im is not None:
            arts.extend([self._spec_im, self._cursor_spec])
        if self._spec_im_r is not None:
            arts.extend([self._spec_im_r, self._cursor_spec_r])
        if self._amp_line_r is not None:
            arts.append(self._amp_line_r)
        if self._wave_line is not None:
            arts.extend([self._wave_line, self._cursor_wave])
        if self._wave_line_r is not None:
            arts.append(self._wave_line_r)
            if self._cursor_wave_r is not None:
                arts.append(self._cursor_wave_r)
        if self._entropy_line is not None:
            arts.extend([self._entropy_line, self._cursor_entropy,
                         self._entropy_thr_line, self._entropy_thr_label])
        if getattr(self, '_events_im', None) is not None:
            arts.extend([self._events_im, self._cursor_events])
        return arts

    @staticmethod
    def _get_vm_artists(vm: dict):
        """Return animated artists for one view-mode cell."""
        arts = [vm['amp_line'],
                vm['cursor_amp'],
                vm['thr_line'], vm['status_text']]
        if vm.get('spec_im') is not None:
            arts.extend([vm['spec_im'], vm['cursor_spec']])
        if vm['amp_line_r'] is not None:
            arts.append(vm['amp_line_r'])
        if vm.get('wave_line') is not None:
            arts.extend([vm['wave_line'], vm['cursor_wave']])
        if vm.get('wave_line_r') is not None:
            arts.append(vm['wave_line_r'])
        if vm.get('entropy_line') is not None:
            arts.extend([vm['entropy_line'], vm['cursor_entropy'],
                         vm['entropy_thr_line']])
        if vm.get('events_im') is not None:
            arts.extend([vm['events_im'], vm['cursor_events']])
        # #28 / #29: sticky saturation / drop indicator text — always
        # present even when empty so the blitter keeps them in sync.
        if vm.get('sat_text') is not None:
            arts.append(vm['sat_text'])
        if vm.get('drop_text') is not None:
            arts.append(vm['drop_text'])
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
        if self._ax_spec is not None:
            self._apply_spec_yticks(self._ax_spec, e)
        if self._ax_spec_r is not None:
            self._apply_spec_yticks(self._ax_spec_r, e)

    def _get_t_axis(self, sr: int, disp_secs: float = DISPLAY_SECONDS) -> np.ndarray:
        """Return a time axis array for the given sample rate + buffer length, cached."""
        key = (sr, disp_secs)
        if key != self._t_axis_key:
            self._t_axis_key = key
            ts = max(1, int(disp_secs * sr / CHUNK_FRAMES)) * CHUNK_FRAMES
            self._t_axis = np.linspace(0.0, disp_secs,
                                       ts,
                                       endpoint=False, dtype=np.float32)
        return self._t_axis

    @property
    def _sel(self) -> RecordingEntity:
        if 0 <= self._selected_idx < len(self._entities):
            return self._entities[self._selected_idx]
        return None

    # ──────────────────────────────────────────────────────────────────────
    # #32: events strip helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_events_rgba(e: 'RecordingEntity') -> np.ndarray | None:
        """Build a 2-row RGBA array (det, rec) for the events imshow.

        Downsamples the per-sample mask buffers to one column per
        CHUNK_FRAMES block so the image width matches ``e._n_cols``
        (and therefore aligns with the spectrogram/entropy grid).

        "Off" cells are painted with the axis face colour (Catppuccin
        mantle) at full opacity — NOT transparent — so that matplotlib
        blitting always fully overwrites previous frame content. A
        previous implementation used alpha=0 for "off" cells, which
        left ghost bars on screen after the ring buffer cycled because
        transparent pixels don't erase what the renderer painted on
        earlier frames.
        """
        n_cols = e._n_cols
        total = e._total_samples
        if total <= 0 or n_cols <= 0:
            return None
        det = e.detect_mask_buffer[:n_cols * CHUNK_FRAMES].reshape(
            n_cols, CHUNK_FRAMES).any(axis=1)
        rec = e.record_mask_buffer[:n_cols * CHUNK_FRAMES].reshape(
            n_cols, CHUNK_FRAMES).any(axis=1)
        # Axis face colour = Catppuccin mantle (#181825) — keep in sync
        # with the rcParam set in __init__.
        bg_r, bg_g, bg_b = 0x18 / 255.0, 0x18 / 255.0, 0x25 / 255.0
        rgba = np.empty((2, n_cols, 4), dtype=np.float32)
        rgba[..., 0] = bg_r
        rgba[..., 1] = bg_g
        rgba[..., 2] = bg_b
        rgba[..., 3] = 1.0  # fully opaque everywhere
        # Row 0 = detect (top): Catppuccin yellow where True.
        rgba[0, det, 0] = 0.976  # 0xF9
        rgba[0, det, 1] = 0.886  # 0xE2
        rgba[0, det, 2] = 0.686  # 0xAF
        # Row 1 = record (bottom): Catppuccin green where True.
        rgba[1, rec, 0] = 0.651  # 0xA6
        rgba[1, rec, 1] = 0.890  # 0xE3
        rgba[1, rec, 2] = 0.631  # 0xA1
        return rgba

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

        # #7: persistent monitor bar at the very top — visible in both
        # Config and View mode so the user can always toggle which
        # stream is routed to the output speakers.
        self._monitor_bar = self._build_monitor_bar()
        vbox.addWidget(self._monitor_bar)

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

        # Config panel — single compact row: Trigger | Display | Transport | Settings
        self._config_widgets: list[QWidget] = []

        hl = self._hline()
        params_panel = self._build_params()
        spec_panel = self._build_spec_params()
        transport_panel = self._build_transport()
        settings_panel = self._build_settings()

        config_row = QWidget()
        config_row.setStyleSheet(f'background-color: {C["mantle"]};')
        config_row.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        config_h = QHBoxLayout(config_row)
        config_h.setContentsMargins(0, 0, 0, 0)
        config_h.setSpacing(0)
        config_h.addWidget(params_panel)
        config_h.addWidget(spec_panel)
        config_h.addWidget(transport_panel)
        config_h.addWidget(settings_panel, stretch=1)

        vbox.addWidget(hl)
        vbox.addWidget(config_row)
        self._config_widgets.extend([hl, config_row])

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
        self._btn_config_mode.setToolTip('Return to Config mode to edit parameters')
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
        self._vm_cols_spin.setToolTip('Number of columns in the View mode grid')
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
        self._vm_height_spin = QSpinBox()
        self._vm_height_spin.setToolTip('Row height for each recording tile in View mode')
        self._vm_height_spin.setRange(120, 700)
        self._vm_height_spin.setValue(self._vm_panel_height)
        self._vm_height_spin.setSuffix(' px')
        self._vm_height_spin.setFixedWidth(90)
        self._vm_height_spin.valueChanged.connect(self._on_vm_height_changed)
        h.addWidget(self._vm_height_spin)

        return w

    def _hline(self) -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.HLine)
        f.setFrameShadow(QFrame.Plain)
        return f

    # ── Audio monitor bar (#7) ───────────────────────────────────────

    # Sentinel userData for "no monitor source" in the source combo.
    _MON_OFF = '__off__'

    def _build_monitor_bar(self) -> QWidget:
        """Thin persistent bar exposing the audio-monitor controls.

        Kept deliberately compact — one row, always visible in both
        Config and View modes. Contains the global output device
        dropdown and a single "Monitor" combo that picks which
        RecordingEntity (if any) is currently routed to the output.
        """
        w = QWidget()
        w.setStyleSheet(
            f'QWidget#monitor_bar {{ background-color: {C["mantle"]}; '
            f'border-bottom: 1px solid {C["surface0"]}; }}')
        w.setObjectName('monitor_bar')
        w.setFixedHeight(34)
        h = QHBoxLayout(w)
        h.setContentsMargins(10, 3, 10, 3)
        h.setSpacing(8)

        icon = QLabel('\U0001F3A7')  # headphones
        icon.setStyleSheet(f'color: {C["mauve"]}; font-size: 12pt;')
        icon.setToolTip('Audio monitor loopback — routes one input stream to an output device')
        h.addWidget(icon)

        lbl_src = QLabel('Monitor:')
        lbl_src.setStyleSheet(f'color: {C["subtext"]}; font-size: 9pt;')
        h.addWidget(lbl_src)

        self._monitor_src_combo = QComboBox()
        self._monitor_src_combo.setToolTip(
            'Which recording stream to route to the output — only one '
            'at a time (switching stops the previous). Independent of '
            'acquisition / recording state.')
        self._monitor_src_combo.setFixedWidth(180)
        self._monitor_src_combo.addItem('Off', userData=self._MON_OFF)
        self._monitor_src_combo.currentIndexChanged.connect(self._on_monitor_source_changed)
        h.addWidget(self._monitor_src_combo)

        h.addSpacing(8)

        lbl_out = QLabel('Output:')
        lbl_out.setStyleSheet(f'color: {C["subtext"]}; font-size: 9pt;')
        h.addWidget(lbl_out)

        self._monitor_out_combo = QComboBox()
        self._monitor_out_combo.setToolTip(
            'Output audio device (speakers/headphones) used for monitor loopback')
        self._monitor_out_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._monitor_out_combo.setMinimumWidth(200)
        self._populate_monitor_output_combo()
        self._monitor_out_combo.currentIndexChanged.connect(self._on_monitor_output_changed)
        h.addWidget(self._monitor_out_combo, stretch=1)

        btn_refresh = QPushButton('\u21BB')
        btn_refresh.setObjectName('btn_small')
        btn_refresh.setFixedSize(26, 22)
        btn_refresh.setToolTip('Rescan available output devices')
        btn_refresh.clicked.connect(self._on_refresh_monitor_outputs)
        h.addWidget(btn_refresh)

        self._monitor_status = QLabel('')
        self._monitor_status.setStyleSheet(
            f'color: {C["subtext"]}; font-size: 9pt; min-width: 40px;')
        h.addWidget(self._monitor_status)

        return w

    def _populate_monitor_output_combo(self):
        """Fill the output-device combo; ``None`` entry = disabled."""
        combo = self._monitor_out_combo
        prev = combo.currentData() if combo.count() > 0 else None
        combo.blockSignals(True)
        combo.clear()
        combo.addItem('\u2014 None (disabled)', userData=None)
        restore_idx = 0
        try:
            default_out = sd.default.device[1]
        except Exception:
            default_out = -1
        default_idx = 0
        for dev_id, info in list_output_devices():
            api = host_api_name(info)
            label = f"{info['name']}  [{api}]" if api else info['name']
            combo.addItem(label, userData=dev_id)
            idx = combo.count() - 1
            if prev is not None and prev == dev_id:
                restore_idx = idx
            if dev_id == default_out and default_idx == 0:
                default_idx = idx
        combo.setCurrentIndex(restore_idx if prev is not None else default_idx)
        combo.blockSignals(False)

    def _refresh_monitor_source_combo(self):
        """Rebuild the monitor-source dropdown from the entity list."""
        combo = self._monitor_src_combo
        prev = combo.currentData() if combo.count() > 0 else self._MON_OFF
        combo.blockSignals(True)
        combo.clear()
        combo.addItem('Off', userData=self._MON_OFF)
        restore_idx = 0
        for i, e in enumerate(self._entities):
            token = id(e)
            combo.addItem(e.name, userData=token)
            if prev == token:
                restore_idx = combo.count() - 1
        combo.setCurrentIndex(restore_idx)
        combo.blockSignals(False)
        # Sync the monitor backend with whatever ended up selected.
        data = combo.currentData()
        self._apply_monitor_source(data)

    def _apply_monitor_source(self, source_token):
        """Switch the monitor to a source token from the combo.

        ``source_token`` is either :attr:`_MON_OFF` or ``id(entity)``.
        When switching to a live entity, the output stream is re-opened
        at that entity's sample rate / channel count so the playback
        isn't speed-shifted.
        """
        if source_token == self._MON_OFF or source_token is None:
            self._monitor.set_source(None)
            self._update_monitor_status()
            return
        ent = next((e for e in self._entities if id(e) == source_token), None)
        if ent is None:
            self._monitor.set_source(None)
            self._update_monitor_status()
            return
        # Re-open the output stream at the source's SR if needed so the
        # loopback doesn't play back at the wrong pitch.
        out_dev = self._monitor_out_combo.currentData()
        want_ch = 2 if ent.channel_mode == 'Stereo' else 1
        if (out_dev is not None
                and (self._monitor.samplerate != ent.sample_rate
                     or self._monitor.channels != want_ch
                     or not self._monitor.running)):
            self._monitor.set_output_device(out_dev,
                                            samplerate=ent.sample_rate,
                                            channels=want_ch)
        self._monitor.set_source(source_token)
        self._update_monitor_status()

    def _on_monitor_source_changed(self, _idx: int):
        self._apply_monitor_source(self._monitor_src_combo.currentData())

    def _on_monitor_output_changed(self, _idx: int):
        dev = self._monitor_out_combo.currentData()
        # Pick the SR/channels of the currently-selected source so the
        # first playback doesn't have to reopen the stream.
        src = self._monitor_src_combo.currentData()
        sr = SAMPLE_RATE
        ch = 1
        if src != self._MON_OFF and src is not None:
            ent = next((e for e in self._entities if id(e) == src), None)
            if ent is not None:
                sr = ent.sample_rate
                ch = 2 if ent.channel_mode == 'Stereo' else 1
        if dev is None:
            self._monitor.set_output_device(None)
        else:
            ok = self._monitor.set_output_device(dev, samplerate=sr, channels=ch)
            if not ok:
                QMessageBox.warning(
                    self, 'Monitor Output',
                    f'Could not open output device:\n'
                    f'{self._monitor_out_combo.currentText()}\n\n'
                    f'{self._monitor.last_error or ""}')
                # Revert to "None".
                self._monitor_out_combo.blockSignals(True)
                self._monitor_out_combo.setCurrentIndex(0)
                self._monitor_out_combo.blockSignals(False)
        self._update_monitor_status()

    def _on_refresh_monitor_outputs(self):
        self._populate_monitor_output_combo()

    def _update_monitor_status(self):
        """Reflect backend state on the little status label."""
        if not hasattr(self, '_monitor_status'):
            return
        if self._monitor.source_id is None:
            self._monitor_status.setText('off')
            self._monitor_status.setStyleSheet(
                f'color: {C["surface2"]}; font-size: 9pt; min-width: 40px;')
            return
        if not self._monitor.running:
            self._monitor_status.setText('no output')
            self._monitor_status.setStyleSheet(
                f'color: {C["peach"]}; font-size: 9pt; min-width: 40px;')
            return
        self._monitor_status.setText('\u25B6 live')
        self._monitor_status.setStyleSheet(
            f'color: {C["green"]}; font-size: 9pt; min-width: 40px; font-weight: bold;')

    # ── Transport ─────────────────────────────────────────────────────────

    def _build_transport(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet(f'background-color: {C["mantle"]};')
        w.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        outer = QHBoxLayout(w)
        outer.setContentsMargins(6, 2, 6, 2)
        outer.setSpacing(6)

        # Left column: buttons grid
        btn_box = QGroupBox('CONTROLS')
        btn_g = QGridLayout(btn_box)
        btn_g.setSpacing(4)
        btn_g.setContentsMargins(8, 4, 8, 4)

        self._btn_start_acq = QPushButton('Start Acq')
        self._btn_stop_acq  = QPushButton('Stop Acq')
        self._btn_start_acq.setObjectName('btn_start_acq')
        self._btn_stop_acq .setObjectName('btn_stop_acq')
        self._btn_start_acq.setToolTip('Start audio acquisition (live monitoring) for the selected recording')
        self._btn_stop_acq .setToolTip('Stop audio acquisition for the selected recording')

        self._btn_start_rec = QPushButton('Start Rec')
        self._btn_stop_rec  = QPushButton('Stop Rec')
        self._btn_start_rec.setObjectName('btn_start_rec')
        self._btn_stop_rec .setObjectName('btn_stop_rec')
        self._btn_start_rec.setToolTip('Enable threshold-triggered WAV recording for the selected recording')
        self._btn_stop_rec .setToolTip('Disable threshold-triggered WAV recording for the selected recording')

        self._btn_reset = QPushButton('Reset')
        self._btn_reset.setObjectName('btn_browse')
        self._btn_reset.setToolTip('Reset all trigger and display parameters to their defaults')

        self._btn_view_mode = QPushButton('\u25a3 View')
        self._btn_view_mode.setObjectName('btn_view_mode')
        self._btn_view_mode.setToolTip('Switch to View mode — a read-only monitoring grid of all recordings')
        self._btn_view_mode.setStyleSheet(
            f'QPushButton {{ background-color: {C["surface0"]}; color: {C["mauve"]}; '
            f'border: 1px solid {C["mauve"]}; border-radius: 5px; '
            f'padding: 4px 8px; font-weight: bold; min-width: 0px; }}'
            f'QPushButton:hover {{ background-color: {C["surface1"]}; }}'
        )

        self._btn_save    = QPushButton('\U0001f4be Save')
        self._btn_save_as = QPushButton('\U0001f4be Save As')
        self._btn_load    = QPushButton('\U0001f4c2 Load')
        self._btn_save   .setToolTip('Save configuration to the current file')
        self._btn_save_as.setToolTip('Save configuration to a new file')
        self._btn_load   .setToolTip('Load configuration from a file (.json or legacy .chirp)')
        for btn in (self._btn_save, self._btn_save_as, self._btn_load):
            btn.setObjectName('btn_browse')

        btn_g.addWidget(self._btn_start_acq, 0, 0)
        btn_g.addWidget(self._btn_stop_acq,  0, 1)
        btn_g.addWidget(self._btn_start_rec, 1, 0)
        btn_g.addWidget(self._btn_stop_rec,  1, 1)
        btn_g.addWidget(self._btn_save,      2, 0)
        btn_g.addWidget(self._btn_save_as,   2, 1)
        btn_g.addWidget(self._btn_load,      3, 0)
        btn_g.addWidget(self._btn_reset,     3, 1)
        btn_g.addWidget(self._btn_view_mode, 4, 0, 1, 2)

        # Right column: status labels
        status_box = QGroupBox('STATUS')
        status_v   = QVBoxLayout(status_box)
        status_v.setSpacing(2)
        status_v.setContentsMargins(8, 4, 8, 4)
        self._lbl_acq_status  = QLabel('ACQ  \u25cf  STOPPED')
        self._lbl_rec_status  = QLabel('REC  \u25cf  STOPPED')
        self._lbl_trig_status = QLabel('TRIG \u25cf  IDLE')
        self._lbl_entropy     = QLabel('ENT  \u2014')
        self._lbl_acq_status .setObjectName('status_off')
        self._lbl_rec_status .setObjectName('status_off')
        self._lbl_trig_status.setObjectName('trig_idle')
        self._lbl_entropy    .setObjectName('trig_idle')
        mono = QFont('Consolas', 9)
        for lbl in (self._lbl_acq_status, self._lbl_rec_status, self._lbl_trig_status,
                     self._lbl_entropy):
            lbl.setFont(mono)
        status_v.addWidget(self._lbl_acq_status)
        status_v.addWidget(self._lbl_rec_status)
        status_v.addWidget(self._lbl_trig_status)
        status_v.addWidget(self._lbl_entropy)
        self._blink_counter = 0

        outer.addWidget(btn_box)
        outer.addWidget(status_box)
        return w

    # ── Trigger Parameters ────────────────────────────────────────────────

    def _build_params(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet(f'background-color: {C["mantle"]};')
        w.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        outer = QHBoxLayout(w)
        outer.setContentsMargins(6, 2, 6, 2)

        # Hidden threshold spinbox (synced from amplitude graph drag)
        self._sb_thr = QDoubleSpinBox()
        self._sb_thr.setRange(0.0, 1.0)
        self._sb_thr.setSingleStep(0.005)
        self._sb_thr.setDecimals(3)
        self._sb_thr.setValue(DEFAULT_THRESHOLD)
        self._sb_thr.hide()

        trig_box = QGroupBox('TRIGGER')
        trig_g   = QGridLayout(trig_box)
        trig_g.setVerticalSpacing(4)
        trig_g.setHorizontalSpacing(8)
        trig_g.setContentsMargins(8, 4, 8, 4)

        self._sb_mc = self._param_row(trig_g, 0, 0, 'Min Cross',
            sb_min=0.0, sb_max=60.0, sb_step=0.001, sb_dec=3, suffix=' s',
            sb_init=DEFAULT_MIN_CROSS)
        self._sb_hold = self._param_row(trig_g, 1, 0, 'Hold',
            sb_min=0.0, sb_max=60.0, sb_step=0.1, sb_dec=2, suffix=' s',
            sb_init=DEFAULT_HOLD)
        self._sb_pre = self._param_row(trig_g, 2, 0, 'Pre-Trigger',
            sb_min=0.0, sb_max=60.0, sb_step=0.1, sb_dec=2, suffix=' s',
            sb_init=DEFAULT_PRE_TRIG)
        self._sb_post_trig = self._param_row(trig_g, 3, 0, 'Post-Trigger',
            sb_min=0.0, sb_max=60.0, sb_step=0.1, sb_dec=2, suffix=' s',
            sb_init=DEFAULT_POST_TRIG)
        self._sb_maxr = self._param_row(trig_g, 4, 0, 'Max Rec',
            sb_min=1.0, sb_max=3600.0, sb_step=1.0, sb_dec=1, suffix=' s',
            sb_init=DEFAULT_MAX_REC)

        self._sb_mc.setToolTip('Min Cross: minimum time the signal must stay above the threshold to start a recording')
        self._sb_hold.setToolTip('Hold: duration of silence after the signal drops before a recording is considered finished')
        self._sb_pre.setToolTip('Pre-Trigger: audio kept before the trigger point (lookback saved to the WAV)')
        self._sb_post_trig.setToolTip('Post-Trigger: audio kept after the last above-threshold sample (tail of the saved WAV)')
        self._sb_maxr.setToolTip('Max Rec: maximum length of a single WAV segment — longer events are split')

        # Band filter row (row 5)
        self._chk_freq = QCheckBox('Band filter')
        self._chk_freq.setChecked(False)
        self._chk_freq.setToolTip('Apply a 4th-order Butterworth band-pass filter to the trigger signal and spectrogram input')

        self._sb_freq_lo = QDoubleSpinBox()
        self._sb_freq_lo.setRange(1.0, SAMPLE_RATE / 2 - 1)
        self._sb_freq_lo.setValue(DEFAULT_FREQ_LO)
        self._sb_freq_lo.setSingleStep(100.0)
        self._sb_freq_lo.setDecimals(0)
        self._sb_freq_lo.setSuffix(' Hz')
        self._sb_freq_lo.setFixedWidth(100)
        self._sb_freq_lo.setEnabled(False)
        self._sb_freq_lo.setToolTip('Band-pass filter low cutoff (Hz)')

        self._sb_freq_hi = QDoubleSpinBox()
        self._sb_freq_hi.setRange(1.0, SAMPLE_RATE / 2 - 1)
        self._sb_freq_hi.setValue(DEFAULT_FREQ_HI)
        self._sb_freq_hi.setSingleStep(100.0)
        self._sb_freq_hi.setDecimals(0)
        self._sb_freq_hi.setSuffix(' Hz')
        self._sb_freq_hi.setFixedWidth(100)
        self._sb_freq_hi.setEnabled(False)
        self._sb_freq_hi.setToolTip('Band-pass filter high cutoff (Hz)')

        self._chk_freq.toggled.connect(lambda on: (
            self._sb_freq_lo.setEnabled(on),
            self._sb_freq_hi.setEnabled(on),
        ))

        lbl_lo = QLabel('Lo')
        lbl_lo.setObjectName('param_label')
        lbl_hi = QLabel('Hi')
        lbl_hi.setObjectName('param_label')

        filt_row = QHBoxLayout()
        filt_row.setSpacing(8)
        filt_row.addWidget(self._chk_freq)
        filt_row.addWidget(lbl_lo)
        filt_row.addWidget(self._sb_freq_lo)
        filt_row.addWidget(lbl_hi)
        filt_row.addWidget(self._sb_freq_hi)
        filt_row.addStretch()
        trig_g.addLayout(filt_row, 5, 0, 1, 3)

        # Detect mode row (row 6)
        lbl_detect = QLabel('Detect Mode')
        lbl_detect.setObjectName('param_label')
        lbl_detect.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        self._combo_detect_mode = QComboBox()
        self._combo_detect_mode.setFixedWidth(160)
        for dm in ('Amplitude Only', 'Spectral Only', 'Amp AND Spectral', 'Amp OR Spectral'):
            self._combo_detect_mode.addItem(dm)
        self._combo_detect_mode.setCurrentText('Amplitude Only')
        self._combo_detect_mode.setToolTip(
            'Trigger detection mode:\n'
            '  • Amplitude Only — trigger when signal amplitude exceeds threshold\n'
            '  • Spectral Only — trigger when spectral entropy falls below threshold (tonal sound)\n'
            '  • Amp AND Spectral — both conditions must be met\n'
            '  • Amp OR Spectral — either condition triggers')

        detect_row = QHBoxLayout()
        detect_row.setSpacing(8)
        detect_row.addWidget(lbl_detect)
        detect_row.addWidget(self._combo_detect_mode)
        detect_row.addStretch()
        trig_g.addLayout(detect_row, 6, 0, 1, 3)

        # Entropy threshold row (row 7)
        self._sb_entropy_thr = self._param_row(trig_g, 7, 0, 'Entropy Thr',
            sb_min=0.0, sb_max=1.0, sb_step=0.01, sb_dec=2, suffix='',
            sb_init=0.50)
        self._sb_entropy_thr.setEnabled(False)
        self._sb_entropy_thr.setToolTip('Spectral entropy threshold — triggers when entropy falls below this value (0 = pure tone, 1 = white noise)')

        self._combo_detect_mode.currentTextChanged.connect(self._on_detect_mode_changed)

        # Auto-calibrate row (row 8)
        self._btn_calibrate = QPushButton('Auto Calibrate')
        self._btn_calibrate.setObjectName('btn_small')
        self._btn_calibrate.setFixedWidth(110)
        self._btn_calibrate.setToolTip(
            'Measure ambient noise for 3 seconds and set threshold automatically')
        self._lbl_calib_status = QLabel('')
        self._lbl_calib_status.setObjectName('param_label')

        self._sb_calib_dur = QDoubleSpinBox()
        self._sb_calib_dur.setRange(1.0, 10.0)
        self._sb_calib_dur.setValue(3.0)
        self._sb_calib_dur.setSingleStep(0.5)
        self._sb_calib_dur.setDecimals(1)
        self._sb_calib_dur.setSuffix(' s')
        self._sb_calib_dur.setFixedWidth(80)
        self._sb_calib_dur.setToolTip('Calibration duration')

        self._sb_calib_margin = QDoubleSpinBox()
        self._sb_calib_margin.setRange(1.1, 10.0)
        self._sb_calib_margin.setValue(3.0)
        self._sb_calib_margin.setSingleStep(0.5)
        self._sb_calib_margin.setDecimals(1)
        self._sb_calib_margin.setSuffix('x')
        self._sb_calib_margin.setFixedWidth(75)
        self._sb_calib_margin.setToolTip('Margin multiplier above noise floor')

        lbl_dur = QLabel('Dur')
        lbl_dur.setObjectName('param_label')
        lbl_margin = QLabel('Margin')
        lbl_margin.setObjectName('param_label')

        calib_row = QHBoxLayout()
        calib_row.setSpacing(8)
        calib_row.addWidget(self._btn_calibrate)
        calib_row.addWidget(lbl_dur)
        calib_row.addWidget(self._sb_calib_dur)
        calib_row.addWidget(lbl_margin)
        calib_row.addWidget(self._sb_calib_margin)
        calib_row.addWidget(self._lbl_calib_status)
        calib_row.addStretch()
        trig_g.addLayout(calib_row, 8, 0, 1, 3)

        # #10 / c24: sync + apply-all controls
        sync_row = QHBoxLayout()
        sync_row.setSpacing(10)
        self._chk_shared_trigger = QCheckBox('Sync trigger across all recordings')
        self._chk_shared_trigger.setToolTip(
            'When enabled, trigger parameter changes propagate to all recordings')
        self._btn_apply_all = QPushButton('Apply All Settings \u2192')
        self._btn_apply_all.setObjectName('btn_small')
        self._btn_apply_all.setFixedWidth(150)
        self._btn_apply_all.setToolTip(
            'Copy ALL settings (trigger + display + output) from the selected '
            'recording to every other recording (one-shot)')
        sync_row.addWidget(self._chk_shared_trigger)
        sync_row.addWidget(self._btn_apply_all)
        sync_row.addStretch()
        trig_g.addLayout(sync_row, 9, 0, 1, 3)

        outer.addWidget(trig_box)
        return w

    # ── Display parameters ───────────────────────────────────────────────

    def _build_spec_params(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet(f'background-color: {C["mantle"]};')
        w.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        outer = QHBoxLayout(w)
        outer.setContentsMargins(6, 2, 6, 2)

        box  = QGroupBox('DISPLAY')
        grid = QGridLayout(box)
        grid.setVerticalSpacing(4)
        grid.setHorizontalSpacing(8)
        grid.setContentsMargins(8, 4, 8, 4)

        lbl_buf = QLabel('Buffer')
        lbl_buf.setObjectName('param_label')
        lbl_buf.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        self._buf_combo = QComboBox()
        for s in RecordingEntity.SUPPORTED_DISPLAY_SECONDS:
            label = f'{int(s)}s' if s == int(s) else f'{s}s'
            self._buf_combo.addItem(label, userData=s)
        self._buf_combo.setCurrentText(f'{int(DISPLAY_SECONDS)}s')
        self._buf_combo.setFixedWidth(90)
        self._buf_combo.setToolTip('Length of visible history (seconds) in the live display')

        self._sb_gain = self._param_row(grid, 0, 0, 'Gain',
            sb_min=-20.0, sb_max=60.0, sb_step=1.0, sb_dec=1, suffix=' dB',
            sb_init=0.0)

        self._sb_floor = self._param_row(grid, 1, 0, 'dB Floor',
            sb_min=-120.0, sb_max=0.0, sb_step=1.0, sb_dec=1, suffix=' dB',
            sb_init=SPEC_DB_MIN)

        self._sb_ceil = self._param_row(grid, 2, 0, 'dB Ceil',
            sb_min=-120.0, sb_max=0.0, sb_step=1.0, sb_dec=1, suffix=' dB',
            sb_init=SPEC_DB_MAX)

        self._sb_gain.setToolTip('Gain applied to the spectrogram (dB) — brightens or darkens the image')
        self._sb_floor.setToolTip('Minimum dB value shown in the spectrogram colormap')
        self._sb_ceil.setToolTip('Maximum dB value shown in the spectrogram colormap')

        lbl_fft = QLabel('FFT')
        lbl_fft.setObjectName('param_label')
        lbl_fft.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        self._combo_fft = QComboBox()
        self._combo_fft.setFixedWidth(90)
        for sz in SpectrogramAccumulator.FFT_SIZES:
            self._combo_fft.addItem(str(sz), userData=sz)
        self._combo_fft.setCurrentText(str(SPECTROGRAM_NPERSEG))
        self._combo_fft.setToolTip('FFT size (nperseg). Larger = better frequency resolution, worse time resolution')

        lbl_win = QLabel('Win')
        lbl_win.setObjectName('param_label')
        lbl_win.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        self._combo_win = QComboBox()
        self._combo_win.setFixedWidth(90)
        for wn in SpectrogramAccumulator.WINDOW_TYPES:
            self._combo_win.addItem(wn.capitalize(), userData=wn)
        self._combo_win.setCurrentIndex(0)
        self._combo_win.setToolTip('FFT window function')

        lbl_fscale = QLabel('Scale')
        lbl_fscale.setObjectName('param_label')
        lbl_fscale.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        self._combo_fscale = QComboBox()
        self._combo_fscale.setFixedWidth(90)
        self._combo_fscale.addItems(['Linear', 'Log', 'Mel'])
        self._combo_fscale.setCurrentText('Mel')
        self._combo_fscale.setToolTip('Frequency axis scale for the spectrogram: Linear, Log, or Mel')

        lbl_dfl = QLabel('Lo')
        lbl_dfl.setObjectName('param_label')
        lbl_dfl.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        self._sb_disp_freq_lo = QDoubleSpinBox()
        self._sb_disp_freq_lo.setRange(0.0, SAMPLE_RATE / 2 - 1)
        self._sb_disp_freq_lo.setValue(0.0)
        self._sb_disp_freq_lo.setSingleStep(100.0)
        self._sb_disp_freq_lo.setDecimals(0)
        self._sb_disp_freq_lo.setSuffix(' Hz')
        self._sb_disp_freq_lo.setFixedWidth(90)
        self._sb_disp_freq_lo.setToolTip('Lowest frequency shown in the spectrogram (Hz)')

        lbl_dfh = QLabel('Hi')
        lbl_dfh.setObjectName('param_label')
        lbl_dfh.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        self._sb_disp_freq_hi = QDoubleSpinBox()
        self._sb_disp_freq_hi.setRange(1.0, SAMPLE_RATE / 2)
        self._sb_disp_freq_hi.setValue(SAMPLE_RATE / 2)
        self._sb_disp_freq_hi.setSingleStep(100.0)
        self._sb_disp_freq_hi.setDecimals(0)
        self._sb_disp_freq_hi.setSuffix(' Hz')
        self._sb_disp_freq_hi.setFixedWidth(90)
        self._sb_disp_freq_hi.setToolTip('Highest frequency shown in the spectrogram (Hz)')

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
        grid.addWidget(lbl_buf,            5, 3)
        grid.addWidget(self._buf_combo,    5, 4)

        lbl_dmode = QLabel('View')
        lbl_dmode.setObjectName('param_label')
        lbl_dmode.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        self._combo_display_mode = QComboBox()
        self._combo_display_mode.addItems(['Spectrogram', 'Waveform', 'Both'])
        self._combo_display_mode.setCurrentText('Spectrogram')
        self._combo_display_mode.setFixedWidth(90)
        self._combo_display_mode.setToolTip('Visualization mode — Spectrogram, raw Waveform, or Both')
        grid.addWidget(lbl_dmode,                  6, 3)
        grid.addWidget(self._combo_display_mode,   6, 4)

        self._chk_shared_spec = QCheckBox('Sync across all recordings')
        self._chk_shared_spec.setToolTip('When enabled, display settings (gain, FFT, scale, etc.) apply to all recordings')
        grid.addWidget(self._chk_shared_spec, 7, 0, 1, 5)

        outer.addWidget(box)
        return w

    def _param_row(self, grid, row, col, label,
                   sb_min, sb_max, sb_step, sb_dec, suffix,
                   sb_init) -> QDoubleSpinBox:
        lbl = QLabel(label)
        lbl.setObjectName('param_label')
        lbl.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)

        sb = QDoubleSpinBox()
        sb.setRange(sb_min, sb_max)
        sb.setSingleStep(sb_step)
        sb.setDecimals(sb_dec)
        sb.setValue(sb_init)
        sb.setSuffix(suffix)
        sb.setFixedWidth(100)
        sb.setAlignment(Qt.AlignRight)

        grid.addWidget(lbl, row, col)
        grid.addWidget(sb,  row, col + 1)
        return sb

    # ── Settings ──────────────────────────────────────────────────────────

    def _build_settings(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet(f'background-color: {C["mantle"]};')
        w.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        outer = QVBoxLayout(w)
        outer.setContentsMargins(6, 2, 6, 2)
        outer.setSpacing(2)

        output_box = QGroupBox('OUTPUT')
        output_g   = QGridLayout(output_box)
        output_g.setVerticalSpacing(4)
        output_g.setHorizontalSpacing(6)
        output_g.setContentsMargins(8, 4, 8, 4)
        lbl_folder = QLabel('Folder')
        lbl_folder.setObjectName('param_label')
        self._folder_edit = QLineEdit(RECORDINGS_DIR)
        self._folder_edit.setPlaceholderText('Path to recordings folder...')
        self._folder_edit.setToolTip('Output folder where triggered WAV files are saved')
        btn_browse = QPushButton('Browse...')
        btn_browse.setObjectName('btn_browse')
        btn_browse.setFixedWidth(70)
        btn_browse.setToolTip('Browse for the output folder')
        btn_browse.clicked.connect(self._on_browse)
        lbl_pfx = QLabel('Prefix')
        lbl_pfx.setObjectName('param_label')
        self._prefix_edit = QLineEdit()
        self._prefix_edit.setPlaceholderText('e.g. bird1_')
        self._prefix_edit.setToolTip('Optional prefix added to the start of each saved WAV filename')
        lbl_sfx = QLabel('Suffix')
        lbl_sfx.setObjectName('param_label')
        self._suffix_edit = QLineEdit()
        self._suffix_edit.setPlaceholderText('e.g. _cage3')
        self._suffix_edit.setToolTip('Optional suffix added to the end of each saved WAV filename')
        output_g.addWidget(lbl_folder,          0, 0)
        output_g.addWidget(self._folder_edit,   0, 1, 1, 3)
        output_g.addWidget(btn_browse,          0, 4)
        output_g.addWidget(lbl_pfx,             1, 0)
        output_g.addWidget(self._prefix_edit,   1, 1)
        output_g.addWidget(lbl_sfx,             1, 2)
        output_g.addWidget(self._suffix_edit,   1, 3)
        output_g.setColumnStretch(1, 1)
        output_g.setColumnStretch(3, 1)

        device_box = QGroupBox('INPUT DEVICE')
        device_v   = QVBoxLayout(device_box)
        device_v.setSpacing(4)
        device_v.setContentsMargins(8, 4, 8, 4)
        dev_row1 = QHBoxLayout()
        dev_row1.setSpacing(4)
        self._device_combo = QComboBox()
        self._device_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._device_combo.setToolTip('Audio input device used by this recording')
        self._populate_device_combo()
        btn_refresh = QPushButton('Refresh')
        btn_refresh.setObjectName('btn_browse')
        btn_refresh.setFixedWidth(60)
        btn_refresh.setToolTip('Rescan available audio devices')
        btn_refresh.clicked.connect(self._on_refresh_devices)
        dev_row1.addWidget(self._device_combo, stretch=1)
        dev_row1.addWidget(btn_refresh)
        dev_row2 = QHBoxLayout()
        dev_row2.setSpacing(4)
        lbl_ch = QLabel('Mode')
        self._chan_combo = QComboBox()
        self._chan_combo.addItems(['Mono', 'Left', 'Right', 'Stereo'])
        self._chan_combo.setCurrentIndex(0)
        self._chan_combo.setToolTip('Channel mode: Mono, single channel (Left/Right), or Stereo (both)')
        lbl_trig = QLabel('Trigger')
        self._trig_combo = QComboBox()
        self._trig_combo.addItems(['Average', 'Any Channel', 'Both Channels', 'Left Channel', 'Right Channel'])
        self._trig_combo.setCurrentIndex(0)
        self._trig_combo.setEnabled(False)
        self._trig_combo.setToolTip('How the stereo trigger is computed (only used in Stereo mode)')
        lbl_sr = QLabel('Rate')
        self._sr_combo = QComboBox()
        for r in RecordingEntity.SUPPORTED_RATES:
            self._sr_combo.addItem(f'{r} Hz', userData=r)
        self._sr_combo.setCurrentText(f'{SAMPLE_RATE} Hz')
        self._sr_combo.setFixedWidth(90)
        self._sr_combo.setToolTip('Audio sample rate — changing this rebuilds the audio pipeline')
        for lbl in (lbl_ch, lbl_trig, lbl_sr):
            lbl.setStyleSheet(f'color: {C["subtext"]}; font-size: 9pt;')
        dev_row2.addWidget(lbl_ch)
        dev_row2.addWidget(self._chan_combo)
        dev_row2.addSpacing(8)
        dev_row2.addWidget(lbl_trig)
        dev_row2.addWidget(self._trig_combo)
        dev_row2.addSpacing(8)
        dev_row2.addWidget(lbl_sr)
        dev_row2.addWidget(self._sr_combo)
        dev_row2.addStretch()

        # Row 3 — WAV simulation controls. Visible only when this
        # entity's input source is 'wav_file'.
        self._wav_ctrl_row = QHBoxLayout()
        self._wav_ctrl_row.setSpacing(6)
        self._btn_wav_reset = QPushButton('\u23EE Reset')
        self._btn_wav_reset.setObjectName('btn_small')
        self._btn_wav_reset.setFixedHeight(22)
        self._btn_wav_reset.setToolTip('Rewind the WAV playback to the start of the file')
        self._btn_wav_reset.clicked.connect(self._on_wav_reset_clicked)
        self._chk_wav_loop = QCheckBox('Loop')
        self._chk_wav_loop.setToolTip('When on, the WAV file restarts from the beginning after it ends')
        self._chk_wav_loop.setChecked(True)
        self._chk_wav_loop.toggled.connect(self._on_wav_loop_toggled)
        self._lbl_wav_time = QLabel('0:00 / 0:00')
        self._lbl_wav_time.setStyleSheet(
            f'color: {C["teal"]}; font-family: Consolas; font-size: 9pt;')
        self._lbl_wav_time.setToolTip('Elapsed / total duration of the WAV file')
        self._wav_ctrl_row.addWidget(self._btn_wav_reset)
        self._wav_ctrl_row.addWidget(self._chk_wav_loop)
        self._wav_ctrl_row.addSpacing(8)
        self._wav_ctrl_row.addWidget(self._lbl_wav_time)
        self._wav_ctrl_row.addStretch()

        # Hold the row's widgets in a list so we can toggle visibility
        # when the input source switches.
        self._wav_ctrl_widgets = [
            self._btn_wav_reset, self._chk_wav_loop, self._lbl_wav_time,
        ]
        # NB: don't reuse the name `w` here — it's the panel widget
        # bound at the top of this method and the parent of `outer`.
        # Shadowing it drops the only Python reference, Python GCs the
        # QWidget, and Qt cascades the delete to `outer` → crash on
        # the next `outer.addLayout` call.
        for _w in self._wav_ctrl_widgets:
            _w.setVisible(False)

        device_v.addLayout(dev_row1)
        device_v.addLayout(dev_row2)
        device_v.addLayout(self._wav_ctrl_row)

        ref_box = QGroupBox('REFERENCE DATE')
        ref_g   = QGridLayout(ref_box)
        ref_g.setVerticalSpacing(4)
        ref_g.setHorizontalSpacing(6)
        self._chk_ref_date = QCheckBox('Days post hatch')
        self._chk_ref_date.setToolTip('When enabled, saved files are organized into day-post-hatch subfolders')
        self._date_line = QLineEdit(datetime.date.today().strftime('%Y-%m-%d'))
        self._date_line.setPlaceholderText('YYYY-MM-DD')
        self._date_line.setFixedWidth(90)
        self._date_line.setToolTip('Reference (hatch) date in YYYY-MM-DD format')
        self._btn_pick_date = QPushButton('\u2026')
        self._btn_pick_date.setObjectName('btn_small')
        self._btn_pick_date.setFixedSize(28, 28)
        self._btn_pick_date.setToolTip('Pick the reference date from a calendar')
        self._lbl_day_count = QLabel('Day: —')
        self._lbl_day_count.setStyleSheet(f'color: {C["yellow"]}; font-size: 9pt; font-weight: bold;')
        lbl_dph_pfx = QLabel('Folder prefix')
        lbl_dph_pfx.setObjectName('param_label')
        self._dph_prefix_edit = QLineEdit()
        self._dph_prefix_edit.setPlaceholderText('e.g. day_')
        self._dph_prefix_edit.setToolTip('Optional prefix added to the day-post-hatch subfolder name')
        date_row = QHBoxLayout()
        date_row.setSpacing(4)
        date_row.addWidget(self._date_line)
        date_row.addWidget(self._btn_pick_date)
        date_row.addWidget(self._lbl_day_count)
        date_row.addStretch()
        pfx_row = QHBoxLayout()
        pfx_row.setSpacing(4)
        pfx_row.addWidget(lbl_dph_pfx)
        pfx_row.addWidget(self._dph_prefix_edit)
        ref_g.addWidget(self._chk_ref_date,  0, 0)
        ref_g.addLayout(date_row,            1, 0)
        ref_g.addLayout(pfx_row,             2, 0)

        # Top row: Output + Ref Date side by side
        top_row = QHBoxLayout()
        top_row.setSpacing(6)
        top_row.addWidget(output_box, stretch=3)
        top_row.addWidget(ref_box, stretch=1)
        outer.addLayout(top_row)
        outer.addWidget(device_box)
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
        self._btn_save    .clicked.connect(self._save_settings)
        self._btn_save_as .clicked.connect(self._save_settings_as)
        self._btn_load    .clicked.connect(self._load_settings)
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
        # `activated` fires once per user selection (including re-click
        # of the current item, unlike `currentIndexChanged`). Using only
        # `activated` keeps the WAV-sim picker fireable on re-select and
        # avoids the double-prompt that both signals firing caused.
        # Programmatic combo updates are wrapped in `blockSignals`, so
        # we don't need `currentIndexChanged` for those.
        self._device_combo.activated.connect(self._on_device_changed)
        self._chan_combo.currentTextChanged.connect(self._on_channel_mode_changed)
        self._trig_combo.currentTextChanged.connect(self._on_trigger_mode_changed)
        self._sr_combo.currentIndexChanged.connect(self._on_sample_rate_changed)
        self._buf_combo.currentIndexChanged.connect(self._on_display_buffer_changed)

        # Auto-calibrate
        self._btn_calibrate.clicked.connect(self._on_calibrate)

        # Freq filter write-through
        self._chk_freq  .toggled       .connect(self._on_freq_filter_toggled)
        self._sb_freq_lo.valueChanged  .connect(self._on_freq_filter_param)
        self._sb_freq_hi.valueChanged  .connect(self._on_freq_filter_param)

        # Trigger params write-through
        self._sb_mc  .valueChanged.connect(lambda _: self._write_trigger_params())
        self._sb_hold     .valueChanged.connect(lambda _: self._write_trigger_params())
        self._sb_pre .valueChanged.connect(lambda _: self._write_trigger_params())
        self._sb_post_trig.valueChanged.connect(lambda _: self._write_trigger_params())
        self._sb_maxr.valueChanged.connect(lambda _: self._write_trigger_params())
        self._combo_detect_mode.currentTextChanged.connect(lambda _: self._write_trigger_params())
        self._sb_entropy_thr.valueChanged.connect(lambda _: self._write_trigger_params())

        # Spectrogram display write-through
        self._sb_gain .valueChanged.connect(lambda _: self._write_spec_params())
        self._sb_floor.valueChanged.connect(lambda _: self._write_spec_params())
        self._sb_ceil .valueChanged.connect(lambda _: self._write_spec_params())
        self._combo_fft   .currentIndexChanged.connect(self._on_fft_params_changed)
        self._combo_win   .currentIndexChanged.connect(self._on_fft_params_changed)
        self._combo_fscale.currentTextChanged .connect(self._on_freq_scale_changed)
        self._sb_disp_freq_lo.valueChanged.connect(self._on_disp_freq_changed)
        self._sb_disp_freq_hi.valueChanged.connect(self._on_disp_freq_changed)
        self._chk_shared_spec.toggled.connect(self._on_shared_spec_toggled)
        self._combo_display_mode.currentTextChanged.connect(self._on_display_mode_changed)

        # #10 / c24: global settings scope
        self._chk_shared_trigger.toggled.connect(self._on_shared_trigger_toggled)
        self._btn_apply_all.clicked.connect(self._on_apply_all_settings)

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
        # #28 / #29: sticky session-flag resets.
        self._sidebar.clear_sat_requested.connect(self._on_clear_sat)
        self._sidebar.clear_drops_requested.connect(self._on_clear_drops)

    # ──────────────────────────────────────────────────────────────────────
    # Write-through: widgets → selected entity
    # ──────────────────────────────────────────────────────────────────────

    def _write_trigger_params(self):
        e = self._sel
        if not e:
            return
        e.min_cross_sec = self._sb_mc.value()
        e.hold_sec      = self._sb_hold.value()
        e.pre_trig_sec  = self._sb_pre.value()
        e.post_trig_sec = self._sb_post_trig.value()
        e.max_rec_sec   = self._sb_maxr.value()
        e.spectral_trigger_mode = self._combo_detect_mode.currentText()
        e.spectral_threshold    = self._sb_entropy_thr.value()
        # #10 / c24: propagate to all when sync is on.
        if self._chk_shared_trigger.isChecked():
            for ent in self._entities:
                if ent is not e:
                    ent.min_cross_sec = e.min_cross_sec
                    ent.hold_sec      = e.hold_sec
                    ent.post_trig_sec = e.post_trig_sec
                    ent.max_rec_sec   = e.max_rec_sec
                    ent.pre_trig_sec  = e.pre_trig_sec
                    ent.spectral_trigger_mode = e.spectral_trigger_mode
                    ent.spectral_threshold    = e.spectral_threshold
        self._mark_dirty()

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
        self._mark_dirty()

    def _on_freq_filter_toggled(self, on: bool):
        self._sb_freq_lo.setEnabled(on)
        self._sb_freq_hi.setEnabled(on)
        e = self._sel
        if e:
            e.freq_filter_enabled = on
            e.bpf.reset()
            e.bpf_r.reset()
            if self._chk_shared_trigger.isChecked():
                for ent in self._entities:
                    if ent is not e:
                        ent.freq_filter_enabled = on
                        ent.bpf.reset()
                        ent.bpf_r.reset()
            self._mark_dirty()

    def _on_freq_filter_param(self, _val):
        e = self._sel
        if e:
            e.freq_lo = self._sb_freq_lo.value()
            e.freq_hi = self._sb_freq_hi.value()
            if self._chk_shared_trigger.isChecked():
                for ent in self._entities:
                    if ent is not e:
                        ent.freq_lo = e.freq_lo
                        ent.freq_hi = e.freq_hi
            self._mark_dirty()

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
        e.pre_trig_sec  = self._sb_pre.value()
        e.post_trig_sec = self._sb_post_trig.value()
        e.max_rec_sec   = self._sb_maxr.value()
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
        # Device_id should only track live-device selections. The
        # WAV-sim sentinel leaves device_id alone (input_source /
        # wav_file_path are managed by _handle_wav_sim_selection).
        sel = self._device_combo.currentData()
        if sel != self.WAV_SIM_SENTINEL:
            e.device_id = sel
        e.spectral_trigger_mode = self._combo_detect_mode.currentText()
        e.spectral_threshold    = self._sb_entropy_thr.value()
        e.display_mode  = self._combo_display_mode.currentText()

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
        _set(self._sb_hold,      e.hold_sec)
        _set(self._sb_pre,  e.pre_trig_sec)
        _set(self._sb_post_trig, e.post_trig_sec)
        _set(self._sb_maxr, e.max_rec_sec)

        self._chk_freq.blockSignals(True)
        self._chk_freq.setChecked(e.freq_filter_enabled)
        self._chk_freq.blockSignals(False)
        self._sb_freq_lo.setEnabled(e.freq_filter_enabled)
        self._sb_freq_hi.setEnabled(e.freq_filter_enabled)
        _set(self._sb_freq_lo, e.freq_lo)
        _set(self._sb_freq_hi, e.freq_hi)

        _set(self._sb_gain,  e.gain_db)
        _set(self._sb_floor, e.db_floor)
        _set(self._sb_ceil,  e.db_ceil)

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

        # Device combo — WAV-sim sentinel when e uses a file, else
        # match by live device_id.
        self._device_combo.blockSignals(True)
        # Always refresh the sentinel's label with the entity's WAV path.
        self._device_combo.setItemText(0, self._wav_sim_label(e.wav_file_path))
        if e.input_source == 'wav_file':
            self._device_combo.setCurrentIndex(0)
        else:
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

        # Spectral trigger mode
        self._combo_detect_mode.blockSignals(True)
        self._combo_detect_mode.setCurrentText(e.spectral_trigger_mode)
        self._combo_detect_mode.blockSignals(False)
        _set(self._sb_entropy_thr, e.spectral_threshold)
        ent_on = (e.spectral_trigger_mode != 'Amplitude Only')
        self._sb_entropy_thr.setEnabled(ent_on)

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

        # Show/hide WAV transport row based on this entity's source.
        self._refresh_wav_controls()

        # Display mode combo
        self._combo_display_mode.blockSignals(True)
        self._combo_display_mode.setCurrentText(e.display_mode)
        self._combo_display_mode.blockSignals(False)

        # Rebuild axes if stereo layout, sample rate, display buffer, display mode, or entropy visibility differs
        want_stereo = (e.channel_mode == 'Stereo')
        axes_changed = (e.sample_rate, e.display_seconds) != self._t_axis_key
        display_mode_changed = (e.display_mode != getattr(self, '_display_mode', 'Spectrogram'))
        want_entropy = (e.spectral_trigger_mode != 'Amplitude Only')
        has_entropy = (self._ax_entropy is not None)
        if want_stereo != self._is_stereo_layout or axes_changed or display_mode_changed or want_entropy != has_entropy:
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
        # #7: wire this entity into the shared audio monitor so its
        # capture forwards samples whenever it becomes the selected
        # source (the monitor itself gates on source_id).
        e.set_monitor(self._monitor)
        self._entities.append(e)
        idx = self._sidebar.add_item(name)
        self._switch_selection(idx)
        self._refresh_monitor_source_combo()
        self._mark_dirty()

    def _remove_recording(self, idx: int):
        if len(self._entities) <= 1:
            return  # don't delete last
        if 0 <= idx < len(self._entities):
            e = self._entities.pop(idx)
            # #7: if this entity was the monitor source, disable first
            # so the monitor doesn't hold a stale token.
            if self._monitor.source_id == id(e):
                self._monitor.set_source(None)
            e.close()
            self._sidebar.remove_item(idx)
            # Re-select
            if self._selected_idx >= len(self._entities):
                self._selected_idx = len(self._entities) - 1
            elif self._selected_idx >= idx:
                self._selected_idx = max(0, self._selected_idx - 1)
            self._switch_selection(self._selected_idx)
            self._refresh_monitor_source_combo()
            self._mark_dirty()

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
            # Keep the monitor-source combo labels in sync with renames.
            self._refresh_monitor_source_combo()
            self._mark_dirty()

    # ── #28 / #29: sticky-flag reset handlers ────────────────────────────

    def _on_clear_sat(self, idx: int):
        """Clear the sticky saturation flag on the idx-th stream."""
        if 0 <= idx < len(self._entities):
            self._entities[idx].clear_saturation_flag()
            # Push a fresh update so the badge goes grey immediately
            # without waiting for the next plot tick.
            self._sidebar.update_item_saturation_sticky(idx, False)

    def _on_clear_drops(self, idx: int):
        """Clear the sticky drop stats on the idx-th stream."""
        if 0 <= idx < len(self._entities):
            self._entities[idx].clear_drop_flag()
            self._sidebar.update_item_drop_sticky(idx, False, 0)

    # ──────────────────────────────────────────────────────────────────────
    # Transport callbacks (operate on selected entity)
    # ──────────────────────────────────────────────────────────────────────

    def _on_start_acq(self):
        e = self._sel
        if e:
            for ent in self._entities:
                ent.reset_display()
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
            e.reset_display()
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
        self._sb_thr      .setValue(DEFAULT_THRESHOLD)
        self._sb_mc       .setValue(DEFAULT_MIN_CROSS)
        self._sb_hold     .setValue(DEFAULT_HOLD)
        self._sb_post_trig.setValue(DEFAULT_POST_TRIG)
        self._sb_maxr     .setValue(DEFAULT_MAX_REC)
        self._sb_pre      .setValue(DEFAULT_PRE_TRIG)
        self._chk_freq.setChecked(False)
        self._sb_freq_lo.setValue(DEFAULT_FREQ_LO)
        self._sb_freq_hi.setValue(DEFAULT_FREQ_HI)
        self._combo_detect_mode.setCurrentText('Amplitude Only')
        self._sb_entropy_thr.setValue(0.5)
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
        self._combo_display_mode.setCurrentText('Spectrogram')

    # ──────────────────────────────────────────────────────────────────────
    # Save / Load settings
    # ──────────────────────────────────────────────────────────────────────

    def _build_settings_data(self) -> dict:
        if self._selected_idx >= 0:
            self._flush_params_to_entity(self._selected_idx)
        from chirp.config import build_settings_dict
        return build_settings_dict(
            self._entities,
            view_mode={
                'columns':      self._vm_n_cols,
                'panel_height': self._vm_panel_height,
            },
        )

    def _write_settings_to_path(self, path: str, data: dict) -> bool:
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self._current_config_path = path
            self._mark_clean()  # #11 / c22
            return True
        except Exception as exc:
            QMessageBox.warning(self, 'Save Error', f'Could not save settings:\n{exc}')
            return False

    def _save_settings(self):
        """Save to current path if known, otherwise prompt."""
        if self._current_config_path:
            self._write_settings_to_path(self._current_config_path, self._build_settings_data())
        else:
            self._save_settings_as()

    def _save_settings_as(self):
        """Always prompt for a save path."""
        data = self._build_settings_data()
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Settings', '', 'Chirp Settings (*.json);;All Files (*)')
        if not path:
            return
        if not path.endswith('.json'):
            path += '.json'
        self._write_settings_to_path(path, data)

    def _load_settings(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Load Settings', '', 'Chirp Settings (*.json *.chirp);;All Files (*)')
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
        # #7: drop any monitor source before closing the entities it
        # might be pointing at.
        self._monitor.set_source(None)
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
        self._vm_height_spin.blockSignals(True)
        self._vm_height_spin.setValue(self._vm_panel_height)
        self._vm_height_spin.blockSignals(False)

        # Create entities from saved data
        warnings = []
        for rec_d in data['recordings']:
            ent, warn = RecordingEntity.from_dict(rec_d)
            # #7: re-wire the monitor on every freshly-loaded entity.
            ent.set_monitor(self._monitor)
            self._entities.append(ent)
            self._sidebar.add_item(ent.name)
            if warn:
                warnings.append(warn)
        # Rebuild monitor-source combo from the loaded entities.
        self._refresh_monitor_source_combo()

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

        self._current_config_path = path
        self._mark_clean()  # #11 / c22
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
            if self._chk_shared_trigger.isChecked():
                for ent in self._entities:
                    if ent is not e:
                        ent.threshold = val
            self._mark_dirty()
        self._sync_thr_line(val)

    def _sync_thr_line(self, val: float):
        self._threshold_line.set_ydata([val, val])
        self._threshold_label.set_y(val + 0.03)
        self._threshold_label.set_text(f'thr = {val:.3f}')
        self._canvas.draw_idle()

    def _sync_entropy_thr_line(self, val: float):
        if self._entropy_thr_line is not None:
            self._entropy_thr_line.set_ydata([val, val])
            self._entropy_thr_label.set_y(val + 0.03)
            self._entropy_thr_label.set_text(f'ent = {val:.3f}')
            self._canvas.draw_idle()

    def _set_thr_silent(self, val: float):
        self._sb_thr.blockSignals(True)
        self._sb_thr.setValue(val)
        self._sb_thr.blockSignals(False)

    # ──────────────────────────────────────────────────────────────────────
    # Auto-calibrate threshold
    # ──────────────────────────────────────────────────────────────────────

    def _on_calibrate(self):
        e = self._sel
        if not e:
            return
        if not e.acq_running:
            self._lbl_calib_status.setText('Start acquisition first')
            QTimer.singleShot(3000, lambda: self._lbl_calib_status.setText(''))
            return

        duration = self._sb_calib_dur.value()
        self._calib_samples = []
        self._calib_remaining = duration
        self._btn_calibrate.setEnabled(False)
        self._lbl_calib_status.setText(f'Calibrating... {duration:.1f}s')
        self._lbl_calib_status.setStyleSheet(f'color: {C["yellow"]};')

        self._calib_timer = QTimer()
        self._calib_timer.setInterval(100)  # check every 100ms
        self._calib_timer.timeout.connect(self._calib_tick)
        self._calib_start_time = datetime.datetime.now()
        self._calib_timer.start()

    def _calib_tick(self):
        e = self._sel
        if not e or not e.acq_running:
            self._calib_timer.stop()
            self._btn_calibrate.setEnabled(True)
            self._lbl_calib_status.setText('Acquisition stopped')
            self._lbl_calib_status.setStyleSheet(f'color: {C["red"]};')
            QTimer.singleShot(3000, lambda: (
                self._lbl_calib_status.setText(''),
                self._lbl_calib_status.setStyleSheet(''),
            ))
            return

        duration = self._sb_calib_dur.value()
        elapsed = (datetime.datetime.now() - self._calib_start_time).total_seconds()
        remaining = max(0.0, duration - elapsed)
        self._lbl_calib_status.setText(f'Calibrating... {remaining:.1f}s')

        # Collect current amplitude data from the buffer
        # We sample the most recent chunk's peak from abs_amp_buffer
        wh = e.write_head
        n = CHUNK_FRAMES
        if wh >= n:
            chunk_data = e.abs_amp_buffer[wh - n:wh]
        else:
            chunk_data = e.abs_amp_buffer[:max(1, wh)]
        if len(chunk_data) > 0:
            self._calib_samples.append(float(np.max(chunk_data)))

        if elapsed >= duration:
            self._calib_timer.stop()
            self._finish_calibrate()

    def _finish_calibrate(self):
        e = self._sel
        self._btn_calibrate.setEnabled(True)

        if not self._calib_samples:
            self._lbl_calib_status.setText('No data collected')
            self._lbl_calib_status.setStyleSheet(f'color: {C["red"]};')
            QTimer.singleShot(3000, lambda: (
                self._lbl_calib_status.setText(''),
                self._lbl_calib_status.setStyleSheet(''),
            ))
            return

        # Use the 95th percentile of collected peaks as the noise floor
        noise_floor = float(np.percentile(self._calib_samples, 95))
        margin = self._sb_calib_margin.value()
        new_threshold = min(1.0, noise_floor * margin)

        # Apply the new threshold
        if e:
            e.threshold = new_threshold
        self._set_thr_silent(new_threshold)
        self._sync_thr_line(new_threshold)

        self._lbl_calib_status.setText(
            f'Done: noise={noise_floor:.4f}, thr={new_threshold:.3f}')
        self._lbl_calib_status.setStyleSheet(f'color: {C["green"]};')
        QTimer.singleShot(5000, lambda: (
            self._lbl_calib_status.setText(''),
            self._lbl_calib_status.setStyleSheet(''),
        ))

    # ──────────────────────────────────────────────────────────────────────
    # Matplotlib mouse events
    # ──────────────────────────────────────────────────────────────────────

    def _on_mpl_press(self, event):
        if self._view_mode:
            return
        e = self._sel
        if not e or event.button != 1:
            return
        if event.inaxes is self._ax_amp:
            _, y_disp = self._ax_amp.transData.transform((0.0, e.threshold))
            if abs(event.y - y_disp) <= 12:
                self._dragging = True
                self._timer.stop()
        elif self._ax_entropy is not None and event.inaxes is self._ax_entropy:
            _, y_disp = self._ax_entropy.transData.transform((0.0, e.spectral_threshold))
            if abs(event.y - y_disp) <= 12:
                self._dragging_entropy = True
                self._timer.stop()

    def _on_mpl_motion(self, event):
        if self._dragging:
            if event.inaxes is not self._ax_amp:
                return
            val = float(np.clip(event.ydata, 0.0, 1.0))
            e = self._sel
            if e:
                e.threshold = val
            self._set_thr_silent(val)
            self._sync_thr_line(val)
        elif self._dragging_entropy:
            if self._ax_entropy is None or event.inaxes is not self._ax_entropy:
                return
            val = float(np.clip(event.ydata, 0.0, 1.0))
            e = self._sel
            if e:
                e.spectral_threshold = val
            self._sb_entropy_thr.blockSignals(True)
            self._sb_entropy_thr.setValue(val)
            self._sb_entropy_thr.blockSignals(False)
            self._sync_entropy_thr_line(val)

    def _on_mpl_release(self, event):
        if self._dragging or self._dragging_entropy:
            self._dragging = False
            self._dragging_entropy = False
            self._timer.start()

    @staticmethod
    def _zoom_axis_centered(ax, event_ydata, step, anchor_zero=False):
        """Zoom an axis centered on the mouse Y position. Returns (new_lo, new_hi).
        If anchor_zero, the lower limit is pinned at 0."""
        scale = 0.85 if step > 0 else 1.0 / 0.85
        ylo, yhi = ax.get_ylim()
        if anchor_zero:
            new_hi = max(0.01, yhi * scale)
            ax.set_ylim(0.0, new_hi)
            return 0.0, new_hi
        # Zoom centered on mouse position
        y = event_ydata if event_ydata is not None else (ylo + yhi) / 2
        new_lo = y - (y - ylo) * scale
        new_hi = y + (yhi - y) * scale
        if new_hi - new_lo < 0.01:
            new_hi = new_lo + 0.01
        ax.set_ylim(new_lo, new_hi)
        return new_lo, new_hi

    def _on_scroll(self, event):
        if self._view_mode:
            for i, vm in enumerate(self._vm_axes):
                if event.inaxes is vm['ax_amp'] or (vm.get('ax_wave') and event.inaxes is vm['ax_wave']):
                    _, new_ymax = self._zoom_axis_centered(vm['ax_amp'], event.ydata, event.step, anchor_zero=True)
                    if vm.get('ax_wave') is not None:
                        vm['ax_wave'].set_ylim(-new_ymax, new_ymax)
                    if i < len(self._entities):
                        self._entities[i].amp_ylim = new_ymax
                    self._recapture_bg()
                    return
                if vm.get('ax_entropy') is not None and event.inaxes is vm['ax_entropy']:
                    self._zoom_axis_centered(vm['ax_entropy'], event.ydata, event.step)
                    self._recapture_bg()
                    return
            return

        # Config mode — entropy axis (centered zoom)
        if self._ax_entropy is not None and event.inaxes is self._ax_entropy:
            self._zoom_axis_centered(self._ax_entropy, event.ydata, event.step)
            self._recapture_bg()
            return

        # Config mode — waveform axes (symmetric around zero)
        wave_axes = [ax for ax in [getattr(self, '_ax_wave', None),
                                    getattr(self, '_ax_wave_r', None)] if ax is not None]
        if event.inaxes in wave_axes:
            e = self._sel
            if e:
                scale = 0.85 if event.step > 0 else 1.0 / 0.85
                new_ylim = max(0.01, e.amp_ylim * scale)
                e.amp_ylim = new_ylim
                for wax in wave_axes:
                    wax.set_ylim(-new_ylim, new_ylim)
                self._ax_amp.set_ylim(0.0, new_ylim)
                self._recapture_bg()
            return

        # Config mode — amplitude axis (anchored at zero)
        if event.inaxes is not self._ax_amp:
            return
        _, new_ymax = self._zoom_axis_centered(self._ax_amp, event.ydata, event.step, anchor_zero=True)
        e = self._sel
        if e:
            e.amp_ylim = new_ymax
            for wax in wave_axes:
                wax.set_ylim(-new_ymax, new_ymax)
        self._recapture_bg()

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

    def _on_shared_trigger_toggled(self, on: bool):
        """When shared-trigger is turned on, push current entity's trigger
        params to all others immediately (#10 / c24)."""
        if on:
            e = self._sel
            if not e:
                return
            for ent in self._entities:
                if ent is not e:
                    ent.threshold     = e.threshold
                    ent.min_cross_sec = e.min_cross_sec
                    ent.hold_sec      = e.hold_sec
                    ent.post_trig_sec = e.post_trig_sec
                    ent.max_rec_sec   = e.max_rec_sec
                    ent.pre_trig_sec  = e.pre_trig_sec
                    ent.freq_filter_enabled = e.freq_filter_enabled
                    ent.freq_lo       = e.freq_lo
                    ent.freq_hi       = e.freq_hi
                    ent.spectral_trigger_mode = e.spectral_trigger_mode
                    ent.spectral_threshold    = e.spectral_threshold
            self._mark_dirty()

    def _on_apply_all_settings(self):
        """One-shot: copy ALL user-configurable settings from the selected
        entity to every other entity (#10 / c24)."""
        e = self._sel
        if not e:
            return
        self._flush_params_to_entity(self._selected_idx)
        for ent in self._entities:
            if ent is not e:
                # Trigger params
                ent.threshold     = e.threshold
                ent.min_cross_sec = e.min_cross_sec
                ent.hold_sec      = e.hold_sec
                ent.post_trig_sec = e.post_trig_sec
                ent.max_rec_sec   = e.max_rec_sec
                ent.pre_trig_sec  = e.pre_trig_sec
                ent.freq_filter_enabled = e.freq_filter_enabled
                ent.freq_lo       = e.freq_lo
                ent.freq_hi       = e.freq_hi
                ent.spectral_trigger_mode = e.spectral_trigger_mode
                ent.spectral_threshold    = e.spectral_threshold
                # Display params
                ent.gain_db       = e.gain_db
                ent.db_floor      = e.db_floor
                ent.db_ceil       = e.db_ceil
                ent.freq_scale    = e.freq_scale
                ent.display_freq_lo = e.display_freq_lo
                ent.display_freq_hi = e.display_freq_hi
                ent.rebuild_freq_mapping()
                ent.change_fft_params(e.spec_nperseg, e.spec_window)
                ent.change_analysis_fft_params(e.analysis_nperseg, e.analysis_window)
                ent.display_mode  = e.display_mode
                # Output params
                ent.output_dir       = e.output_dir
                ent.filename_prefix  = e.filename_prefix
                ent.filename_suffix  = e.filename_suffix
                ent.dph_folder_prefix = e.dph_folder_prefix
                ent.ref_date         = e.ref_date
        self._mark_dirty()
        # Brief flash to confirm
        self._btn_apply_all.setText('\u2713 Applied!')
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(1500, lambda: self._btn_apply_all.setText('Apply All Settings \u2192'))

    def _on_display_mode_changed(self, mode: str):
        e = self._sel
        if not e:
            return
        e.display_mode = mode
        stereo = e.channel_mode == 'Stereo'
        self._setup_axes(stereo=stereo)
        self._update_spec_yticks(e)
        self._mark_dirty()

    def _on_detect_mode_changed(self, mode: str):
        self._sb_entropy_thr.setEnabled(mode != 'Amplitude Only')
        self._write_trigger_params()
        e = self._sel
        if e:
            stereo = e.channel_mode == 'Stereo'
            self._setup_axes(stereo=stereo)
            self._update_spec_yticks(e)

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
            self._mark_dirty()

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
            self._mark_dirty()

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
            self._mark_dirty()

    def _on_prefix_changed(self):
        e = self._sel
        if e:
            e.filename_prefix = self._prefix_edit.text()
            self._mark_dirty()

    def _on_suffix_changed(self):
        e = self._sel
        if e:
            e.filename_suffix = self._suffix_edit.text()
            self._mark_dirty()

    def _on_ref_date_toggled(self, on: bool):
        self._date_line.setEnabled(on)
        self._btn_pick_date.setEnabled(on)
        self._dph_prefix_edit.setEnabled(on)
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
            if e:
                e.ref_date = d
                if self._chk_ref_date.isChecked():
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

    # Sentinel userData for the WAV-file-simulation virtual device entry.
    WAV_SIM_SENTINEL = '__wav_sim__'

    def _wav_sim_label(self, path: str | None) -> str:
        if path:
            return f'\u25B6 WAV sim: {os.path.basename(path)}'
        return '\u25B6 <WAV file simulation...>'

    def _populate_device_combo(self, keep_current: bool = False):
        prev_name = self._device_combo.currentText() if keep_current else None
        prev_data = self._device_combo.currentData() if keep_current else None
        self._device_combo.blockSignals(True)
        self._device_combo.clear()

        # Virtual entry at the top for WAV-file simulation. Its label
        # reflects the currently selected entity's WAV path (if any) so
        # the user can see what file will play.
        e = self._sel if hasattr(self, '_sel') else None
        wav_path = e.wav_file_path if (e is not None) else None
        self._device_combo.addItem(self._wav_sim_label(wav_path),
                                   userData=self.WAV_SIM_SENTINEL)

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
        if keep_current and prev_data == self.WAV_SIM_SENTINEL:
            restore_idx = 0
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

        # WAV-file simulation sentinel — prompt for a file and route
        # input through WavFileCapture instead of a live device.
        if device_id == self.WAV_SIM_SENTINEL:
            self._handle_wav_sim_selection(e)
            self._refresh_transport_ui()
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
        self._refresh_wav_controls()

    # ── WAV simulation transport ─────────────────────────────────────

    @staticmethod
    def _format_mmss(seconds: float) -> str:
        if seconds < 0 or seconds != seconds:  # NaN guard
            seconds = 0.0
        total = int(seconds)
        return f'{total // 60}:{total % 60:02d}'

    def _refresh_wav_controls(self):
        """Show/hide the WAV transport row based on the selected
        entity's input source, and sync the loop checkbox.
        """
        from chirp.audio import WavFileCapture
        e = self._sel
        show = bool(e) and e.input_source == 'wav_file' \
               and isinstance(e.capture, WavFileCapture)
        for w in self._wav_ctrl_widgets:
            w.setVisible(show)
        if show:
            self._chk_wav_loop.blockSignals(True)
            self._chk_wav_loop.setChecked(bool(e.wav_loop))
            self._chk_wav_loop.blockSignals(False)
            self._update_wav_time_label()
        else:
            self._lbl_wav_time.setText('0:00 / 0:00')

    def _update_wav_time_label(self):
        """Refresh the ``passed / total`` label from the live capture."""
        from chirp.audio import WavFileCapture
        e = self._sel
        if not e or not isinstance(e.capture, WavFileCapture):
            return
        cap = e.capture
        passed = self._format_mmss(cap.position_sec)
        total  = self._format_mmss(cap.duration_sec)
        self._lbl_wav_time.setText(f'{passed} / {total}')

    def _on_wav_reset_clicked(self):
        from chirp.audio import WavFileCapture
        e = self._sel
        if not e or not isinstance(e.capture, WavFileCapture):
            return
        e.capture.reset_position()
        # Clear the display ring so the rewind isn't visually confusing.
        e.reset_display()
        self._update_wav_time_label()

    def _on_wav_loop_toggled(self, checked: bool):
        from chirp.audio import WavFileCapture
        e = self._sel
        if not e:
            return
        e.wav_loop = bool(checked)
        if isinstance(e.capture, WavFileCapture):
            e.capture.set_loop(checked)

    def _handle_wav_sim_selection(self, e: RecordingEntity):
        """Prompt for a WAV file and switch ``e`` to WAV-simulation mode.

        If the user cancels the dialog, revert the combo to whatever
        live device was previously active. On success, display the
        chosen filename on the combo entry.
        """
        start_dir = os.path.dirname(e.wav_file_path) if e.wav_file_path else ''
        path, _ = QFileDialog.getOpenFileName(
            self, 'Pick a WAV file to simulate input',
            start_dir, 'WAV files (*.wav)')
        if not path:
            # User cancelled — revert selection to current device/source.
            self._device_combo.blockSignals(True)
            target = e.device_id if e.input_source == 'device' else None
            for i in range(self._device_combo.count()):
                if self._device_combo.itemData(i) == target:
                    self._device_combo.setCurrentIndex(i)
                    break
            self._device_combo.blockSignals(False)
            return

        ok, warning = e.use_wav_file(path, loop=e.wav_loop)
        if not ok:
            QMessageBox.warning(self, 'WAV File Error',
                                f'Could not open WAV file:\n{path}')
            return

        # Update the sentinel label to show the chosen filename.
        self._device_combo.blockSignals(True)
        self._device_combo.setItemText(0, self._wav_sim_label(path))
        self._device_combo.setCurrentIndex(0)
        self._device_combo.blockSignals(False)

        # If the session SR changed to match the file, sync the combo.
        self._sr_combo.blockSignals(True)
        self._sr_combo.setCurrentText(f'{e.sample_rate} Hz')
        self._sr_combo.blockSignals(False)

        self._refresh_wav_controls()

        if warning:
            QMessageBox.information(self, 'WAV File Simulation', warning)

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
        is_wav_sim = (device_id == self.WAV_SIM_SENTINEL)
        if device_id is not None and not is_wav_sim:
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

        if is_wav_sim:
            # Re-open the WAV with the new channel count.
            if e.wav_file_path:
                e.use_wav_file(e.wav_file_path, loop=e.wav_loop)
        else:
            e.change_device(device_id, need_ch)
        if not want_stereo:
            e.amp_buffer_r[:] = 0.0
            e.spec_buffer_r[:] = SPEC_DB_MIN
        e.bpf.reset()
        e.bpf_r.reset()
        # #7: if this entity is the monitor source, resync so the output
        # stream has the right channel count.
        if self._monitor.source_id == id(e):
            self._apply_monitor_source(id(e))

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
        # #7: if this entity is the monitor source, the output stream
        # needs to be reopened at the new SR so playback isn't pitched.
        if self._monitor.source_id == id(e):
            self._apply_monitor_source(id(e))
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

        # Outer grid: rows × cols, each cell has inner subplots
        outer_gs = self._fig.add_gridspec(
            rows, cols, hspace=0.35, wspace=0.15,
            top=0.97, bottom=0.03, left=0.05, right=0.99)

        n_disp = N_DISPLAY_ROWS

        for i, e in enumerate(self._entities):
            r, c = divmod(i, cols)
            dmode = e.display_mode
            show_spec = dmode in ('Spectrogram', 'Both')
            show_wave = dmode in ('Waveform', 'Both')
            show_entropy = (e.spectral_trigger_mode != 'Amplitude Only')

            # Build inner subplot rows
            inner_rows = []
            inner_ratios = []
            if show_spec:
                inner_rows.append('spec')
                inner_ratios.append(3)
            if show_wave:
                inner_rows.append('wave')
                inner_ratios.append(1 if show_spec else 3)
            inner_rows.append('amp')
            inner_ratios.append(1)
            # #32: thin detect/record events strip under the amp axis,
            # mirrors the config-mode layout.
            inner_rows.append('events')
            inner_ratios.append(0.3)
            if show_entropy:
                inner_rows.append('entropy')
                inner_ratios.append(1)

            inner = outer_gs[r, c].subgridspec(
                len(inner_rows), 1, height_ratios=inner_ratios, hspace=0.08)
            inner_idx = {key: idx for idx, key in enumerate(inner_rows)}

            # Create axes based on display mode
            ax_spec = None
            ax_wave = None
            spec_im = None
            cursor_spec = None
            wave_line = None
            wave_line_r = None
            cursor_wave = None
            ax_entropy = None
            entropy_line = None
            entropy_thr_line = None
            cursor_entropy = None
            ax_events = None
            events_im = None
            cursor_events = None

            e_disp = e.display_seconds
            e_ts = e._total_samples
            e_t_axis = np.linspace(0.0, e_disp, e_ts, endpoint=False, dtype=np.float32)

            # First axis (anchor for sharex)
            if show_spec:
                ax_spec = self._fig.add_subplot(inner[inner_idx['spec']])
                first_ax = ax_spec
            elif show_wave:
                ax_wave = self._fig.add_subplot(inner[inner_idx['wave']])
                first_ax = ax_wave

            if show_spec:
                db_floor = min(e.db_floor, e.db_ceil - 0.1)
                dummy = np.full((n_disp, e._n_cols), db_floor, dtype=np.float32)
                spec_im = ax_spec.imshow(
                    dummy, aspect='auto', origin='lower',
                    extent=[0.0, e_disp, 0, n_disp],
                    vmin=db_floor, vmax=e.db_ceil,
                    cmap=COLORMAP, interpolation='nearest',
                )
                cursor_spec = ax_spec.axvline(x=0.0, color=C['green'], linewidth=1.0, alpha=0.7)
                self._apply_spec_yticks(ax_spec, e)
                ax_spec.tick_params(labelbottom=False)
                ax_spec.set_ylabel('Freq', fontsize=7)

            if show_wave:
                if ax_wave is None:
                    ax_wave = self._fig.add_subplot(inner[inner_idx['wave']], sharex=first_ax)
                amp_ylim = e.amp_ylim
                (wave_line,) = ax_wave.plot(
                    e_t_axis, np.zeros(e_ts),
                    color=C['teal'], linewidth=0.6, antialiased=False,
                    label='L' if e.channel_mode == 'Stereo' else None,
                )
                wave_line_r = None
                if e.channel_mode == 'Stereo':
                    (wave_line_r,) = ax_wave.plot(
                        e_t_axis, np.zeros(e_ts),
                        color=C['pink'], linewidth=0.6, antialiased=False, label='R',
                    )
                    ax_wave.legend(loc='upper right', fontsize=7,
                                   facecolor=C['mantle'], edgecolor=C['surface1'],
                                   labelcolor=C['text'])
                ax_wave.set_xlim(0.0, e_disp)
                ax_wave.set_ylim(-amp_ylim, amp_ylim)
                cursor_wave = ax_wave.axvline(x=0.0, color=C['green'], linewidth=1.0, alpha=0.7)
                ax_wave.tick_params(labelbottom=False)
                ax_wave.set_ylabel('Wave', fontsize=7)

            ax_amp = self._fig.add_subplot(inner[inner_idx['amp']], sharex=first_ax)

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

            # The amp axis is no longer the bottom of this cell (events
            # strip and/or entropy axis come below). Hide its x tick
            # labels; whichever axis ends up last will carry the label.
            ax_amp.tick_params(labelbottom=False)
            ax_amp.set_ylabel('Amp', fontsize=7)

            # #32: detect/record events strip (row 0 = det / yellow,
            # row 1 = rec / green). Same construction as config mode.
            ax_events = self._fig.add_subplot(
                inner[inner_idx['events']], sharex=first_ax)
            rgba0 = np.empty((2, max(1, e._n_cols), 4), dtype=np.float32)
            rgba0[..., 0] = 0x18 / 255.0
            rgba0[..., 1] = 0x18 / 255.0
            rgba0[..., 2] = 0x25 / 255.0
            rgba0[..., 3] = 1.0
            events_im = ax_events.imshow(
                rgba0, aspect='auto', origin='upper',
                extent=[0.0, e_disp, 0, 2],
                interpolation='nearest',
            )
            ax_events.set_xlim(0.0, e_disp)
            ax_events.set_ylim(0, 2)
            ax_events.set_yticks([0.5, 1.5])
            ax_events.set_yticklabels(['rec', 'det'], fontsize=6)
            ax_events.tick_params(axis='y', length=0, pad=2)
            cursor_events = ax_events.axvline(
                x=0.0, color=C['green'], linewidth=1.0, alpha=0.7)
            # Events is either the bottom axis in this cell, or entropy
            # is below it. Default: hide its x ticks — entropy (if
            # present) or the fallback below handles the label.
            ax_events.tick_params(labelbottom=False)

            # Entropy subplot
            if show_entropy:
                ax_entropy = self._fig.add_subplot(inner[inner_idx['entropy']], sharex=first_ax)
                ent_t = np.linspace(0.0, e_disp, e._n_cols, endpoint=False, dtype=np.float32)
                (entropy_line,) = ax_entropy.plot(
                    ent_t, np.ones(e._n_cols),
                    color=C['mauve'], linewidth=0.6, antialiased=False,
                )
                ax_entropy.set_xlim(0.0, e_disp)
                ax_entropy.set_ylim(0.0, 1.05)
                entropy_thr_line = ax_entropy.axhline(
                    y=e.spectral_threshold, color=C['peach'], linewidth=1.0,
                    linestyle=(0, (6, 3)),
                )
                cursor_entropy = ax_entropy.axvline(
                    x=0.0, color=C['green'], linewidth=1.0, alpha=0.7)
                ax_entropy.set_ylabel('Ent', fontsize=7)
                ax_entropy.tick_params(labelbottom=False)
                # Entropy is the bottom axis when present — it carries
                # the x-label on the bottom row of cells.
                if r >= rows - 1:
                    ax_entropy.tick_params(labelbottom=True)
                    ax_entropy.set_xlabel('Time (s)', fontsize=7)
            elif r >= rows - 1:
                # No entropy: events strip is the bottom axis. Give it
                # the x-label on the bottom row of cells.
                ax_events.tick_params(labelbottom=True)
                ax_events.set_xlabel('Time (s)', fontsize=7)

            # Title and status on the topmost axis
            top_ax = ax_spec if ax_spec is not None else ax_wave
            title_obj = top_ax.set_title(e.name, loc='left', fontsize=9,
                                         color=C['text'], fontweight='bold', pad=3)
            status_text = top_ax.text(
                0.99, 1.02, '', transform=top_ax.transAxes, fontsize=8,
                ha='right', va='bottom', fontfamily='Consolas')
            # #28 / #29: sticky saturation + drop indicators, mirroring
            # the sidebar 'S' / 'D' badges so view-mode (monitoring
            # grid) surfaces the same session-wide flags. Placed at the
            # top-right corner INSIDE the top axis with a backing box so
            # they're readable over the spectrogram without crowding
            # the title / status row above the axes. Display-only here
            # — user flips to config mode to clear.
            _badge_bbox = dict(facecolor=C['mantle'], edgecolor=C['red'],
                               linewidth=0.6, boxstyle='round,pad=0.25',
                               alpha=0.9)
            sat_text = top_ax.text(
                0.985, 0.97, 'SAT', transform=top_ax.transAxes, fontsize=8,
                ha='right', va='top', fontfamily='Consolas',
                color=C['red'], fontweight='bold', bbox=_badge_bbox,
                visible=False)
            drop_text = top_ax.text(
                0.985, 0.82, '', transform=top_ax.transAxes, fontsize=8,
                ha='right', va='top', fontfamily='Consolas',
                color=C['red'], fontweight='bold', bbox=_badge_bbox,
                visible=False)

            self._vm_axes.append({
                'ax_spec': ax_spec, 'ax_amp': ax_amp, 'ax_wave': ax_wave,
                'ax_entropy': ax_entropy, 'ax_events': ax_events,
                'spec_im': spec_im,
                'amp_line': amp_line, 'amp_line_r': amp_line_r,
                'wave_line': wave_line, 'wave_line_r': wave_line_r,
                'entropy_line': entropy_line,
                'events_im': events_im,
                'cursor_spec': cursor_spec, 'cursor_amp': cursor_amp,
                'cursor_wave': cursor_wave, 'cursor_entropy': cursor_entropy,
                'cursor_events': cursor_events,
                'thr_line': thr_line, 'entropy_thr_line': entropy_thr_line,
                'title': title_obj, 'status_text': status_text,
                'sat_text': sat_text, 'drop_text': drop_text,
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
        if self._view_mode:
            self._setup_view_mode_axes()

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

            amp_color_l = '#ff5555' if e.saturated else C['blue']
            amp_color_r = '#ff5555' if e.saturated else C['pink']

            # Spectrogram
            if vm.get('spec_im') is not None:
                vm['spec_im'].set_data(e.resample_spec(e.spec_buffer))
                clim_lo = min(e.db_floor, e.db_ceil - 0.1)
                vm['spec_im'].set_clim(clim_lo, e.db_ceil)
                vm['cursor_spec'].set_xdata([cursor_x, cursor_x])

            # Waveform
            if vm.get('wave_line') is not None:
                wave_color_l = '#ff5555' if e.saturated else C['teal']
                vm['wave_line'].set_color(wave_color_l)
                vm['wave_line'].set_ydata(e.amp_buffer)
                vm['cursor_wave'].set_xdata([cursor_x, cursor_x])
            if vm.get('wave_line_r') is not None and e.channel_mode == 'Stereo':
                wave_color_r = '#ff5555' if e.saturated else C['pink']
                vm['wave_line_r'].set_color(wave_color_r)
                vm['wave_line_r'].set_ydata(e.amp_buffer_r)

            # Amplitude envelope
            vm['amp_line'].set_color(amp_color_l)
            vm['amp_line'].set_ydata(e.abs_amp_buffer)
            if vm['amp_line_r'] is not None and e.channel_mode == 'Stereo':
                vm['amp_line_r'].set_color(amp_color_r)
                vm['amp_line_r'].set_ydata(e.abs_amp_buffer_r)

            vm['cursor_amp'].set_xdata([cursor_x, cursor_x])
            vm['thr_line'].set_ydata([e.threshold, e.threshold])

            # Entropy
            if vm.get('entropy_line') is not None:
                col_cursor_x = (e.col_head / e._n_cols) * e.display_seconds
                vm['entropy_line'].set_ydata(e.entropy_buffer)
                vm['cursor_entropy'].set_xdata([col_cursor_x, col_cursor_x])
                vm['entropy_thr_line'].set_ydata([e.spectral_threshold, e.spectral_threshold])

            # #32: detect / record events strip
            if vm.get('events_im') is not None:
                rgba = self._build_events_rgba(e)
                if rgba is not None:
                    vm['events_im'].set_data(rgba)
                vm['cursor_events'].set_xdata([cursor_x, cursor_x])

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

            # #28 / #29: sticky saturation / drop indicators. Mirror
            # the sidebar 'S' / 'D' badges so view mode (which hides
            # the sidebar) still surfaces session-wide flags. Toggle
            # visibility rather than blanking the text — an empty
            # string with a bbox still renders an empty pill outline.
            if vm.get('sat_text') is not None:
                vm['sat_text'].set_visible(
                    bool(getattr(e, 'saturated_ever', False)))
            if vm.get('drop_text') is not None:
                cap = getattr(e, 'capture', None)
                has_ever = bool(getattr(cap, 'has_ever_dropped', False))
                if has_ever:
                    total = int(getattr(cap, 'drop_count_total', 0))
                    vm['drop_text'].set_text(f'DROP×{total}')
                    vm['drop_text'].set_visible(True)
                else:
                    vm['drop_text'].set_visible(False)

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
        # 1. Ingestion now happens on per-entity background threads
        # (#19 / c21). The main thread only reads the ring buffers and
        # updates the display. Drop / saturation badges still need
        # polling here.
        for idx, e in enumerate(self._entities):
            if hasattr(e.capture, 'consume_drop_count'):
                n_drops = e.capture.consume_drop_count()
                if hasattr(self, '_sidebar'):
                    try:
                        self._sidebar.update_item_drops(idx, n_drops)
                    except Exception:
                        pass
            if hasattr(self, '_sidebar'):
                # #28: sticky saturation flag.
                try:
                    self._sidebar.update_item_saturation_sticky(
                        idx, bool(getattr(e, 'saturated_ever', False)))
                except Exception:
                    pass
                # #29: sticky persistent-drops flag.
                try:
                    has_ever = bool(getattr(e.capture, 'has_ever_dropped', False))
                    total    = int(getattr(e.capture, 'drop_count_total', 0))
                    self._sidebar.update_item_drop_sticky(idx, has_ever, total)
                except Exception:
                    pass

        # 2. Branch on mode
        if self._view_mode:
            self._update_plot_view_mode()
            return

        # Refresh the WAV transport time label for the selected entity
        # on every plot tick (50 ms) so "passed / total" stays live.
        self._update_wav_time_label()

        # 3. Update main display for selected entity (blitting)
        e = self._sel
        if e and self._bg is not None:
            cursor_x = (e.write_head / e.sample_rate) % e.display_seconds
            amp_color_l = '#ff5555' if e.saturated else C['blue']
            amp_color_r = '#ff5555' if e.saturated else C['pink']

            # Spectrogram
            if self._spec_im is not None:
                self._spec_im.set_data(e.resample_spec(e.spec_buffer))
                clim_lo = min(e.db_floor, e.db_ceil - 0.1)
                self._spec_im.set_clim(clim_lo, e.db_ceil)
                self._cursor_spec.set_xdata([cursor_x, cursor_x])
            if self._is_stereo_layout and self._spec_im_r is not None:
                self._spec_im_r.set_data(e.resample_spec(e.spec_buffer_r))
                clim_lo = min(e.db_floor, e.db_ceil - 0.1)
                self._spec_im_r.set_clim(clim_lo, e.db_ceil)
                self._cursor_spec_r.set_xdata([cursor_x, cursor_x])

            # Waveform
            if self._wave_line is not None:
                wave_color_l = '#ff5555' if e.saturated else C['teal']
                self._wave_line.set_color(wave_color_l)
                self._wave_line.set_ydata(e.amp_buffer)
                self._cursor_wave.set_xdata([cursor_x, cursor_x])
            if self._wave_line_r is not None:
                wave_color_r = '#ff5555' if e.saturated else C['pink']
                self._wave_line_r.set_color(wave_color_r)
                self._wave_line_r.set_ydata(e.amp_buffer_r)
                if self._cursor_wave_r is not None:
                    self._cursor_wave_r.set_xdata([cursor_x, cursor_x])

            # Amplitude envelope
            self._amp_line.set_color(amp_color_l)
            self._amp_line.set_ydata(e.abs_amp_buffer)
            if self._is_stereo_layout and self._amp_line_r is not None:
                self._amp_line_r.set_color(amp_color_r)
                self._amp_line_r.set_ydata(e.abs_amp_buffer_r)
            self._cursor_amp .set_xdata([cursor_x, cursor_x])

            # Entropy trace
            if self._entropy_line is not None:
                col_cursor_x = (e.col_head / e._n_cols) * e.display_seconds
                self._entropy_line.set_ydata(e.entropy_buffer)
                self._cursor_entropy.set_xdata([col_cursor_x, col_cursor_x])
                self._entropy_thr_line.set_ydata([e.spectral_threshold, e.spectral_threshold])
                self._entropy_thr_label.set_y(e.spectral_threshold + 0.03)
                self._entropy_thr_label.set_text(f'ent = {e.spectral_threshold:.3f}')

            # #32: detect / record events strip
            if self._events_im is not None:
                rgba = self._build_events_rgba(e)
                if rgba is not None:
                    self._events_im.set_data(rgba)
                self._cursor_events.set_xdata([cursor_x, cursor_x])

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

        # 7. Entropy display
        if e and e.acq_running:
            ent_val = e.spectral_entropy
            below = ent_val < e.spectral_threshold
            if e.spectral_trigger_mode != 'Amplitude Only' and below:
                self._lbl_entropy.setText(f'ENT  {ent_val:.3f} \u25bc')
                self._lbl_entropy.setStyleSheet(f'color: {C["green"]}; font-size: 10pt;')
            else:
                self._lbl_entropy.setText(f'ENT  {ent_val:.3f}')
                self._lbl_entropy.setStyleSheet(f'color: {C["subtext"]}; font-size: 10pt;')
        else:
            self._lbl_entropy.setText('ENT  \u2014')
            self._lbl_entropy.setStyleSheet(f'color: {C["subtext"]}; font-size: 10pt;')

    # ──────────────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        active = any(e.acq_running or e.rec_enabled for e in self._entities)
        if active:
            reply = QMessageBox.warning(
                self, 'Chirp',
                'Acquisition or recording is still running.\n'
                'Are you sure you want to quit?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                event.ignore()
                return
        self._timer.stop()

        # #19 / c21: stop all ingestion threads before flushing, so no
        # new chunks are processed while we're draining pending events.
        for e in self._entities:
            e.stop_acq()

        # #17 / c16: flush any in-flight trigger events to the writer
        # pool, then drain the pool so non-daemon worker threads finish
        # writing before the interpreter exits. Without this, daemon
        # threads from the old launcher would be killed mid-write and
        # the most recent WAV would be left truncated on disk.
        from chirp.recording import writer as _writer
        for e in self._entities:
            try:
                e.recorder.flush_all(
                    output_dir=e.output_dir,
                    filename_prefix=e.filename_prefix,
                    filename_suffix=e.filename_suffix,
                    sample_rate=e.sample_rate,
                    filename_stream=e.name,
                    reason='app shutdown',
                )
            except Exception as exc:
                print(f'[Chirp] flush_all failed for {e.name}: {exc}')

        pending = _writer.pending()
        if pending:
            # Show a non-cancellable modal so the user knows we're
            # waiting on disk I/O and not just frozen.
            try:
                msg = QMessageBox(self)
                msg.setIcon(QMessageBox.Information)
                msg.setWindowTitle('Chirp')
                msg.setText(f'Finishing {pending} pending recording(s)…')
                msg.setStandardButtons(QMessageBox.NoButton)
                msg.show()
                QApplication.processEvents()
                _writer.drain(timeout=30.0)
                msg.close()
            except Exception:
                _writer.drain(timeout=30.0)
        _writer.shutdown(timeout=5.0)

        # #7: close the monitor loopback before closing entities — the
        # output stream's callback could otherwise read a buffer that
        # feeders are tearing down.
        try:
            self._monitor.close()
        except Exception:
            pass

        for e in self._entities:
            e.close()
        super().closeEvent(event)


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)
    win = ChirpWindow()
    win.showMaximized()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
