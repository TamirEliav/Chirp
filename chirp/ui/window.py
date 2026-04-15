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
    QGridLayout, QGroupBox, QPushButton, QSlider, QLabel, QLineEdit,
    QFileDialog, QFrame, QSizePolicy, QDoubleSpinBox, QComboBox, QCheckBox,
    QScrollArea, QStackedLayout, QDialog, QCalendarWidget, QMessageBox, QSpinBox,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QSize, QDate, QPointF
from PyQt5.QtGui import QFont, QPainter, QColor, QPainterPath, QPen, QPolygonF

# Re-exports used throughout the window code. The star-import brings in
# all the module-level constants and the palette (C, QSS) so the class
# body below keeps referring to them by bare name.
from chirp.constants import *  # noqa: F401,F403
from chirp.audio import AudioCapture  # noqa: F401
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
        self._dragging_entropy = False
        self._current_config_path: str | None = None

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

        self.setWindowTitle(f'Chirp v{__version__} — Triggered Sound Recording')
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

        # Row 1: transport
        hl = self._hline()
        transport = self._build_transport()
        vbox.addWidget(hl)
        vbox.addWidget(transport)
        self._config_widgets.extend([hl, transport])

        # Row 2: trigger params + display side by side
        hl2 = self._hline()
        params_panel = self._build_params()
        spec_panel = self._build_spec_params()
        combined = QWidget()
        combined.setStyleSheet(f'background-color: {C["mantle"]};')
        combined_h = QHBoxLayout(combined)
        combined_h.setContentsMargins(0, 0, 0, 0)
        combined_h.setSpacing(0)
        combined_h.addWidget(params_panel, stretch=1)
        combined_h.addWidget(spec_panel, stretch=1)
        vbox.addWidget(hl2)
        vbox.addWidget(combined)
        self._config_widgets.extend([hl2, combined])

        # Row 3: settings (output folder, filename, device)
        hl3 = self._hline()
        settings = self._build_settings()
        vbox.addWidget(hl3)
        vbox.addWidget(settings)
        self._config_widgets.extend([hl3, settings])

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
        self._vm_height_sl = QSlider(Qt.Horizontal)
        self._vm_height_sl.setToolTip('Row height for each recording tile in View mode')
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
        self._btn_start_acq.setToolTip('Start audio acquisition (live monitoring) for the selected recording')
        self._btn_stop_acq .setToolTip('Stop audio acquisition for the selected recording')
        acq_h.addWidget(self._btn_start_acq)
        acq_h.addWidget(self._btn_stop_acq)

        rec_box = QGroupBox('RECORDING')
        rec_h   = QHBoxLayout(rec_box)
        rec_h.setSpacing(10)
        self._btn_start_rec = QPushButton('Start Rec')
        self._btn_stop_rec  = QPushButton('Stop Rec')
        self._btn_start_rec.setObjectName('btn_start_rec')
        self._btn_stop_rec .setObjectName('btn_stop_rec')
        self._btn_start_rec.setToolTip('Enable threshold-triggered WAV recording for the selected recording')
        self._btn_stop_rec .setToolTip('Disable threshold-triggered WAV recording for the selected recording')
        rec_h.addWidget(self._btn_start_rec)
        rec_h.addWidget(self._btn_stop_rec)

        status_box = QGroupBox('STATUS')
        status_v   = QVBoxLayout(status_box)
        status_v.setSpacing(4)
        self._lbl_acq_status  = QLabel('ACQ  \u25cf  STOPPED')
        self._lbl_rec_status  = QLabel('REC  \u25cf  STOPPED')
        self._lbl_trig_status = QLabel('TRIG \u25cf  IDLE')
        self._lbl_entropy     = QLabel('ENT  —')
        self._lbl_acq_status .setObjectName('status_off')
        self._lbl_rec_status .setObjectName('status_off')
        self._lbl_trig_status.setObjectName('trig_idle')
        self._lbl_entropy    .setObjectName('trig_idle')
        mono = QFont('Consolas', 10)
        for lbl in (self._lbl_acq_status, self._lbl_rec_status, self._lbl_trig_status,
                     self._lbl_entropy):
            lbl.setFont(mono)
        status_v.addWidget(self._lbl_acq_status)
        status_v.addWidget(self._lbl_rec_status)
        status_v.addWidget(self._lbl_trig_status)
        status_v.addWidget(self._lbl_entropy)
        self._blink_counter = 0

        self._btn_reset = QPushButton('Reset Params')
        self._btn_reset.setObjectName('btn_browse')
        self._btn_reset.setToolTip('Reset all trigger and display parameters to their defaults')

        self._btn_view_mode = QPushButton('\u25a3  View Mode')
        self._btn_view_mode.setObjectName('btn_view_mode')
        self._btn_view_mode.setToolTip('Switch to View mode — a read-only monitoring grid of all recordings')
        self._btn_view_mode.setStyleSheet(
            f'QPushButton {{ background-color: {C["surface0"]}; color: {C["mauve"]}; '
            f'border: 1px solid {C["mauve"]}; border-radius: 5px; '
            f'padding: 6px 16px; font-weight: bold; min-width: 0px; }}'
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

        h.addWidget(acq_box)
        h.addWidget(rec_box)
        h.addWidget(self._btn_reset)
        h.addWidget(self._btn_save)
        h.addWidget(self._btn_save_as)
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

        _TRIG_SCALE = 10  # slider integer ticks per second

        trig_box = QGroupBox('TRIGGER')
        trig_g   = QGridLayout(trig_box)
        trig_g.setVerticalSpacing(8)
        trig_g.setHorizontalSpacing(10)
        trig_g.setColumnStretch(1, 3)

        self._sl_mc, self._sb_mc = self._param_row(trig_g, 0, 0, 'Min Cross',
            sl_min=0, sl_max=600, sl_init=int(DEFAULT_MIN_CROSS * _TRIG_SCALE),
            sb_min=0.0, sb_max=60.0, sb_step=0.1, sb_dec=2, suffix=' s',
            scale=_TRIG_SCALE)
        self._sl_hold, self._sb_hold = self._param_row(trig_g, 1, 0, 'Hold',
            sl_min=0, sl_max=600, sl_init=int(DEFAULT_HOLD * _TRIG_SCALE),
            sb_min=0.0, sb_max=60.0, sb_step=0.1, sb_dec=2, suffix=' s',
            scale=_TRIG_SCALE)
        self._sl_post_trig, self._sb_post_trig = self._param_row(trig_g, 2, 0, 'Post-Trigger',
            sl_min=0, sl_max=600, sl_init=int(DEFAULT_POST_TRIG * _TRIG_SCALE),
            sb_min=0.0, sb_max=60.0, sb_step=0.1, sb_dec=2, suffix=' s',
            scale=_TRIG_SCALE)
        self._sl_maxr, self._sb_maxr = self._param_row(trig_g, 3, 0, 'Max Rec',
            sl_min=10, sl_max=36000, sl_init=int(DEFAULT_MAX_REC * _TRIG_SCALE),
            sb_min=1.0, sb_max=3600.0, sb_step=1.0, sb_dec=1, suffix=' s',
            scale=_TRIG_SCALE)
        self._sl_pre, self._sb_pre = self._param_row(trig_g, 4, 0, 'Pre-Trigger',
            sl_min=0, sl_max=600, sl_init=int(DEFAULT_PRE_TRIG * _TRIG_SCALE),
            sb_min=0.0, sb_max=60.0, sb_step=0.1, sb_dec=2, suffix=' s',
            scale=_TRIG_SCALE)

        for _tw in (self._sl_mc, self._sb_mc):
            _tw.setToolTip('Min Cross: minimum time the signal must stay above the threshold to start a recording')
        for _tw in (self._sl_hold, self._sb_hold):
            _tw.setToolTip('Hold: duration of silence after the signal drops before a recording is considered finished')
        for _tw in (self._sl_post_trig, self._sb_post_trig):
            _tw.setToolTip('Post-Trigger: audio kept after the last above-threshold sample (tail of the saved WAV)')
        for _tw in (self._sl_maxr, self._sb_maxr):
            _tw.setToolTip('Max Rec: maximum length of a single WAV segment — longer events are split')
        for _tw in (self._sl_pre, self._sb_pre):
            _tw.setToolTip('Pre-Trigger: audio kept before the trigger point (lookback saved to the WAV)')

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
        _ENTROPY_SCALE = 100  # slider integer ticks per unit
        self._sl_entropy_thr, self._sb_entropy_thr = self._param_row(trig_g, 7, 0, 'Entropy Thr',
            sl_min=0, sl_max=100, sl_init=50,
            sb_min=0.0, sb_max=1.0, sb_step=0.01, sb_dec=2, suffix='',
            scale=_ENTROPY_SCALE)
        self._sl_entropy_thr.setEnabled(False)
        self._sb_entropy_thr.setEnabled(False)
        for _tw in (self._sl_entropy_thr, self._sb_entropy_thr):
            _tw.setToolTip('Spectral entropy threshold — triggers when entropy falls below this value (0 = pure tone, 1 = white noise)')

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

        outer.addWidget(trig_box)
        return w

    # ── Display parameters ───────────────────────────────────────────────

    def _build_spec_params(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet(f'background-color: {C["mantle"]};')
        outer = QHBoxLayout(w)
        outer.setContentsMargins(14, 6, 14, 6)

        box  = QGroupBox('DISPLAY')
        grid = QGridLayout(box)
        grid.setVerticalSpacing(8)
        grid.setHorizontalSpacing(10)
        grid.setColumnStretch(1, 3)
        grid.setColumnStretch(4, 1)

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

        for _tw in (self._sl_gain, self._sb_gain):
            _tw.setToolTip('Gain applied to the spectrogram (dB) — brightens or darkens the image')
        for _tw in (self._sl_floor, self._sb_floor):
            _tw.setToolTip('Minimum dB value shown in the spectrogram colormap')
        for _tw in (self._sl_ceil, self._sb_ceil):
            _tw.setToolTip('Maximum dB value shown in the spectrogram colormap')

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
                   sl_min, sl_max, sl_init,
                   sb_min, sb_max, sb_step, sb_dec, suffix,
                   scale) -> tuple:
        lbl = QLabel(label)
        lbl.setObjectName('param_label')
        lbl.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)

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

        output_box = QGroupBox('OUTPUT')
        output_g   = QGridLayout(output_box)
        output_g.setVerticalSpacing(4)
        output_g.setHorizontalSpacing(6)
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
        device_v.addLayout(dev_row1)
        device_v.addLayout(dev_row2)

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

        outer.addWidget(output_box, stretch=10)
        outer.addWidget(ref_box, stretch=3)
        outer.addWidget(device_box, stretch=5)
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
        self._device_combo.currentIndexChanged.connect(self._on_device_changed)
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
        self._sl_mc  .valueChanged.connect(lambda _: self._write_trigger_params())
        self._sb_mc  .valueChanged.connect(lambda _: self._write_trigger_params())
        self._sl_hold     .valueChanged.connect(lambda _: self._write_trigger_params())
        self._sb_hold     .valueChanged.connect(lambda _: self._write_trigger_params())
        self._sl_post_trig.valueChanged.connect(lambda _: self._write_trigger_params())
        self._sb_post_trig.valueChanged.connect(lambda _: self._write_trigger_params())
        self._sl_maxr.valueChanged.connect(lambda _: self._write_trigger_params())
        self._sb_maxr.valueChanged.connect(lambda _: self._write_trigger_params())
        self._sl_pre .valueChanged.connect(lambda _: self._write_trigger_params())
        self._sb_pre .valueChanged.connect(lambda _: self._write_trigger_params())
        self._combo_detect_mode.currentTextChanged.connect(lambda _: self._write_trigger_params())
        self._sl_entropy_thr.valueChanged.connect(lambda _: self._write_trigger_params())
        self._sb_entropy_thr.valueChanged.connect(lambda _: self._write_trigger_params())

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
        self._combo_display_mode.currentTextChanged.connect(self._on_display_mode_changed)

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
        e.post_trig_sec = self._sb_post_trig.value()
        e.max_rec_sec   = self._sb_maxr.value()
        e.pre_trig_sec  = self._sb_pre.value()
        e.spectral_trigger_mode = self._combo_detect_mode.currentText()
        e.spectral_threshold    = self._sb_entropy_thr.value()

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
        e.post_trig_sec = self._sb_post_trig.value()
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

        _TRIG_SCALE = 10
        _set(self._sb_thr,  e.threshold)
        _set(self._sb_mc,   e.min_cross_sec)
        _set(self._sl_mc,   int(round(e.min_cross_sec * _TRIG_SCALE)))
        _set(self._sb_hold,      e.hold_sec)
        _set(self._sl_hold,      int(round(e.hold_sec * _TRIG_SCALE)))
        _set(self._sb_post_trig, e.post_trig_sec)
        _set(self._sl_post_trig, int(round(e.post_trig_sec * _TRIG_SCALE)))
        _set(self._sb_maxr, e.max_rec_sec)
        _set(self._sl_maxr, int(round(e.max_rec_sec * _TRIG_SCALE)))
        _set(self._sb_pre,  e.pre_trig_sec)
        _set(self._sl_pre,  int(round(e.pre_trig_sec * _TRIG_SCALE)))

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

        # Spectral trigger mode
        self._combo_detect_mode.blockSignals(True)
        self._combo_detect_mode.setCurrentText(e.spectral_trigger_mode)
        self._combo_detect_mode.blockSignals(False)
        _ENTROPY_SCALE = 100
        _set(self._sb_entropy_thr, e.spectral_threshold)
        _set(self._sl_entropy_thr, int(round(e.spectral_threshold * _ENTROPY_SCALE)))
        ent_on = (e.spectral_trigger_mode != 'Amplitude Only')
        self._sl_entropy_thr.setEnabled(ent_on)
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

        self._current_config_path = path
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

    def _on_display_mode_changed(self, mode: str):
        e = self._sel
        if not e:
            return
        e.display_mode = mode
        stereo = e.channel_mode == 'Stereo'
        self._setup_axes(stereo=stereo)
        self._update_spec_yticks(e)

    def _on_detect_mode_changed(self, mode: str):
        self._sl_entropy_thr.setEnabled(mode != 'Amplitude Only')
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

            # Only show x-labels on bottom row
            if r < rows - 1:
                ax_amp.tick_params(labelbottom=False)
            else:
                ax_amp.set_xlabel('Time (s)', fontsize=7)

            ax_amp.set_ylabel('Amp', fontsize=7)

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
                # Move x-label from amp to entropy
                ax_amp.tick_params(labelbottom=False)
                if r >= rows - 1:
                    ax_entropy.tick_params(labelbottom=True)
                    ax_entropy.set_xlabel('Time (s)', fontsize=7)
                    ax_amp.set_xlabel('')

            # Title and status on the topmost axis
            top_ax = ax_spec if ax_spec is not None else ax_wave
            title_obj = top_ax.set_title(e.name, loc='left', fontsize=9,
                                         color=C['text'], fontweight='bold', pad=3)
            status_text = top_ax.text(
                0.99, 1.02, '', transform=top_ax.transAxes, fontsize=8,
                ha='right', va='bottom', fontfamily='Consolas')

            self._vm_axes.append({
                'ax_spec': ax_spec, 'ax_amp': ax_amp, 'ax_wave': ax_wave,
                'ax_entropy': ax_entropy,
                'spec_im': spec_im,
                'amp_line': amp_line, 'amp_line_r': amp_line_r,
                'wave_line': wave_line, 'wave_line_r': wave_line_r,
                'entropy_line': entropy_line,
                'cursor_spec': cursor_spec, 'cursor_amp': cursor_amp,
                'cursor_wave': cursor_wave, 'cursor_entropy': cursor_entropy,
                'thr_line': thr_line, 'entropy_thr_line': entropy_thr_line,
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
        # 1. Ingest chunks for ALL entities. Cap the per-tick drain at
        # MAX_DRAIN_PER_TICK so a stalled main thread cannot stretch a
        # single tick into a multi-second processing block — that would
        # starve the GUI loop and snowball drops on every other stream.
        # Anything left in the queue past the cap is intentionally
        # dropped on the floor and counted via `consume_drop_count`,
        # which the sidebar surfaces as a drop-indicator badge (#13).
        MAX_DRAIN_PER_TICK = 8
        for idx, e in enumerate(self._entities):
            drained = 0
            while drained < MAX_DRAIN_PER_TICK:
                try:
                    e.ingest_chunk(e.queue.get_nowait())
                except queue.Empty:
                    break
                drained += 1
            else:
                # Cap hit: anything still queued is over-budget. Account
                # for it so the sidebar shows a drop badge, then drain
                # the queue so we don't fall further behind.
                overflow = 0
                while True:
                    try:
                        e.queue.get_nowait()
                        overflow += 1
                    except queue.Empty:
                        break
                if overflow and hasattr(e.capture, 'drop_count'):
                    e.capture.drop_count += overflow
            # Surface the drop badge to the sidebar (#13 / c15).
            if hasattr(e.capture, 'consume_drop_count'):
                n_drops = e.capture.consume_drop_count()
                if hasattr(self, '_sidebar'):
                    try:
                        self._sidebar.update_item_drops(idx, n_drops)
                    except Exception:
                        pass

        # 2. Branch on mode
        if self._view_mode:
            self._update_plot_view_mode()
            return

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
