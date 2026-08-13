"""Top-level Qt window — ChirpWindow + main() entry point.

Extracted from the monolith in the Phase 1 refactor (plan: c08). This
is still the largest file in the project (~2700 lines) — Phase 2 and
Phase 3 fixes will chip away at it:

  - #13 (c15): bounded per-tick queue drain + drop badge
  - #17 (c16): shutdown flushes in-flight events and awaits the writer pool
  - #19 (c21): move ingest_chunk off the Qt main thread
  - #11 (c22): save-button tooltip + dirty-state indicator
"""

import ctypes
import datetime
import json
import os
import math
import sys
import threading
import time
from contextlib import contextmanager

import numpy as np
import sounddevice as sd

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QPushButton, QLabel, QLineEdit,
    QFileDialog, QFrame, QSizePolicy, QDoubleSpinBox, QComboBox, QCheckBox,
    QScrollArea, QStackedLayout, QDialog, QCalendarWidget, QMessageBox, QSpinBox,
    QMenu, QAction, QActionGroup, QSlider, QSplitter,
    QRadioButton, QButtonGroup, QDialogButtonBox, QColorDialog,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QSize, QDate, QPointF
from PyQt5.QtGui import (
    QFont, QFontMetrics, QPainter, QColor, QPainterPath, QPen, QPolygonF,
    QCursor,
)

# Re-exports used throughout the window code. The star-import brings in
# all the module-level constants and the palette (C, QSS) so the class
# body below keeps referring to them by bare name.
from chirp import __version__
from chirp.constants import *  # noqa: F401,F403
from chirp.audio import AudioCapture, AudioMonitor  # noqa: F401
from chirp.audio.devices import list_output_devices, host_api_name
from chirp.config import DEFAULT_VIEW_MODE, DEFAULT_MONITOR  # noqa: F401
from chirp.error_log import log as _err_log
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
        self._vm_n_cols = 1
        self._vm_panel_height = 300
        # Grid fill order: 'column' fills down the first column (streams
        # 1..rows) before the next; 'row' fills across the top row first.
        # Persisted in the view_mode config section.
        self._vm_fill_order = DEFAULT_VIEW_MODE['fill_order']
        # Phase 4: view-mode grid is rendered by pyqtgraph/OpenGL. Built
        # lazily on first entry; swapped into the central scroll area in
        # place of the config panel. Set to use software rendering in
        # headless tests via ``_pg_use_opengl``.
        self._pg_grid = None
        self._pg_use_opengl = True
        # Remote-desktop rendering resilience: an RDP attach swaps the
        # display driver out from under live OpenGL contexts, which is
        # what actually froze the app under Windows Remote Desktop
        # (AnyDesk merely mirrors the console session, so it never hit
        # this). ``_pg_use_opengl`` stays the CONFIGURED value (what the
        # user saved); ``_gl_effective`` clamps it to raster whenever a
        # remote session is (or becomes) active. Mid-run transitions are
        # tracked via WM_WTSSESSION_CHANGE (see nativeEvent).
        self._remote_display_active = _is_remote_session()
        if self._remote_display_active:
            print('[Chirp] Windows remote session detected — rendering '
                  'in raster mode (OpenGL restored on console login)')
        # 3c: view-mode 'Active only' filter (persisted in view_mode).
        self._vm_active_only = True
        self._vm_visible_sig: tuple = ()
        # Phase 4 audio-priority: adaptively skip view-mode render frames
        # when a render costs more than ~half a tick, so DSP / capture
        # threads keep the GIL and audio never drops. EMA of view render
        # time (ms) drives a frame-skip count (0 = full rate, up to 3 =
        # ~1/4 rate). Off-screen culling handles the rest.
        self._view_render_ema = 0.0
        self._view_skip = 0
        self._view_skip_left = 0
        # H3: the same audio-priority frame-skip for CONFIG mode. The
        # config panel's per-tick cost scales with display_seconds ×
        # sample_rate (full-buffer log10 + setData) and numpy holds the
        # GIL — at high SR / long windows an unthrottled 20 Hz render
        # starves the capture callbacks and ingest threads, which is
        # exactly the input_overflow / ring-overrun failure mode.
        self._cfg_render_ema = 0.0
        self._cfg_skip = 0
        self._cfg_skip_left = 0
        # H3: sidebar mini-amp previews update round-robin (one entity
        # per tick) — get_mini_amplitude is a full-buffer reduction and
        # N streams × 20 Hz of it is pure GIL burn for a 30px preview.
        self._mini_amp_rr = 0
        # TODO#1 (RDP): capture-recovery throttle state. Recovery runs
        # on a background worker thread — PortAudio calls on a dead
        # WASAPI device (close/reopen/terminate) can block for seconds
        # or forever, and doing that on the GUI thread froze the whole
        # app the moment AnyDesk/RDP churned the audio endpoints.
        self._recovery_last_attempt = 0.0
        self._recovery_needs_refresh = False
        self._recovery_thread: threading.Thread | None = None
        self._recovery_backoff = 3.0   # doubles on failure, max 30 s

        # Capture-engine settings + inserted-silence auto-recovery state.
        # The audio dict mirrors the config file's ``audio`` section; the
        # recovery bookkeeping is per-entity (how long its duty cycle has
        # been high) and per-device (when that endpoint was last reset,
        # so a persistent fault can't become a restart loop).
        from chirp.config.schema import DEFAULT_AUDIO
        self._audio_cfg = dict(DEFAULT_AUDIO)
        self._zero_high_since: dict[int, float] = {}
        self._zero_recover_last: dict[tuple, float] = {}
        self._zero_recover_thread: threading.Thread | None = None
        self.zero_recovery_count = 0

        # #7: shared audio-monitor loopback. One output stream; each
        # RecordingEntity is wired into it in `_add_recording` and the
        # monitor itself gates on `source_id` so only the chosen stream
        # plays. Created before `_build_ui` so the UI can reference it.
        self._monitor = AudioMonitor()

        self._build_figure()
        self._build_ui()
        self._connect_signals()

        self._timer = QTimer(self)
        self._timer.setInterval(ANIMATION_INTERVAL)
        self._timer.timeout.connect(self._update_plot)
        self._timer.start()

        # Seed the initial state from the startup preference (empty /
        # last / pinned file). Runs after the timer exists because a
        # config load stops and restarts it. Always leaves at least one
        # recording present.
        self._apply_startup_config()

        self._update_title()
        self.resize(1400, 850)

        # Remote-desktop resilience: get WM_WTSSESSION_CHANGE delivered
        # so a mid-run RDP attach/detach can swap the render backend.
        self._register_session_notification()

        # Display-change resilience: when the desktop resolution / DPI
        # changes under a maximized window (RDP connect/disconnect,
        # monitor hot-plug, docking), Windows can leave the window at
        # its stale pre-change frame so the right/bottom edges hang off
        # the new screen until the user minimizes and re-maximizes.
        # WM_DISPLAYCHANGE arrives in a burst while the new mode
        # settles, so the fix is debounced.
        self._remax_timer = QTimer(self)
        self._remax_timer.setSingleShot(True)
        self._remax_timer.setInterval(500)
        self._remax_timer.timeout.connect(self._remaximize_after_display_change)

    # ──────────────────────────────────────────────────────────────────────
    # Remote-desktop rendering resilience
    # ──────────────────────────────────────────────────────────────────────

    #: WM_WTSSESSION_CHANGE wParam values (wtsapi32).
    _WM_WTSSESSION_CHANGE = 0x02B1
    _WTS_CONSOLE_CONNECT  = 0x1
    _WTS_REMOTE_CONNECT   = 0x3
    #: Broadcast when the desktop resolution / color depth changes.
    _WM_DISPLAYCHANGE = 0x007E

    @property
    def _gl_effective(self) -> bool:
        """OpenGL actually usable right now: the configured setting
        clamped to raster while a remote-desktop session drives the
        display (RDP swaps the display driver and kills GL contexts)."""
        return self._pg_use_opengl and not self._remote_display_active

    def _register_session_notification(self) -> None:
        if sys.platform != 'win32':
            return
        try:
            ctypes.windll.wtsapi32.WTSRegisterSessionNotification(
                int(self.winId()), 0)   # NOTIFY_FOR_THIS_SESSION
        except Exception as exc:
            print(f'[Chirp] WTS session notification unavailable: {exc}')

    def nativeEvent(self, event_type, message):
        if sys.platform == 'win32':
            try:
                import ctypes.wintypes
                msg = ctypes.wintypes.MSG.from_address(int(message))
                if msg.message == self._WM_WTSSESSION_CHANGE:
                    self._on_wts_session_change(int(msg.wParam))
                elif msg.message == self._WM_DISPLAYCHANGE:
                    if self.isMaximized() and not self.isMinimized():
                        self._remax_timer.start()
            except Exception:
                pass
        return super().nativeEvent(event_type, message)

    def _on_wts_session_change(self, wparam: int) -> None:
        """React to the display moving between the console and a remote
        session. On remote attach every live pyqtgraph viewport is
        swapped to raster BEFORE the lost GL context can wedge the
        paint path; on console return the configured backend comes
        back."""
        if wparam == self._WTS_REMOTE_CONNECT:
            if not self._remote_display_active:
                self._remote_display_active = True
                print('[Chirp] remote session attached — switching to '
                      'raster rendering')
                self._apply_render_backend()
        elif wparam == self._WTS_CONSOLE_CONNECT:
            if self._remote_display_active:
                self._remote_display_active = False
                print('[Chirp] console session restored — '
                      + ('OpenGL re-enabled' if self._pg_use_opengl
                         else 'staying in raster mode (configured)'))
                self._apply_render_backend()

    def _apply_render_backend(self) -> None:
        """Swap every live pyqtgraph viewport between OpenGL and raster
        to match ``_gl_effective``. pyqtgraph's GraphicsView supports
        this at runtime (``useOpenGL`` replaces the viewport widget);
        the global config option covers widgets built afterwards."""
        try:
            on = self._gl_effective
            import pyqtgraph as pg
            pg.setConfigOptions(useOpenGL=on)
            for gv in self.findChildren(pg.GraphicsView):
                try:
                    gv.useOpenGL(on)
                except Exception:
                    pass
        except Exception:
            # Best-effort — partially-constructed windows (test stubs)
            # and headless environments must not crash here.
            pass

    def _remaximize_after_display_change(self) -> None:
        """Re-fit a maximized window after the desktop mode changed.

        Windows does not reliably re-maximize an already-maximized
        window when the resolution/DPI changes, leaving it framed for
        the OLD screen (right side cut off). The manual workaround is
        minimize + maximize; this performs the equivalent
        showNormal/showMaximized round-trip automatically, and only
        when the frame is actually stale."""
        try:
            if not self.isMaximized() or self.isMinimized():
                return
            handle = self.windowHandle()
            screen = handle.screen() if handle else None
            if screen is None:
                return
            avail = screen.availableGeometry()
            geo = self.geometry()
            # A correctly maximized window fills the available area
            # (give or take the hidden resize borders). If it already
            # does, don't flicker the window for nothing.
            if (abs(geo.width() - avail.width()) <= 24
                    and abs(geo.height() - avail.height()) <= 24):
                return
            print('[Chirp] display mode changed — re-fitting maximized '
                  f'window ({geo.width()}x{geo.height()} → '
                  f'{avail.width()}x{avail.height()})')
            self.showNormal()
            self.showMaximized()
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────────────
    # Dirty-state tracking (#11 / c22)
    # ──────────────────────────────────────────────────────────────────────

    def _mark_dirty(self):
        """Flag that in-memory config has changed since last save/load."""
        if not self._config_dirty:
            self._config_dirty = True
            self._update_title()
            self._update_save_tooltip()
            self._update_save_button()

    def _mark_clean(self):
        """Reset dirty flag after a successful save or load."""
        self._config_dirty = False
        self._update_title()
        self._update_save_tooltip()
        self._update_save_button()

    def _update_save_button(self):
        """4c (TODO#12): Save is enabled only when there are unsaved
        changes (Save As stays always available)."""
        if hasattr(self, '_btn_save'):
            self._btn_save.setEnabled(self._config_dirty)

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
    # M4: busy feedback for blocking device operations
    # ──────────────────────────────────────────────────────────────────────

    @contextmanager
    def _busy_cursor(self):
        """Wait cursor around device open/close, sample-rate rebuilds and
        WAV loads — they block the GUI thread for up to several seconds
        (PortAudio open, buffer rebuild, 10 s ingest join). Without any
        feedback the app looks hung and users force-kill it, orphaning
        the writer pool. Moving these off the GUI thread is a larger
        refactor; the cursor at least tells the user work is happening.

        No-op when no QApplication exists (stub-window unit tests)."""
        app = QApplication.instance()
        if app is not None:
            QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        try:
            if app is not None:
                QApplication.processEvents()
            yield
        finally:
            if app is not None:
                QApplication.restoreOverrideCursor()

    # ──────────────────────────────────────────────────────────────────────
    # Central plot panel (pyqtgraph)
    # ──────────────────────────────────────────────────────────────────────

    def _build_figure(self):
        # Phase C: the hidden matplotlib figure is gone — the pyqtgraph
        # ConfigPlotPanel is the config-mode central view; it rebuilds
        # its own layout from the entity signature (display mode /
        # stereo / spectral / amp scale / display window) on each tick.
        from chirp.ui.config_panel import ConfigPlotPanel
        self._config_panel = ConfigPlotPanel(use_opengl=self._gl_effective)
        self._config_panel.thresholdChanged.connect(self._on_threshold_dragged)
        self._config_panel.spectralThresholdChanged.connect(
            self._on_spectral_threshold_dragged)
        self._config_panel.ampScaleChanged.connect(self._on_amp_scale_menu)








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

        # Main horizontal splitter: sidebar | right pane. The handle is a
        # draggable border so the user can widen/narrow the sidebar.
        self._main_splitter = QSplitter(Qt.Horizontal)
        self._main_splitter.setChildrenCollapsible(False)
        self._main_splitter.setHandleWidth(5)

        # Sidebar
        self._sidebar = RecordingSidebar()
        self._main_splitter.addWidget(self._sidebar)

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

        # Vertical splitter: plots (top) | config area (bottom). The handle
        # is a draggable border between the display plots and the
        # configuration panels.
        self._plot_config_splitter = QSplitter(Qt.Vertical)
        self._plot_config_splitter.setChildrenCollapsible(False)
        self._plot_config_splitter.setHandleWidth(5)

        # Plot area inside a scroll area (scrollable in view mode). Phase 4b:
        # config mode shows the pyqtgraph ConfigPlotPanel; view mode swaps in
        # the MultiStreamGrid. The matplotlib canvas is no longer mounted.
        self._canvas_scroll = QScrollArea()
        self._canvas_scroll.setWidgetResizable(True)
        self._canvas_scroll.setWidget(self._config_panel)
        self._canvas_scroll.setFrameShape(QFrame.NoFrame)
        self._plot_config_splitter.addWidget(self._canvas_scroll)

        # Config area — one compact row of panels:
        #   Controls | Trigger | Display | [Status / Output / Ref-date / Input]
        # The last column is a vertical stack of wide horizontal panels.
        self._config_widgets: list[QWidget] = []

        controls_box = self._build_controls_box()
        params_panel = self._build_params()
        spec_panel   = self._build_spec_params()
        right_stack  = self._build_config_right_stack()

        config_row = QWidget()
        config_row.setStyleSheet(f'background-color: {C["mantle"]};')
        config_row.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        config_h = QHBoxLayout(config_row)
        config_h.setContentsMargins(6, 2, 6, 2)
        config_h.setSpacing(6)
        config_h.addWidget(controls_box)
        config_h.addWidget(params_panel)
        config_h.addWidget(spec_panel)
        config_h.addWidget(right_stack, stretch=1)

        # Wrap the divider + row so the whole config area is one splitter
        # child that view mode can hide/show as a unit.
        config_container = QWidget()
        cc_v = QVBoxLayout(config_container)
        cc_v.setContentsMargins(0, 0, 0, 0)
        cc_v.setSpacing(0)
        cc_v.addWidget(self._hline())
        cc_v.addWidget(config_row)
        self._plot_config_splitter.addWidget(config_container)
        self._plot_config_splitter.setStretchFactor(0, 1)  # plots grow
        self._plot_config_splitter.setStretchFactor(1, 0)  # config keeps hint

        vbox.addWidget(self._plot_config_splitter, stretch=1)

        # View-mode toolbar (hidden initially)
        self._view_toolbar = self._build_view_toolbar()
        vbox.addWidget(self._view_toolbar)
        self._view_toolbar.hide()

        self._config_widgets.append(config_container)

        self._main_splitter.addWidget(right)
        self._main_splitter.setStretchFactor(0, 0)  # sidebar keeps width
        self._main_splitter.setStretchFactor(1, 1)  # right pane grows
        self._main_splitter.setSizes([250, 1200])
        hbox.addWidget(self._main_splitter)

        self._collect_lockable_widgets()

    def _collect_lockable_widgets(self) -> None:
        """Gather every per-stream *configuration* control that the
        parameter lock disables. Deliberately EXCLUDES display params
        (DISPLAY panel), the audio monitor bar, and transport actions —
        those stay usable while a stream is locked."""
        self._lockable_widgets = [
            # Trigger params
            self._sb_thr, self._sb_mc, self._sb_min_total, self._sb_hold,
            self._sb_pre, self._sb_post_trig, self._sb_maxr,
            self._chk_freq, self._sb_freq_lo, self._sb_freq_hi,
            self._combo_detect_mode, self._sb_entropy_thr, self._sb_entropy_mc,
            self._combo_rec_mode,
            self._btn_calibrate, self._sb_calib_dur, self._sb_calib_margin,
            # Output
            self._folder_edit, self._btn_browse_out,
            self._prefix_edit, self._suffix_edit,
            # Reference date
            self._chk_ref_date, self._date_line, self._btn_pick_date,
            self._dph_prefix_edit,
            # Input device
            self._device_combo, self._btn_dev_refresh,
            self._chan_combo, self._trig_combo, self._sr_combo,
            self._btn_wav_reset, self._chk_wav_loop,
            # Per-stream extras that also change persisted config
            self._btn_stream_color, self._btn_reset,
        ]

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

        h.addSpacing(14)

        # 3c: show only acquiring streams in the grid.
        self._chk_vm_active_only = QCheckBox('Active only')
        self._chk_vm_active_only.setChecked(True)
        self._chk_vm_active_only.setToolTip(
            'Show only streams whose acquisition is running')
        self._chk_vm_active_only.toggled.connect(self._on_vm_active_only)
        h.addWidget(self._chk_vm_active_only)

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

        lbl_o = QLabel('Order:')
        lbl_o.setStyleSheet(f'color: {C["subtext"]}; font-size: 9pt;')
        h.addWidget(lbl_o)
        self._vm_order_combo = QComboBox()
        self._vm_order_combo.setToolTip(
            'Panel fill order in the View grid.\n'
            'Column: fill down the first column (streams 1..N), then the '
            'next column.\nRow: fill across the top row first.')
        self._vm_order_combo.addItem('Column', userData='column')
        self._vm_order_combo.addItem('Row', userData='row')
        self._vm_order_combo.setCurrentIndex(
            0 if self._vm_fill_order == 'column' else 1)
        self._vm_order_combo.setFixedWidth(90)
        self._vm_order_combo.currentIndexChanged.connect(
            self._on_vm_fill_order_changed)
        h.addWidget(self._vm_order_combo)

        h.addSpacing(10)

        lbl_h = QLabel('Height:')
        lbl_h.setStyleSheet(f'color: {C["subtext"]}; font-size: 9pt;')
        h.addWidget(lbl_h)
        self._vm_height_spin = QSpinBox()
        self._vm_height_spin.setToolTip('Row height for each recording tile in View mode')
        self._vm_height_spin.setRange(80, 700)
        self._vm_height_spin.setValue(self._vm_panel_height)
        self._vm_height_spin.setSuffix(' px')
        self._vm_height_spin.setFixedWidth(90)
        self._vm_height_spin.valueChanged.connect(self._on_vm_height_changed)
        h.addWidget(self._vm_height_spin)

        h.addSpacing(10)

        self._btn_vm_fit = QPushButton('Fit to screen')
        self._btn_vm_fit.setToolTip(
            'Set the tile height so every stream fits the window without '
            'scrolling (uses the current column count)')
        self._btn_vm_fit.setStyleSheet(
            f'QPushButton {{ background-color: {C["surface0"]}; color: {C["blue"]}; '
            f'border: 1px solid {C["blue"]}; border-radius: 5px; '
            f'padding: 5px 12px; font-weight: bold; min-width: 0px; }}'
            f'QPushButton:hover {{ background-color: {C["surface1"]}; }}'
        )
        self._btn_vm_fit.clicked.connect(self._on_vm_fit_to_screen)
        h.addWidget(self._btn_vm_fit)

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

        # 4d (TODO#13): master enable/mute toggle. Turning the monitor
        # off keeps both combo selections, so turning it back on
        # restores the exact same routing without re-picking devices.
        self._monitor_muted = False
        self._btn_monitor_mute = QPushButton('\U0001F50A')
        self._btn_monitor_mute.setCheckable(True)
        self._btn_monitor_mute.setChecked(True)
        self._btn_monitor_mute.setFixedSize(30, 24)
        self._btn_monitor_mute.setToolTip(
            'Enable / disable the audio monitor. Disabling keeps the '
            'source and output selections, so re-enabling restores the '
            'same routing.')
        self._btn_monitor_mute.setStyleSheet(
            f'QPushButton {{ background-color: {C["surface0"]}; '
            f'border: 1px solid {C["surface1"]}; border-radius: 4px; '
            f'min-width: 0px; padding: 0px; font-size: 10pt; }}'
            f'QPushButton:!checked {{ color: {C["surface2"]}; }}'
        )
        self._btn_monitor_mute.toggled.connect(self._on_monitor_mute_toggled)
        h.addWidget(self._btn_monitor_mute)

        # 4d (TODO#10): follow the selection — the monitor re-targets to
        # whichever stream is selected (Config mode) or clicked in the
        # View-mode grid.
        self._chk_monitor_follow = QCheckBox('Follow')
        self._chk_monitor_follow.setToolTip(
            'Monitor follows the selection: selecting a stream in the '
            'sidebar (Config mode) or clicking a tile (View mode) routes '
            'that stream to the monitor output.')
        self._chk_monitor_follow.toggled.connect(self._on_monitor_follow_toggled)
        h.addWidget(self._chk_monitor_follow)

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
        # Same guard as the input-device combo: long output device
        # names must not inflate the window's minimum width.
        self._monitor_out_combo.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self._monitor_out_combo.setMinimumContentsLength(20)
        self._populate_monitor_output_combo()
        self._monitor_out_combo.currentIndexChanged.connect(self._on_monitor_output_changed)
        h.addWidget(self._monitor_out_combo, stretch=1)

        btn_refresh = QPushButton('\u21BB')
        btn_refresh.setObjectName('btn_small')
        btn_refresh.setFixedSize(26, 22)
        btn_refresh.setToolTip('Rescan available output devices')
        btn_refresh.clicked.connect(self._on_refresh_monitor_outputs)
        h.addWidget(btn_refresh)

        h.addSpacing(8)

        # Output gain: 0\u2013200%, unity at 100%. Session-scoped like the
        # rest of the monitor bar (source/output are not persisted).
        lbl_gain = QLabel('Gain:')
        lbl_gain.setStyleSheet(f'color: {C["subtext"]}; font-size: 9pt;')
        h.addWidget(lbl_gain)
        self._monitor_gain_slider = QSlider(Qt.Horizontal)
        self._monitor_gain_slider.setRange(0, 200)
        self._monitor_gain_slider.setValue(100)
        self._monitor_gain_slider.setFixedWidth(110)
        self._monitor_gain_slider.setToolTip(
            'Monitor output gain (0\u2013200%, 100% = unity). Applied to the '
            'loopback playback only \u2014 recordings are unaffected. Boosted '
            'output above full scale is clipped.')
        self._monitor_gain_slider.valueChanged.connect(
            self._on_monitor_gain_changed)
        h.addWidget(self._monitor_gain_slider)
        self._monitor_gain_label = QLabel('100%')
        self._monitor_gain_label.setStyleSheet(
            f'color: {C["subtext"]}; font-size: 9pt; min-width: 34px;')
        h.addWidget(self._monitor_gain_label)

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
        # 4d: while muted, selections are remembered (the combos hold
        # them) but the backend stays silent and the output stream
        # stays closed.
        if self._monitor_muted:
            self._monitor.set_source(None)
            self._update_monitor_status()
            return
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

    def _sync_monitor_source_combo(self) -> None:
        """Point the source combo at whatever the monitor backend is
        actually playing (used by monitor-follow / tile clicks) without
        re-triggering the apply handler."""
        combo = self._monitor_src_combo
        target = self._monitor.source_id
        combo.blockSignals(True)
        pick = 0
        for i in range(combo.count()):
            if combo.itemData(i) == (target if target is not None
                                     else self._MON_OFF):
                pick = i
                break
        combo.setCurrentIndex(pick)
        combo.blockSignals(False)

    def _on_monitor_source_changed(self, _idx: int):
        self._apply_monitor_source(self._monitor_src_combo.currentData())

    def _on_monitor_mute_toggled(self, enabled: bool):
        """4d (TODO#13): master monitor enable. OFF closes the output
        stream (silence, device released) but leaves both combos alone;
        ON re-applies whatever they hold."""
        self._monitor_muted = not enabled
        self._btn_monitor_mute.setText('\U0001F50A' if enabled else '\U0001F507')
        if self._monitor_muted:
            self._monitor.set_source(None)
            self._monitor.set_output_device(None)   # close the stream
            self._update_monitor_status()
        else:
            # Re-open the output and re-apply the source selection.
            self._on_monitor_output_changed(0)
            self._apply_monitor_source(self._monitor_src_combo.currentData())

    def _on_monitor_follow_toggled(self, on: bool):
        """4d (TODO#10): follow-selection just turned on — snap the
        monitor to the currently selected stream right away."""
        if on and self._sel is not None:
            self._apply_monitor_source(id(self._sel))
            self._sync_monitor_source_combo()

    def _on_monitor_output_changed(self, _idx: int):
        if self._monitor_muted:
            # Remember the choice (the combo holds it); apply on unmute.
            self._update_monitor_status()
            return
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

    def _on_monitor_gain_changed(self, value: int):
        self._monitor.set_gain(value / 100.0)
        self._monitor_gain_label.setText(f'{value}%')

    def _update_monitor_status(self):
        """Reflect backend state on the little status label."""
        if not hasattr(self, '_monitor_status'):
            return
        if self._monitor_muted:
            self._monitor_status.setText('muted')
            self._monitor_status.setStyleSheet(
                f'color: {C["surface2"]}; font-size: 9pt; min-width: 40px;')
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

    # \u2500\u2500 Monitor settings persistence \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def _build_monitor_settings(self) -> dict:
        """Snapshot the audio-monitor UI state for the config file.

        The output device is stored by NAME (indices shift between
        sessions / machines); the source is stored as its position in
        the recordings list (-1 = Off) since ``id(entity)`` tokens are
        not stable across a reload.
        """
        out_dev = self._monitor_out_combo.currentData()
        out_name, out_api = '', ''
        if out_dev is not None:
            try:
                info = sd.query_devices(out_dev)
                out_name = info.get('name', '') or ''
                out_api = host_api_name(info)
            except Exception:
                pass
        # Source index from the selected monitor token.
        src_token = self._monitor_src_combo.currentData()
        src_index = -1
        if src_token not in (None, self._MON_OFF):
            for i, e in enumerate(self._entities):
                if id(e) == src_token:
                    src_index = i
                    break
        return {
            'output_device_name':    out_name,
            'output_device_hostapi': out_api,
            'gain_percent':          int(self._monitor_gain_slider.value()),
            'muted':                 bool(self._monitor_muted),
            'follow':                bool(self._chk_monitor_follow.isChecked()),
            'source_index':          src_index,
        }

    def _resolve_output_device(self, name: str, hostapi: str):
        """Return the output-device combo index whose device matches
        ``name`` (preferring the same host API), or None if not found."""
        if not name:
            return None
        combo = self._monitor_out_combo
        fallback = None
        for i in range(combo.count()):
            dev_id = combo.itemData(i)
            if dev_id is None:
                continue
            try:
                info = sd.query_devices(dev_id)
            except Exception:
                continue
            if info.get('name', '') == name:
                if not hostapi or host_api_name(info) == hostapi:
                    return i
                if fallback is None:
                    fallback = i
        if fallback is not None:
            return fallback
        # Substring fallback (Windows truncation / API-suffix drift).
        for i in range(combo.count()):
            dev_id = combo.itemData(i)
            if dev_id is None:
                continue
            try:
                info = sd.query_devices(dev_id)
            except Exception:
                continue
            if name in info.get('name', '') or info.get('name', '') in name:
                return i
        return None

    def _apply_monitor_settings(self, monitor: dict) -> None:
        """Restore persisted audio-monitor settings after a config load.

        Order matters: gain + mute + follow first, then output device,
        then source (so the source's apply opens the output at the right
        SR). All combo writes block signals; the backend is driven
        explicitly via ``_apply_monitor_source`` / ``set_output_device``.
        """
        # Follow toggle.
        follow = bool(monitor.get('follow', False))
        self._chk_monitor_follow.blockSignals(True)
        self._chk_monitor_follow.setChecked(follow)
        self._chk_monitor_follow.blockSignals(False)

        # Gain.
        gain = int(monitor.get('gain_percent', 100))
        gain = max(0, min(200, gain))
        self._monitor_gain_slider.blockSignals(True)
        self._monitor_gain_slider.setValue(gain)
        self._monitor_gain_slider.blockSignals(False)
        self._monitor.set_gain(gain / 100.0)
        self._monitor_gain_label.setText(f'{gain}%')

        # Mute state \u2014 set the flag and the button without re-running the
        # toggle handler (which would try to re-open the output before we
        # have selected the device below).
        muted = bool(monitor.get('muted', False))
        self._monitor_muted = muted
        self._btn_monitor_mute.blockSignals(True)
        self._btn_monitor_mute.setChecked(not muted)
        self._btn_monitor_mute.setText('\U0001F507' if muted else '\U0001F50A')
        self._btn_monitor_mute.blockSignals(False)

        # Output device (resolved by name).
        out_idx = self._resolve_output_device(
            monitor.get('output_device_name', ''),
            monitor.get('output_device_hostapi', ''))
        self._monitor_out_combo.blockSignals(True)
        if out_idx is not None:
            self._monitor_out_combo.setCurrentIndex(out_idx)
        self._monitor_out_combo.blockSignals(False)

        # Source (by list position). Point the combo at it, then apply.
        src_index = int(monitor.get('source_index', -1))
        self._monitor_src_combo.blockSignals(True)
        pick = 0
        if 0 <= src_index < len(self._entities):
            token = id(self._entities[src_index])
            for i in range(self._monitor_src_combo.count()):
                if self._monitor_src_combo.itemData(i) == token:
                    pick = i
                    break
        self._monitor_src_combo.setCurrentIndex(pick)
        self._monitor_src_combo.blockSignals(False)

        if muted:
            # Keep selections but stay silent / output closed.
            self._monitor.set_source(None)
            self._monitor.set_output_device(None)
            self._update_monitor_status()
        else:
            # Open the output on the chosen device, then route the source.
            self._on_monitor_output_changed(0)
            self._apply_monitor_source(self._monitor_src_combo.currentData())

    # ── Transport ─────────────────────────────────────────────────────────

    def _build_controls_box(self) -> QGroupBox:
        # Left column: buttons grid
        btn_box = QGroupBox('CONTROLS')
        btn_box.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
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

        # 2b: force-trigger toggle — manual segment while REC is on.
        self._btn_force_trig = QPushButton('● Force Trigger')
        self._btn_force_trig.setCheckable(True)
        self._btn_force_trig.setEnabled(False)
        self._btn_force_trig.setToolTip(
            'Force recording NOW (toggle): press to open a recording '
            'segment immediately (pre-trigger lookback included), press '
            'again to close it immediately (no hold / post-trigger tail). '
            'Max Rec splitting still applies. Enabled while REC is on in '
            'Triggered mode.')
        self._btn_force_trig.setStyleSheet(
            f'QPushButton {{ background-color: {C["surface0"]}; color: {C["peach"]}; '
            f'border: 1px solid {C["peach"]}; border-radius: 5px; '
            f'padding: 4px 8px; font-weight: bold; min-width: 0px; }}'
            f'QPushButton:hover {{ background-color: {C["surface1"]}; }}'
            f'QPushButton:checked {{ background-color: {C["peach"]}; '
            f'color: {C["mantle"]}; }}'
            f'QPushButton:disabled {{ color: {C["surface2"]}; '
            f'border-color: {C["surface1"]}; }}'
        )

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
        self._btn_startup = QPushButton('⚙ Startup')
        self._btn_advanced = QPushButton('⚙ Advanced')
        self._btn_advanced.setToolTip(
            'Capture engine tuning (buffer sizes) and inserted-silence '
            'auto-recovery')
        self._btn_save   .setToolTip('Save configuration to the current file')
        self._btn_save_as.setToolTip('Save configuration to a new file')
        self._btn_load   .setToolTip('Load configuration from a file (.json or legacy .chirp)')
        self._btn_startup.setToolTip('Choose which configuration Chirp loads at startup '
                                     '(empty / last used / a specific file)')
        for btn in (self._btn_save, self._btn_save_as, self._btn_load,
                    self._btn_startup, self._btn_advanced):
            btn.setObjectName('btn_browse')

        # Width diet: the global QSS gives every button min-width 120px
        # + 20px side padding; across the two grid columns that made
        # CONTROLS one of the widest panels and helped push the config
        # row past a laptop screen. These buttons render fine narrower —
        # relax the floor and let the grid distribute the width.
        for btn in (self._btn_start_acq, self._btn_stop_acq,
                    self._btn_start_rec, self._btn_stop_rec,
                    self._btn_save, self._btn_save_as,
                    self._btn_load, self._btn_reset, self._btn_startup,
                    self._btn_advanced):
            btn.setStyleSheet(btn.styleSheet()
                              + 'min-width: 0px; padding: 6px 10px;')

        btn_g.addWidget(self._btn_start_acq, 0, 0)
        btn_g.addWidget(self._btn_stop_acq,  0, 1)
        btn_g.addWidget(self._btn_start_rec, 1, 0)
        btn_g.addWidget(self._btn_stop_rec,  1, 1)
        btn_g.addWidget(self._btn_force_trig, 2, 0, 1, 2)
        btn_g.addWidget(self._btn_save,      3, 0)
        btn_g.addWidget(self._btn_save_as,   3, 1)
        btn_g.addWidget(self._btn_load,      4, 0)
        btn_g.addWidget(self._btn_reset,     4, 1)
        btn_g.addWidget(self._btn_startup,   5, 0)
        btn_g.addWidget(self._btn_advanced,  5, 1)
        btn_g.addWidget(self._btn_view_mode, 6, 0, 1, 2)
        return btn_box

    def _build_status_box(self) -> QGroupBox:
        """STATUS panel — a single wide horizontal strip: the four status
        readouts, then the parameter-lock toggle and the recognition-color
        swatch for the selected stream."""
        status_box = QGroupBox('STATUS')
        status_box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        grid = QGridLayout(status_box)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(3)
        grid.setContentsMargins(8, 4, 8, 4)
        self._lbl_acq_status  = QLabel('ACQ  \u25cf  STOPPED')
        self._lbl_rec_status  = QLabel('REC  \u25cf  STOPPED')
        self._lbl_trig_status = QLabel('TRIG \u25cf  IDLE')
        self._lbl_entropy     = QLabel('ENT  \u2014')
        self._lbl_acq_status .setObjectName('status_off')
        self._lbl_rec_status .setObjectName('status_off')
        self._lbl_trig_status.setObjectName('trig_idle')
        self._lbl_entropy    .setObjectName('trig_idle')
        mono = QFont('Consolas', 9)
        # These labels change text at runtime (STOPPED→RUNNING, live
        # entropy value, blink glyphs, error notes). A QLabel's minimum
        # width follows its text, so without a guard every text change
        # re-negotiates the whole config row's width — and growth past
        # the fixed width of a maximized window clips the right-hand
        # panels. Reserve the widest normal state up front (QSS renders
        # them at 10pt) and ignore the live text's size hint; anything
        # longer (e.g. the display-halted note) is clipped, not
        # propagated into the layout.
        fm = QFontMetrics(QFont('Consolas', 10))
        reserve = max(fm.horizontalAdvance(s) for s in (
            'ACQ  ●  RUNNING', 'REC  ●  RUNNING',
            'TRIG ●  IDLE', 'ENT  0.943 ▼')) + 16
        for lbl in (self._lbl_acq_status, self._lbl_rec_status, self._lbl_trig_status,
                     self._lbl_entropy):
            lbl.setFont(mono)
            # A small floor (not the full reserve) so the four readouts
            # still fit when the right-hand column is narrow; the grid
            # never lets adjacent cells overlap, it clips instead.
            lbl.setMinimumWidth(min(reserve, 96))
            lbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        # Two rows × two columns of readouts so they fit even in a narrow
        # column (a single row overlapped them on a laptop-width screen).
        # Lock + color live in a third column on the right.
        grid.addWidget(self._lbl_acq_status,  0, 0)
        grid.addWidget(self._lbl_rec_status,  0, 1)
        grid.addWidget(self._lbl_trig_status, 1, 0)
        grid.addWidget(self._lbl_entropy,     1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        # Per-stream parameter-lock toggle (mirrors the sidebar lock icon).
        self._btn_lock_cfg = QPushButton('\U0001f513 Unlocked')
        self._btn_lock_cfg.setObjectName('btn_small')
        self._btn_lock_cfg.setCursor(Qt.PointingHandCursor)
        self._btn_lock_cfg.setToolTip(
            'Lock this stream’s configuration against accidental edits '
            '(display params and the audio monitor stay editable). '
            'Unlocking asks for confirmation and names the stream.')
        self._btn_lock_cfg.clicked.connect(
            lambda: self._on_toggle_lock(self._selected_idx))
        grid.addWidget(self._btn_lock_cfg, 0, 2)

        # Per-stream recognition color picker — a small labeled swatch
        # button. The chosen color frames this stream's config panel + view
        # tile and tints its sidebar left edge.
        color_row = QHBoxLayout()
        color_row.setSpacing(4)
        color_row.setContentsMargins(0, 0, 0, 0)
        lbl_color = QLabel('Color')
        lbl_color.setFont(mono)
        lbl_color.setStyleSheet(f'color: {C["subtext"]};')
        self._btn_stream_color = QPushButton()
        self._btn_stream_color.setFixedSize(18, 14)
        self._btn_stream_color.setCursor(Qt.PointingHandCursor)
        self._btn_stream_color.setToolTip(
            'Choose a recognition color for the selected stream — frames '
            'its config panel and view-mode tile and tints its sidebar '
            'left edge.')
        self._btn_stream_color.clicked.connect(
            lambda: self._on_change_stream_color(self._selected_idx))
        color_row.addWidget(lbl_color)
        color_row.addWidget(self._btn_stream_color)
        color_row.addStretch()
        grid.addLayout(color_row, 1, 2)
        self._blink_counter = 0
        # #58: ``_update_plot`` exception bookkeeping. The Qt-timer
        # slot used to swallow exceptions silently — Qt logs the
        # traceback to stderr (invisible in the packaged build) and
        # keeps firing, but the half-finished slot leaves blit cache
        # invariants broken. The display freezes; the user assumes
        # the app is dead and force-kills it (per #56 that orphans
        # the writer pool). Counters are bumped from inside the
        # top-level guard added in the same PR.
        self._update_plot_err_count        = 0  # consecutive errors
        self._update_plot_err_total        = 0  # session-wide
        self._update_plot_last_err: str | None = None
        self._update_plot_freeze_threshold = 5  # ticks before sticky note
        return status_box

    # ── Trigger Parameters ────────────────────────────────────────────────

    def _build_params(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet(f'background-color: {C["mantle"]};')
        w.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        outer = QHBoxLayout(w)
        outer.setContentsMargins(6, 2, 6, 2)

        # Hidden threshold spinbox (synced from amplitude graph drag).
        # 6 decimals so the linear value carries the full dynamic range:
        # on the dB scale the floor (-80 dB) is 1e-4 linear, and 3 decimals
        # gave essentially no resolution below ~-40 dB (0.001 = -60 dB,
        # 0.000 = unrepresentable). 1e-6 steps keep fine dB resolution
        # everywhere from the floor to 0 dB.
        self._sb_thr = QDoubleSpinBox()
        self._sb_thr.setRange(0.0, 1.0)
        self._sb_thr.setSingleStep(0.0001)
        self._sb_thr.setDecimals(6)
        self._sb_thr.setValue(DEFAULT_THRESHOLD)
        self._sb_thr.hide()

        trig_box = QGroupBox('TRIGGER')
        trig_g   = QGridLayout(trig_box)
        trig_g.setVerticalSpacing(4)
        trig_g.setHorizontalSpacing(8)
        trig_g.setContentsMargins(8, 4, 8, 4)

        # 2a: recording mode — Triggered (threshold state machine) or
        # Continuous (record everything while REC is on; new file every
        # Max Rec). Per-stream, persisted in the config.
        lbl_rmode = QLabel('Rec Mode')
        lbl_rmode.setObjectName('param_label')
        self._combo_rec_mode = QComboBox()
        self._combo_rec_mode.setFixedWidth(160)
        for rm in ('Triggered', 'Continuous'):
            self._combo_rec_mode.addItem(rm)
        self._combo_rec_mode.setToolTip(
            'Recording mode:\n'
            '  • Triggered — threshold-based event detection (all params below)\n'
            '  • Continuous — record everything while REC is on; a new file '
            'starts every Max Rec seconds (other trigger params are ignored)')
        rmode_row = QHBoxLayout()
        rmode_row.setSpacing(8)
        rmode_row.addWidget(lbl_rmode)
        rmode_row.addWidget(self._combo_rec_mode)
        rmode_row.addStretch()
        trig_g.addLayout(rmode_row, 0, 0, 1, 3)

        self._sb_mc = self._param_row(trig_g, 1, 0, 'Min Cross',
            sb_min=0.0, sb_max=60.0, sb_step=0.001, sb_dec=3, suffix=' s',
            sb_init=DEFAULT_MIN_CROSS)
        self._sb_min_total = self._param_row(trig_g, 2, 0, 'Min Total Cross',
            sb_min=0.0, sb_max=3600.0, sb_step=0.01, sb_dec=3, suffix=' s',
            sb_init=DEFAULT_MIN_TOTAL_CROSS)
        self._sb_hold = self._param_row(trig_g, 3, 0, 'Hold',
            sb_min=0.0, sb_max=60.0, sb_step=0.1, sb_dec=2, suffix=' s',
            sb_init=DEFAULT_HOLD)
        self._sb_pre = self._param_row(trig_g, 4, 0, 'Pre-Trigger',
            sb_min=0.0, sb_max=60.0, sb_step=0.1, sb_dec=2, suffix=' s',
            sb_init=DEFAULT_PRE_TRIG)
        self._sb_post_trig = self._param_row(trig_g, 5, 0, 'Post-Trigger',
            sb_min=0.0, sb_max=60.0, sb_step=0.1, sb_dec=2, suffix=' s',
            sb_init=DEFAULT_POST_TRIG)
        self._sb_maxr = self._param_row(trig_g, 6, 0, 'Max Rec',
            sb_min=1.0, sb_max=3600.0, sb_step=1.0, sb_dec=1, suffix=' s',
            sb_init=DEFAULT_MAX_REC)

        self._sb_mc.setToolTip('Min Cross: minimum time the signal must stay above the threshold to start a recording')
        self._sb_min_total.setToolTip(
            'Min Total Cross: minimum ACCUMULATED above-threshold duration '
            'over the whole event — files whose total crossing time is '
            'shorter are discarded instead of saved (0 = keep everything)')
        self._sb_hold.setToolTip('Hold: duration of silence after the signal drops before a recording is considered finished')
        self._sb_pre.setToolTip('Pre-Trigger: audio kept before the trigger point (lookback saved to the WAV)')
        self._sb_post_trig.setToolTip('Post-Trigger: audio kept after the last above-threshold sample (tail of the saved WAV)')
        self._sb_maxr.setToolTip('Max Rec: maximum length of a single WAV segment — longer events are split')

        # Band filter row (row 7)
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
        trig_g.addLayout(filt_row, 7, 0, 1, 3)

        # Detect mode row (row 8)
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
        trig_g.addLayout(detect_row, 8, 0, 1, 3)

        # Entropy threshold row (row 9)
        self._sb_entropy_thr = self._param_row(trig_g, 9, 0, 'Entropy Thr',
            sb_min=0.0, sb_max=1.0, sb_step=0.01, sb_dec=2, suffix='',
            sb_init=0.50)
        self._sb_entropy_thr.setEnabled(False)
        self._sb_entropy_thr.setToolTip('Spectral entropy threshold — triggers when entropy falls below this value (0 = pure tone, 1 = white noise)')

        # 2c: entropy debounce duration (row 10)
        self._sb_entropy_mc = self._param_row(trig_g, 10, 0, 'Entropy Min Dur',
            sb_min=0.0, sb_max=10.0, sb_step=0.05, sb_dec=2, suffix=' s',
            sb_init=0.0)
        self._sb_entropy_mc.setEnabled(False)
        self._sb_entropy_mc.setToolTip(
            'Entropy Min Duration: entropy must stay below the threshold '
            'continuously for this long before the spectral condition turns '
            'ON (debounce; 0 = instantaneous). Evaluated per FFT chunk; the '
            'amplitude Min Cross still applies to the combined detection.')

        self._combo_detect_mode.currentTextChanged.connect(self._on_detect_mode_changed)

        # Auto-calibrate row (row 11)
        self._btn_calibrate = QPushButton('Auto Calibrate')
        self._btn_calibrate.setObjectName('btn_small')
        self._btn_calibrate.setFixedWidth(110)
        self._btn_calibrate.setToolTip(
            'Measure ambient noise for 3 seconds and set threshold automatically')
        self._lbl_calib_status = QLabel('')
        self._lbl_calib_status.setObjectName('param_label')
        # Result text ('Threshold set to …') can be long — never let it
        # widen the TRIGGER panel; clip instead.
        self._lbl_calib_status.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Preferred)

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
        trig_g.addLayout(calib_row, 11, 0, 1, 3)

        # 4a (v3.3.0): the live-sync checkboxes are gone \u2014 bulk editing
        # is the all-streams table.
        sync_row = QHBoxLayout()
        sync_row.setSpacing(10)
        # 4b: side-by-side editable table of every stream's parameters.
        self._btn_config_table = QPushButton('All Streams Table\u2026')
        self._btn_config_table.setObjectName('btn_small')
        self._btn_config_table.setFixedWidth(140)
        self._btn_config_table.setToolTip(
            'Open a table showing all parameters of all streams side by '
            'side \u2014 double-click a cell to edit')
        sync_row.addWidget(self._btn_config_table)
        sync_row.addStretch()
        trig_g.addLayout(sync_row, 12, 0, 1, 3)

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

        # Single-column stack (label | control) to keep the DISPLAY panel
        # narrow — gain/floor/ceil occupy rows 0–2 (added via _param_row
        # at col 0), the rest continue down the same column.
        grid.addWidget(lbl_fft,            3, 0)
        grid.addWidget(self._combo_fft,    3, 1)
        grid.addWidget(lbl_win,            4, 0)
        grid.addWidget(self._combo_win,    4, 1)
        grid.addWidget(lbl_fscale,         5, 0)
        grid.addWidget(self._combo_fscale, 5, 1)
        grid.addWidget(lbl_dfl,            6, 0)
        grid.addWidget(self._sb_disp_freq_lo,  6, 1)
        grid.addWidget(lbl_dfh,            7, 0)
        grid.addWidget(self._sb_disp_freq_hi,  7, 1)
        grid.addWidget(lbl_buf,            8, 0)
        grid.addWidget(self._buf_combo,    8, 1)

        lbl_dmode = QLabel('View')
        lbl_dmode.setObjectName('param_label')
        lbl_dmode.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        self._combo_display_mode = QComboBox()
        self._combo_display_mode.addItems(['Spectrogram', 'Waveform', 'Both'])
        self._combo_display_mode.setCurrentText('Spectrogram')
        self._combo_display_mode.setFixedWidth(90)
        self._combo_display_mode.setToolTip('Visualization mode — Spectrogram, raw Waveform, or Both')
        grid.addWidget(lbl_dmode,                  9, 0)
        grid.addWidget(self._combo_display_mode,   9, 1)

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

    def _build_config_right_stack(self) -> QWidget:
        """Right-hand column of the config row: STATUS, OUTPUT, REFERENCE
        DATE and INPUT DEVICE stacked vertically as wide horizontal
        panels."""
        w = QWidget()
        w.setStyleSheet(f'background-color: {C["mantle"]};')
        w.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        outer = QVBoxLayout(w)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        status_box = self._build_status_box()

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
        self._btn_browse_out = QPushButton('Browse...')
        btn_browse = self._btn_browse_out
        btn_browse.setObjectName('btn_browse')
        # QSS min-width would defeat the fixed width — clear it.
        btn_browse.setStyleSheet('min-width: 0px; padding: 6px 6px;')
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
        # Long WASAPI device names must NOT dictate the layout's minimum
        # width: by default a combo's minimum tracks its longest item,
        # which let this one demand >1500px, overflow the fixed width of
        # a maximized window, and push the right-hand config panels off
        # screen. Cap the minimum at ~20 chars; the Expanding policy
        # still lets it use all the width actually available.
        self._device_combo.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self._device_combo.setMinimumContentsLength(16)
        self._device_combo.setToolTip('Audio input device used by this recording')
        self._populate_device_combo()
        self._btn_dev_refresh = QPushButton('Refresh')
        btn_refresh = self._btn_dev_refresh
        btn_refresh.setObjectName('btn_browse')
        # QSS min-width would defeat the fixed width — clear it.
        btn_refresh.setStyleSheet('min-width: 0px; padding: 6px 6px;')
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
        # Same minimum-width cap as the device combo: 'Both Channels'
        # etc. must not set the floor for the whole INPUT DEVICE panel.
        for combo in (self._chan_combo, self._trig_combo):
            combo.setSizeAdjustPolicy(
                QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(5)
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
        ref_h   = QHBoxLayout(ref_box)
        ref_h.setSpacing(8)
        ref_h.setContentsMargins(8, 4, 8, 4)
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
        ref_h.addWidget(self._chk_ref_date)
        ref_h.addSpacing(8)
        ref_h.addWidget(self._date_line)
        ref_h.addWidget(self._btn_pick_date)
        ref_h.addWidget(self._lbl_day_count)
        ref_h.addSpacing(14)
        ref_h.addWidget(lbl_dph_pfx)
        ref_h.addWidget(self._dph_prefix_edit)
        ref_h.addStretch()

        # Stack the four panels vertically; each spans the column width in
        # the (stretched) right-hand column of the config row.
        outer.addWidget(status_box)
        outer.addWidget(output_box)
        outer.addWidget(ref_box)
        outer.addWidget(device_box)
        outer.addStretch()
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
        self._btn_startup .clicked.connect(self._open_startup_prefs)
        self._btn_advanced.clicked.connect(self._open_advanced_settings)
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
        self._sb_min_total.valueChanged.connect(lambda _: self._write_trigger_params())
        self._sb_hold     .valueChanged.connect(lambda _: self._write_trigger_params())
        self._sb_pre .valueChanged.connect(lambda _: self._write_trigger_params())
        self._sb_post_trig.valueChanged.connect(lambda _: self._write_trigger_params())
        self._sb_maxr.valueChanged.connect(lambda _: self._write_trigger_params())
        self._combo_detect_mode.currentTextChanged.connect(lambda _: self._write_trigger_params())
        self._sb_entropy_thr.valueChanged.connect(lambda _: self._write_trigger_params())
        self._sb_entropy_mc.valueChanged.connect(lambda _: self._write_trigger_params())
        # 2a / 2b: recording mode + force-trigger toggle
        self._combo_rec_mode.currentTextChanged.connect(self._on_rec_mode_changed)
        self._btn_force_trig.toggled.connect(self._on_force_trigger_toggled)

        # Spectrogram display write-through
        self._sb_gain .valueChanged.connect(lambda _: self._write_spec_params())
        self._sb_floor.valueChanged.connect(lambda _: self._write_spec_params())
        self._sb_ceil .valueChanged.connect(lambda _: self._write_spec_params())
        self._combo_fft   .currentIndexChanged.connect(self._on_fft_params_changed)
        self._combo_win   .currentIndexChanged.connect(self._on_fft_params_changed)
        self._combo_fscale.currentTextChanged .connect(self._on_freq_scale_changed)
        self._sb_disp_freq_lo.valueChanged.connect(self._on_disp_freq_changed)
        self._sb_disp_freq_hi.valueChanged.connect(self._on_disp_freq_changed)
        self._combo_display_mode.currentTextChanged.connect(self._on_display_mode_changed)

        # 4b: editable all-streams table.
        self._btn_config_table.clicked.connect(self._open_config_table)

        # Plot interaction (threshold drag, scroll-zoom, amp-scale menu) is
        # handled by the pyqtgraph ConfigPlotPanel signals wired in
        # _build_figure — no matplotlib canvas events to connect.

        # Sidebar
        self._sidebar.selection_changed.connect(self._switch_selection)
        self._sidebar.add_requested.connect(self._add_recording)
        self._sidebar.delete_requested.connect(self._remove_recording)
        self._sidebar.move_requested.connect(self._move_recording)
        self._sidebar.item_renamed.connect(self._on_item_renamed)
        # #28 / #29: sticky session-flag resets.
        self._sidebar.clear_sat_requested.connect(self._on_clear_sat)
        self._sidebar.clear_drops_requested.connect(self._on_clear_drops)
        # #43 / #44 / #48: sticky error-flag reset.
        self._sidebar.clear_errors_requested.connect(self._on_clear_errors)
        # Per-stream enable switch.
        self._sidebar.toggle_enabled_requested.connect(
            self._on_toggle_stream_enabled)
        # Per-stream parameter lock (icon on each item) + bulk lock/unlock.
        self._sidebar.toggle_lock_requested.connect(self._on_toggle_lock)
        self._sidebar.lock_all_requested.connect(self._on_lock_all)
        self._sidebar.unlock_all_requested.connect(self._on_unlock_all)

    # ──────────────────────────────────────────────────────────────────────
    # Write-through: widgets → selected entity
    # ──────────────────────────────────────────────────────────────────────

    def _write_trigger_params(self):
        e = self._sel
        if not e:
            return
        e.min_cross_sec = self._sb_mc.value()
        e.min_total_cross_sec = self._sb_min_total.value()
        e.hold_sec      = self._sb_hold.value()
        e.pre_trig_sec  = self._sb_pre.value()
        e.post_trig_sec = self._sb_post_trig.value()
        e.max_rec_sec   = self._sb_maxr.value()
        e.spectral_trigger_mode = self._combo_detect_mode.currentText()
        e.spectral_threshold    = self._sb_entropy_thr.value()
        e.entropy_min_cross_sec = self._sb_entropy_mc.value()
        self._mark_dirty()

    def _write_spec_params(self):
        e = self._sel
        if not e:
            return
        e.gain_db  = self._sb_gain.value()
        e.db_floor = self._sb_floor.value()
        e.db_ceil  = self._sb_ceil.value()
        self._mark_dirty()

    def _on_freq_filter_toggled(self, on: bool):
        self._sb_freq_lo.setEnabled(on)
        self._sb_freq_hi.setEnabled(on)
        e = self._sel
        if e:
            e.freq_filter_enabled = on
            e.bpf.reset()
            e.bpf_r.reset()
            self._mark_dirty()

    def _on_freq_filter_param(self, _val):
        e = self._sel
        if e:
            e.freq_lo = self._sb_freq_lo.value()
            e.freq_hi = self._sb_freq_hi.value()
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
        e.min_total_cross_sec = self._sb_min_total.value()
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
        e.entropy_min_cross_sec = self._sb_entropy_mc.value()
        e.rec_mode      = self._combo_rec_mode.currentText()
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
        _set(self._sb_min_total, e.min_total_cross_sec)
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
        # #50: validate the loaded folder too — a config saved while a
        # USB drive was mounted must not silently lose recordings when
        # the drive is unplugged before reopening Chirp.
        self._apply_folder_validation(e, e.output_dir)

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
        _set(self._sb_entropy_mc, e.entropy_min_cross_sec)
        ent_on = (e.spectral_trigger_mode != 'Amplitude Only')
        self._sb_entropy_thr.setEnabled(ent_on)
        self._sb_entropy_mc.setEnabled(ent_on)

        # 2a / 2b: recording mode combo + widget enable states + force
        # trigger button state for this stream.
        self._combo_rec_mode.blockSignals(True)
        self._combo_rec_mode.setCurrentText(
            getattr(e, 'rec_mode', 'Triggered'))
        self._combo_rec_mode.blockSignals(False)
        self._apply_rec_mode_ui(e)

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

        # Rebuild the pyqtgraph config panel for the newly selected
        # entity so its layout + threshold lines reflect this stream
        # (layout auto-corrects on the next tick too, but the threshold
        # lines are only set here / on drag).
        if getattr(self, '_config_panel', None) is not None and e is not None:
            self._config_panel.rebuild(e)
            self._config_panel.set_threshold(e.threshold)
            self._config_panel.set_spectral_threshold(e.spectral_threshold)

        # Reflect this stream's recognition color on the config-mode
        # swatch button + the config panel's frame.
        if e is not None:
            self._refresh_stream_color_ui(e)

        # Reflect this stream's parameter-lock state on the config-area
        # controls (disable lockable widgets, update the lock toggle,
        # freeze the threshold lines). Done last so it overrides the
        # enabled-state the widget population above may have set.
        self._apply_lock_ui(e)

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
        # 4d: monitor-follow — route the newly selected stream to the
        # audio monitor.
        if (self._chk_monitor_follow.isChecked()
                and 0 <= new_idx < len(self._entities)):
            self._apply_monitor_source(id(self._entities[new_idx]))
            self._sync_monitor_source_combo()

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
        self._sync_stream_colors()
        self._switch_selection(idx)
        self._refresh_monitor_source_combo()
        self._refresh_config_table()
        self._mark_dirty()

    def _next_free_color(self) -> str:
        """Pick the first palette color not already used by a stream,
        falling back to a position-based color when all are taken."""
        used = {e.color for e in self._entities if e.color}
        for c in STREAM_COLORS:
            if c not in used:
                return c
        return default_stream_color(len(self._entities))

    def _sync_stream_colors(self) -> None:
        """Assign a default color to any stream that lacks one, then push
        every stream's color to its sidebar item."""
        for i, e in enumerate(self._entities):
            if not getattr(e, 'color', ''):
                e.color = self._next_free_color()
            self._sidebar.set_item_color(i, e.color)

    def _on_change_stream_color(self, idx: int) -> None:
        """Open a color picker for the idx-th stream and apply the pick."""
        if not (0 <= idx < len(self._entities)):
            return
        e = self._entities[idx]
        initial = QColor(e.color) if e.color else QColor(C['blue'])
        col = QColorDialog.getColor(initial, self, 'Choose stream color')
        if not col.isValid():
            return
        e.color = col.name()
        self._sidebar.set_item_color(idx, e.color)
        # Reflect on the config-mode swatch button + panel frame for the
        # selected stream. View-mode tiles pull ``e.color`` on every tick
        # (update_all), so the grid repaints its rectangle on its own.
        if idx == self._selected_idx:
            self._refresh_stream_color_ui(e)
        self._mark_dirty()

    def _refresh_stream_color_ui(self, e) -> None:
        """Sync the config-mode color swatch button + config-panel frame
        to entity ``e``'s recognition color."""
        col = getattr(e, 'color', '') or C['surface1']
        btn = getattr(self, '_btn_stream_color', None)
        if btn is not None:
            btn.setStyleSheet(
                f'QPushButton {{ background-color: {col}; min-width: 0px; '
                f'min-height: 0px; padding: 0px; '
                f'border: 1px solid {C["surface1"]}; border-radius: 3px; }}'
                f'QPushButton:hover {{ border: 1px solid {C["text"]}; }}')
        panel = getattr(self, '_config_panel', None)
        if panel is not None:
            panel.set_color(getattr(e, 'color', '') or None)

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
            self._refresh_config_table()
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
            self._refresh_config_table()
            self._mark_dirty()

    # ── #28 / #29: sticky-flag reset handlers ────────────────────────────

    def _on_clear_sat(self, idx: int):
        """Clear the sticky saturation flag on the idx-th stream."""
        if 0 <= idx < len(self._entities):
            self._entities[idx].clear_saturation_flag()
            # Push a fresh update so the badge goes grey immediately
            # without waiting for the next plot tick.
            from chirp.ui.status_util import compose_saturation_state
            self._sidebar.update_item_saturation_sticky(
                idx, *compose_saturation_state(self._entities[idx]))

    def _on_clear_drops(self, idx: int):
        """Clear the sticky drop stats on the idx-th stream."""
        if 0 <= idx < len(self._entities):
            self._entities[idx].clear_drop_flag()
            self._sidebar.update_item_drop_sticky(idx, False, 0)

    def _update_error_sticky(self, idx: int, e) -> None:
        """#43 / #44 / #48: compose ingest / OS-drop / open / writer
        error state into a single sticky badge. Shared with the
        view-mode tile headers via ``status_util.compose_error_state``.
        """
        from chirp.ui.status_util import compose_error_state
        any_err, tip = compose_error_state(e)
        self._sidebar.update_item_error_sticky(idx, any_err, tip)

    def _on_clear_errors(self, idx: int):
        """#43 / #44 / #48: clear sticky error flags on the idx-th
        stream. Resets the entity's ingest error counters and the
        capture's OS-drop / open-error stats, plus the global
        writer-pool error stats (which aren't per-stream but are
        surfaced on every sidebar badge for visibility).
        """
        if 0 <= idx < len(self._entities):
            self._entities[idx].clear_error_flag()
            # Writer errors are global — clearing once clears for all.
            from chirp.recording import writer as _writer
            _writer.reset_error_stats()
            self._sidebar.update_item_error_sticky(
                idx, False, 'No pipeline errors recorded for this stream.')

    # ──────────────────────────────────────────────────────────────────────
    # Transport callbacks (operate on selected entity)
    # ──────────────────────────────────────────────────────────────────────

    def _on_start_acq(self):
        e = self._sel
        if e:
            for ent in self._entities:
                ent.reset_display()
            with self._busy_cursor():
                e.start_acq()
            # L6: a failed (re)open used to be a silently dead button —
            # the entity now latches the reason in last_ingest_error.
            if not e.acq_running:
                QMessageBox.warning(
                    self, 'Acquisition Error',
                    f'Could not start acquisition:\n'
                    f'{e.last_ingest_error or "unknown error"}')
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
        # Disabled streams are skipped by the bulk START actions (their
        # per-stream buttons still work for deliberate one-off use).
        active = [e for e in self._entities if e.stream_enabled]
        for e in active:
            e.reset_display()
        for e in active:
            e.start_acq()
        self._refresh_transport_ui()

    def _on_stop_all_acq(self):
        # Stop applies to every stream — including a disabled one the
        # user started manually; stopping is always safe.
        for e in self._entities:
            e.stop_acq()
        self._refresh_transport_ui()

    def _on_start_all_rec(self):
        for e in self._entities:
            if e.stream_enabled:
                e.start_rec()
        self._refresh_transport_ui()

    def _on_stop_all_rec(self):
        for e in self._entities:
            e.stop_rec()
        self._refresh_transport_ui()

    def _on_toggle_stream_enabled(self, idx: int):
        """Flip a stream's enable switch. Disabling also stops the
        stream immediately (acquisition + recording) so one click
        silences it; the configuration is fully preserved."""
        if not (0 <= idx < len(self._entities)):
            return
        e = self._entities[idx]
        e.stream_enabled = not e.stream_enabled
        if not e.stream_enabled:
            e.stop_acq()
        self._sidebar.set_item_stream_enabled(idx, e.stream_enabled)
        self._refresh_transport_ui()
        self._mark_dirty()

    # ── Per-stream parameter lock ─────────────────────────────────────────

    def _apply_lock_ui(self, e) -> None:
        """Reflect entity ``e``'s lock state on the config-area controls:
        disable the lockable widgets, update the STATUS lock toggle, and
        freeze the draggable threshold lines. Display params, the monitor
        bar and transport buttons are intentionally left untouched."""
        locked = bool(getattr(e, 'params_locked', False)) if e is not None else False
        for wdg in getattr(self, '_lockable_widgets', ()):
            wdg.setEnabled(not locked)
        # The stereo trigger combo is only meaningful in Stereo mode; keep
        # that gating even when unlocking (don't force-enable it).
        if not locked and e is not None:
            self._trig_combo.setEnabled(e.channel_mode == 'Stereo')
        btn = getattr(self, '_btn_lock_cfg', None)
        if btn is not None:
            btn.setText('\U0001f512 Locked' if locked else '\U0001f513 Unlocked')
            btn.setStyleSheet(
                f'QPushButton {{ color: {C["peach"] if locked else C["subtext"]}; '
                f'font-weight: bold; }}')
        panel = getattr(self, '_config_panel', None)
        if panel is not None:
            panel.set_threshold_locked(locked)

    def _on_toggle_lock(self, idx: int) -> None:
        """Toggle one stream's parameter lock. Unlocking is confirmed with
        a dialog naming the stream so a stream locked by one user isn't
        unlocked by another by mistake."""
        if not (0 <= idx < len(self._entities)):
            return
        e = self._entities[idx]
        if e.params_locked:
            if not self._confirm_unlock([e]):
                return
            e.params_locked = False
        else:
            e.params_locked = True
        self._sidebar.set_item_params_locked(idx, e.params_locked)
        if idx == self._selected_idx:
            self._apply_lock_ui(e)
        self._mark_dirty()

    def _on_lock_all(self) -> None:
        """Lock every stream's configuration."""
        if not self._entities:
            return
        for i, e in enumerate(self._entities):
            e.params_locked = True
            self._sidebar.set_item_params_locked(i, True)
        if self._sel is not None:
            self._apply_lock_ui(self._sel)
        self._mark_dirty()

    def _on_unlock_all(self) -> None:
        """Unlock every currently-locked stream, after one confirmation
        that names them."""
        locked = [e for e in self._entities if e.params_locked]
        if not locked:
            return
        if not self._confirm_unlock(locked):
            return
        for i, e in enumerate(self._entities):
            if e.params_locked:
                e.params_locked = False
                self._sidebar.set_item_params_locked(i, False)
        if self._sel is not None:
            self._apply_lock_ui(self._sel)
        self._mark_dirty()

    def _confirm_unlock(self, entities: list) -> bool:
        """Confirmation dialog for unlocking one or more streams. Names
        the stream(s) so the user can be sure they're unlocking the right
        one. Returns True if the user confirmed."""
        names = ', '.join(f'“{e.name}”' for e in entities)
        if len(entities) == 1:
            msg = (f'Unlock the parameters for stream {names}?\n\n'
                   'Its configuration will become editable again.')
        else:
            msg = (f'Unlock the parameters for these {len(entities)} '
                   f'streams?\n\n{names}\n\n'
                   'Their configuration will become editable again.')
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle('Confirm unlock')
        box.setText(msg)
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setDefaultButton(QMessageBox.No)
        return box.exec_() == QMessageBox.Yes

    def _on_reset_params(self):
        e = self._sel
        if not e:
            return
        self._sb_thr      .setValue(DEFAULT_THRESHOLD)
        self._sb_mc       .setValue(DEFAULT_MIN_CROSS)
        self._sb_min_total.setValue(DEFAULT_MIN_TOTAL_CROSS)
        self._sb_hold     .setValue(DEFAULT_HOLD)
        self._sb_post_trig.setValue(DEFAULT_POST_TRIG)
        self._sb_maxr     .setValue(DEFAULT_MAX_REC)
        self._sb_pre      .setValue(DEFAULT_PRE_TRIG)
        self._chk_freq.setChecked(False)
        self._sb_freq_lo.setValue(DEFAULT_FREQ_LO)
        self._sb_freq_hi.setValue(DEFAULT_FREQ_HI)
        self._combo_detect_mode.setCurrentText('Amplitude Only')
        self._sb_entropy_thr.setValue(0.5)
        self._sb_entropy_mc.setValue(0.0)
        self._combo_rec_mode.setCurrentText('Triggered')
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
        # The line edits above write through on ``editingFinished``,
        # which setText() does not emit — push them into the entity by
        # hand or Reset would leave the old folder/prefix/suffix live
        # (and unsaved-but-different from what the fields show).
        self._on_folder_changed()
        self._on_prefix_changed()
        self._on_suffix_changed()
        self._on_dph_prefix_changed()
        self._mark_dirty()

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
                'use_opengl':   self._pg_use_opengl,
                'active_only':  self._vm_active_only,
                'fill_order':   self._vm_fill_order,
            },
            monitor=self._build_monitor_settings(),
            audio=self._build_audio_settings(),
        )

    def _build_audio_settings(self) -> dict:
        """Capture-engine tuning, as the next stream would open, plus the
        inserted-silence auto-recovery settings."""
        from chirp.audio import shared_stream as _shared
        from chirp.dsp import envelope as _env
        blocksize, latency, exclusive = _shared.current_params()
        method, cutoff = _env.current_params()
        out = dict(self._audio_cfg)
        out['capture_blocksize'] = blocksize
        out['capture_latency'] = latency
        out['capture_exclusive'] = exclusive
        out['envelope_method'] = method
        out['envelope_cutoff_hz'] = cutoff
        return out

    def _write_settings_to_path(self, path: str, data: dict) -> bool:
        # #52: atomic settings write. Serialize to JSON in memory
        # first so a formatting error can't leave the target
        # truncated. Then write to a sibling ``<path>.tmp``, fsync,
        # and ``os.replace`` onto the canonical path — either the
        # previous good config survives or the new one fully
        # replaces it. A crash between truncation and json.dump used
        # to wipe the user's settings file.
        tmp_path = path + '.tmp'
        try:
            payload = json.dumps(data, indent=2, ensure_ascii=False)
            with open(tmp_path, 'w', encoding='utf-8') as f:
                f.write(payload)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    # Some filesystems / platforms don't support
                    # fsync on writable text-mode file objects;
                    # fall back to relying on the OS flush.
                    pass
            os.replace(tmp_path, path)
            self._current_config_path = path
            self._remember_last_config(path)
            self._mark_clean()  # #11 / c22
            return True
        except Exception as exc:
            # Best-effort cleanup of the tmp sibling so a failed save
            # doesn't leak files next to the user's config.
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
            QMessageBox.warning(self, 'Save Error', f'Could not save settings:\n{exc}')
            return False

    def _save_settings(self) -> bool:
        """Save to current path if known, otherwise prompt. Returns True
        when the config was written, False if the write failed or the
        user cancelled the path prompt (used by the quit flow)."""
        if self._current_config_path:
            return self._write_settings_to_path(
                self._current_config_path, self._build_settings_data())
        return self._save_settings_as()

    def _save_settings_as(self) -> bool:
        """Always prompt for a save path. Returns True when written,
        False if the user cancelled the dialog or the write failed."""
        data = self._build_settings_data()
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Settings', '', 'Chirp Settings (*.json);;All Files (*)')
        if not path:
            return False
        if not path.endswith('.json'):
            path += '.json'
        return self._write_settings_to_path(path, data)

    # ── Startup configuration preference ───────────────────────────────

    @staticmethod
    def _remember_last_config(path: str) -> None:
        """Record ``path`` as the most-recently used config so the
        'last configuration' startup mode can reload it next launch.
        Best-effort — never raises (QSettings access is guarded)."""
        from chirp.config import startup as _startup
        _startup.set_last_config(path)

    def _apply_startup_config(self) -> None:
        """Decide what to load at launch based on the persisted startup
        preference (empty / last / pinned file). Falls back to a single
        fresh recording whenever the chosen source is unset, missing, or
        fails to load — the app must always come up usable."""
        from chirp.config import startup as _startup
        path = _startup.resolve_startup_path()
        loaded = False
        if path:
            if os.path.exists(path):
                loaded = self._load_settings_from_path(path, silent=True)
            else:
                print(f'[Chirp] startup config not found, starting empty: '
                      f'{path!r}')
        if not loaded:
            # Empty config (historical default) or fallback.
            self._add_recording()

    # ── Advanced (capture engine + auto-recovery) ─────────────────────

    def _open_advanced_settings(self):
        """Capture-engine tuning and inserted-silence auto-recovery.

        These live here rather than in the per-stream panels because
        they are properties of the machine and its audio hardware, not
        of any one recording: the buffer sizes apply to every capture
        stream, and the recovery watchdog acts on a whole device.
        """
        from chirp.audio import shared_stream as _shared
        from chirp.constants import (CAPTURE_BLOCKSIZE_MAX,
                                     CAPTURE_BLOCKSIZE_MIN)
        from chirp.dsp import envelope as _env

        cur_bs, cur_lat, cur_excl = _shared.current_params()
        cur_env_method, cur_env_cut = _env.current_params()
        cfg = dict(self._audio_cfg)

        dlg = QDialog(self)
        dlg.setWindowTitle('Advanced Settings')
        outer = QVBoxLayout(dlg)
        # Four explained groups no longer fit a laptop screen at once, so
        # the groups scroll and the OK/Cancel row stays pinned below.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        page = QWidget()
        v = QVBoxLayout(page)
        scroll.setWidget(page)
        outer.addWidget(scroll, 1)

        # ── Capture engine ────────────────────────────────────────────
        cap_box = QGroupBox('CAPTURE ENGINE')
        cap_form = QGridLayout(cap_box)
        row = 0

        cap_form.addWidget(QLabel('Input buffer (latency):'), row, 0)
        sb_lat = QDoubleSpinBox()
        sb_lat.setRange(0.0, 2.0)
        sb_lat.setSingleStep(0.05)
        sb_lat.setDecimals(2)
        sb_lat.setSuffix(' s')
        sb_lat.setSpecialValueText('device default')
        sb_lat.setValue(float(cur_lat) if isinstance(cur_lat, (int, float))
                        else 0.0)
        cap_form.addWidget(sb_lat, row, 1)
        row += 1
        hint_lat = QLabel(
            'How late the machine may be before captured audio is lost.\n'
            'This is the setting that protects against dropouts and\n'
            'inserted silence. "device default" can be as little as 10 ms\n'
            'on WASAPI — an explicit 0.25–0.5 s is far safer.')
        hint_lat.setStyleSheet(f'color: {C["subtext"]};')
        cap_form.addWidget(hint_lat, row, 0, 1, 2)
        row += 1

        cap_form.addWidget(QLabel('Callback block size:'), row, 0)
        cb_bs = QComboBox()
        sizes = [b for b in (1024, 2048, 4096, 8192, 16384, 32768, 65536)
                 if CAPTURE_BLOCKSIZE_MIN <= b <= CAPTURE_BLOCKSIZE_MAX]
        for b in sizes:
            cb_bs.addItem(f'{b} frames', b)
        idx = cb_bs.findData(int(cur_bs))
        cb_bs.setCurrentIndex(idx if idx >= 0 else 0)
        cap_form.addWidget(cb_bs, row, 1)
        row += 1
        hint_bs = QLabel(
            'How often Chirp is handed audio. Larger = fewer wake-ups\n'
            '(less chance of being late) but more delay before the\n'
            'monitor and spectrogram update. Does NOT affect recordings.')
        hint_bs.setStyleSheet(f'color: {C["subtext"]};')
        cap_form.addWidget(hint_bs, row, 0, 1, 2)
        row += 1

        chk_excl = QCheckBox('WASAPI exclusive mode (bypass the Windows '
                             'audio mixer)')
        chk_excl.setChecked(bool(cur_excl))
        cap_form.addWidget(chk_excl, row, 0, 1, 2)
        row += 1
        hint_excl = QLabel(
            'Hands the input endpoint to Chirp alone, so captured audio\n'
            'comes straight from the driver instead of through the shared\n'
            'Windows audio engine — the layer that inserts the silent gaps.\n'
            'While it is on, no other application can use that input, and\n'
            'the hardware must accept the stream format natively. Only\n'
            'WASAPI devices support it; on MME / DirectSound / WDM-KS\n'
            'entries the request is logged and ignored. If an exclusive\n'
            'open is refused, Chirp opens shared and logs it — check\n'
            'chirp_errors.log for "mode=EXCLUSIVE" to confirm it took.')
        hint_excl.setStyleSheet(f'color: {C["subtext"]};')
        cap_form.addWidget(hint_excl, row, 0, 1, 2)
        row += 1

        lbl_apply = QLabel('All three take effect the next time a stream '
                           'opens (Stop Acq → Start Acq, or a config load).')
        lbl_apply.setStyleSheet(f'color: {C["peach"]};')
        cap_form.addWidget(lbl_apply, row, 0, 1, 2)
        v.addWidget(cap_box)

        # ── Auto-recovery ─────────────────────────────────────────────
        rec_box = QGroupBox('INSERTED-SILENCE AUTO-RECOVERY')
        rec_form = QGridLayout(rec_box)
        r = 0
        chk = QCheckBox('Restart acquisition automatically when the input '
                        'is being zero-filled')
        chk.setChecked(bool(cfg.get('auto_recover_zero_runs', True)))
        rec_form.addWidget(chk, r, 0, 1, 2)
        r += 1

        rec_form.addWidget(QLabel('Trigger above:'), r, 0)
        sb_pct = QDoubleSpinBox()
        sb_pct.setRange(0.1, 100.0)
        sb_pct.setSingleStep(1.0)
        sb_pct.setDecimals(1)
        sb_pct.setSuffix(' % digital zeros')
        sb_pct.setValue(float(cfg.get('zero_recover_percent', 5.0)))
        rec_form.addWidget(sb_pct, r, 1)
        r += 1

        rec_form.addWidget(QLabel('Sustained for:'), r, 0)
        sb_sec = QDoubleSpinBox()
        sb_sec.setRange(1.0, 600.0)
        sb_sec.setSingleStep(5.0)
        sb_sec.setDecimals(0)
        sb_sec.setSuffix(' s')
        sb_sec.setValue(float(cfg.get('zero_recover_seconds', 15.0)))
        rec_form.addWidget(sb_sec, r, 1)
        r += 1

        rec_form.addWidget(QLabel('Wait between attempts:'), r, 0)
        sb_cool = QDoubleSpinBox()
        sb_cool.setRange(5.0, 3600.0)
        sb_cool.setSingleStep(30.0)
        sb_cool.setDecimals(0)
        sb_cool.setSuffix(' s')
        sb_cool.setValue(float(cfg.get('zero_recover_cooldown_sec', 120.0)))
        rec_form.addWidget(sb_cool, r, 1)
        r += 1

        hint_rec = QLabel(
            'A capture session can latch into zero-filling the audio; the\n'
            'only reliable reset is stopping every stream that uses that\n'
            'device and starting again. This does it for you — recording\n'
            'resumes automatically and each intervention is logged.')
        hint_rec.setStyleSheet(f'color: {C["subtext"]};')
        rec_form.addWidget(hint_rec, r, 0, 1, 2)
        v.addWidget(rec_box)

        # ── Capture-stall (RDP) reconnect ─────────────────────────────
        stall_box = QGroupBox('LOST-DEVICE AUTO-RECONNECT')
        stall_form = QGridLayout(stall_box)
        chk_stall = QCheckBox(
            'Reopen a stream automatically when its device stops '
            'delivering audio')
        chk_stall.setChecked(
            bool(cfg.get('auto_recover_capture_stall', True)))
        stall_form.addWidget(chk_stall, 0, 0, 1, 2)
        hint_stall = QLabel(
            'A remote-desktop connect/disconnect can rip out the Windows\n'
            'audio endpoint; this closes the dead stream and reopens the\n'
            'device by name. It is not free — the teardown costs audio —\n'
            'so if the device recovers on its own, or you are using a\n'
            'WDM-KS input (which survives RDP session churn), turning\n'
            'this off avoids reconnects that do more harm than good.\n'
            'Detection is unaffected: the stream is still marked stalled,\n'
            'the "!" badge still lights, and chirp_errors.log still gets\n'
            'the capture_dead line — you just reconnect it yourself with\n'
            'Stop Acq / Start Acq.')
        hint_stall.setStyleSheet(f'color: {C["subtext"]};')
        stall_form.addWidget(hint_stall, 1, 0, 1, 2)
        v.addWidget(stall_box)

        # ── Amplitude envelope ────────────────────────────────────────
        env_box = QGroupBox('AMPLITUDE ENVELOPE (TRIGGER)')
        env_form = QGridLayout(env_box)
        er = 0
        env_form.addWidget(QLabel('Estimator:'), er, 0)
        cb_env = QComboBox()
        cb_env.addItem('Hilbert (analytic signal)', 'hilbert')
        cb_env.addItem('Rectify + low-pass', 'rectify')
        _ei = cb_env.findData(cur_env_method)
        cb_env.setCurrentIndex(_ei if _ei >= 0 else 0)
        env_form.addWidget(cb_env, er, 1)
        er += 1

        env_form.addWidget(QLabel('Low-pass cutoff:'), er, 0)
        sb_env_cut = QDoubleSpinBox()
        sb_env_cut.setRange(_env.ENVELOPE_CUTOFF_MIN_HZ,
                            _env.ENVELOPE_CUTOFF_MAX_HZ)
        sb_env_cut.setSingleStep(5.0)
        sb_env_cut.setDecimals(1)
        sb_env_cut.setSuffix(' Hz')
        sb_env_cut.setValue(float(cur_env_cut))
        env_form.addWidget(sb_env_cut, er, 1)
        er += 1

        hint_env = QLabel(
            'How the trigger measures "how loud is it right now" from the\n'
            'band-filtered signal. Hilbert is exact and lag-free but is\n'
            'recomputed per block, so it has a small artifact at every\n'
            'block edge. Rectify + low-pass is the classic envelope\n'
            'follower: continuous across blocks, at the cost of a delay\n'
            'of roughly 1/cutoff. They cost the same to run — choose on\n'
            'artifact-vs-delay, not speed. The cutoff must sit below\n'
            'your signal band (to remove the tone itself) and above the\n'
            'rate at which the calls turn on and off. Both are scaled so\n'
            'a steady tone reads its true amplitude — switching does not\n'
            'move your thresholds. Applies to every stream on the next\n'
            'block; no restart needed.')
        hint_env.setStyleSheet(f'color: {C["subtext"]};')
        env_form.addWidget(hint_env, er, 0, 1, 2)
        v.addWidget(env_box)

        def _sync_env_enabled():
            sb_env_cut.setEnabled(cb_env.currentData() == 'rectify')
        cb_env.currentIndexChanged.connect(lambda _i: _sync_env_enabled())
        _sync_env_enabled()

        def _sync_rec_enabled():
            on = chk.isChecked()
            for w in (sb_pct, sb_sec, sb_cool):
                w.setEnabled(on)
        chk.toggled.connect(_sync_rec_enabled)
        _sync_rec_enabled()

        v.addStretch(1)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        outer.addWidget(bb)
        # Tall enough to show a group or two without the user resizing,
        # but capped so the dialog never grows past a small screen.
        dlg.resize(dlg.sizeHint().width(), 620)

        if dlg.exec_() != QDialog.Accepted:
            return

        latency = sb_lat.value()
        # 0 means "whatever the device says" — stored as 'high' so the
        # config keeps working with the string form the schema accepts.
        _shared.configure(cb_bs.currentData(),
                          latency if latency > 0 else 'high',
                          bool(chk_excl.isChecked()))
        env_method, env_cutoff = _env.configure(cb_env.currentData(),
                                                sb_env_cut.value())
        self._audio_cfg.update({
            'capture_exclusive': bool(chk_excl.isChecked()),
            'auto_recover_zero_runs': bool(chk.isChecked()),
            'zero_recover_percent': float(sb_pct.value()),
            'zero_recover_seconds': float(sb_sec.value()),
            'zero_recover_cooldown_sec': float(sb_cool.value()),
            'auto_recover_capture_stall': bool(chk_stall.isChecked()),
            'envelope_method': env_method,
            'envelope_cutoff_hz': env_cutoff,
        })
        self._zero_high_since.clear()
        self._mark_dirty()

    def _open_startup_prefs(self):
        """Modal to choose what Chirp loads on startup."""
        from chirp.config import startup as _startup
        dlg = QDialog(self)
        dlg.setWindowTitle('Startup Configuration')
        v = QVBoxLayout(dlg)
        v.addWidget(QLabel('When Chirp starts, load:'))

        rb_empty = QRadioButton('Empty configuration (one fresh recording)')
        rb_last  = QRadioButton('Last used configuration')
        rb_file  = QRadioButton('A specific configuration file:')
        grp = QButtonGroup(dlg)
        for rb in (rb_empty, rb_last, rb_file):
            grp.addButton(rb)
            v.addWidget(rb)

        file_row = QHBoxLayout()
        file_edit = QLineEdit(_startup.get_startup_file())
        file_btn  = QPushButton('Browse…')
        file_row.addWidget(file_edit, 1)
        file_row.addWidget(file_btn)
        v.addLayout(file_row)

        last = _startup.get_last_config()
        lbl_last = QLabel(f'Last used: {last or "(none yet)"}')
        lbl_last.setStyleSheet(f'color: {C["subtext"]};')
        v.addWidget(lbl_last)

        {_startup.MODE_EMPTY: rb_empty,
         _startup.MODE_LAST:  rb_last,
         _startup.MODE_FILE:  rb_file}[_startup.get_startup_mode()].setChecked(True)

        def _sync_enabled():
            on = rb_file.isChecked()
            file_edit.setEnabled(on)
            file_btn.setEnabled(on)
        for rb in (rb_empty, rb_last, rb_file):
            rb.toggled.connect(_sync_enabled)
        _sync_enabled()

        def _browse():
            p, _ = QFileDialog.getOpenFileName(
                dlg, 'Choose Startup Configuration', file_edit.text(),
                'Chirp Settings (*.json *.chirp);;All Files (*)')
            if p:
                file_edit.setText(p)
        file_btn.clicked.connect(_browse)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        v.addWidget(bb)

        if dlg.exec_() != QDialog.Accepted:
            return
        if rb_file.isChecked():
            mode = _startup.MODE_FILE
        elif rb_last.isChecked():
            mode = _startup.MODE_LAST
        else:
            mode = _startup.MODE_EMPTY
        _startup.set_startup_mode(mode)
        _startup.set_startup_file(file_edit.text().strip())

    def _load_settings(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Load Settings', '', 'Chirp Settings (*.json *.chirp);;All Files (*)')
        if not path:
            return
        self._load_settings_from_path(path)

    def _load_settings_from_path(self, path: str, *, silent: bool = False) -> bool:
        """Read + apply a settings file at ``path``. Returns True on a
        successful load, False if the file could not be read/parsed or
        the schema rejected it.

        Split out from ``_load_settings`` so the startup path (see
        ``_apply_startup_config``) can reuse the exact same load/rebuild
        logic. ``silent`` suppresses the hard-error modals (used at
        startup, where a missing last/pinned config falls back to an
        empty config rather than nagging with a dialog); the
        load-with-warnings info modal is always shown."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as exc:
            if not silent:
                QMessageBox.warning(self, 'Load Error', f'Could not read file:\n{exc}')
            else:
                print(f'[Chirp] could not read startup config {path!r}: {exc}')
            return False

        # #55: route through ``load_settings_dict`` so version /
        # migration / unknown-key warnings actually run on real loads.
        # The previous code path validated only ``'recordings' in data``
        # and instantiated entities directly, leaving the schema
        # version guard, the migration chain, and the unknown-key
        # warnings as dead code (only exercised by tests).
        from chirp.config import load_settings_dict, parse_audio_settings
        # Capture tuning must be applied BEFORE the entities are built —
        # constructing a RecordingEntity opens its capture, which is
        # when the blocksize / latency take effect.
        audio_warnings: list[str] = []
        try:
            audio_cfg, audio_warnings = parse_audio_settings(data)
            self._audio_cfg = dict(audio_cfg)
            self._zero_high_since.clear()
            from chirp.audio import shared_stream as _shared
            _shared.configure(audio_cfg.get('capture_blocksize'),
                              audio_cfg.get('capture_latency'),
                              audio_cfg.get('capture_exclusive'))
            from chirp.dsp import envelope as _env
            _env.configure(audio_cfg.get('envelope_method'),
                           audio_cfg.get('envelope_cutoff_hz'))
        except Exception as exc:
            print(f'[Chirp] could not apply audio settings: {exc}')
        try:
            entities, view_mode, monitor, schema_warnings = load_settings_dict(data)
        except ValueError as exc:
            # Invalid format / future-version file / pre-migration
            # error. Leave the existing UI state intact — bailing here
            # is safer than half-loading a config that the schema
            # already rejected.
            if not silent:
                QMessageBox.warning(
                    self, 'Load Error',
                    f'Could not load settings:\n{exc}')
            else:
                print(f'[Chirp] startup config {path!r} rejected: {exc}')
            return False

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

        # Restore view mode globals (already validated by
        # load_settings_dict — view_mode is a dict with the right keys
        # even if the on-disk file had garbage).
        self._vm_n_cols = view_mode['columns']
        self._vm_panel_height = view_mode['panel_height']
        self._pg_use_opengl = bool(view_mode.get('use_opengl', True))
        self._vm_active_only = bool(view_mode.get('active_only', True))
        self._vm_fill_order = view_mode.get('fill_order', 'column')
        # Drop any grid built with the previous OpenGL setting so it is
        # recreated with the new one on next view-mode entry, and swap
        # the live config-panel viewport to match (clamped to raster
        # while a remote session drives the display).
        self._pg_grid = None
        self._apply_render_backend()
        self._vm_cols_spin.blockSignals(True)
        self._vm_cols_spin.setValue(self._vm_n_cols)
        self._vm_cols_spin.blockSignals(False)
        self._vm_height_spin.blockSignals(True)
        self._vm_height_spin.setValue(self._vm_panel_height)
        self._vm_height_spin.blockSignals(False)
        self._chk_vm_active_only.blockSignals(True)
        self._chk_vm_active_only.setChecked(self._vm_active_only)
        self._chk_vm_active_only.blockSignals(False)
        self._vm_order_combo.blockSignals(True)
        self._vm_order_combo.setCurrentIndex(
            0 if self._vm_fill_order == 'column' else 1)
        self._vm_order_combo.blockSignals(False)

        # Per-entity device warnings come back inside ``schema_warnings``
        # already (load_settings_dict appends RecordingEntity.from_dict's
        # warning). Surface them all in a single modal at the end.
        warnings = list(audio_warnings) + list(schema_warnings)
        for ent in entities:
            # #7: re-wire the monitor on every freshly-loaded entity.
            ent.set_monitor(self._monitor)
            self._entities.append(ent)
            idx = self._sidebar.add_item(ent.name)
            self._sidebar.set_item_stream_enabled(idx, ent.stream_enabled)
            self._sidebar.set_item_params_locked(idx, ent.params_locked)
        # Assign default colors to any stream that lacks one and push
        # every color to its sidebar item.
        self._sync_stream_colors()
        # Rebuild monitor-source combo from the loaded entities.
        self._refresh_monitor_source_combo()
        # Restore the persisted audio-monitor settings now that the
        # entities exist (source is resolved by list position).
        self._apply_monitor_settings(monitor)

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
            self._refresh_transport_ui()

        # If in view mode, rebuild the pyqtgraph view grid.
        if self._view_mode:
            self._rebuild_view()

        self._current_config_path = path
        self._remember_last_config(path)
        self._mark_clean()  # #11 / c22
        self._timer.start()

        if warnings:
            # Combine schema warnings (unknown keys, version notes,
            # device fallbacks) into one modal. Cap the displayed
            # count at 20 so a really gnarly file doesn't produce a
            # screen-tall dialog.
            shown = warnings[:20]
            extra = len(warnings) - len(shown)
            body = '\n'.join(f'• {w}' for w in shown)
            if extra > 0:
                body += f'\n\n…and {extra} more warning(s).'
            QMessageBox.information(
                self, 'Settings Loaded with Warnings',
                f'Settings loaded with {len(warnings)} warning(s):\n\n{body}')
        return True

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

        # 2b: force-trigger availability follows REC state + rec mode.
        self._refresh_force_trig_button(e)

    # ──────────────────────────────────────────────────────────────────────
    # Threshold sync
    # ──────────────────────────────────────────────────────────────────────

    def _on_thr_spinbox(self, val: float):
        e = self._sel
        if e:
            e.threshold = val
            self._mark_dirty()
        self._sync_thr_line(val)

    def _sync_thr_line(self, val: float):
        # Phase 4b: drive the pyqtgraph config panel's threshold line.
        if getattr(self, '_config_panel', None) is not None:
            self._config_panel.set_threshold(val)

    def _sync_entropy_thr_line(self, val: float):
        if getattr(self, '_config_panel', None) is not None:
            self._config_panel.set_spectral_threshold(val)

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
        # M8: reset the entity's envelope-peak accumulator so stale
        # peaks from before the calibration window don't leak in.
        e.consume_env_peak()
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

        # M8: collect the max trigger-ENVELOPE peak since the last tick.
        # Every ingested chunk contributes (the entity accumulates the
        # scalar on its DSP thread), and it is the same statistic the
        # trigger compares against the threshold — the old code sampled
        # |filtered| from abs_amp_buffer once per 100 ms, missing most
        # chunks and systematically underestimating the noise floor.
        peak = e.consume_env_peak()
        if peak is not None:
            self._calib_samples.append(peak)

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
            self._mark_dirty()
        # Silent so the spinbox handler doesn't fight the value we just
        # wrote; the dirty mark above is what that handler would have
        # done.
        self._set_thr_silent(new_threshold)
        self._sync_thr_line(new_threshold)

        self._lbl_calib_status.setText(
            f'Done: noise={noise_floor:.4f}, thr={new_threshold:.3f}')
        self._lbl_calib_status.setStyleSheet(f'color: {C["green"]};')
        QTimer.singleShot(5000, lambda: (
            self._lbl_calib_status.setText(''),
            self._lbl_calib_status.setStyleSheet(''),
        ))

    # Phase 4b: matplotlib mouse handlers (threshold drag, scroll-zoom,
    # amp-scale right-click) were removed. Threshold drag is now a
    # movable pyqtgraph InfiniteLine (ConfigPlotPanel.thresholdChanged /
    # spectralThresholdChanged), scroll-zoom is the pyqtgraph ViewBox's
    # built-in wheel zoom, and the amp-scale toggle is the panel's
    # context menu (ampScaleChanged).

    def _on_amp_scale_menu(self, value: str) -> None:
        """Phase 4b: the config panel's amplitude-scale context menu was
        used. Apply to the selected entity."""
        e = self._sel
        if e is not None:
            self._set_amp_scale(e, value)

    def _apply_amp_scale_to_axes(self, e) -> None:
        """Shim (Phase 4b): the pyqtgraph config panel owns the amp-axis
        scale now. Rebuild it so the Y range / label / threshold line
        reflect ``e.amp_scale``. Kept under its old name because several
        handlers still call it (e.g. ``_load_params_from_entity``)."""
        if e is None or getattr(self, '_config_panel', None) is None:
            return
        if not self._view_mode and e is self._sel:
            self._config_panel.rebuild(e)
            self._config_panel.set_threshold(e.threshold)

    def _set_amp_scale(self, e, scale: str) -> None:
        """Pick Linear or Log (dB) for ``e``. Config mode rebuilds the
        panel; view mode needs nothing (each tile reads ``amp_scale`` on
        every render tick)."""
        if scale not in ('linear', 'log'):
            return
        if scale == getattr(e, 'amp_scale', 'linear'):
            return
        e.amp_scale = scale
        self._mark_dirty()
        if not self._view_mode and e is self._sel:
            self._apply_amp_scale_to_axes(e)

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
        self._mark_dirty()

    # ── 4b: all-streams config table ─────────────────────────────────

    def _open_config_table(self):
        from chirp.ui.config_table import ConfigTableDialog
        if getattr(self, '_config_table_dlg', None) is None:
            self._config_table_dlg = ConfigTableDialog(self)
        self._config_table_dlg.show()
        self._config_table_dlg.raise_()
        self._config_table_dlg.activateWindow()

    def _refresh_config_table(self):
        """Refresh the table if it's open (entity added/removed/renamed)."""
        dlg = getattr(self, '_config_table_dlg', None)
        if dlg is not None and dlg.isVisible():
            dlg.refresh()

    def _apply_table_edit(self, idx: int, key: str, value):
        """Apply one table edit to entity ``idx`` through the same code
        paths the per-stream widgets use (side effects included)."""
        if not (0 <= idx < len(self._entities)):
            return
        e = self._entities[idx]
        if key == 'spec_nperseg':
            e.change_fft_params(int(value), e.spec_window)
        elif key == 'spec_window':
            e.change_fft_params(e.spec_nperseg, value)
        elif key == 'display_seconds':
            e.change_display_seconds(float(value))
        else:
            if key in ('threshold', 'min_cross_sec', 'min_total_cross_sec',
                       'hold_sec',
                       'pre_trig_sec', 'post_trig_sec', 'max_rec_sec',
                       'freq_lo', 'freq_hi', 'spectral_threshold',
                       'entropy_min_cross_sec', 'gain_db', 'db_floor',
                       'db_ceil', 'display_freq_lo', 'display_freq_hi'):
                value = float(value)
            setattr(e, key, value)
            if key in ('freq_scale', 'display_freq_lo', 'display_freq_hi'):
                e.rebuild_freq_mapping()
            if key in ('freq_filter_enabled', 'freq_lo', 'freq_hi'):
                e.bpf.reset()
                e.bpf_r.reset()
            if key == 'stream_enabled':
                # Same semantics as the sidebar On/Off button:
                # disabling stops the stream immediately.
                if not value:
                    e.stop_acq()
                self._sidebar.set_item_stream_enabled(idx, bool(value))
                self._refresh_transport_ui()
        self._mark_dirty()
        # Keep the per-stream panel in sync when the edited stream is
        # the selected one.
        if idx == self._selected_idx:
            self._load_params_from_entity(idx)

    def _on_display_mode_changed(self, mode: str):
        e = self._sel
        if not e:
            return
        e.display_mode = mode
        self._mark_dirty()

    def _on_detect_mode_changed(self, mode: str):
        spectral = (mode != 'Amplitude Only')
        self._sb_entropy_thr.setEnabled(spectral)
        self._sb_entropy_mc.setEnabled(spectral)
        self._write_trigger_params()

    # ── 2a / 2b: recording mode + force trigger ─────────────────────────

    def _on_rec_mode_changed(self, mode: str):
        e = self._sel
        if not e:
            return
        e.rec_mode = mode
        self._apply_rec_mode_ui(e)
        self._mark_dirty()

    def _apply_rec_mode_ui(self, e) -> None:
        """Grey out the trigger-detection params in Continuous mode —
        only Max Rec matters there. Force Trigger is a Triggered-mode
        tool (Continuous already records everything)."""
        continuous = (getattr(e, 'rec_mode', 'Triggered') == 'Continuous')
        for wdg in (self._sb_mc, self._sb_min_total, self._sb_hold,
                    self._sb_pre,
                    self._sb_post_trig, self._combo_detect_mode,
                    self._chk_freq, self._btn_calibrate):
            wdg.setEnabled(not continuous)
        spectral = (e.spectral_trigger_mode != 'Amplitude Only')
        self._sb_entropy_thr.setEnabled(not continuous and spectral)
        self._sb_entropy_mc.setEnabled(not continuous and spectral)
        band_on = bool(e.freq_filter_enabled)
        self._sb_freq_lo.setEnabled(not continuous and band_on)
        self._sb_freq_hi.setEnabled(not continuous and band_on)
        self._refresh_force_trig_button(e)

    def _refresh_force_trig_button(self, e) -> None:
        """Force Trigger is enabled while REC is on in Triggered mode;
        its checked state mirrors the entity's flag."""
        enabled = bool(e is not None and e.rec_enabled
                       and getattr(e, 'rec_mode', 'Triggered') == 'Triggered')
        self._btn_force_trig.blockSignals(True)
        self._btn_force_trig.setEnabled(enabled)
        self._btn_force_trig.setChecked(
            bool(e is not None and e.force_rec_active))
        self._btn_force_trig.blockSignals(False)

    def _on_force_trigger_toggled(self, checked: bool):
        e = self._sel
        if e is None:
            return
        if checked and not e.rec_enabled:
            # Shouldn't happen (button disabled), but never force
            # without REC — the recorder would ignore the mask anyway.
            self._btn_force_trig.blockSignals(True)
            self._btn_force_trig.setChecked(False)
            self._btn_force_trig.blockSignals(False)
            return
        e.set_force_trigger(checked)

    def _on_freq_scale_changed(self, scale: str):
        e = self._sel
        if e:
            e.freq_scale = scale
            e.rebuild_freq_mapping()
            self._mark_dirty()

    def _on_fft_params_changed(self):
        e = self._sel
        if not e:
            return
        nperseg = self._combo_fft.currentData()
        window  = self._combo_win.currentData()
        if nperseg and window:
            e.change_fft_params(nperseg, window)
            self._mark_dirty()

    # ──────────────────────────────────────────────────────────────────────
    # Folder & device
    # ──────────────────────────────────────────────────────────────────────

    # #50: output-folder validation. The writer worker used to be the
    # first code that noticed a bad path — at which point the failure
    # surfaced as a swallowed stdout print while the transport UI
    # happily showed "REC RUNNING". Validate synchronously at every
    # user-visible entry point (browse / text-edit / config-load) and
    # stamp the result on the entity so the sidebar can light up.
    _FOLDER_INVALID_STYLE = 'QLineEdit { border: 1px solid #f38ba8; }'

    def _probe_output_dir(self, path: str) -> tuple[bool, str]:
        """#50: return ``(ok, reason)`` for ``path``.

        Checks (in order):
          1. non-empty string
          2. if the path doesn't exist, attempt ``os.makedirs(...,
             exist_ok=True)`` — first-run with the default
             ``./recordings`` is the common case and silently creating
             it is the right behaviour. Only fall through to "invalid"
             when the create fails (read-only filesystem, no permission,
             illegal name on this OS, etc.).
          3. ``os.path.isdir`` (catches the "exists but is a file" case)
          4. writable (best-effort ``open(w)`` on a ``.chirp_write_test``
             sibling, then cleanup) — catches removed / disconnected
             drives where the path string is fine but the mount is gone.
        """
        import os as _os
        if not isinstance(path, str) or not path.strip():
            return (False, 'output folder is empty')
        if not _os.path.exists(path):
            try:
                _os.makedirs(path, exist_ok=True)
            except OSError as exc:
                return (False, f'could not create directory: {exc}')
        if not _os.path.isdir(path):
            return (False, f'not a directory: {path!r}')
        probe = _os.path.join(path, '.chirp_write_test')
        try:
            with open(probe, 'w') as f:
                f.write('')
            try:
                _os.remove(probe)
            except OSError:
                pass
        except OSError as exc:
            return (False, f'not writable: {exc}')
        return (True, '')

    def _apply_folder_validation(self, e, path: str) -> None:
        """#50: run the probe, stamp ``output_dir_valid`` +
        ``output_dir_error`` on the entity, and style the textbox red
        on failure. Safe to call with ``e=None`` (style-only path)."""
        ok, reason = self._probe_output_dir(path)
        if e is not None:
            e.output_dir_valid = ok
            e.output_dir_error = None if ok else reason
        self._folder_edit.setStyleSheet(
            '' if ok else self._FOLDER_INVALID_STYLE)
        if not ok:
            self._folder_edit.setToolTip(reason)
        else:
            self._folder_edit.setToolTip('')

    def _on_browse(self):
        e = self._sel
        start = e.output_dir if e else RECORDINGS_DIR
        chosen = QFileDialog.getExistingDirectory(self, 'Select output folder', start)
        if chosen:
            if e:
                e.output_dir = chosen
                self._mark_dirty()
            self._folder_edit.setText(chosen)
            self._apply_folder_validation(e, chosen)

    def _on_folder_changed(self):
        text = self._folder_edit.text().strip()
        e = self._sel
        if e:
            e.output_dir = text if text else RECORDINGS_DIR
            self._mark_dirty()
        self._apply_folder_validation(e, e.output_dir if e else text)

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
            self._mark_dirty()

    def _on_ref_date_text_changed(self):
        e = self._sel
        if e and self._chk_ref_date.isChecked():
            d = self._parse_date_text()
            if d:
                e.ref_date = d
                days = (datetime.date.today() - d).days
                self._lbl_day_count.setText(f'Day: {days}')
                self._mark_dirty()

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
                self._mark_dirty()

    def _on_dph_prefix_changed(self):
        e = self._sel
        if e:
            e.dph_folder_prefix = self._dph_prefix_edit.text()
            self._mark_dirty()

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
        self._chan_combo.setEnabled(max_ch >= 2)
        need_ch = 2 if e.channel_mode != 'Mono' else 1
        with self._busy_cursor():
            ok = e.change_device(device_id, need_ch)
        # ``device_id`` is set by change_device even when the open
        # failed, and it is what to_dict() serializes — so the config
        # differs from the last saved file either way.
        self._mark_dirty()
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
        self._mark_dirty()

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

        with self._busy_cursor():
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
        self._mark_dirty()

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
                    self._mark_dirty()
                    return
            except Exception:
                pass

        if is_wav_sim:
            # Re-open the WAV with the new channel count.
            if e.wav_file_path:
                with self._busy_cursor():
                    e.use_wav_file(e.wav_file_path, loop=e.wav_loop)
        else:
            with self._busy_cursor():
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
        self._mark_dirty()

    def _on_trigger_mode_changed(self, mode: str):
        e = self._sel
        if e:
            e.trigger_mode = mode
            self._mark_dirty()

    def _on_sample_rate_changed(self, _index: int):
        e = self._sel
        if not e:
            return
        new_sr = self._sr_combo.currentData()
        if new_sr is None or new_sr == e.sample_rate:
            return
        # #46: ``change_sample_rate`` is a multi-second operation —
        # close stream, drain queue, rebuild filters/buffers, reopen.
        # Block re-entrancy (rapid wheel-scroll, chained signal)
        # explicitly: disable the combo until the rebuild completes,
        # and gate the body behind ``_sr_change_busy``. Without this
        # the prior call can be mid-``sd.InputStream.close()`` when
        # the next one tries to touch the stream — double-close at
        # best, PortAudio crash at worst.
        if getattr(self, '_sr_change_busy', False):
            return
        self._sr_change_busy = True
        self._sr_combo.setEnabled(False)
        # M4: wait cursor for the whole multi-second rebuild. (No-op
        # without a QApplication — stub-window unit tests.)
        _app = QApplication.instance()
        if _app is not None:
            QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
            QApplication.processEvents()
        try:
            e.change_sample_rate(new_sr)
            # (4a: the live-sync SR broadcast is gone with the sync
            # checkboxes — use the all-streams table for bulk changes.)
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
            # TODO#7: the entity followed the Nyquist (up or down) in
            # change_sample_rate — mirror its values into the spinboxes
            # instead of only clamping down, so the UI and the mapping
            # can't disagree.
            self._sb_disp_freq_lo.blockSignals(True)
            self._sb_disp_freq_lo.setValue(e.display_freq_lo)
            self._sb_disp_freq_lo.blockSignals(False)
            self._sb_disp_freq_hi.blockSignals(True)
            self._sb_disp_freq_hi.setValue(e.display_freq_hi)
            self._sb_disp_freq_hi.blockSignals(False)
            self._refresh_transport_ui()
            self._mark_dirty()
        finally:
            if _app is not None:
                QApplication.restoreOverrideCursor()
            self._sr_change_busy = False
            self._sr_combo.setEnabled(True)

    def _on_display_buffer_changed(self, _index: int):
        e = self._sel
        if not e:
            return
        new_secs = self._buf_combo.currentData()
        if new_secs is None or new_secs == e.display_seconds:
            return
        e.change_display_seconds(new_secs)
        self._refresh_transport_ui()
        self._mark_dirty()

    # ──────────────────────────────────────────────────────────────────────
    # View Mode
    # ──────────────────────────────────────────────────────────────────────

    def _toggle_view_mode(self):
        self._view_mode = not self._view_mode

        if self._view_mode:
            # Save current amplitude zoom (linear scale only — the dB
            # axis is fixed) from the config panel's viewbox.
            e = self._sel
            if e:
                ymax = self._config_panel.current_amp_ylim()
                if ymax is not None:
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
            # Phase 4: render all streams via the pyqtgraph/OpenGL grid.
            self._rebuild_view()
        else:
            # Hide view toolbar
            self._view_toolbar.hide()
            # Show sidebar + config panels
            self._sidebar.show()
            for w in self._config_widgets:
                w.show()
            # Swap the config panel back into the scroll area.
            self._restore_config_canvas()
            if self._sel:
                self._load_params_from_entity(self._selected_idx)
            # A maximized window sometimes fails to re-lay-out the batch
            # of just-re-shown config widgets at full width (symptom:
            # right-hand panels clipped until a manual minimize +
            # maximize). Kick an explicit layout pass once the show
            # events have settled.
            QTimer.singleShot(0, self._force_full_relayout)

    def _force_full_relayout(self) -> None:
        """Invalidate and re-activate the top-level layout so every
        re-shown widget is measured at the window's current size."""
        root = self.centralWidget()
        if root is None or root.layout() is None:
            return
        root.layout().invalidate()
        root.layout().activate()

    def _visible_view_entities(self) -> tuple[list, list[int]]:
        """3c: the (entities, indices) shown in the view grid — filtered
        to acquiring streams when 'Active only' is on."""
        pairs = [(e, i) for i, e in enumerate(self._entities)]
        if self._vm_active_only:
            pairs = [(e, i) for e, i in pairs if e.acq_running]
        ents = [p[0] for p in pairs]
        idxs = [p[1] for p in pairs]
        return ents, idxs

    def _rebuild_view(self) -> None:
        """Phase 4: build/populate the pyqtgraph view-mode grid and swap it
        into the central scroll area (detaching the config panel without
        deleting it)."""
        if self._pg_grid is None:
            from chirp.ui.central_plots import MultiStreamGrid
            self._pg_grid = MultiStreamGrid(use_opengl=self._gl_effective)
            # 3a: tile interactions — badges clear by ENTITY index; a
            # plain tile click re-targets the monitor when follow mode
            # is on (4d).
            self._pg_grid.clear_sat_requested.connect(self._on_clear_sat)
            self._pg_grid.clear_drops_requested.connect(self._on_clear_drops)
            self._pg_grid.clear_errors_requested.connect(self._on_clear_errors)
            self._pg_grid.tile_clicked.connect(self._on_view_tile_clicked)
        self._pg_grid.set_tile_height(self._vm_panel_height)
        ents, idxs = self._visible_view_entities()
        self._vm_visible_sig = tuple(id(e) for e in ents)
        self._pg_grid.rebuild(
            ents, cols=self._vm_n_cols, indices=idxs,
            fill_order=self._vm_fill_order,
            empty_hint=("No active streams — start acquisition or untick "
                        "'Active only'" if self._vm_active_only
                        else 'No recordings'))
        if self._canvas_scroll.widget() is not self._pg_grid:
            # takeWidget detaches the current widget (the config panel,
            # which we still hold via self._config_panel) without
            # deleting it.
            self._canvas_scroll.takeWidget()
            self._canvas_scroll.setWidget(self._pg_grid)

    def _refresh_view(self) -> None:
        """Phase 4: per-tick update of the pyqtgraph view-mode grid.
        Rebuilds the grid when the visible (active) set changed (3c)."""
        if self._pg_grid is None:
            return
        ents, _idxs = self._visible_view_entities()
        if tuple(id(e) for e in ents) != self._vm_visible_sig:
            self._rebuild_view()
            ents, _idxs = self._visible_view_entities()
        self._pg_grid.update_all(ents,
                                 monitor_source_id=self._monitor.source_id)

    def _restore_config_canvas(self) -> None:
        """Swap the pyqtgraph config panel back into the scroll area when
        leaving view mode (detaching the grid without deleting it)."""
        if self._canvas_scroll.widget() is not self._config_panel:
            self._canvas_scroll.takeWidget()
            self._canvas_scroll.setWidget(self._config_panel)
        e = self._sel
        if e is not None:
            self._config_panel.rebuild(e)
            self._config_panel.set_threshold(e.threshold)
            self._config_panel.set_spectral_threshold(e.spectral_threshold)
            self._refresh_stream_color_ui(e)

    def _on_threshold_dragged(self, thr_linear: float) -> None:
        """Phase 4b: the amplitude threshold line was dragged in the config
        panel — push it into the selected entity + spinbox (routing through
        the spinbox handler so shared-trigger broadcast still applies)."""
        e = self._sel
        if e is None:
            return
        # Update the spinbox; its valueChanged handler writes the entity,
        # broadcasts to all if shared-trigger is on, and marks dirty. The
        # line is already where the user left it, so re-sync is cheap.
        self._sb_thr.setValue(thr_linear)

    def _on_spectral_threshold_dragged(self, val: float) -> None:
        """Phase 4b: the spectral (entropy) threshold line was dragged."""
        e = self._sel
        if e is None:
            return
        e.spectral_threshold = float(val)
        sb = getattr(self, '_sb_entropy_thr', None)
        if sb is not None:
            sb.blockSignals(True)
            sb.setValue(float(val))
            sb.blockSignals(False)
        self._mark_dirty()


    def _on_vm_cols_changed(self, val):
        self._vm_n_cols = val
        self._mark_dirty()
        if self._view_mode:
            self._rebuild_view()

    def _on_vm_height_changed(self, val):
        self._vm_panel_height = val
        self._mark_dirty()
        if self._view_mode and self._pg_grid is not None:
            self._pg_grid.set_tile_height(val)

    def _on_vm_fill_order_changed(self, _idx):
        order = self._vm_order_combo.currentData() or 'column'
        self._vm_fill_order = order
        self._mark_dirty()
        if self._view_mode:
            self._rebuild_view()

    def _on_vm_active_only(self, on: bool):
        """3c: 'Active only' toggled — rebuild the grid with the new
        filter (persisted via the view_mode config section)."""
        self._vm_active_only = bool(on)
        self._mark_dirty()
        if self._view_mode:
            self._rebuild_view()

    def _on_view_tile_clicked(self, idx: int):
        """3a/4d: a view-mode tile was clicked. When monitor-follow is
        on, route that stream to the audio monitor."""
        if not (0 <= idx < len(self._entities)):
            return
        chk = getattr(self, '_chk_monitor_follow', None)
        if chk is not None and chk.isChecked():
            self._apply_monitor_source(id(self._entities[idx]))
            self._sync_monitor_source_combo()

    def _on_vm_fit_to_screen(self):
        """Pick a tile height so all visible streams fit the viewport
        without scrolling, given the current column count."""
        if not self._view_mode or self._pg_grid is None:
            return
        n = len(self._visible_view_entities()[0])
        if n == 0:
            return
        cols = max(1, min(self._vm_n_cols, n))
        rows = math.ceil(n / cols)
        # Viewport height minus the grid's top/bottom margins (4+4) and the
        # inter-row spacing (4 px between rows) — matches MultiStreamGrid.
        vp = self._canvas_scroll.viewport().height()
        avail = vp - 8 - 4 * (rows - 1)
        tile = int(avail / rows) if rows else vp
        lo, hi = self._vm_height_spin.minimum(), self._vm_height_spin.maximum()
        tile = max(lo, min(hi, tile))
        if tile == self._vm_height_spin.value():
            # Value unchanged (already clamped there) — apply directly since
            # valueChanged won't fire.
            self._pg_grid.set_tile_height(tile)
        else:
            self._vm_height_spin.setValue(tile)  # → _on_vm_height_changed


    # ──────────────────────────────────────────────────────────────────────
    # Audio ingestion + plot refresh
    # ──────────────────────────────────────────────────────────────────────

    def _update_plot(self):
        # 1. Ingestion now happens on per-entity background threads
        # (#19 / c21). The main thread only reads the ring buffers and
        # updates the display. Drop / saturation badges still need
        # polling here.
        # A/V sync: how far behind the audio monitor is running right
        # now (its jitter buffer + the output device's latency). The
        # SAME figure goes to every stream, not just the monitored one,
        # so the whole grid stays comparable with itself as well as with
        # what is being heard.
        try:
            mon_delay = self._monitor.playback_delay_sec
        except Exception:
            mon_delay = 0.0
        for idx, e in enumerate(self._entities):
            # Display pacing: advance every entity's paced cursor, not
            # just the visible ones — an off-screen stream that skipped
            # ticks would jump when it came back into view.
            try:
                e.advance_display(monitor_delay_sec=mon_delay or None)
                e.publish_display()
            except Exception:
                pass
            if hasattr(e.capture, 'consume_drop_count'):
                n_drops = e.capture.consume_drop_count()
                if hasattr(self, '_sidebar'):
                    try:
                        self._sidebar.update_item_drops(idx, n_drops)
                    except Exception:
                        pass
            # M2: drain the transient error counters. The sticky badge
            # reads the has_ever_* flags, but the throttled log lines
            # to chirp_errors.log are emitted inside these consume
            # calls — before this poll, OS-level input overflows never
            # reached the log at all.
            try:
                if hasattr(e.capture, 'consume_os_drop_count'):
                    e.capture.consume_os_drop_count()
                # Zero-insertion (input_underflow) counter — the log
                # line marking "zeros are entering the recorded audio"
                # is emitted inside this consume call.
                if hasattr(e.capture, 'consume_underflow_count'):
                    e.capture.consume_underflow_count()
                e.consume_ingest_error_count()
                # Inserted-silence (zero-run) detector — the throttled
                # zero_run log line is emitted inside this consume.
                e.consume_zero_run_count()
                # Pick up the path of any clipped WAV published since
                # the last tick, so the "S" badge can name the file.
                e.poll_saturated_file()
                # M3: detect a dead ingest thread while acq claims to
                # be running (BaseException escaped the chunk guard).
                e.check_ingest_alive()
            except Exception:
                pass
            if hasattr(self, '_sidebar'):
                # #28: sticky saturation flag + the last clipped file.
                try:
                    from chirp.ui.status_util import compose_saturation_state
                    sat, sat_tip = compose_saturation_state(e)
                    self._sidebar.update_item_saturation_sticky(
                        idx, sat, sat_tip)
                except Exception:
                    pass
                # #29: sticky persistent-drops flag.
                try:
                    has_ever = bool(getattr(e.capture, 'has_ever_dropped', False))
                    total    = int(getattr(e.capture, 'drop_count_total', 0))
                    self._sidebar.update_item_drop_sticky(idx, has_ever, total)
                except Exception:
                    pass
                # #43 / #44 / #48: aggregate sticky error flag. Composes
                # four independent failure modes into a single badge so
                # the user doesn't need a dashboard to notice something
                # broke.
                try:
                    self._update_error_sticky(idx, e)
                except Exception:
                    pass

        # M2: drain the writer pool's transient error counter once per
        # tick (the sticky stats the badges read are unaffected).
        try:
            from chirp.recording import writer as _writer_mod
            _writer_mod.consume_error_count()
        except Exception:
            pass

        # Inserted-silence auto-recovery: reset an endpoint whose audio
        # has been substantially digital silence for a sustained period.
        try:
            self._zero_recovery_tick()
        except Exception:
            pass

        # TODO#1 (RDP): capture-stall watchdog + auto-reconnect.
        try:
            self._capture_watchdog_tick()
        except Exception:
            pass

        # #58: top-level guard around the main display body. The
        # individual sidebar updates above are already try/except'd
        # per-update; everything from here through the end of the slot
        # used to run unguarded. A single matplotlib / numpy / shape
        # exception (NaN sample, axes-rebuild race, buffer-resize
        # race) propagated out of the slot — Qt swallowed it but the
        # half-finished slot left blit-cache invariants broken
        # (``_axes_changed`` half-set, background bbox captured
        # mid-reallocation), and subsequent ticks compounded the
        # problem until the display froze. The user assumed the app
        # was dead and force-killed it (per #56 that orphans the
        # writer pool — silent data loss).
        try:
            self._update_plot_body()
        except Exception as exc:
            self._on_update_plot_error(exc)

    def _on_update_plot_error(self, exc: Exception) -> None:
        """#58: handle an exception escaping ``_update_plot_body``.

        Bumps the consecutive-error counter, stashes the message, and
        after ``_update_plot_freeze_threshold`` straight failures shows
        a sticky note in the trigger-status label so the user knows the
        display is degraded (the mic and writer keep working — they're
        on background threads — so the user should NOT force-kill).
        """
        import traceback
        self._update_plot_err_count += 1
        self._update_plot_err_total += 1
        self._update_plot_last_err = f'{type(exc).__name__}: {exc}'[:200]
        # Stderr trace so a developer can debug from the console; the
        # GUI will surface a sticky note after threshold is reached.
        traceback.print_exc()
        # (Phase C: the matplotlib blit-cache re-baseline that used to
        # live here is gone — pyqtgraph repaints from scratch each
        # frame, so there is no cached background to repair.)
        # Sticky note after N consecutive failures.
        if (self._update_plot_err_count >= self._update_plot_freeze_threshold
                and getattr(self, '_lbl_trig_status', None) is not None):
            try:
                self._lbl_trig_status.setText(
                    f'TRIG  DISPLAY HALTED ({self._update_plot_err_count} '
                    f'errors) — acquisition still running')
                self._lbl_trig_status.setObjectName('trig_active')
                self._lbl_trig_status.style().unpolish(self._lbl_trig_status)
                self._lbl_trig_status.style().polish(self._lbl_trig_status)
            except Exception:
                pass

    # ── Inserted-silence auto-recovery ────────────────────────────────
    #
    # Field evidence: a capture session can latch into a state where the
    # driver/engine periodically zero-fills the audio, and the ONLY
    # reliable reset is stopping acquisition on every stream that holds
    # the endpoint and starting again — stopping just one is not enough,
    # because the OS keeps the session alive for the remaining client.
    # An overnight episode ran 96 minutes unattended. This watchdog
    # performs that reset automatically.

    def _zero_recovery_tick(self):
        """Poll each stream's zero-sample duty cycle; when one stays
        above the configured threshold for the configured time, reset
        every stream on its device. Cheap enough for the UI tick — it
        reads one float per entity."""
        cfg = self._audio_cfg
        if not cfg.get('auto_recover_zero_runs', True):
            self._zero_high_since.clear()
            return
        t = self._zero_recover_thread
        if t is not None and t.is_alive():
            return
        now = time.monotonic()
        thr = float(cfg.get('zero_recover_percent', 5.0)) / 100.0
        hold = float(cfg.get('zero_recover_seconds', 15.0))
        cooldown = float(cfg.get('zero_recover_cooldown_sec', 120.0))
        for e in self._entities:
            key = id(e)
            if (not getattr(e, 'acq_running', False)
                    or getattr(e, 'input_source', '') != 'device'):
                self._zero_high_since.pop(key, None)
                continue
            if float(getattr(e, 'zero_sample_frac', 0.0)) < thr:
                self._zero_high_since.pop(key, None)
                continue
            since = self._zero_high_since.setdefault(key, now)
            if now - since < hold:
                continue
            dev_key = (getattr(e, 'device_id', None),
                       getattr(e, 'sample_rate', None))
            if now - self._zero_recover_last.get(dev_key, -1e9) < cooldown:
                continue
            # Every stream on this endpoint has to go down together, or
            # the OS session survives and the fault with it.
            group = [g for g in self._entities
                     if getattr(g, 'input_source', '') == 'device'
                     and (getattr(g, 'device_id', None),
                          getattr(g, 'sample_rate', None)) == dev_key]
            self._zero_recover_last[dev_key] = now
            for g in group:
                self._zero_high_since.pop(id(g), None)
            self._start_zero_recovery(group, e)
            return

    def _start_zero_recovery(self, group, trigger) -> None:
        pct = float(getattr(trigger, 'zero_sample_frac', 0.0)) * 100.0
        names = ', '.join(getattr(g, 'name', '?') for g in group)
        self.zero_recovery_count += 1
        msg = (f'inserted-silence auto-recovery #{self.zero_recovery_count}: '
               f'{getattr(trigger, "name", "?")} at {pct:.1f}% digital '
               f'zeros — restarting acquisition on all streams of this '
               f'device ({names})')
        print(f'[Chirp] {msg}')
        _err_log('zero_run_recovery', getattr(trigger, 'name', ''), msg)
        t = threading.Thread(target=self._zero_recovery_worker,
                             args=(list(group),),
                             name='chirp-zero-recovery', daemon=True)
        self._zero_recover_thread = t
        t.start()

    def _zero_recovery_worker(self, group) -> None:
        """Stop every stream on the endpoint, let the OS tear the capture
        session down, then bring them back. Runs off the GUI thread —
        ``stop_acq`` joins an ingest thread and closes PortAudio streams,
        either of which can block."""
        try:
            was_rec = {id(g): bool(getattr(g, 'rec_enabled', False))
                       for g in group}
            for g in group:
                try:
                    g.stop_acq()
                except Exception as exc:
                    print(f'[Chirp] zero-recovery: stop failed for '
                          f'{getattr(g, "name", "?")}: {exc}')
            # The last close releases the endpoint; give the driver a
            # moment to actually tear the session down before we ask for
            # a new one.
            time.sleep(0.75)
            for g in group:
                try:
                    g.start_acq()
                    if was_rec.get(id(g)):
                        g.rec_enabled = True
                except Exception as exc:
                    print(f'[Chirp] zero-recovery: restart failed for '
                          f'{getattr(g, "name", "?")}: {exc}')
                    _err_log('zero_run_recovery', getattr(g, 'name', ''),
                             f'restart after auto-recovery failed: {exc}')
        except Exception as exc:
            print(f'[Chirp] zero-recovery round crashed: {exc}')

    def _capture_watchdog_tick(self):
        """TODO#1: detect live-device captures that stopped delivering
        frames (a remote-desktop session change ripping out Windows
        audio endpoints) and auto-reconnect them.

        The GUI tick only DETECTS and dispatches: every PortAudio call
        the recovery needs (closing the dead stream, re-enumerating,
        reopening, terminate/initialize) can block for seconds — or
        indefinitely on a WASAPI device Windows tore out — so the
        actual work runs on a daemon worker thread. One worker at a
        time; attempt rounds are throttled with exponential backoff
        (3 s doubling to 30 s) so a device that never comes back
        doesn't get hammered.
        """
        # Detection always runs — the `!` badge and the ``capture_dead``
        # log line are how the user finds out at all. Only the automatic
        # reconnect is optional: on a rig where the device comes back on
        # its own (WDM-KS inputs largely ride out RDP session churn), the
        # teardown-and-reopen costs more audio than it saves.
        recover = bool(self._audio_cfg.get('auto_recover_capture_stall', True))
        stalled = [e for e in self._entities
                   if e.check_capture_stalled(recover=recover)]
        if not stalled or not recover:
            self._recovery_backoff = 3.0
            return
        if (self._recovery_thread is not None
                and self._recovery_thread.is_alive()):
            return   # a recovery round is still running (possibly hung
                     # in PortAudio — the GUI must stay responsive)
        now = time.monotonic()
        if now - self._recovery_last_attempt < self._recovery_backoff:
            return
        self._recovery_last_attempt = now
        healthy = [e for e in self._entities
                   if e.acq_running and e.input_source == 'device'
                   and not e.capture_stalled]
        # A global PortAudio refresh (needed when the endpoint list
        # itself went stale) kills every open stream — only allowed when
        # NO healthy live-device stream exists, and the worker closes
        # the monitor's output stream first (terminating PortAudio under
        # an open output stream is exactly the kind of native hang that
        # froze the app under AnyDesk).
        do_refresh = (not healthy) and self._recovery_needs_refresh
        t = threading.Thread(
            target=self._capture_recovery_worker,
            args=(list(stalled), do_refresh),
            name='chirp-capture-recovery', daemon=True)
        self._recovery_thread = t
        t.start()

    def _capture_recovery_worker(self, stalled, do_refresh: bool):
        """Background body of one capture-recovery round (TODO#1).

        Runs entirely off the GUI thread; a PortAudio call that hangs
        here leaks this worker but leaves the app usable (the sticky
        `!` badge keeps telling the user the stream is down). Entities
        whose capture resumed on its own between detection and this
        round are skipped by ``attempt_capture_recovery``'s
        ``capture_stalled`` guard.
        """
        try:
            if do_refresh:
                try:
                    # Terminating PortAudio with the monitor's output
                    # stream open is a native-level hang — close it
                    # first. The monitor bar keeps its device selection;
                    # the user re-enables it after the churn settles.
                    self._monitor.close()
                except Exception:
                    pass
                from chirp.audio.devices import refresh_portaudio
                refresh_portaudio()
                self._recovery_needs_refresh = False
            any_fail = False
            for e in stalled:
                try:
                    if not e.attempt_capture_recovery():
                        any_fail = True
                except Exception as exc:
                    any_fail = True
                    print(f'[Chirp] capture recovery failed for '
                          f'{e.name}: {exc}')
            # If reopening by name failed, the PortAudio device list
            # itself is probably stale — allow a global refresh on the
            # next round, and back off so a permanently-gone device
            # isn't hammered every 3 s.
            self._recovery_needs_refresh = any_fail
            self._recovery_backoff = (min(self._recovery_backoff * 2, 30.0)
                                      if any_fail else 3.0)
        except Exception as exc:
            print(f'[Chirp] capture recovery round crashed: {exc}')

    def _update_plot_body(self):
        """#58: extracted from ``_update_plot`` so the top-level
        try/except in the slot can wrap the entire main-display path
        without indenting the whole body."""
        # 2. Branch on mode
        if self._view_mode:
            # Phase 4: pyqtgraph/OpenGL grid renders the multi-stream view,
            # at an adaptive rate. If recent renders are expensive we skip
            # ticks so DSP/capture threads keep the GIL (audio-priority).
            if self._view_skip_left > 0:
                self._view_skip_left -= 1
                self._update_plot_err_count = 0
                return
            t0 = time.perf_counter()
            self._refresh_view()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self._view_render_ema = (0.7 * self._view_render_ema + 0.3 * dt_ms
                                     if self._view_render_ema else dt_ms)
            target = ANIMATION_INTERVAL * 0.5  # keep render duty <= ~50%
            if self._view_render_ema > target:
                self._view_skip = min(
                    3, int(math.ceil(self._view_render_ema / target)) - 1)
            else:
                self._view_skip = 0
            self._view_skip_left = self._view_skip
            self._update_plot_err_count = 0
            return

        # Refresh the WAV transport time label for the selected entity
        # on every plot tick (50 ms) so "passed / total" stays live.
        self._update_wav_time_label()

        e = self._sel

        # 3+4. Heavy render work (config panel + sidebar mini-amps),
        # gated by the same adaptive audio-priority frame-skip used in
        # view mode (H3): if recent renders cost more than ~half a tick,
        # skip ticks so DSP / capture threads keep the GIL. Status dots
        # and badges (cheap, change-detected) still update every tick.
        for i, ent in enumerate(self._entities):
            self._sidebar.update_item_status(i, ent.acq_running, ent.rec_enabled,
                                             ent.recorder.is_recording)
        if self._cfg_skip_left > 0:
            self._cfg_skip_left -= 1
        else:
            t0 = time.perf_counter()
            # Main display for the selected entity via the pyqtgraph
            # config panel (rebuilds its layout automatically when
            # display_mode / stereo / spectral / amp_scale change).
            if e is not None:
                self._config_panel.update_from_entity(e)
            # Sidebar mini-amp previews: one entity per tick
            # (round-robin) — a full-buffer reduction per stream per
            # tick is pure GIL burn for a 30px preview.
            if self._entities:
                i = self._mini_amp_rr % len(self._entities)
                self._mini_amp_rr = (i + 1) % len(self._entities)
                self._sidebar.update_item_amp(
                    i, self._entities[i].get_mini_amplitude())
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self._cfg_render_ema = (0.7 * self._cfg_render_ema + 0.3 * dt_ms
                                    if self._cfg_render_ema else dt_ms)
            target = ANIMATION_INTERVAL * 0.5  # keep render duty <= ~50%
            if self._cfg_render_ema > target:
                self._cfg_skip = min(
                    3, int(math.ceil(self._cfg_render_ema / target)) - 1)
            else:
                self._cfg_skip = 0
            self._cfg_skip_left = self._cfg_skip

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

        # #58: tick completed cleanly — reset the consecutive-error
        # counter so a transient blip (one bad chunk) doesn't leave
        # the "DISPLAY HALTED" sticky note up after recovery.
        self._update_plot_err_count = 0

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

        # Offer to persist unsaved configuration changes before quitting.
        # Shown independently of the acquisition warning above (either,
        # both, or neither can appear); Cancel aborts the whole quit.
        if self._config_dirty:
            reply = QMessageBox.question(
                self, 'Chirp',
                'You have unsaved changes to the configuration.\n'
                'Save them before quitting?',
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save)
            if reply == QMessageBox.Cancel:
                event.ignore()
                return
            if reply == QMessageBox.Save and not self._save_settings():
                # Save failed or the user backed out of the Save-As
                # dialog — don't quit and silently drop the changes.
                event.ignore()
                return

        self._timer.stop()

        # #56: close-event teardown used to run bare `stop_acq` /
        # `close` / writer-drain calls — one exception from any of
        # them would skip the rest, orphaning ingest threads and
        # silently discarding in-flight recordings from every
        # remaining entity. Now each step gets its own try/except and
        # errors are collected for a single post-teardown modal.
        teardown_errors: list[str] = []

        # #19 / c21: stop all ingestion threads before flushing, so no
        # new chunks are processed while we're draining pending events.
        for e in self._entities:
            try:
                e.stop_acq()
            except Exception as exc:
                teardown_errors.append(f'stop_acq({e.name}): {exc}')
                print(f'[Chirp] stop_acq failed for {e.name}: {exc}')

        # #17 / c16: flush any in-flight trigger events to the writer
        # pool, then drain the pool so non-daemon worker threads finish
        # writing before the interpreter exits. Without this, daemon
        # threads from the old launcher would be killed mid-write and
        # the most recent WAV would be left truncated on disk.
        # Routed through the entity helper so the flush lands in the
        # same ref_date day-subfolder every other flush path uses
        # (M6 — the old manual flush_all here passed the bare
        # ``output_dir`` and stranded shutdown WAVs outside the day
        # folder their siblings live in).
        from chirp.recording import writer as _writer
        for e in self._entities:
            try:
                e._flush_active_events(reason='app shutdown')
            except Exception as exc:
                teardown_errors.append(f'flush_all({e.name}): {exc}')
                print(f'[Chirp] flush_all failed for {e.name}: {exc}')

        try:
            pending = _writer.pending()
        except Exception as exc:
            pending = 0
            teardown_errors.append(f'writer.pending: {exc}')
        if pending:
            # Show a non-cancellable modal so the user knows we're
            # waiting on disk I/O and not just frozen.
            msg = None
            try:
                msg = QMessageBox(self)
                msg.setIcon(QMessageBox.Information)
                msg.setWindowTitle('Chirp')
                msg.setText(f'Finishing {pending} pending recording(s)…')
                msg.setStandardButtons(QMessageBox.NoButton)
                msg.show()
                QApplication.processEvents()
            except Exception:
                msg = None
            try:
                drained = _writer.drain(timeout=30.0)
                if not drained:
                    # #56: drain timeout means queued WAVs will be
                    # abandoned. Record it so the user sees a warning
                    # instead of a "clean" exit.
                    remaining = _writer.pending()
                    teardown_errors.append(
                        f'writer drain timed out with {remaining} '
                        f'recording(s) still queued — they will be lost')
            except Exception as exc:
                teardown_errors.append(f'writer.drain: {exc}')
            finally:
                if msg is not None:
                    try:
                        msg.close()
                    except Exception:
                        pass
        # #7: close the monitor loopback before closing entities — the
        # output stream's callback could otherwise read a buffer that
        # feeders are tearing down.
        try:
            self._monitor.close()
        except Exception as exc:
            teardown_errors.append(f'monitor.close: {exc}')

        for e in self._entities:
            try:
                e.close()
            except Exception as exc:
                teardown_errors.append(f'close({e.name}): {exc}')
                print(f'[Chirp] close failed for {e.name}: {exc}')

        # Writer shutdown must come AFTER every entity is closed: an
        # ``e.close()`` can still flush a straggler event (e.g. when an
        # earlier stop_acq raised) and a submit() after shutdown would
        # lazily resurrect a fresh pool whose non-daemon workers park on
        # queue.get() forever — the interpreter would never exit (H2).
        try:
            _writer.shutdown(timeout=30.0)
        except Exception as exc:
            teardown_errors.append(f'writer.shutdown: {exc}')

        # Flush + stop the async error logger so the tail of
        # chirp_errors.log isn't lost when the interpreter exits.
        try:
            from chirp import error_log as _errlog
            _errlog.shutdown(timeout=2.0)
        except Exception as exc:
            teardown_errors.append(f'error_log.shutdown: {exc}')

        # #56: if anything went wrong, show a modal so the user knows
        # the session did not exit cleanly before the window dies.
        if teardown_errors:
            try:
                detail = '\n'.join(f'• {line}' for line in teardown_errors[:12])
                extra = ''
                if len(teardown_errors) > 12:
                    extra = f'\n…and {len(teardown_errors) - 12} more.'
                QMessageBox.warning(
                    self, 'Chirp — shutdown incomplete',
                    'Some steps failed while closing the app. Any recordings '
                    'listed below may not have been saved.\n\n' + detail + extra,
                )
            except Exception:
                # If even the modal fails, swallow — we're exiting
                # anyway and the console log still has the details.
                pass

        super().closeEvent(event)


def _is_remote_session() -> bool:
    """True when this process runs inside a Windows remote-desktop
    session (``GetSystemMetrics(SM_REMOTESESSION)``). RDP swaps the
    display driver, so OpenGL must be avoided; screen-mirroring tools
    (AnyDesk, TeamViewer) attach to the console session and return
    False here — their rendering path is unaffected."""
    if sys.platform != 'win32':
        return False
    try:
        SM_REMOTESESSION = 0x1000
        return bool(ctypes.windll.user32.GetSystemMetrics(SM_REMOTESESSION))
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────
def _log_opengl_status():
    """Print whether OpenGL is enabled and which renderer the driver
    provides, so the user can tell hardware GPU (e.g. 'NVIDIA ...') from a
    software fallback ('GDI Generic', 'llvmpipe', or an ANGLE/Direct3D
    string). Best-effort — never raises. Requires a live QApplication."""
    try:
        import pyqtgraph as pg
        enabled = bool(pg.getConfigOption('useOpenGL'))
    except Exception:
        enabled = None
    renderer = None
    try:
        from PyQt5.QtGui import QOffscreenSurface, QOpenGLContext
        from OpenGL import GL
        ctx = QOpenGLContext()
        if ctx.create():
            surf = QOffscreenSurface()
            surf.create()
            if ctx.makeCurrent(surf):
                try:
                    raw = GL.glGetString(GL.GL_RENDERER)
                    renderer = raw.decode() if isinstance(raw, bytes) else str(raw)
                finally:
                    ctx.doneCurrent()
    except Exception:
        renderer = None
    state = {True: 'enabled', False: 'disabled'}.get(enabled, 'unknown')
    if renderer:
        print(f'[Chirp] OpenGL: {state} | renderer: {renderer}')
    else:
        print(f'[Chirp] OpenGL: {state} | renderer: <unavailable — '
              f'software raster likely>')


def _app_icon_path() -> str | None:
    """Locate assets/chirp.ico for both dev runs (repo root) and frozen
    PyInstaller builds (bundled next to the executable via --add-data)."""
    if getattr(sys, 'frozen', False):
        bases = [getattr(sys, '_MEIPASS', ''),
                 os.path.dirname(sys.executable)]
    else:
        bases = [os.path.join(os.path.dirname(__file__), '..', '..')]
    for base in bases:
        p = os.path.join(base, 'assets', 'chirp.ico')
        if base and os.path.exists(p):
            return p
    return None


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)
    icon_path = _app_icon_path()
    if icon_path:
        from PyQt5.QtGui import QIcon
        app.setWindowIcon(QIcon(icon_path))
    win = ChirpWindow()
    _log_opengl_status()
    win.showMaximized()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
