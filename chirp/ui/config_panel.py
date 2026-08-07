"""ConfigPlotPanel — pyqtgraph single-stream config/editing view (Phase 4b).

Replaces the matplotlib config-mode canvas so the whole app is
matplotlib-free. Renders the selected RecordingEntity in detail:

  * spectrogram (+ stereo right channel) with GPU inferno LUT,
  * raw waveform (+ stereo), teal,
  * amplitude envelope with a **draggable threshold** line (linear or dB),
  * spectral-entropy trace with a **draggable spectral-threshold** line
    (only when a spectral trigger mode is active),
  * a detect/record events strip,
  * a shared time cursor across all rows.

Layout depends on ``display_mode`` (Spectrogram / Waveform / Both) and
``channel_mode`` (stereo adds right-channel rows); :meth:`rebuild` is
called whenever those — or the selected entity — change. Amplitude/
entropy rows have mouse-wheel Y zoom (the old scroll-zoom feature); the
time (X) axis is fixed to the display window.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QMenu

from chirp.constants import (AMP_DB_EPS, AMP_DB_MAX, AMP_DB_MIN, C,
                             CHUNK_FRAMES)
from chirp.ui.pg_panel import events_rgba, spec_ytick_key, spec_ytick_list

_INFERNO_LUT = (
    matplotlib.colormaps['inferno'](np.linspace(0.0, 1.0, 256))[:, :3] * 255
).astype(np.ubyte)

pg.setConfigOption('imageAxisOrder', 'row-major')


def _amp_to_display(buf, scale):
    if scale == 'log':
        return 20.0 * np.log10(np.maximum(np.abs(buf), AMP_DB_EPS))
    return buf


# H3: target column counts for pre-setData peak decimation. Full display
# buffers reach millions of samples (display_seconds × sample_rate); the
# screen is a few thousand pixels wide. Reducing on our side — BEFORE
# the dB conversion and setData — cuts the per-tick numpy work (which
# holds the GIL, starving the audio threads) from O(buffer) to
# O(buffer) once for the reduction and O(pixels) for everything after.
_MAX_ENV_COLS = 4096    # envelope: max-decimated (values are >= 0)
_MAX_WAVE_COLS = 2048   # waveform: min/max interleaved (2 pts per col)


def _decimate_max(y: np.ndarray, max_cols: int) -> np.ndarray:
    """Peak (max) decimation for non-negative envelope data."""
    n = y.shape[0]
    if n <= max_cols * 2:
        return y
    k = n // max_cols
    m = n // k
    return y[:m * k].reshape(m, k).max(axis=1)


def _decimate_minmax(y: np.ndarray, max_cols: int) -> np.ndarray:
    """Min/max-interleaved decimation for signed waveform data —
    preserves the visual peak envelope in both directions."""
    n = y.shape[0]
    if n <= max_cols * 2:
        return y
    k = n // max_cols
    m = n // k
    z = y[:m * k].reshape(m, k)
    out = np.empty(m * 2, dtype=y.dtype)
    out[0::2] = z.min(axis=1)
    out[1::2] = z.max(axis=1)
    return out


def _thr_to_display(thr, scale):
    if scale == 'log':
        return max(20.0 * np.log10(max(thr, AMP_DB_EPS)), AMP_DB_MIN)
    return thr


def _display_to_thr(yval, scale):
    if scale == 'log':
        return float(np.clip(10.0 ** (yval / 20.0), 0.0, 1.0))
    return float(np.clip(yval, 0.0, 1.0))


class ConfigPlotPanel(pg.GraphicsLayoutWidget):
    thresholdChanged = pyqtSignal(float)          # linear amplitude threshold
    spectralThresholdChanged = pyqtSignal(float)  # entropy threshold
    ampScaleChanged = pyqtSignal(str)             # 'linear' | 'log'

    def __init__(self, parent=None, *, use_opengl: bool = True):
        super().__init__(parent)
        if use_opengl:
            try:
                pg.setConfigOptions(useOpenGL=True, antialias=False)
            except Exception:
                pass
        self.setBackground(C['base'])
        # Recognition color — drawn as a rectangle around the whole panel
        # (spectrogram + amplitude + events) via the QGraphicsView frame.
        self.setObjectName('config_plot_panel')
        self._color: str | None = None
        self._apply_color()
        # Layout signature — rebuild only when it actually changes.
        self._sig = None
        self._amp_scale = 'log'
        self._suppress = False       # guard drag-vs-programmatic set
        self._suppress_spec = False
        # When True the threshold lines are frozen (stream params locked).
        self._locked = False
        # Row references (set in rebuild).
        self._plots: list = []
        self._img = self._img_r = None
        self._wave = self._wave_r = None
        self._amp = self._amp_r = None
        self._entropy = None
        self._events_img = None
        self._thr_line = None
        self._spec_thr_line = None
        self._cursors: list = []
        # H3: per-tick work gates. ``_t_axes`` caches the time ramps
        # (rebuilt only when the length/window changes);
        # ``_last_data_stamp`` skips all data uploads when no new audio
        # arrived and no display parameter moved — only the cursor is
        # repositioned.
        self._t_axes: dict = {}
        self._last_data_stamp = None
        self._spec_plots: list = []
        self._ytick_key: tuple | None = None

    # ── Layout ────────────────────────────────────────────────────────
    @staticmethod
    def _signature(e) -> tuple:
        # display_seconds is part of the layout signature because the
        # fixed X ranges are set at rebuild time.
        return (e.display_mode, e.channel_mode == 'Stereo',
                e.spectral_trigger_mode != 'Amplitude Only',
                getattr(e, 'amp_scale', 'log'),
                float(e.display_seconds))

    def current_amp_ylim(self):
        """Top of the amplitude plot's Y range when the linear scale is
        active (the user may have wheel-zoomed it); None on the dB scale,
        whose range is fixed."""
        if self._amp is None or self._amp_scale == 'log':
            return None
        try:
            vb = self._amp.getViewBox()
            return float(vb.viewRange()[1][1])
        except Exception:
            return None

    def set_color(self, color: str | None) -> None:
        """Set the recognition color drawn as a rectangle around the
        panel. ``None`` falls back to a neutral frame."""
        color = color or None
        if color == self._color:
            return
        self._color = color
        self._apply_color()

    def _apply_color(self) -> None:
        col = self._color or C['surface1']
        # objectName selector so the border styles the view frame only.
        self.setStyleSheet(
            f'QGraphicsView#config_plot_panel {{ border: 3px solid {col}; '
            f'border-radius: 4px; }}')

    def rebuild_if_needed(self, e) -> None:
        sig = self._signature(e)
        if sig != self._sig:
            self.rebuild(e)

    def rebuild(self, e) -> None:
        self.clear()
        self._plots = []
        self._cursors = []
        self._img = self._img_r = None
        self._wave = self._wave_r = None
        self._amp = self._amp_r = None
        self._entropy = None
        self._events_img = None
        self._thr_line = self._spec_thr_line = None
        self._amp_scale = getattr(e, 'amp_scale', 'log')

        show_spec = e.display_mode in ('Spectrogram', 'Both')
        show_wave = e.display_mode in ('Waveform', 'Both')
        stereo = e.channel_mode == 'Stereo'
        has_entropy = e.spectral_trigger_mode != 'Amplitude Only'
        disp = float(e.display_seconds)

        row = 0

        def add_plot(height=None, mouse_y=False, link=True):
            nonlocal row
            p = self.addPlot(row=row, col=0)
            row += 1
            p.setMenuEnabled(False)
            p.setMouseEnabled(x=False, y=mouse_y)
            p.setXRange(0, disp, padding=0)
            if height:
                p.setMaximumHeight(height)
            if link and self._plots:
                p.setXLink(self._plots[0])
            self._plots.append(p)
            cur = pg.InfiniteLine(angle=90, pen=pg.mkPen(C['red'], width=1))
            p.addItem(cur)
            self._cursors.append(cur)
            return p

        def add_image(p):
            img = pg.ImageItem()
            img.setLookupTable(_INFERNO_LUT)
            # Float images require levels at paint time; set a sane default
            # (the entity's dB range) so a paint that lands before the first
            # update_from_entity — or on an empty buffer — doesn't raise.
            img.setLevels([min(e.db_floor, e.db_ceil - 0.1), e.db_ceil])
            p.addItem(img)
            return img

        self._spec_plots = []
        if show_spec:
            p = add_plot()
            p.setLabel('left', 'Freq')
            self._img = add_image(p)
            self._spec_plots.append(p)
            if stereo:
                p = add_plot()
                p.setLabel('left', 'Freq R')
                self._img_r = add_image(p)
                self._spec_plots.append(p)

        if show_wave:
            p = add_plot(mouse_y=True)
            p.setLabel('left', 'Wave')
            self._wave = p.plot(pen=pg.mkPen(C['teal'], width=1))
            self._wave.setDownsampling(auto=True, method='peak')
            self._wave.setClipToView(True)
            if stereo:
                p = add_plot(mouse_y=True)
                p.setLabel('left', 'Wave R')
                self._wave_r = p.plot(pen=pg.mkPen(C['pink'], width=1))
                self._wave_r.setDownsampling(auto=True, method='peak')
                self._wave_r.setClipToView(True)

        # Amplitude row (always present) with draggable threshold.
        # Mouse-wheel Y zoom is enabled on BOTH scales so the user can zoom
        # in to refine the threshold line — especially on the dB scale where
        # the interesting range is often a narrow band near the noise floor.
        amp_p = add_plot(mouse_y=True)
        amp_p.setLabel('left', 'Amp (dB)' if self._amp_scale == 'log' else 'Amp')
        self._amp = amp_p.plot(pen=pg.mkPen(C['blue'], width=1))
        self._amp.setDownsampling(auto=True, method='peak')
        self._amp.setClipToView(True)
        if stereo:
            self._amp_r = amp_p.plot(pen=pg.mkPen(C['pink'], width=1))
            self._amp_r.setDownsampling(auto=True, method='peak')
            self._amp_r.setClipToView(True)
        if self._amp_scale == 'log':
            amp_p.setYRange(AMP_DB_MIN, AMP_DB_MAX, padding=0)
        else:
            amp_p.setYRange(0.0, getattr(e, 'amp_ylim', 1.05), padding=0)
        self._thr_line = pg.InfiniteLine(
            angle=0, movable=not self._locked,
            pen=pg.mkPen(C['mauve'], width=2, style=pg.QtCore.Qt.DashLine))
        self._thr_line.setValue(_thr_to_display(e.threshold, self._amp_scale))
        self._thr_line.sigPositionChanged.connect(self._on_thr_dragged)
        amp_p.addItem(self._thr_line)

        if has_entropy:
            ent_p = add_plot(mouse_y=True)
            ent_p.setLabel('left', 'Entropy')
            ent_p.setYRange(0.0, 1.0, padding=0)
            self._entropy = ent_p.plot(pen=pg.mkPen(C['peach'], width=1))
            self._spec_thr_line = pg.InfiniteLine(
                angle=0, movable=not self._locked,
                pen=pg.mkPen(C['peach'], width=2, style=pg.QtCore.Qt.DashLine))
            self._spec_thr_line.setValue(e.spectral_threshold)
            self._spec_thr_line.sigPositionChanged.connect(
                self._on_spec_thr_dragged)
            ent_p.addItem(self._spec_thr_line)

        # Detect/record events strip. Taller than the old 44 px and with
        # the 'det'/'rec' row labels the matplotlib version had (TODO#14).
        ev_p = add_plot(height=80)
        ev_p.setLabel('left', 'Events')
        ev_p.getAxis('left').setTicks([[(0.5, 'det'), (1.5, 'rec')]])
        ev_p.setLabel('bottom', 'Time', units='s')
        self._events_img = pg.ImageItem()
        ev_p.addItem(self._events_img)

        self._sig = self._signature(e)
        # Fresh curves need a full data upload on the next tick — and
        # fresh spectrogram axes need their frequency ticks re-applied.
        self._t_axes.clear()
        self._last_data_stamp = None
        self._ytick_key = None

    # ── Amplitude-scale context menu (replaces the old right-click on
    #    the matplotlib amp axis) ────────────────────────────────────────
    def contextMenuEvent(self, ev):
        menu = QMenu(self)
        a_lin = menu.addAction('Amplitude scale: Linear')
        a_log = menu.addAction('Amplitude scale: Log (dB)')
        a_lin.setCheckable(True)
        a_log.setCheckable(True)
        a_lin.setChecked(self._amp_scale != 'log')
        a_log.setChecked(self._amp_scale == 'log')
        chosen = menu.exec_(ev.globalPos())
        if chosen is a_lin:
            self.ampScaleChanged.emit('linear')
        elif chosen is a_log:
            self.ampScaleChanged.emit('log')

    # ── Drag handlers ─────────────────────────────────────────────────
    def _on_thr_dragged(self):
        if self._suppress or self._thr_line is None:
            return
        self.thresholdChanged.emit(
            _display_to_thr(self._thr_line.value(), self._amp_scale))

    def _on_spec_thr_dragged(self):
        if self._suppress_spec or self._spec_thr_line is None:
            return
        self.spectralThresholdChanged.emit(
            float(np.clip(self._spec_thr_line.value(), 0.0, 1.0)))

    def set_threshold(self, thr_linear: float) -> None:
        if self._thr_line is None:
            return
        self._suppress = True
        try:
            self._thr_line.setValue(_thr_to_display(thr_linear, self._amp_scale))
        finally:
            self._suppress = False

    def set_spectral_threshold(self, val: float) -> None:
        if self._spec_thr_line is None:
            return
        self._suppress_spec = True
        try:
            self._spec_thr_line.setValue(float(val))
        finally:
            self._suppress_spec = False

    def set_threshold_locked(self, locked: bool) -> None:
        """Freeze/unfreeze the draggable amplitude + entropy threshold
        lines. The flag is remembered so lines recreated by a later
        rebuild come back frozen when the stream is locked."""
        self._locked = bool(locked)
        if self._thr_line is not None:
            self._thr_line.setMovable(not self._locked)
        if self._spec_thr_line is not None:
            self._spec_thr_line.setMovable(not self._locked)

    # ── Per-tick update ───────────────────────────────────────────────
    def _t_axis(self, n: int, disp: float) -> np.ndarray:
        key = (n, disp)
        t = self._t_axes.get(key)
        if t is None:
            t = np.linspace(0.0, disp, n, dtype=np.float32)
            # Bounded cache — decimated lengths are few, but a display
            # window change could otherwise accumulate stale ramps.
            if len(self._t_axes) > 8:
                self._t_axes.clear()
            self._t_axes[key] = t
        return t

    def update_from_entity(self, e) -> None:
        self.rebuild_if_needed(e)
        disp = float(e.display_seconds)
        # Paced cursor (see RecordingEntity.advance_display).
        cursor_x = (e.display_head / e.sample_rate) % disp
        for cur in self._cursors:
            cur.setValue(cursor_x)

        # H3: skip every data upload when no new audio has been ingested
        # and no display parameter changed — the cursor above is the
        # only thing moving. This makes an idle (acquisition-stopped)
        # panel essentially free and de-duplicates ticks that land
        # between chunks.
        # ``display_head`` is in the key because under pacing the view
        # keeps changing between ingests — the revealed strip grows every
        # tick — so keying on _samples_total alone would freeze it.
        stamp = (id(e), e._samples_total, e.display_head,
                 e.db_floor, e.db_ceil, e.gain_db,
                 e.freq_scale, e.display_freq_lo, e.display_freq_hi,
                 e.amp_ylim, bool(getattr(e, 'saturated', False)))
        if stamp == self._last_data_stamp:
            return
        self._last_data_stamp = stamp

        # Frequency y-tick labels for the spectrogram rows (mel/log/
        # linear mapped) — recomputed only when the mapping changes.
        key = spec_ytick_key(e)
        if key != self._ytick_key and self._spec_plots:
            self._ytick_key = key
            ticks = [spec_ytick_list(e)]
            for p in self._spec_plots:
                p.getAxis('left').setTicks(ticks)

        clim_lo = min(e.db_floor, e.db_ceil - 0.1)
        if self._img is not None:
            spec = e.resample_spec(e.view('spec_buffer'))
            self._img.setImage(spec, autoLevels=False)
            self._img.setLevels([clim_lo, e.db_ceil])
            self._img.setRect(pg.QtCore.QRectF(0.0, 0.0, disp,
                                               float(spec.shape[0])))
        if self._img_r is not None:
            spec_r = e.resample_spec(e.view('spec_buffer_r'))
            self._img_r.setImage(spec_r, autoLevels=False)
            self._img_r.setLevels([clim_lo, e.db_ceil])
            self._img_r.setRect(pg.QtCore.QRectF(0.0, 0.0, disp,
                                                 float(spec_r.shape[0])))

        # H3: decimate to screen resolution BEFORE the dB conversion and
        # setData — the full buffers are millions of samples at high
        # SR × long windows and the numpy work holds the GIL.
        scale = getattr(e, 'amp_scale', 'log')
        if self._wave is not None:
            color = C['red'] if getattr(e, 'saturated', False) else C['teal']
            self._wave.setPen(pg.mkPen(color, width=1))
            w = _decimate_minmax(e.view('amp_buffer'), _MAX_WAVE_COLS)
            self._wave.setData(self._t_axis(w.shape[0], disp), w)
        if self._wave_r is not None:
            w_r = _decimate_minmax(e.view('amp_buffer_r'), _MAX_WAVE_COLS)
            self._wave_r.setData(self._t_axis(w_r.shape[0], disp), w_r)
        if self._amp is not None:
            color = C['red'] if getattr(e, 'saturated', False) else C['blue']
            self._amp.setPen(pg.mkPen(color, width=1))
            a = _amp_to_display(
                _decimate_max(e.view('abs_amp_buffer'), _MAX_ENV_COLS), scale)
            self._amp.setData(self._t_axis(a.shape[0], disp), a)
        if self._amp_r is not None:
            a_r = _amp_to_display(
                _decimate_max(e.view('abs_amp_buffer_r'), _MAX_ENV_COLS),
                scale)
            self._amp_r.setData(self._t_axis(a_r.shape[0], disp), a_r)

        if self._entropy is not None:
            nc = e._n_cols
            self._entropy.setData(self._t_axis(nc, disp),
                                  e.view('entropy_buffer'))

        if self._events_img is not None:
            rgba = events_rgba(e)
            if rgba is not None:
                self._events_img.setImage(rgba, autoLevels=False)
                self._events_img.setRect(pg.QtCore.QRectF(0.0, 0.0, disp, 2.0))
