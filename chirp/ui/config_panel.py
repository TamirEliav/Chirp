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

_INFERNO_LUT = (
    matplotlib.colormaps['inferno'](np.linspace(0.0, 1.0, 256))[:, :3] * 255
).astype(np.ubyte)

pg.setConfigOption('imageAxisOrder', 'row-major')


def _amp_to_display(buf, scale):
    if scale == 'log':
        return 20.0 * np.log10(np.maximum(np.abs(buf), AMP_DB_EPS))
    return buf


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
        # Layout signature — rebuild only when it actually changes.
        self._sig = None
        self._amp_scale = 'log'
        self._suppress = False       # guard drag-vs-programmatic set
        self._suppress_spec = False
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

    # ── Layout ────────────────────────────────────────────────────────
    @staticmethod
    def _signature(e) -> tuple:
        return (e.display_mode, e.channel_mode == 'Stereo',
                e.spectral_trigger_mode != 'Amplitude Only',
                getattr(e, 'amp_scale', 'log'))

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
            p.addItem(img)
            return img

        if show_spec:
            p = add_plot()
            p.setLabel('left', 'Freq')
            self._img = add_image(p)
            if stereo:
                p = add_plot()
                p.setLabel('left', 'Freq R')
                self._img_r = add_image(p)

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
        amp_p = add_plot(mouse_y=(self._amp_scale != 'log'))
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
            angle=0, movable=True,
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
                angle=0, movable=True,
                pen=pg.mkPen(C['peach'], width=2, style=pg.QtCore.Qt.DashLine))
            self._spec_thr_line.setValue(e.spectral_threshold)
            self._spec_thr_line.sigPositionChanged.connect(
                self._on_spec_thr_dragged)
            ent_p.addItem(self._spec_thr_line)

        # Detect/record events strip.
        ev_p = add_plot(height=44)
        ev_p.setLabel('left', 'Events')
        ev_p.getAxis('left').setStyle(showValues=False)
        ev_p.setLabel('bottom', 'Time', units='s')
        self._events_img = pg.ImageItem()
        ev_p.addItem(self._events_img)

        self._sig = self._signature(e)

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

    # ── Per-tick update ───────────────────────────────────────────────
    def update_from_entity(self, e) -> None:
        self.rebuild_if_needed(e)
        disp = float(e.display_seconds)
        cursor_x = (e.write_head / e.sample_rate) % disp
        for cur in self._cursors:
            cur.setValue(cursor_x)

        clim_lo = min(e.db_floor, e.db_ceil - 0.1)
        if self._img is not None:
            spec = e.resample_spec(e.spec_buffer)
            self._img.setImage(spec, autoLevels=False)
            self._img.setLevels([clim_lo, e.db_ceil])
            self._img.setRect(pg.QtCore.QRectF(0.0, 0.0, disp,
                                               float(spec.shape[0])))
        if self._img_r is not None:
            spec_r = e.resample_spec(e.spec_buffer_r)
            self._img_r.setImage(spec_r, autoLevels=False)
            self._img_r.setLevels([clim_lo, e.db_ceil])
            self._img_r.setRect(pg.QtCore.QRectF(0.0, 0.0, disp,
                                                 float(spec_r.shape[0])))

        scale = getattr(e, 'amp_scale', 'log')
        n = e.abs_amp_buffer.shape[0]
        t = np.linspace(0.0, disp, n, dtype=np.float32)
        if self._wave is not None:
            color = C['red'] if getattr(e, 'saturated', False) else C['teal']
            self._wave.setPen(pg.mkPen(color, width=1))
            self._wave.setData(t, e.amp_buffer)
        if self._wave_r is not None:
            self._wave_r.setData(t, e.amp_buffer_r)
        if self._amp is not None:
            color = C['red'] if getattr(e, 'saturated', False) else C['blue']
            self._amp.setPen(pg.mkPen(color, width=1))
            self._amp.setData(t, _amp_to_display(e.abs_amp_buffer, scale))
        if self._amp_r is not None:
            self._amp_r.setData(t, _amp_to_display(e.abs_amp_buffer_r, scale))

        if self._entropy is not None:
            nc = e._n_cols
            tcol = np.linspace(0.0, disp, nc, dtype=np.float32)
            self._entropy.setData(tcol, e.entropy_buffer)

        if self._events_img is not None:
            rgba = self._events_rgba(e)
            if rgba is not None:
                self._events_img.setImage(rgba, autoLevels=False)
                self._events_img.setRect(pg.QtCore.QRectF(0.0, 0.0, disp, 2.0))

    @staticmethod
    def _events_rgba(e):
        nc = e._n_cols
        need = nc * CHUNK_FRAMES
        if e.detect_mask_buffer.shape[0] < need:
            return None
        det = e.detect_mask_buffer[:need].reshape(nc, CHUNK_FRAMES).any(axis=1)
        rec = e.record_mask_buffer[:need].reshape(nc, CHUNK_FRAMES).any(axis=1)
        rgba = np.zeros((2, nc, 4), dtype=np.float32)
        # Background (surface0-ish) so empty cells aren't pure black.
        rgba[..., 3] = 1.0
        rgba[..., 0] = 0.19
        rgba[..., 1] = 0.20
        rgba[..., 2] = 0.27
        # Row 0 = detect (yellow), row 1 = record (green).
        rgba[0, det] = (0.976, 0.886, 0.686, 1.0)
        rgba[1, rec] = (0.651, 0.890, 0.631, 1.0)
        return rgba
