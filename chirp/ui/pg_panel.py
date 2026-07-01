"""pyqtgraph-based stream visualization (redesign Phase 4).

Renders one RecordingEntity's spectrogram + amplitude envelope (+ optional
raw waveform and spectral-entropy trace) using pyqtgraph with an
OpenGL-accelerated viewport. This replaces the matplotlib FigureCanvas
render path; the wins over the old renderer:

  * The colormap is applied by a Qt/GPU lookup table (``setLookupTable`` +
    ``setLevels``) — no per-frame Python colormap recompute / ``set_clim``
    renormalize, which dominated the matplotlib view-mode cost.
  * Curves use pyqtgraph auto-downsampling + clip-to-view, so a 10-second
    envelope at 44.1 kHz draws ~viewport-width points, not 441k.
  * The panel only repaints when :meth:`update_from_entity` is called, so
    the window can drive it at an adaptive rate and skip off-screen
    streams entirely (audio-priority degradation).

The panel reads the entity's existing display ring buffers (the same ones
the matplotlib path read), so no change to the DSP/ingest side is needed.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pyqtgraph as pg

from chirp.constants import C, N_DISPLAY_ROWS, SPEC_DB_MAX, SPEC_DB_MIN

# Inferno LUT (256 RGB rows) built once from matplotlib's colormap so the
# spectrogram matches the previous renderer's look. Applied on the GPU/Qt
# side rather than recomputed per frame.
_INFERNO_LUT = (
    matplotlib.colormaps['inferno'](np.linspace(0.0, 1.0, 256))[:, :3] * 255
).astype(np.ubyte)

# pyqtgraph image data is indexed [row, col] == [freq, time] in row-major.
pg.setConfigOption('imageAxisOrder', 'row-major')


def _amp_to_display(buf: np.ndarray, scale: str) -> np.ndarray:
    """Linear envelope → display units (dB when scale == 'log')."""
    if scale == 'log':
        return 20.0 * np.log10(np.maximum(np.abs(buf), 1e-4))
    return buf


class StreamPlotPanel(pg.GraphicsLayoutWidget):
    """A vertically-stacked spectrogram + amplitude view for one entity.

    Call :meth:`update_from_entity` on a timer to refresh. Construction is
    headless-safe (no OpenGL required); pass ``use_opengl=False`` in tests.
    """

    def __init__(self, parent=None, *, use_opengl: bool = True,
                 show_waveform: bool = False):
        super().__init__(parent)
        if use_opengl:
            # OpenGL viewport for GPU-accelerated blitting. Antialiasing
            # off — it's costly and unnecessary for spectrogram pixels.
            try:
                pg.setConfigOptions(useOpenGL=True, antialias=False)
                import pyqtgraph.opengl  # noqa: F401  (ensures GL available)
            except Exception:
                pass
        self.setBackground(C['base'])
        self._show_waveform = show_waveform

        # ── Spectrogram ────────────────────────────────────────────────
        self._spec_plot = self.addPlot(row=0, col=0)
        self._spec_plot.setMenuEnabled(False)
        self._spec_plot.setMouseEnabled(x=False, y=False)
        self._spec_plot.setLabel('left', 'Frequency')
        self._spec_plot.getAxis('bottom').setStyle(showValues=False)
        self._spec_plot.setClipToView(True)
        self._img = pg.ImageItem()
        self._img.setLookupTable(_INFERNO_LUT)
        self._img.setLevels([SPEC_DB_MIN, SPEC_DB_MAX])
        self._spec_plot.addItem(self._img)
        self._spec_cursor = pg.InfiniteLine(
            angle=90, pen=pg.mkPen(C['red'], width=1))
        self._spec_plot.addItem(self._spec_cursor)

        # ── Amplitude envelope ─────────────────────────────────────────
        self._amp_plot = self.addPlot(row=1, col=0)
        self._amp_plot.setMenuEnabled(False)
        self._amp_plot.setMaximumHeight(150)
        self._amp_plot.setXLink(self._spec_plot)
        self._amp_plot.setLabel('left', 'Amp')
        self._amp_plot.setLabel('bottom', 'Time', units='s')
        self._amp_curve = self._amp_plot.plot(pen=pg.mkPen(C['blue'], width=1))
        self._amp_curve.setDownsampling(auto=True, method='peak')
        self._amp_curve.setClipToView(True)
        self._thr_line = pg.InfiniteLine(
            angle=0, movable=False, pen=pg.mkPen(C['mauve'], width=1,
                                                 style=pg.QtCore.Qt.DashLine))
        self._amp_plot.addItem(self._thr_line)
        self._amp_cursor = pg.InfiniteLine(
            angle=90, pen=pg.mkPen(C['red'], width=1))
        self._amp_plot.addItem(self._amp_cursor)

        # Optional raw waveform overlay (teal), hidden by default.
        self._wave_curve = None
        if show_waveform:
            self._wave_curve = self._amp_plot.plot(
                pen=pg.mkPen(C['teal'], width=1))
            self._wave_curve.setDownsampling(auto=True, method='peak')
            self._wave_curve.setClipToView(True)

        self._t_axis: np.ndarray | None = None
        self._t_len = 0

    def set_title(self, text: str) -> None:
        """Set the per-tile title shown above the spectrogram (stream name)."""
        self._spec_plot.setTitle(text, color=C['text'], size='9pt')

    def _time_axis(self, n: int, disp_secs: float) -> np.ndarray:
        """Cached 0..disp_secs ramp of length ``n`` (recomputed on resize)."""
        if self._t_axis is None or self._t_len != n:
            self._t_axis = np.linspace(0.0, disp_secs, n, dtype=np.float32)
            self._t_len = n
        return self._t_axis

    def update_from_entity(self, e) -> None:
        """Repaint from the entity's current display buffers. Cheap enough
        to call at the render rate; pyqtgraph applies the LUT/levels and
        curve downsampling itself."""
        disp_secs = float(e.display_seconds)

        # Spectrogram: resample to the display freq mapping (mel/log/linear),
        # add gain — same transform the matplotlib path used. The LUT +
        # levels are applied by pyqtgraph, not recomputed here.
        spec = e.resample_spec(e.spec_buffer)          # (rows, cols) dB
        n_rows = spec.shape[0]
        self._img.setImage(spec, autoLevels=False)
        clim_lo = min(e.db_floor, e.db_ceil - 0.1)
        self._img.setLevels([clim_lo, e.db_ceil])
        # Map image pixels onto time (x) × freq-row (y) coordinates.
        self._img.setRect(pg.QtCore.QRectF(0.0, 0.0, disp_secs, float(n_rows)))

        cursor_x = (e.write_head / e.sample_rate) % disp_secs
        self._spec_cursor.setValue(cursor_x)
        self._amp_cursor.setValue(cursor_x)

        # Amplitude envelope.
        scale = getattr(e, 'amp_scale', 'log')
        env = _amp_to_display(e.abs_amp_buffer, scale)
        t = self._time_axis(env.shape[0], disp_secs)
        self._amp_curve.setData(t, env)
        # Threshold line in display units.
        thr = e.threshold
        self._thr_line.setValue(
            20.0 * np.log10(max(thr, 1e-4)) if scale == 'log' else thr)

        if self._wave_curve is not None:
            self._wave_curve.setData(t, e.amp_buffer)
            color = C['red'] if getattr(e, 'saturated', False) else C['teal']
            self._wave_curve.setPen(pg.mkPen(color, width=1))
