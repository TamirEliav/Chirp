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

from chirp.constants import (C, CHUNK_FRAMES, N_DISPLAY_ROWS, SPEC_DB_MAX,
                             SPEC_DB_MIN)

# Inferno LUT (256 RGB rows) built once from matplotlib's colormap so the
# spectrogram matches the previous renderer's look. Applied on the GPU/Qt
# side rather than recomputed per frame.
_INFERNO_LUT = (
    matplotlib.colormaps['inferno'](np.linspace(0.0, 1.0, 256))[:, :3] * 255
).astype(np.ubyte)

# pyqtgraph image data is indexed [row, col] == [freq, time] in row-major.
pg.setConfigOption('imageAxisOrder', 'row-major')

# Empty cell in the detect/record strip (Catppuccin surface0).
EVENTS_BG = np.array([49, 50, 68, 255], dtype=np.ubyte)


def _amp_to_display(buf: np.ndarray, scale: str) -> np.ndarray:
    """Linear envelope → display units (dB when scale == 'log')."""
    if scale == 'log':
        return 20.0 * np.log10(np.maximum(np.abs(buf), 1e-4))
    return buf


def spec_ytick_list(e) -> list:
    """pyqtgraph ``setTicks`` structure for a spectrogram axis.

    The spectrogram image is drawn in display-ROW coordinates (0 ..
    N_DISPLAY_ROWS) after the mel/log/linear resampling — the axis must
    therefore label row positions with the frequencies they map to via
    ``e.display_freqs``, exactly like the retired matplotlib
    ``_apply_spec_yticks`` did. Returns ``[(row_pos, label), ...]``.
    """
    freqs = getattr(e, 'display_freqs', None)
    if freqs is None or len(freqs) == 0:
        return []
    n_dst = len(freqs)
    f_lo, f_hi = float(freqs[0]), float(freqs[-1])
    if getattr(e, 'freq_scale', 'Mel') == 'Linear':
        step = max(500, round((f_hi - f_lo) / 8 / 500) * 500) or 5000
        tick_freqs = np.arange(np.ceil(f_lo / step) * step, f_hi + 1, step)
    else:
        tick_freqs = np.array([50, 100, 200, 500, 1000, 2000, 5000,
                               10000, 20000, 50000, 100000], dtype=float)
        tick_freqs = tick_freqs[(tick_freqs >= f_lo) & (tick_freqs <= f_hi)]
    rows = np.interp(tick_freqs, freqs, np.arange(n_dst))
    return [(float(r), f'{f/1000:.0f}k' if f >= 1000 else f'{int(f)}')
            for r, f in zip(rows, tick_freqs)]


def spec_ytick_key(e) -> tuple:
    """Cache key for the tick set — recompute only when the frequency
    mapping actually changes."""
    return (getattr(e, 'freq_scale', ''), getattr(e, 'display_freq_lo', 0.0),
            getattr(e, 'display_freq_hi', 0.0), getattr(e, 'sample_rate', 0))


def events_rgba(e):
    """2-row RGBA strip for the detect/record events display: row 0
    (bottom) = detect mask, row 1 (top) = record mask, one column per
    CHUNK_FRAMES block. Shared by the Config panel and the View-mode
    tiles (moved here from config_panel in Phase 3 of v3.3.0).

    uint8 RGBA — direct colour, so pyqtgraph needs no levels (a float
    image without levels raises in ImageItem.render).
    """
    nc = e._n_cols
    need = nc * CHUNK_FRAMES
    if e.detect_mask_buffer.shape[0] < need:
        return None
    det = e.detect_mask_buffer[:need].reshape(nc, CHUNK_FRAMES).any(axis=1)
    rec = e.record_mask_buffer[:need].reshape(nc, CHUNK_FRAMES).any(axis=1)
    disc_buf = getattr(e, 'discard_mask_buffer', None)
    disc = (disc_buf[:need].reshape(nc, CHUNK_FRAMES).any(axis=1)
            if disc_buf is not None else None)
    rgba = np.zeros((2, nc, 4), dtype=np.ubyte)
    # Background (Catppuccin surface0) so empty cells aren't pure black.
    rgba[...] = EVENTS_BG
    # Row 0 = detect (yellow), row 1 = record (green).
    rgba[0, det] = (249, 226, 175, 255)
    rgba[1, rec] = (166, 227, 161, 255)
    # Discarded events (min-total-crossing filter dropped them, no WAV
    # written) paint red over the record row — applied last so red wins
    # over any residual green in an overlapping column.
    if disc is not None:
        rgba[1, disc] = (243, 139, 168, 255)
    return rgba


def _decimate_max(y: np.ndarray, max_cols: int) -> np.ndarray:
    """H3: peak (max) decimation for non-negative envelope data —
    reduces the per-tick numpy work (GIL-holding) from O(buffer) to a
    single reduction; everything downstream is O(pixels)."""
    n = y.shape[0]
    if n <= max_cols * 2:
        return y
    k = n // max_cols
    m = n // k
    return y[:m * k].reshape(m, k).max(axis=1)


_MAX_ENV_COLS = 2048  # view-mode tiles are small; 2k columns is plenty


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

        # ── Detect/record events strip (Phase 3 / TODO#14) ─────────────
        self._ev_plot = self.addPlot(row=2, col=0)
        self._ev_plot.setMenuEnabled(False)
        self._ev_plot.setMouseEnabled(x=False, y=False)
        self._ev_plot.setMaximumHeight(46)
        self._ev_plot.setXLink(self._spec_plot)
        self._ev_plot.getAxis('left').setTicks(
            [[(0.5, 'det'), (1.5, 'rec')]])
        self._ev_plot.getAxis('bottom').setStyle(showValues=False)
        self._events_img = pg.ImageItem()
        self._ev_plot.addItem(self._events_img)
        self._ev_cursor = pg.InfiniteLine(
            angle=90, pen=pg.mkPen(C['red'], width=1))
        self._ev_plot.addItem(self._ev_cursor)

        self._t_axis: np.ndarray | None = None
        self._t_len = 0
        self._ytick_key: tuple | None = None

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
        # Frequency y-tick labels — the rows are mel/log/linear-mapped,
        # so raw row indices are meaningless to the user. Recomputed
        # only when the mapping changes.
        key = spec_ytick_key(e)
        if key != self._ytick_key:
            self._ytick_key = key
            self._spec_plot.getAxis('left').setTicks([spec_ytick_list(e)])

        cursor_x = (e.display_head / e.sample_rate) % disp_secs
        self._spec_cursor.setValue(cursor_x)
        self._amp_cursor.setValue(cursor_x)
        self._ev_cursor.setValue(cursor_x)

        # Detect/record events strip.
        rgba = events_rgba(e)
        if rgba is not None:
            self._events_img.setImage(rgba, autoLevels=False)
            self._events_img.setRect(
                pg.QtCore.QRectF(0.0, 0.0, disp_secs, 2.0))

        # Amplitude envelope — peak-decimated to display resolution
        # before the dB conversion (H3).
        scale = getattr(e, 'amp_scale', 'log')
        env = _amp_to_display(_decimate_max(e.abs_amp_buffer, _MAX_ENV_COLS),
                              scale)
        t = self._time_axis(env.shape[0], disp_secs)
        self._amp_curve.setData(t, env)
        # Threshold line in display units.
        thr = e.threshold
        self._thr_line.setValue(
            20.0 * np.log10(max(thr, 1e-4)) if scale == 'log' else thr)

        if self._wave_curve is not None:
            # Separate ramp — the envelope axis above is decimated and
            # no longer matches the raw buffer length.
            wave = e.amp_buffer
            t_w = np.linspace(0.0, disp_secs, wave.shape[0], dtype=np.float32)
            self._wave_curve.setData(t_w, wave)
            color = C['red'] if getattr(e, 'saturated', False) else C['teal']
            self._wave_curve.setPen(pg.mkPen(color, width=1))
