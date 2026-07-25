"""View-mode grid — StreamTile (header + plots) per stream (Phase 3 / v3.3.0).

The view-mode monitoring grid is where rendering cost dominates with many
streams. Each stream gets a :class:`StreamTile`: a thin header row
(stream name, acq/rec/trig status dots, sticky S/D/! badges with
click-to-clear, a 🎧 marker when the stream feeds the audio monitor)
above the pyqtgraph :class:`StreamPlotPanel` (OpenGL spectrogram +
amplitude + events strip).

Efficiency features:
  * Per-tile GPU LUT spectrogram + downsampled curves (from StreamPlotPanel).
  * Off-screen culling — :meth:`update_all` skips tiles scrolled out of the
    viewport, so a tall grid only pays for what's visible. Header state is
    change-detected (same pattern as the sidebar) so idle headers are free.
  * "Active only" filtering is handled by the window — the grid just
    renders whatever (entity, index) pairs it was last rebuilt with.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QGridLayout, QHBoxLayout, QLabel, QVBoxLayout,
                             QWidget)

from chirp.constants import C
from chirp.ui.pg_panel import StreamPlotPanel
from chirp.ui.status_util import compose_error_state


def grid_cell(i: int, n: int, cols: int, fill_order: str = 'column') -> tuple[int, int]:
    """Return the ``(row, col)`` grid position for the ``i``-th tile.

    ``fill_order='column'`` fills down the first column (tiles 0..rows-1)
    before moving to the next column — so with 8 streams in 2 columns,
    streams 1–4 land in column 0 and 5–8 in column 1. ``'row'`` fills
    across the top row first (the legacy behavior). Pure function so the
    layout math is unit-testable without building any Qt widgets.
    """
    c = max(1, int(cols))
    n = max(1, int(n))
    if str(fill_order).lower().startswith('col'):
        rows = (n + c - 1) // c   # ceil
        return i % rows, i // rows
    return i // c, i % c


class StreamTile(QWidget):
    """One view-mode tile: header (status + badges) above the plots.

    ``entity_idx`` is the index into the window's FULL entity list (not
    the grid position) so click-to-clear and monitor-follow keep working
    when the grid shows a filtered subset.
    """

    clicked              = pyqtSignal(int)
    clear_sat_requested  = pyqtSignal(int)
    clear_drops_requested = pyqtSignal(int)
    clear_errors_requested = pyqtSignal(int)

    def __init__(self, entity_idx: int, name: str, *,
                 use_opengl: bool = True, color: str | None = None,
                 parent=None):
        super().__init__(parent)
        self.entity_idx = entity_idx
        self._color = color or None
        # The recognition color is drawn as a rectangle around the whole
        # tile (see _apply_color). WA_StyledBackground + an objectName
        # selector confine the border to the tile frame so it does not
        # cascade into child widgets.
        self.setObjectName('stream_tile')
        self.setAttribute(Qt.WA_StyledBackground, True)
        v = QVBoxLayout(self)
        # Inset the content so the colored frame is fully visible around it.
        v.setContentsMargins(3, 3, 3, 3)
        v.setSpacing(0)

        # ── Header row ────────────────────────────────────────────────
        head = QWidget()
        head.setObjectName('tile_header')
        head.setFixedHeight(24)
        head.setAttribute(Qt.WA_StyledBackground, True)
        head.setStyleSheet(
            f'QWidget#tile_header {{ background-color: {C["mantle"]}; }}')
        h = QHBoxLayout(head)
        h.setContentsMargins(8, 2, 8, 2)
        h.setSpacing(6)

        self._lbl_name = QLabel(name)
        self._lbl_name.setStyleSheet(
            f'color: {C["text"]}; font-weight: bold; font-size: 9pt;')
        h.addWidget(self._lbl_name)

        self._lbl_mon = QLabel('\U0001F3A7')  # headphones — monitor source
        self._lbl_mon.setToolTip('This stream is routed to the audio monitor')
        self._lbl_mon.setStyleSheet(f'color: {C["mauve"]}; font-size: 9pt;')
        self._lbl_mon.hide()
        h.addWidget(self._lbl_mon)

        h.addStretch()

        # Sticky badges (click-to-clear) — same vocabulary as the sidebar.
        self._lbl_sat = QLabel('S')
        self._lbl_drop = QLabel('D')
        self._lbl_err = QLabel('!')
        for lbl in (self._lbl_sat, self._lbl_drop, self._lbl_err):
            lbl.setFixedWidth(14)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(
                f'color: {C["surface2"]}; font-weight: bold; font-size: 9pt;')
            lbl.setCursor(Qt.PointingHandCursor)
            h.addWidget(lbl)
        self._sat_lit = False
        self._drop_lit = False
        self._err_lit = False
        self._err_tip = ''
        self._drop_total = -1

        # Status dots: acq (blue), rec (green), trig (red).
        self._lbl_acq = QLabel('●')
        self._lbl_rec = QLabel('●')
        self._lbl_trig = QLabel('●')
        self._lbl_acq.setToolTip('Acquisition status')
        self._lbl_rec.setToolTip('Recording (armed) status')
        self._lbl_trig.setToolTip('Trigger status (currently writing a WAV)')
        for lbl in (self._lbl_acq, self._lbl_rec, self._lbl_trig):
            lbl.setFixedWidth(12)
            lbl.setStyleSheet(f'color: {C["surface2"]}; font-size: 8pt;')
            h.addWidget(lbl)
        self._last_status = None

        v.addWidget(head)

        # ── Plots ─────────────────────────────────────────────────────
        self.panel = StreamPlotPanel(use_opengl=use_opengl,
                                     show_waveform=False)
        v.addWidget(self.panel, stretch=1)

        self._apply_color()

    # ── Header updates (change-detected) ─────────────────────────────
    def set_name(self, name: str) -> None:
        if self._lbl_name.text() != name:
            self._lbl_name.setText(name)

    def set_color(self, color: str | None) -> None:
        """Update the recognition color (the rectangle around the tile)."""
        color = color or None
        if color == self._color:
            return
        self._color = color
        self._apply_color()

    def _apply_color(self) -> None:
        """Draw a colored rectangle around the whole tile in the stream
        color. Falls back to a neutral surface color when unset. Scoped
        by objectName so the border stays on the tile frame and does not
        cascade into child widgets."""
        col = self._color or C['surface1']
        self.setStyleSheet(
            f'QWidget#stream_tile {{ background-color: {C["base"]}; '
            f'border: 3px solid {col}; border-radius: 4px; }}')

    def update_status(self, acq: bool, rec: bool, trig: bool) -> None:
        key = (acq, rec, trig)
        if key == self._last_status:
            return
        self._last_status = key
        self._lbl_acq.setStyleSheet(
            f'color: {C["blue"] if acq else C["surface2"]}; font-size: 8pt;')
        self._lbl_rec.setStyleSheet(
            f'color: {C["green"] if rec else C["surface2"]}; font-size: 8pt;')
        self._lbl_trig.setStyleSheet(
            f'color: {C["red"] if trig else C["surface2"]}; font-size: 8pt;')

    def update_badges(self, e) -> None:
        """Refresh the sticky S/D/! badges from the entity's flags."""
        sat = bool(getattr(e, 'saturated_ever', False))
        if sat != self._sat_lit:
            self._sat_lit = sat
            color = C['red'] if sat else C['surface2']
            self._lbl_sat.setStyleSheet(
                f'color: {color}; font-weight: bold; font-size: 9pt;')
            self._lbl_sat.setToolTip(
                'Saturation detected during this session — click to clear.'
                if sat else 'No saturation detected on this stream.')

        cap = getattr(e, 'capture', None)
        drop = bool(getattr(cap, 'has_ever_dropped', False))
        total = int(getattr(cap, 'drop_count_total', 0))
        if drop != self._drop_lit or total != self._drop_total:
            self._drop_lit = drop
            self._drop_total = total
            color = C['red'] if drop else C['surface2']
            self._lbl_drop.setStyleSheet(
                f'color: {color}; font-weight: bold; font-size: 9pt;')
            self._lbl_drop.setToolTip(
                f'Dropped {total} chunk{"s" if total != 1 else ""} since '
                f'last reset — click to clear.'
                if drop else 'No dropped chunks recorded for this stream.')

        any_err, tip = compose_error_state(e)
        if any_err != self._err_lit or tip != self._err_tip:
            self._err_lit = any_err
            self._err_tip = tip
            color = C['peach'] if any_err else C['surface2']
            self._lbl_err.setStyleSheet(
                f'color: {color}; font-weight: bold; font-size: 9pt;')
            self._lbl_err.setToolTip(tip)

    def set_monitored(self, on: bool) -> None:
        self._lbl_mon.setVisible(bool(on))

    # ── Clicks ────────────────────────────────────────────────────────
    def mousePressEvent(self, event):
        child = self.childAt(event.pos())
        if child is self._lbl_sat and self._sat_lit:
            self.clear_sat_requested.emit(self.entity_idx)
            return
        if child is self._lbl_drop and self._drop_lit:
            self.clear_drops_requested.emit(self.entity_idx)
            return
        if child is self._lbl_err and self._err_lit:
            self.clear_errors_requested.emit(self.entity_idx)
            return
        self.clicked.emit(self.entity_idx)
        super().mousePressEvent(event)


class MultiStreamGrid(QWidget):
    # Re-emitted from tiles with the ENTITY index (into the window's
    # full list), so filtered grids keep addressing the right stream.
    tile_clicked           = pyqtSignal(int)
    clear_sat_requested    = pyqtSignal(int)
    clear_drops_requested  = pyqtSignal(int)
    clear_errors_requested = pyqtSignal(int)

    def __init__(self, parent=None, *, use_opengl: bool = True):
        super().__init__(parent)
        self._use_opengl = use_opengl
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(4, 4, 4, 4)
        self._grid.setSpacing(4)
        self._tiles: list[StreamTile] = []
        self._cols = 1
        self._tile_min_h = 300
        self.setStyleSheet(f'background-color: {C["base"]};')
        # Empty-state hint (active-only filter with nothing running).
        self._hint = QLabel('', self)
        self._hint.setAlignment(Qt.AlignCenter)
        self._hint.setStyleSheet(
            f'color: {C["subtext"]}; font-size: 11pt; padding: 40px;')
        self._hint.hide()

    def rebuild(self, entities, cols: int | None = None,
                indices: list[int] | None = None,
                fill_order: str = 'column',
                empty_hint: str = 'No recordings') -> None:
        """(Re)create one tile per entity in a ``cols``-wide grid.

        ``indices[i]`` is the position of ``entities[i]`` in the
        window's full entity list (defaults to 0..n-1 when the grid
        shows everything).

        ``fill_order`` controls how tiles map to grid cells:
          * ``'column'`` — fill down the first column (streams 1..rows)
            before moving to the next column.
          * ``'row'`` — fill across the top row first (legacy behavior).
        """
        if cols:
            self._cols = max(1, int(cols))
        for t in self._tiles:
            self._grid.removeWidget(t)
            t.setParent(None)
            t.deleteLater()
        self._tiles = []
        self._grid.removeWidget(self._hint)
        entities = list(entities)
        if indices is None:
            indices = list(range(len(entities)))
        n = len(entities)
        if n == 0:
            self._hint.setText(empty_hint)
            self._grid.addWidget(self._hint, 0, 0)
            self._hint.show()
            return
        self._hint.hide()
        c = max(1, min(self._cols, n))
        for i, (e, idx) in enumerate(zip(entities, indices)):
            tile = StreamTile(idx, e.name, use_opengl=self._use_opengl,
                              color=getattr(e, 'color', '') or None)
            tile.panel.set_title('')
            tile.setMinimumHeight(self._tile_min_h)
            tile.clicked.connect(self.tile_clicked.emit)
            tile.clear_sat_requested.connect(self.clear_sat_requested.emit)
            tile.clear_drops_requested.connect(self.clear_drops_requested.emit)
            tile.clear_errors_requested.connect(
                self.clear_errors_requested.emit)
            r, col = grid_cell(i, n, c, fill_order)
            self._grid.addWidget(tile, r, col)
            self._tiles.append(tile)

    def set_tile_height(self, h: int) -> None:
        self._tile_min_h = int(h)
        for t in self._tiles:
            t.setMinimumHeight(self._tile_min_h)

    def update_all(self, entities, monitor_source_id=None) -> None:
        """Repaint visible tiles + refresh every header (headers are
        cheap and change-detected; plots of off-screen tiles are
        skipped)."""
        for tile, e in zip(self._tiles, entities):
            tile.set_name(e.name)
            tile.set_color(getattr(e, 'color', '') or None)
            tile.update_status(e.acq_running, e.rec_enabled,
                               e.recorder.is_recording)
            tile.update_badges(e)
            tile.set_monitored(monitor_source_id is not None
                               and id(e) == monitor_source_id)
            if tile.visibleRegion().isEmpty():
                continue
            tile.panel.update_from_entity(e)

    # Back-compat: some code paths iterate panels directly.
    @property
    def _panels(self):
        return [t.panel for t in self._tiles]
