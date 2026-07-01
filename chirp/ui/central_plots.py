"""MultiStreamGrid — pyqtgraph view-mode grid (redesign Phase 4).

The view-mode monitoring grid is where rendering cost dominates with many
streams, so it is ported to pyqtgraph first. Each stream gets a compact
:class:`StreamPlotPanel` tile (OpenGL spectrogram + amplitude). The window
swaps this widget into the central scroll area when entering view mode and
swaps the matplotlib canvas back when leaving, so config-mode editing is
untouched during the migration.

Efficiency features:
  * Per-tile GPU LUT spectrogram + downsampled curves (from StreamPlotPanel).
  * Off-screen culling — :meth:`update_all` skips tiles scrolled out of the
    viewport, so a tall grid only pays for what's visible.
"""

from __future__ import annotations

from PyQt5.QtWidgets import QGridLayout, QWidget

from chirp.constants import C
from chirp.ui.pg_panel import StreamPlotPanel


class MultiStreamGrid(QWidget):
    def __init__(self, parent=None, *, use_opengl: bool = True):
        super().__init__(parent)
        self._use_opengl = use_opengl
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(4, 4, 4, 4)
        self._grid.setSpacing(4)
        self._panels: list[StreamPlotPanel] = []
        self._cols = 1
        self._tile_min_h = 300
        self.setStyleSheet(f'background-color: {C["base"]};')

    def rebuild(self, entities, cols: int | None = None) -> None:
        """(Re)create one tile per entity in a ``cols``-wide grid."""
        if cols:
            self._cols = max(1, int(cols))
        for p in self._panels:
            self._grid.removeWidget(p)
            p.setParent(None)
            p.deleteLater()
        self._panels = []
        n = len(entities)
        c = max(1, min(self._cols, n)) if n else 1
        for i, e in enumerate(entities):
            panel = StreamPlotPanel(use_opengl=self._use_opengl,
                                    show_waveform=False)
            panel.set_title(e.name)
            panel.setMinimumHeight(self._tile_min_h)
            self._grid.addWidget(panel, i // c, i % c)
            self._panels.append(panel)

    def set_tile_height(self, h: int) -> None:
        self._tile_min_h = int(h)
        for p in self._panels:
            p.setMinimumHeight(self._tile_min_h)

    def update_all(self, entities) -> None:
        """Repaint visible tiles. Tiles scrolled out of the viewport are
        skipped (off-screen culling) so a tall grid stays cheap."""
        for panel, e in zip(self._panels, entities):
            if panel.visibleRegion().isEmpty():
                continue
            panel.update_from_entity(e)
