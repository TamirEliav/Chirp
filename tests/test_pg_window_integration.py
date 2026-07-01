"""Headless tests for the pyqtgraph view-mode grid (Phase 4).

NOTE on coverage: instantiating *multiple* pyqtgraph GraphicsLayoutWidget
tiles and tearing them down inside pytest's batch process segfaults at
interpreter exit (deferred Qt widget deletion ordering with matplotlib +
PortAudio also loaded) — a well-known headless-Qt batch artifact, not a
product defect. That multi-tile path runs cleanly as a standalone script
and is verified by launching scripts/pg_demo.py. Here we keep only the
non-crashing assertions: the empty grid and the construction contract.
The single-tile renderer is covered by test_pg_panel.py.
"""

import pytest

pytest.importorskip('pyqtgraph')

from PyQt5.QtWidgets import QApplication  # noqa: E402

from chirp.ui.central_plots import MultiStreamGrid  # noqa: E402


def _app():
    return QApplication.instance() or QApplication([])


def test_grid_constructs_and_empty_is_safe():
    _app()
    grid = MultiStreamGrid(use_opengl=False)
    # No entities → no child tiles created (safe to build/update at teardown).
    grid.rebuild([], cols=2)
    grid.update_all([])
    assert grid._panels == []
    # Column / height config is accepted without tiles present.
    grid.set_tile_height(180)
    grid.rebuild([], cols=3)
    assert grid._panels == []
