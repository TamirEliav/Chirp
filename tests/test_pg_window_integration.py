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


def test_grid_cell_column_major():
    """Column-major: 8 tiles / 2 cols → 1–4 in col 0, 5–8 in col 1
    (the layout the user asked for). Pure function, no widgets."""
    from chirp.ui.central_plots import grid_cell
    n, cols = 8, 2
    pos = [grid_cell(i, n, cols, 'column') for i in range(n)]
    # rows = ceil(8/2) = 4
    assert pos == [(0, 0), (1, 0), (2, 0), (3, 0),
                   (0, 1), (1, 1), (2, 1), (3, 1)]
    # Column 0 holds tiles 0..3, column 1 holds 4..7.
    assert [c for _, c in pos] == [0, 0, 0, 0, 1, 1, 1, 1]


def test_grid_cell_row_major():
    """Row-major (legacy): 8 tiles / 2 cols fill across the top row first."""
    from chirp.ui.central_plots import grid_cell
    n, cols = 8, 2
    pos = [grid_cell(i, n, cols, 'row') for i in range(n)]
    assert pos == [(0, 0), (0, 1), (1, 0), (1, 1),
                   (2, 0), (2, 1), (3, 0), (3, 1)]


def test_grid_cell_uneven_column_major():
    """Uneven fill: 5 tiles / 2 cols → col 0 gets 3 (ceil), col 1 gets 2."""
    from chirp.ui.central_plots import grid_cell
    n, cols = 5, 2
    pos = [grid_cell(i, n, cols, 'column') for i in range(n)]
    # rows = ceil(5/2) = 3
    assert pos == [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1)]
