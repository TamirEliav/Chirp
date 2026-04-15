"""Chirp — UI subpackage (Qt widgets + matplotlib canvas).

Contains everything that touches PyQt5 or matplotlib:

  - `theme`:   Catppuccin Mocha palette + QSS stylesheet. Currently
               re-exports from `chirp.constants`; exists as its own
               module so future theme variants have a clear home.
  - `sidebar`: MiniAmplitudeWidget, RecordingSidebarItem, RecordingSidebar
               — the per-stream list on the left-hand side.
  - `window`:  ChirpWindow, the top-level QMainWindow. Still large;
               Phase 2 + 3 fixes (#13, #17, #19, #11) will chip away
               at the internals but the class boundary stays here.
"""

from chirp.ui.window import ChirpWindow, main
from chirp.ui.sidebar import (
    MiniAmplitudeWidget,
    RecordingSidebar,
    RecordingSidebarItem,
)

__all__ = [
    "ChirpWindow",
    "main",
    "MiniAmplitudeWidget",
    "RecordingSidebar",
    "RecordingSidebarItem",
]
