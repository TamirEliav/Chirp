"""UI theme — Catppuccin Mocha palette and QSS stylesheet.

Thin re-export from `chirp.constants`. Extracted in the Phase 1
refactor (plan: c08) so future theme variants (light mode, high
contrast, etc.) have a clear home without touching `constants.py`,
which is supposed to stay cheap to import for non-UI modules.
"""

from chirp.constants import C, QSS

__all__ = ["C", "QSS"]
