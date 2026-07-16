"""Shared pytest configuration for the Chirp test suite.

Adds the repo root to sys.path so tests can `import chirp` while the
project is still a single-file monolith (pre-refactor). After the
package extraction in Phase 1 of the refactor plan, tests will import
from the `chirp` package directly and this shim becomes redundant.
"""

import os
import sys
from pathlib import Path

# Force a non-interactive matplotlib backend BEFORE chirp is imported,
# so the test suite can run on headless CI without a display server.
# chirp.py calls matplotlib.use('Qt5Agg') at import time, but pytest
# imports conftest first — setting MPLBACKEND in the environment is
# the only knob matplotlib honors before any backend is selected.
os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _disable_timestamp_divergence_check(monkeypatch):
    """Disable the publish-time timestamp sanity check by default.

    Many tests write WAVs with fixed historical onsets (e.g.
    2024-01-01); the watchdog would flag every one of them and pollute
    the singleton writer pool's error stats across tests.
    tests/test_timestamp_divergence.py re-enables it locally."""
    from chirp.recording import writer
    monkeypatch.setattr(writer, 'TIMESTAMP_DIVERGENCE_SEC', None)
