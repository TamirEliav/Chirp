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
