"""Chirp — Sound Analysis & Recording.

Package root. Every subsystem lives in its own submodule after the
Phase 1 refactor; this file is just a re-export surface so downstream
imports like `from chirp import ChirpWindow` keep working.

Subpackage layout:

  chirp.constants       module-level constants + Catppuccin palette + QSS
  chirp.dsp             SpectrogramAccumulator, BandpassFilter, entropy
  chirp.audio           AudioCapture, devices
  chirp.recording       ThresholdRecorder, RecordingEntity, WAV writer
  chirp.config          settings schema (build / load / migrate)
  chirp.ui              ChirpWindow, sidebar widgets, theme

Refactor plan: ~/.claude/plans/robust-yawning-spring.md
"""

__version__ = "2.0.0"

# Re-exports for backward-compatible `from chirp import X` callers and
# for tests that monkeypatch via `chirp.<Class>`. Keep this list in sync
# with anything referenced by name from outside the package.
from chirp.constants import *  # noqa: F401,F403
from chirp.audio import AudioCapture  # noqa: F401
from chirp.dsp import (  # noqa: F401
    BandpassFilter,
    SpectrogramAccumulator,
    normalized_spectral_entropy as _spectral_entropy,
)
from chirp.recording.trigger import ThresholdRecorder  # noqa: F401
from chirp.recording.entity import RecordingEntity  # noqa: F401
from chirp.ui.window import ChirpWindow, main  # noqa: F401
from chirp.ui.sidebar import (  # noqa: F401
    MiniAmplitudeWidget,
    RecordingSidebar,
    RecordingSidebarItem,
)


if __name__ == '__main__':
    main()
