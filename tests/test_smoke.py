"""Smoke test: the chirp module imports cleanly and exposes its top-level classes.

This is the cheapest possible regression gate. If a refactor accidentally
breaks the module import path or removes a top-level symbol, this test
fails fast — long before any behavioral test would.
"""


def test_chirp_imports():
    import chirp  # noqa: F401  — import side-effect is the test


def test_top_level_classes_present():
    import chirp

    expected = (
        "AudioCapture",
        "SpectrogramAccumulator",
        "BandpassFilter",
        "ThresholdRecorder",
        "RecordingEntity",
        "MiniAmplitudeWidget",
        "RecordingSidebarItem",
        "RecordingSidebar",
        "ChirpWindow",
    )
    missing = [name for name in expected if not hasattr(chirp, name)]
    assert not missing, f"chirp module is missing expected top-level classes: {missing}"


def test_version_string():
    import chirp

    assert isinstance(chirp.__version__, str)
    assert chirp.__version__  # non-empty
