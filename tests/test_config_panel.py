"""Headless smoke tests for the pyqtgraph config panel (Phase 4b).

Single-panel construction + updates across display modes / spectral / drag,
catching API + shape errors. Full-window integration is verified by a
standalone smoke run and by launching the app. (Multiple pyqtgraph widgets
torn down inside pytest's batch process can segfault at exit — see
test_pg_window_integration — so this keeps to one panel instance.)
"""

import numpy as np
import pytest

pytest.importorskip('pyqtgraph')

from PyQt5.QtWidgets import QApplication  # noqa: E402

from chirp.constants import CHUNK_FRAMES  # noqa: E402
from chirp.recording.entity import RecordingEntity  # noqa: E402
from chirp.ui.config_panel import ConfigPlotPanel  # noqa: E402


def _app():
    return QApplication.instance() or QApplication([])


def _feed(e, n=10):
    for k in range(n):
        t = (np.arange(CHUNK_FRAMES) + k * CHUNK_FRAMES) / e.sample_rate
        e.ingest_chunk((0.4 * np.sin(2 * np.pi * 1200 * t)).astype(np.float32))


def test_config_panel_renders_all_display_modes():
    _app()
    panel = ConfigPlotPanel(use_opengl=False)
    e = RecordingEntity(name='cfg', device_id=None)
    try:
        _feed(e)
        for mode in ('Spectrogram', 'Waveform', 'Both'):
            e.display_mode = mode
            panel.update_from_entity(e)   # rebuild_if_needed + render
            panel.update_from_entity(e)
        # Spectral mode adds the entropy row.
        e.spectral_trigger_mode = 'Amp AND Spectral'
        e.display_mode = 'Spectrogram'
        panel.update_from_entity(e)
        assert panel._entropy is not None
    finally:
        e.close()


def test_config_panel_threshold_signals():
    _app()
    panel = ConfigPlotPanel(use_opengl=False)
    e = RecordingEntity(name='cfg2', device_id=None)
    got = []
    panel.thresholdChanged.connect(lambda v: got.append(v))
    try:
        e.amp_scale = 'linear'
        panel.rebuild(e)
        # Programmatic set must NOT emit (suppressed).
        panel.set_threshold(0.3)
        assert got == []
        # Simulating a user drag emits the converted linear value.
        panel._thr_line.setValue(0.42)
        assert got and abs(got[-1] - 0.42) < 1e-6
    finally:
        e.close()
