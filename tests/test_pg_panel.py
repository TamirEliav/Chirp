"""Headless smoke tests for the pyqtgraph stream panel (Phase 4).

Constructs the panel without OpenGL under the offscreen Qt platform and
drives one update from a real entity, catching API/shape errors. Visual
correctness + OpenGL are verified by launching scripts/pg_demo.py on a
real display.
"""

import numpy as np
import pytest

pytest.importorskip('pyqtgraph')

from PyQt5.QtWidgets import QApplication  # noqa: E402

from chirp.constants import CHUNK_FRAMES  # noqa: E402
from chirp.recording.entity import RecordingEntity  # noqa: E402
from chirp.ui.pg_panel import StreamPlotPanel  # noqa: E402


def _app():
    return QApplication.instance() or QApplication([])


def test_panel_constructs_headless():
    _app()
    panel = StreamPlotPanel(use_opengl=False)
    assert panel is not None


def test_panel_updates_from_entity():
    _app()
    panel = StreamPlotPanel(use_opengl=False, show_waveform=True)
    e = RecordingEntity(name='pg', device_id=None)
    # Feed a few chunks of tone through the real DSP pipeline so the
    # display buffers are populated, then render.
    sr = e.sample_rate
    for k in range(8):
        t = (np.arange(CHUNK_FRAMES) + k * CHUNK_FRAMES) / sr
        chunk = (0.4 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
        e.ingest_chunk(chunk)
    # Should not raise for either amplitude scale.
    e.amp_scale = 'log'
    panel.update_from_entity(e)
    e.amp_scale = 'linear'
    panel.update_from_entity(e)
    e.close()


def test_panel_updates_when_buffers_empty():
    _app()
    panel = StreamPlotPanel(use_opengl=False)
    e = RecordingEntity(name='pg2', device_id=None)
    # No audio ingested — buffers are at their initial fill; render must
    # still be safe.
    panel.update_from_entity(e)
    e.close()
