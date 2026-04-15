"""Tests for `chirp.recording.entity.RecordingEntity.ingest_chunk`.

These pin behavior of the per-stream processing pipeline. RecordingEntity
constructs an AudioCapture in __init__ which silently swallows
device-open failures in headless environments, so it is safe to build
with `device_id=None`.
"""

import numpy as np

from chirp.recording.entity import RecordingEntity


def _entity():
    return RecordingEntity(name="Test", device_id=None)


# ── #18: saturation must reflect the raw input, not the post-filter signal ──

def test_saturation_detects_clipped_raw_input_even_with_bandpass():
    """A clipped raw chunk should mark the entity saturated even when a
    bandpass filter at frequencies far from the clip's energy attenuates
    the post-filter peak below 0.99.

    Regression for #18 (c11): pre-fix the code measured `trigger_peak`
    (post-filter) and would report `saturated=False` on a clipped DC /
    low-frequency input as long as the band filter excluded its energy.
    """
    e = _entity()
    e.freq_filter_enabled = True
    e.freq_lo = 5000.0
    e.freq_hi = 10000.0

    # 100 Hz clipped tone (peaks at ±1.0). The 5–10 kHz bandpass kills
    # almost all of it post-filter, but the raw peak is exactly 1.0.
    sr = e.sample_rate
    n = 1024
    t = np.arange(n, dtype=np.float32) / sr
    chunk = np.sign(np.sin(2 * np.pi * 100.0 * t)).astype(np.float32)

    e.ingest_chunk(chunk)
    assert e.saturated is True


def test_saturation_false_on_quiet_input():
    e = _entity()
    chunk = np.full(1024, 0.1, dtype=np.float32)
    e.ingest_chunk(chunk)
    assert e.saturated is False


def test_saturation_true_on_unfiltered_clip():
    """Sanity check: clipping with the band filter off still trips."""
    e = _entity()
    e.freq_filter_enabled = False
    chunk = np.full(1024, 0.999, dtype=np.float32)
    e.ingest_chunk(chunk)
    assert e.saturated is True
