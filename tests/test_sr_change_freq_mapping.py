"""Repro + regression pin for TODO #7: spectrogram frequencies after a
sample-rate change.

Feeds a pure tone through the real ingest path at one sample rate,
locates the spectrogram peak row, and checks that ``display_freqs``
maps that row back to the tone's frequency. Then changes the sample
rate (both up and down) and repeats — the mapping must stay correct
and the display range must follow the new Nyquist when it was pinned
to the old one.
"""

import numpy as np
import pytest

from chirp.constants import CHUNK_FRAMES
from chirp.recording.entity import RecordingEntity


def _feed_tone(e: RecordingEntity, freq: float, seconds: float = 0.6):
    sr = e.sample_rate
    n = int(seconds * sr / CHUNK_FRAMES) * CHUNK_FRAMES
    t = np.arange(n) / sr
    tone = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    for off in range(0, n, CHUNK_FRAMES):
        e.ingest_chunk(tone[off:off + CHUNK_FRAMES])


def _peak_freq_from_display(e: RecordingEntity) -> float:
    """Frequency of the strongest display row, via the same
    resample_spec + display_freqs mapping the panels use."""
    disp = e.resample_spec(e.spec_buffer)     # (rows, cols) dB
    col_energy = disp.max(axis=0)
    col = int(np.argmax(col_energy))          # a column with the tone
    row = int(np.argmax(disp[:, col]))
    return float(e.display_freqs[row])


@pytest.mark.parametrize('scale', ['Linear', 'Mel'])
def test_tone_maps_to_correct_display_freq(scale):
    e = RecordingEntity(name='t', device_id=None, sample_rate=44100)
    try:
        e.freq_scale = scale
        e.rebuild_freq_mapping()
        _feed_tone(e, 5000.0)
        got = _peak_freq_from_display(e)
        assert abs(got - 5000.0) < 300.0, f'{scale}: tone at {got:.0f} Hz'
    finally:
        e.close()


@pytest.mark.parametrize('scale', ['Linear', 'Mel'])
@pytest.mark.parametrize('new_sr', [96000, 22050])
def test_tone_maps_correctly_after_sr_change(scale, new_sr):
    e = RecordingEntity(name='t', device_id=None, sample_rate=44100)
    try:
        e.freq_scale = scale
        e.rebuild_freq_mapping()
        _feed_tone(e, 5000.0)
        e.change_sample_rate(new_sr)
        assert e.sample_rate == new_sr
        _feed_tone(e, 5000.0)
        got = _peak_freq_from_display(e)
        assert abs(got - 5000.0) < 400.0, (
            f'{scale} @ {new_sr}: tone shows at {got:.0f} Hz')
    finally:
        e.close()


def test_display_range_follows_nyquist_up():
    """When the display high limit sat AT the old Nyquist (the default),
    raising the SR must widen it to the new Nyquist — otherwise the
    spectrogram silently keeps showing only the old range."""
    e = RecordingEntity(name='t', device_id=None, sample_rate=44100)
    try:
        assert e.display_freq_hi == pytest.approx(22050.0)
        e.change_sample_rate(96000)
        assert e.display_freq_hi == pytest.approx(48000.0)
    finally:
        e.close()


def test_display_range_user_narrowed_is_preserved():
    """A user-narrowed display range must NOT be blown open by an SR
    change (only clamped down if it exceeds the new Nyquist)."""
    e = RecordingEntity(name='t', device_id=None, sample_rate=44100)
    try:
        e.display_freq_hi = 10000.0
        e.rebuild_freq_mapping()
        e.change_sample_rate(96000)
        assert e.display_freq_hi == pytest.approx(10000.0)
        e.change_sample_rate(8000)
        assert e.display_freq_hi == pytest.approx(4000.0)
    finally:
        e.close()
