"""Unit tests for the `chirp.dsp` subpackage.

These tests pin the behavior of the pure-numpy DSP primitives after
their extraction from the monolith (plan: c03). They intentionally do
*not* boot a QApplication or any PyQt machinery — the dsp subpackage is
a leaf of the dependency graph and must stay importable on its own.
"""

import numpy as np
import pytest

from chirp.dsp import (
    BandpassFilter,
    SpectrogramAccumulator,
    normalized_spectral_entropy,
)
from chirp.dsp.entropy import SILENT_MAGNITUDE_THRESHOLD


# ── Spectral entropy ─────────────────────────────────────────────────────────

def test_entropy_pure_tone_is_near_zero():
    """A single-bin spike is a pure tone → entropy ≈ 0."""
    mag = np.zeros(1024, dtype=np.float32)
    mag[42] = 1.0
    assert normalized_spectral_entropy(mag) == pytest.approx(0.0, abs=1e-9)


def test_entropy_white_noise_is_near_one():
    """Flat magnitude spectrum → maximally uniform → entropy == 1.0."""
    mag = np.ones(1024, dtype=np.float32)
    assert normalized_spectral_entropy(mag) == pytest.approx(1.0, abs=1e-9)


def test_entropy_silent_returns_sentinel():
    """Silence (total magnitude below the numeric floor) returns the
    legacy sentinel 1.0 rather than NaN. This pins the monolith's
    behavior — callers that care about "silent vs noisy" must inspect
    the magnitudes directly.
    """
    mag = np.full(1024, SILENT_MAGNITUDE_THRESHOLD / 10.0, dtype=np.float64)
    assert normalized_spectral_entropy(mag) == 1.0


def test_entropy_two_equal_bins_is_fraction_of_log():
    """Two equal bins in a length-N vector → H = log2(2) / log2(N)."""
    n = 1024
    mag = np.zeros(n, dtype=np.float32)
    mag[10] = mag[500] = 1.0
    expected = np.log2(2) / np.log2(n)
    assert normalized_spectral_entropy(mag) == pytest.approx(expected, rel=1e-6)


def test_entropy_monotone_vs_noise_ordering():
    """Pure tone must have strictly lower entropy than white noise."""
    n = 2048
    tone = np.zeros(n, dtype=np.float32)
    tone[100] = 1.0
    noise = np.ones(n, dtype=np.float32)
    assert normalized_spectral_entropy(tone) < normalized_spectral_entropy(noise)


# ── SpectrogramAccumulator ───────────────────────────────────────────────────

def test_spectrogram_column_shape_and_dtype():
    acc = SpectrogramAccumulator(nperseg=512)
    chunk = np.random.default_rng(0).standard_normal(1024).astype(np.float32)
    db, lin = acc.compute_column(chunk)
    # rfft of length N → N//2 + 1 bins
    assert db.shape == (512 // 2 + 1,)
    assert lin.shape == (512 // 2 + 1,)
    assert db.dtype == np.float32
    assert np.all(np.isfinite(db))
    assert np.all(lin >= 0.0)


def test_spectrogram_overlap_continuity():
    """Two consecutive short chunks should produce the same column as
    one long chunk that concatenates them, because the accumulator
    carries its overlap buffer across calls.
    """
    rng = np.random.default_rng(1)
    a = rng.standard_normal(512).astype(np.float32)
    b = rng.standard_normal(512).astype(np.float32)

    acc_split = SpectrogramAccumulator(nperseg=512)
    acc_split.compute_column(a)
    db_split, _ = acc_split.compute_column(b)

    acc_combined = SpectrogramAccumulator(nperseg=512)
    db_combined, _ = acc_combined.compute_column(np.concatenate([a, b]))

    np.testing.assert_allclose(db_split, db_combined, atol=1e-5)


def test_spectrogram_detects_known_tone():
    """A pure 1 kHz tone at 44.1 kHz should peak near bin
    round(1000 * nperseg / 44100).
    """
    sr = 44100
    nperseg = 2048
    f = 1000.0
    t = np.arange(nperseg, dtype=np.float32) / sr
    tone = np.sin(2 * np.pi * f * t).astype(np.float32)
    acc = SpectrogramAccumulator(nperseg=nperseg)
    _, lin = acc.compute_column(tone)
    expected_bin = int(round(f * nperseg / sr))
    assert abs(int(np.argmax(lin)) - expected_bin) <= 1


# ── BandpassFilter ───────────────────────────────────────────────────────────

def test_bandpass_invalid_band_passes_through():
    """A degenerate (lo >= hi) band disables the filter and returns the
    raw chunk with its raw peak — matches the legacy monolith behavior.
    """
    f = BandpassFilter(sample_rate=44100)
    chunk = np.array([0.0, 0.2, -0.5, 0.3], dtype=np.float32)
    out, peak = f.filter_chunk(chunk, 8000.0, 4000.0)  # lo > hi
    np.testing.assert_array_equal(out, chunk)
    assert peak == pytest.approx(0.5)


def test_bandpass_rejects_out_of_band_tone():
    """A 100 Hz tone should be strongly attenuated by a 5–10 kHz band."""
    sr = 44100
    n = sr  # 1 s
    t = np.arange(n, dtype=np.float32) / sr
    low_tone = np.sin(2 * np.pi * 100.0 * t).astype(np.float32)

    f = BandpassFilter(sample_rate=sr)
    out, _ = f.filter_chunk(low_tone, 5000.0, 10000.0)
    # After filter warm-up, the tail should be heavily attenuated.
    tail_peak = float(np.max(np.abs(out[sr // 2:])))
    assert tail_peak < 0.05, f"expected rejection, got tail_peak={tail_peak}"


def test_bandpass_passes_in_band_tone():
    """A 7 kHz tone should survive a 5–10 kHz bandpass."""
    sr = 44100
    n = sr
    t = np.arange(n, dtype=np.float32) / sr
    mid_tone = np.sin(2 * np.pi * 7000.0 * t).astype(np.float32)

    f = BandpassFilter(sample_rate=sr)
    out, _ = f.filter_chunk(mid_tone, 5000.0, 10000.0)
    tail_peak = float(np.max(np.abs(out[sr // 2:])))
    assert tail_peak > 0.7, f"expected pass-through, got tail_peak={tail_peak}"


def test_bandpass_reset_reseeds_state():
    """After reset(), the first filtered chunk should match the very
    first filtered chunk of a fresh filter with the same band.
    """
    sr = 44100
    f1 = BandpassFilter(sample_rate=sr)
    f2 = BandpassFilter(sample_rate=sr)

    rng = np.random.default_rng(7)
    warmup = rng.standard_normal(4096).astype(np.float32)
    probe = rng.standard_normal(1024).astype(np.float32)

    # f1 gets a warmup then reset, f2 is fresh. Both should produce
    # identical output on `probe` when using the same band.
    f1.filter_chunk(warmup, 1000.0, 8000.0)
    f1.reset()
    out1, _ = f1.filter_chunk(probe, 1000.0, 8000.0)
    out2, _ = f2.filter_chunk(probe, 1000.0, 8000.0)
    np.testing.assert_allclose(out1, out2, atol=1e-6)
