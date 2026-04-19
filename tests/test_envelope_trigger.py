"""Regression test for the v2.1.0 min_cross-on-narrowband-signals bug.

Symptom the user reported:
    Signal clearly crossed the threshold for far longer than
    ``min_cross_sec``; the yellow detect strip was lit almost
    continuously; yet no recording was ever triggered.

Root cause (traced to commit d758fbf, c18 / #15):
    The sample-accurate rewrite in v2.1.0 changed the detection
    signal from a chunk-level ``max(|chunk|) >= threshold`` scalar
    into a per-sample ``|filtered[i]| >= threshold`` boolean. For any
    narrowband signal (pure tones, bandpassed bioacoustic calls,
    whistles, etc.), ``|x|`` dips to zero at every waveform zero
    crossing. The opening streak in ``ThresholdRecorder`` resets to
    0 on any below-threshold sample, so a 1 kHz sine at 44.1 kHz
    can never accumulate more than ~20 consecutive above samples,
    no matter how long the signal runs — ``min_cross`` is never
    satisfied, no event opens.

Fix: build ``amp_mask`` from the analytic-signal envelope
(``|scipy.signal.hilbert(filt)|``) instead of from ``|filt|``. The
envelope is smooth for narrowband signals and has no zero-crossing
dips, so ``min_cross`` counts what a human would count — samples
of "signal present".

Pre-fix: these tests fail because no event ever opens.
Post-fix: both mono and stereo tones trigger cleanly.
"""

from __future__ import annotations

import numpy as np
import pytest

from chirp.constants import CHUNK_FRAMES
from chirp.recording.entity import RecordingEntity
from chirp.recording.trigger import ThresholdRecorder


# ── Capture flushes without writing WAVs ────────────────────────────────

@pytest.fixture
def captured_flushes(monkeypatch):
    flushes: list[dict] = []

    def _capture(buf_snapshot, output_dir, prefix='', suffix='',
                 sample_rate=44100, onset_time=None, filename_stream=''):
        flushes.append({'n_chunks': len(buf_snapshot)})

    monkeypatch.setattr(ThresholdRecorder, '_start_flush',
                        staticmethod(_capture))
    yield flushes


def _sine(sample_rate: int, freq: float, n: int,
          amp: float = 0.5, phase: float = 0.0) -> np.ndarray:
    """Generate a continuous sine of length n samples, starting at
    sample index 0 with phase offset ``phase`` radians. Stateless —
    caller must chain the phase across chunks to avoid discontinuity."""
    t = np.arange(n, dtype=np.float64) / sample_rate
    return (amp * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)


def _iter_sine_chunks(sample_rate: int, freq: float, amp: float,
                      total_samples: int, chunk: int = CHUNK_FRAMES):
    """Yield phase-continuous sine chunks of length ``chunk``."""
    phase = 0.0
    produced = 0
    while produced < total_samples:
        n = min(chunk, total_samples - produced)
        x = _sine(sample_rate, freq, n, amp=amp, phase=phase)
        phase += 2 * np.pi * freq * n / sample_rate
        phase = phase % (2 * np.pi)
        produced += n
        yield x


# ── Entity factory ──────────────────────────────────────────────────────

def _make_entity(sample_rate: int = 44100, threshold: float = 0.1,
                 min_cross_sec: float = 0.05):
    e = RecordingEntity(name='env-test', device_id=None,
                        sample_rate=sample_rate)
    e.threshold      = threshold
    e.min_cross_sec  = min_cross_sec
    e.hold_sec       = 0.0
    e.pre_trig_sec   = 0.0
    e.post_trig_sec  = 0.0
    e.max_rec_sec    = 10.0
    e.rec_enabled    = True
    return e


# ── The user's bug ──────────────────────────────────────────────────────

def test_pure_sine_triggers_despite_zero_crossings(captured_flushes):
    """A 1 kHz sine wave whose instantaneous amplitude is well above
    threshold should open an event after ``min_cross_sec`` — which
    it DIDN'T before the envelope fix because ``|sin(2πft)|`` dips
    to zero every half-cycle, resetting the consecutive-above
    streak long before ``min_cross_samps`` is reached."""
    sr = 44100
    e = _make_entity(sample_rate=sr, threshold=0.1, min_cross_sec=0.05)
    try:
        # Feed 0.3s of pure 1 kHz sine at amp 0.5.
        # min_cross_sec=0.05 → need=2205 samples of "signal".
        total = int(0.3 * sr)
        for chunk in _iter_sine_chunks(sr, freq=1000, amp=0.5,
                                       total_samples=total):
            e.ingest_chunk(chunk)
        # Flush the event by passing silence.
        silence = np.zeros(CHUNK_FRAMES, dtype=np.float32)
        for _ in range(4):
            e.ingest_chunk(silence)
        # An event must have opened and flushed. Pre-fix: 0 flushes.
        assert len(captured_flushes) >= 1, (
            'narrowband signal failed to trigger min_cross — '
            'envelope fix regressed')
    finally:
        e.close()


def test_detect_strip_is_solid_during_sine(captured_flushes):
    """The yellow detect indicator must be True continuously
    throughout a pure-tone burst — not flickering at the signal's
    zero-crossing rate. Pre-fix it flickered, which was visible as
    a dashed-looking yellow bar in the UI."""
    sr = 44100
    e = _make_entity(sample_rate=sr, threshold=0.1, min_cross_sec=0.0)
    try:
        # Single chunk of pure 1 kHz sine.
        chunk = _sine(sr, freq=1000, n=CHUNK_FRAMES, amp=0.5)
        e.ingest_chunk(chunk)
        mask = e.detect_mask_buffer[:CHUNK_FRAMES]
        # Edge artifacts from per-chunk Hilbert FFT can dip the first
        # / last ~30 samples; check the middle 80%.
        lo, hi = int(CHUNK_FRAMES * 0.1), int(CHUNK_FRAMES * 0.9)
        solid_fraction = mask[lo:hi].mean()
        assert solid_fraction >= 0.99, (
            f'detect mask was not solid across a sine burst: '
            f'{solid_fraction:.2%} True — envelope fix regressed')
    finally:
        e.close()


def test_subthreshold_sine_does_not_trigger(captured_flushes):
    """Symmetric sanity check: a sine whose envelope is BELOW threshold
    must not trigger. Without this, the envelope fix could trivially
    pass the above tests by just wiring amp_mask to True everywhere."""
    sr = 44100
    e = _make_entity(sample_rate=sr, threshold=0.5, min_cross_sec=0.0)
    try:
        # Amp 0.1, threshold 0.5 → envelope ≈ 0.1, well below.
        total = int(0.2 * sr)
        for chunk in _iter_sine_chunks(sr, freq=1000, amp=0.1,
                                       total_samples=total):
            e.ingest_chunk(chunk)
        silence = np.zeros(CHUNK_FRAMES, dtype=np.float32)
        for _ in range(4):
            e.ingest_chunk(silence)
        assert captured_flushes == [], (
            'subthreshold sine triggered — envelope is too hot')
    finally:
        e.close()


def test_silence_does_not_trigger(captured_flushes):
    """Pure silence must not trigger — rules out a constant-bias bug
    in the envelope path."""
    sr = 44100
    e = _make_entity(sample_rate=sr, threshold=0.01, min_cross_sec=0.0)
    try:
        silence = np.zeros(CHUNK_FRAMES, dtype=np.float32)
        for _ in range(20):
            e.ingest_chunk(silence)
        assert captured_flushes == []
        # And the detect strip must remain entirely dark.
        assert not e.detect_mask_buffer.any()
    finally:
        e.close()


def test_envelope_below_threshold_on_silence_then_sine(captured_flushes):
    """Transitions are what matter for trigger timing. After 2 chunks
    of silence and 2 chunks of sine, the detect strip must be False
    in the silent region and True in the sine region (with edge
    tolerance for Hilbert transients at the silence→sine boundary)."""
    sr = 44100
    e = _make_entity(sample_rate=sr, threshold=0.1, min_cross_sec=0.0)
    try:
        silence = np.zeros(CHUNK_FRAMES, dtype=np.float32)
        e.ingest_chunk(silence)
        e.ingest_chunk(silence)
        # 2 chunks of sine.
        phase = 0.0
        for _ in range(2):
            chunk = _sine(sr, freq=1000, n=CHUNK_FRAMES, amp=0.5,
                          phase=phase)
            phase += 2 * np.pi * 1000 * CHUNK_FRAMES / sr
            e.ingest_chunk(chunk)

        # The amp_buffer has 4*CHUNK_FRAMES samples; indices 0..2048
        # are silence, 2048..4096 are sine.
        db = e.detect_mask_buffer
        # Silent region: must be False (with some slack for Hilbert
        # tail from the sine leaking backward — check the early part
        # of the silent region, well away from the boundary).
        assert not db[:CHUNK_FRAMES].any(), (
            'silence region showed detection — Hilbert tail bled back?')
        # Sine region: must be True in the interior (skip first and
        # last chunk edges where Hilbert has transient artifacts).
        sine_interior = db[2 * CHUNK_FRAMES + 100:
                           4 * CHUNK_FRAMES - 100]
        assert sine_interior.mean() >= 0.99
    finally:
        e.close()
