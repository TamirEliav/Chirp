"""Selectable amplitude-envelope estimator (⚙ Advanced → AMPLITUDE ENVELOPE).

The trigger's "how loud is it right now" measurement is now one of two
methods, chosen app-wide and persisted in the config's ``audio`` section:

  * ``hilbert``  — |analytic signal|, the historical default.
  * ``rectify``  — rectify + 2-pole Butterworth low-pass, carried
    statefully across chunks.

What these tests pin:

* **Calibration parity.** Both methods report a steady tone's PEAK
  amplitude, so switching does not silently move every stream's
  effective threshold. This is what the 2/π rectified-sine correction in
  ``RectifiedEnvelope`` buys, and it is the property most likely to be
  broken by a well-meaning "simplification" of that constant.
* **State continuity.** The follower's IIR history must survive the
  chunk boundary; a per-chunk rebuild would re-run the start-up ramp
  every 1024 samples and shred the envelope.
* **Cache identity.** ``RecordingEntity._trigger_envelope`` caches one
  follower per channel and rebuilds it only on a real config change. A
  cutoff clamped by Nyquist must still compare equal to the REQUEST, or
  the follower is rebuilt (history wiped) on every single chunk.
* **The original bug stays fixed.** The whole reason an envelope exists
  is that ``|filtered|`` dips to zero at every zero crossing and
  ``min_cross`` could never be satisfied for a narrowband signal
  (see test_envelope_trigger.py). The rectify path must trigger too.
"""

from __future__ import annotations

import numpy as np
import pytest

from chirp.constants import CHUNK_FRAMES
from chirp.dsp import envelope as env_mod
from chirp.dsp.envelope import (DEFAULT_ENVELOPE_CUTOFF_HZ,
                                ENVELOPE_CUTOFF_MAX_HZ,
                                ENVELOPE_CUTOFF_MIN_HZ, ENVELOPE_METHODS,
                                RectifiedEnvelope, analytic_envelope)
from chirp.recording.entity import RecordingEntity
from chirp.recording.trigger import ThresholdRecorder


@pytest.fixture(autouse=True)
def _restore_global_method():
    """The estimator is process-global; never leak a selection into the
    next test (or into the rest of the suite, which assumes hilbert)."""
    before = env_mod.current_params()
    yield
    env_mod.configure(*before)


def _sine(sr: int, freq: float, n: int, amp: float = 0.3,
          phase: float = 0.0) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / sr
    return (amp * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)


# ── RectifiedEnvelope ────────────────────────────────────────────────────

@pytest.mark.parametrize('freq', [500.0, 2000.0, 8000.0])
def test_rectify_reads_tone_peak_amplitude(freq):
    """A steady sine of amplitude A must read A, not 2A/π."""
    sr, amp = 44100, 0.3
    x = _sine(sr, freq, sr, amp=amp)
    out = RectifiedEnvelope(sr, 50.0).process(x)
    settled = out[sr // 2:]           # past the follower's rise time
    assert settled.mean() == pytest.approx(amp, rel=0.02)
    assert settled.min() == pytest.approx(amp, rel=0.05)
    assert settled.max() == pytest.approx(amp, rel=0.05)


def test_rectify_and_hilbert_agree_on_a_tone():
    """Calibration parity: the same tone must land at the same level
    under both methods, so a user switching does not have to re-tune
    every stream's threshold."""
    sr, amp = 44100, 0.42
    x = _sine(sr, 1500.0, sr, amp=amp)
    h = analytic_envelope(x)[sr // 2:].mean()
    r = RectifiedEnvelope(sr, 50.0).process(x)[sr // 2:].mean()
    assert r == pytest.approx(h, rel=0.02)


def test_rectify_state_carries_across_chunks():
    """Chunked processing must equal whole-signal processing — that is
    what the persistent ``_zi`` is for. A fresh follower per chunk would
    re-ramp from zero every chunk and fail this badly."""
    sr = 44100
    x = _sine(sr, 1000.0, CHUNK_FRAMES * 8, amp=0.3)
    whole = RectifiedEnvelope(sr, 50.0).process(x)
    eng = RectifiedEnvelope(sr, 50.0)
    chunked = np.concatenate(
        [eng.process(x[i:i + CHUNK_FRAMES])
         for i in range(0, x.size, CHUNK_FRAMES)])
    assert np.allclose(whole, chunked, atol=1e-6)


def test_rectify_output_is_never_negative():
    """An envelope below zero would be nonsense to compare against a
    threshold and would draw a dip below the axis. The low-pass can ring
    negative after a sharp offset, so the clamp must be there."""
    sr = 44100
    burst = np.concatenate([
        _sine(sr, 1000.0, sr // 10, amp=0.9),
        np.zeros(sr // 10, dtype=np.float32),
    ])
    out = RectifiedEnvelope(sr, 200.0).process(burst)
    assert out.min() >= 0.0


def test_rectify_empty_and_dtype_contract():
    eng = RectifiedEnvelope(44100, 50.0)
    assert eng.process(np.empty(0, dtype=np.float32)).shape == (0,)
    out = eng.process(_sine(44100, 1000.0, 512))
    assert out.dtype == np.float32
    assert out.shape == (512,)


def test_rectify_cutoff_clamped_but_request_remembered():
    """At 8 kHz the 5 kHz ceiling is above Nyquist, so the DESIGN must be
    clamped — while ``key`` keeps reporting the request, so a cache
    doesn't rebuild the follower on every chunk chasing an unreachable
    cutoff."""
    eng = RectifiedEnvelope(8000, ENVELOPE_CUTOFF_MAX_HZ)
    assert eng.cutoff_hz < 4000.0
    assert eng.requested_cutoff_hz == ENVELOPE_CUTOFF_MAX_HZ
    assert eng.key == (8000, ENVELOPE_CUTOFF_MAX_HZ)
    # And it still produces a usable envelope rather than blowing up.
    out = eng.process(_sine(8000, 500.0, 8000, amp=0.3))
    assert np.isfinite(out).all()


def test_rectify_reset_clears_history():
    sr = 44100
    eng = RectifiedEnvelope(sr, 50.0)
    eng.process(_sine(sr, 1000.0, sr, amp=0.9))    # charge it up
    eng.reset()
    fresh = eng.process(np.zeros(256, dtype=np.float32))
    assert fresh.max() == 0.0


# ── Global selection ─────────────────────────────────────────────────────

def test_configure_round_trip_and_unknown_method_falls_back():
    assert env_mod.configure('rectify', 80.0) == ('rectify', 80.0)
    assert env_mod.current_params() == ('rectify', 80.0)
    # An unknown method is a typo, not a crash: fall back to hilbert.
    method, _ = env_mod.configure('bogus', 80.0)
    assert method == 'hilbert'
    assert 'hilbert' in ENVELOPE_METHODS and 'rectify' in ENVELOPE_METHODS


def test_configure_clamps_and_survives_junk_cutoff():
    _, cut = env_mod.configure('rectify', 10_000_000.0)
    assert cut == ENVELOPE_CUTOFF_MAX_HZ
    _, cut = env_mod.configure('rectify', 0.0)
    assert cut == ENVELOPE_CUTOFF_MIN_HZ
    _, cut = env_mod.configure('rectify', 'not-a-number')
    assert cut == DEFAULT_ENVELOPE_CUTOFF_HZ


# ── Entity dispatch ──────────────────────────────────────────────────────

def _entity(sr: int = 44100) -> RecordingEntity:
    return RecordingEntity(name='env-method', device_id=None, sample_rate=sr)


def test_entity_uses_selected_method_without_restart():
    """The method is read per chunk, so flipping it in ⚙ Advanced applies
    to a running stream immediately — no Stop Acq / Start Acq."""
    e = _entity()
    try:
        env_mod.configure('hilbert', 50.0)
        assert e._env_lp is None
        e._trigger_envelope(_sine(44100, 1000.0, CHUNK_FRAMES))
        assert e._env_lp is None, 'hilbert path must allocate no follower'

        env_mod.configure('rectify', 50.0)
        e._trigger_envelope(_sine(44100, 1000.0, CHUNK_FRAMES))
        assert isinstance(e._env_lp, RectifiedEnvelope)
    finally:
        e.close()


def test_entity_caches_follower_and_keeps_channels_separate():
    e = _entity()
    try:
        env_mod.configure('rectify', 50.0)
        x = _sine(44100, 1000.0, CHUNK_FRAMES)
        e._trigger_envelope(x)
        e._trigger_envelope(x, right=True)
        left, right = e._env_lp, e._env_lp_r
        assert left is not None and right is not None and left is not right
        # Same config on the next chunk → same objects (history intact).
        e._trigger_envelope(x)
        e._trigger_envelope(x, right=True)
        assert e._env_lp is left and e._env_lp_r is right
        # Changing the cutoff DOES rebuild.
        env_mod.configure('rectify', 120.0)
        e._trigger_envelope(x)
        assert e._env_lp is not left
        assert e._env_lp.requested_cutoff_hz == 120.0
    finally:
        e.close()


def test_entity_rebuilds_follower_after_sample_rate_change():
    """A follower designed at 44.1 kHz has the wrong cutoff at 22.05 kHz;
    change_sample_rate must drop it."""
    e = _entity(44100)
    try:
        env_mod.configure('rectify', 50.0)
        e._trigger_envelope(_sine(44100, 1000.0, CHUNK_FRAMES))
        assert e._env_lp is not None
        e.change_sample_rate(22050)
        assert e._env_lp is None
        e._trigger_envelope(_sine(22050, 1000.0, CHUNK_FRAMES))
        assert e._env_lp.sample_rate == 22050
    finally:
        e.close()


def test_narrowband_tone_still_triggers_under_rectify(monkeypatch):
    """The bug the envelope exists to fix (test_envelope_trigger.py) must
    stay fixed under the rectify estimator: |filtered| dips to zero every
    half-cycle, so a per-sample compare on the raw signal can never
    satisfy min_cross for a pure tone."""
    flushes: list[dict] = []
    monkeypatch.setattr(
        ThresholdRecorder, '_start_flush',
        staticmethod(lambda buf_snapshot, output_dir, prefix='', suffix='',
                     sample_rate=44100, onset_time=None, filename_stream='':
                     flushes.append({'n': len(buf_snapshot)})))
    env_mod.configure('rectify', 50.0)
    sr = 44100
    e = _entity(sr)
    e.threshold = 0.1
    e.min_cross_sec = 0.05
    e.hold_sec = e.pre_trig_sec = e.post_trig_sec = 0.0
    e.max_rec_sec = 10.0
    e.rec_enabled = True
    try:
        phase = 0.0
        for _ in range(int(0.4 * sr) // CHUNK_FRAMES):
            e.ingest_chunk(_sine(sr, 1000.0, CHUNK_FRAMES, amp=0.5,
                                 phase=phase))
            phase = (phase + 2 * np.pi * 1000.0 * CHUNK_FRAMES / sr) % (2 * np.pi)
        for _ in range(4):
            e.ingest_chunk(np.zeros(CHUNK_FRAMES, dtype=np.float32))
        assert len(flushes) >= 1
    finally:
        e.close()


def test_subthreshold_tone_does_not_trigger_under_rectify(monkeypatch):
    """Symmetric guard — the rectify path must not simply read high."""
    flushes: list[dict] = []
    monkeypatch.setattr(
        ThresholdRecorder, '_start_flush',
        staticmethod(lambda buf_snapshot, output_dir, prefix='', suffix='',
                     sample_rate=44100, onset_time=None, filename_stream='':
                     flushes.append({'n': len(buf_snapshot)})))
    env_mod.configure('rectify', 50.0)
    sr = 44100
    e = _entity(sr)
    e.threshold = 0.5
    e.min_cross_sec = 0.05
    e.hold_sec = e.pre_trig_sec = e.post_trig_sec = 0.0
    e.rec_enabled = True
    try:
        phase = 0.0
        for _ in range(int(0.4 * sr) // CHUNK_FRAMES):
            e.ingest_chunk(_sine(sr, 1000.0, CHUNK_FRAMES, amp=0.1,
                                 phase=phase))
            phase = (phase + 2 * np.pi * 1000.0 * CHUNK_FRAMES / sr) % (2 * np.pi)
        assert flushes == []
    finally:
        e.close()
