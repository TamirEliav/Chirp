"""2a/2b/2c: continuous recording mode, force-trigger toggle, and the
entropy min-cross debounce — all implemented entity-side as mask/param
overrides in ``_ingest_chunk_locked`` (the ThresholdRecorder state
machine is untouched).

Uses the standard ``_start_flush`` monkeypatch seam, so streaming steps
aside and each flushed segment's full buffer is observable in memory.
"""

import numpy as np
import pytest

from chirp.constants import CHUNK_FRAMES
from chirp.recording.entity import RecordingEntity
from chirp.recording.trigger import ThresholdRecorder


@pytest.fixture
def flushes(monkeypatch):
    out = []

    def _capture(buf_snapshot, output_dir, prefix='', suffix='',
                 sample_rate=44100, onset_time=None, filename_stream=''):
        out.append({
            'audio': np.concatenate(buf_snapshot),
            'suffix': suffix,
        })

    monkeypatch.setattr(ThresholdRecorder, '_start_flush',
                        staticmethod(_capture))
    return out


def _entity(**kw):
    e = RecordingEntity(name='rm', device_id=None, sample_rate=44100)
    e.threshold = 0.2
    e.min_cross_sec = 0.01
    e.hold_sec = 0.01
    e.pre_trig_sec = 0.0
    e.post_trig_sec = 0.0
    for k, v in kw.items():
        setattr(e, k, v)
    return e


def _quiet():
    return np.full(CHUNK_FRAMES, 0.01, dtype=np.float32)


def _feed(e, chunks):
    for ch in chunks:
        e.ingest_chunk(ch)


# ── 2a: continuous mode ───────────────────────────────────────────────

def test_continuous_records_silence_and_splits_at_max_rec(flushes):
    e = _entity(rec_mode='Continuous')
    e.max_rec_sec = (4 * CHUNK_FRAMES) / e.sample_rate  # split every 4 chunks
    e.rec_enabled = True
    try:
        _feed(e, [_quiet() for _ in range(10)])   # far below threshold
        # 10 chunks → two full 4-chunk parts flushed, 2 chunks pending.
        assert len(flushes) == 2
        for f in flushes:
            assert f['audio'].size == 4 * CHUNK_FRAMES
        assert 'part01' in flushes[0]['suffix']
        assert 'part02' in flushes[1]['suffix']
        # Stop REC → the pending tail flushes too (gapless total).
        e.rec_enabled = False
        _feed(e, [_quiet()])
        assert len(flushes) == 3
        total = sum(f['audio'].size for f in flushes)
        assert total == 10 * CHUNK_FRAMES
    finally:
        e.close()


def test_continuous_off_means_no_recording_of_silence(flushes):
    e = _entity(rec_mode='Triggered')
    e.rec_enabled = True
    try:
        _feed(e, [_quiet() for _ in range(6)])
        e.rec_enabled = False
        _feed(e, [_quiet()])
        assert flushes == []
    finally:
        e.close()


def test_rec_mode_round_trips_through_config():
    e = _entity(rec_mode='Continuous', entropy_min_cross_sec=0.25)
    try:
        d = e.to_dict()
        assert d['rec_mode'] == 'Continuous'
        assert d['entropy_min_cross_sec'] == 0.25
        e2, _warn = RecordingEntity.from_dict(d)
        try:
            assert e2.rec_mode == 'Continuous'
            assert e2.entropy_min_cross_sec == 0.25
        finally:
            e2.close()
    finally:
        e.close()


# ── 2b: force trigger toggle ─────────────────────────────────────────

def test_force_trigger_records_manual_segment(flushes):
    e = _entity()
    e.rec_enabled = True
    try:
        _feed(e, [_quiet() for _ in range(2)])    # no trigger
        assert flushes == []
        e.set_force_trigger(True)
        _feed(e, [_quiet() for _ in range(5)])    # forced despite silence
        e.set_force_trigger(False)
        _feed(e, [_quiet()])                      # flush consumed here
        assert len(flushes) == 1
        # Segment = the 5 forced chunks (no hold/post tail, no re-open).
        assert flushes[0]['audio'].size == 5 * CHUNK_FRAMES
        _feed(e, [_quiet() for _ in range(3)])    # stays closed after
        assert len(flushes) == 1
    finally:
        e.close()


def test_force_trigger_includes_pre_trigger_lookback(flushes):
    e = _entity(pre_trig_sec=(2 * CHUNK_FRAMES) / 44100)
    e.rec_enabled = True
    try:
        _feed(e, [_quiet() for _ in range(4)])    # history for lookback
        e.set_force_trigger(True)
        _feed(e, [_quiet() for _ in range(3)])
        e.set_force_trigger(False)
        _feed(e, [_quiet()])
        assert len(flushes) == 1
        # 3 forced chunks + 2 chunks of pre-trigger lookback.
        assert flushes[0]['audio'].size == 5 * CHUNK_FRAMES
    finally:
        e.close()


def test_stop_rec_clears_force_flag(flushes):
    e = _entity()
    e.rec_enabled = True
    try:
        e.set_force_trigger(True)
        _feed(e, [_quiet() for _ in range(2)])
        e.stop_rec()
        assert e.force_rec_active is False
        _feed(e, [_quiet()])                      # disable-flush closes it
        assert len(flushes) == 1
    finally:
        e.close()


# ── 2c: entropy debounce ─────────────────────────────────────────────

def _tone():
    t = np.arange(CHUNK_FRAMES) / 44100
    return (0.5 * np.sin(2 * np.pi * 5000 * t)).astype(np.float32)


def _noise(rng):
    return (0.5 * rng.uniform(-1, 1, CHUNK_FRAMES)).astype(np.float32)


def test_entropy_debounce_delays_spectral_gate(flushes):
    chunk_sec = CHUNK_FRAMES / 44100
    e = _entity(spectral_trigger_mode='Spectral Only',
                spectral_threshold=0.7,
                entropy_min_cross_sec=4 * chunk_sec)
    e.rec_enabled = True
    rng = np.random.default_rng(1)
    try:
        # Warm up the FFT with noise (high entropy — no trigger).
        _feed(e, [_noise(rng) for _ in range(8)])
        assert not e.recorder.is_recording
        # Two low-entropy chunks — below the 4-chunk debounce: no event.
        _feed(e, [_tone() for _ in range(2)])
        assert not e.recorder.is_recording
        # Noise resets the streak; two more tone chunks still no event.
        _feed(e, [_noise(rng) for _ in range(4)])
        _feed(e, [_tone() for _ in range(2)])
        assert not e.recorder.is_recording
        # Sustained tone crosses the debounce → event opens. (The FFT
        # window is 4096 with 1024 hops, so entropy only drops once the
        # window is mostly tone — allow for that plus the 4-chunk
        # debounce.)
        _feed(e, [_tone() for _ in range(12)])
        assert e.recorder.is_recording
    finally:
        e.rec_enabled = False
        e.close()


def test_entropy_debounce_zero_is_instantaneous(flushes):
    e = _entity(spectral_trigger_mode='Spectral Only',
                spectral_threshold=0.7, entropy_min_cross_sec=0.0)
    e.rec_enabled = True
    rng = np.random.default_rng(2)
    try:
        _feed(e, [_noise(rng) for _ in range(8)])   # warm-up
        assert not e.recorder.is_recording
        # Enough tone chunks for the 4096-sample FFT window to become
        # tonal — with debounce 0 the FIRST below-threshold chunk opens
        # the event.
        _feed(e, [_tone() for _ in range(6)])
        assert e.recorder.is_recording
    finally:
        e.rec_enabled = False
        e.close()
