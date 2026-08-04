"""Signal-level inserted-silence (zero-run) detector.

Field finding (2026-08): the Windows audio engine / device driver can
latch a per-endpoint state where it periodically zero-fills 2-8 ms of
the capture stream in place, raising NO PortAudio status flag — the
only in-app tell is the samples themselves. A live analog input never
produces exact-zero runs of a millisecond (ADC noise floor), so
``RecordingEntity._detect_zero_runs`` counts exact-zero runs >=
``ZERO_RUN_MIN_SEC`` across all captured channels. These tests pin:

* runs >= threshold count once; short runs and nonzero audio don't;
* a run spanning chunk boundaries counts exactly once (carry logic);
* stereo chunks require ALL channels zero;
* WAV-playback input is exempt (loop-seam padding, legit silence);
* consume/clear contract mirrors the other sticky error stats.
"""

import numpy as np

from chirp.recording.entity import RecordingEntity, ZERO_RUN_MIN_SEC


def _entity():
    e = RecordingEntity(name='zr', device_id=None)
    assert e.input_source == 'device'
    return e


def _noise(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.01, 0.5, n).astype(np.float32)
    return x


def _min_run(e):
    return max(2, int(ZERO_RUN_MIN_SEC * e.sample_rate))


def test_run_at_threshold_counts_once():
    e = _entity()
    m = _min_run(e)
    chunk = _noise(1024)
    chunk[100:100 + m] = 0.0
    e.ingest_chunk(chunk)
    assert e.zero_run_count == 1
    assert e.zero_run_count_total == 1
    assert e.has_ever_zero_run is True
    assert e.zero_run_longest == m


def test_short_runs_and_clean_audio_ignored():
    e = _entity()
    m = _min_run(e)
    chunk = _noise(1024)
    chunk[50:50 + m - 1] = 0.0          # one sample short of threshold
    chunk[500:504] = 0.0                # tiny incidental run
    e.ingest_chunk(chunk)
    e.ingest_chunk(_noise(1024, seed=1))
    assert e.zero_run_count == 0
    assert e.has_ever_zero_run is False


def test_multiple_runs_in_one_chunk():
    e = _entity()
    m = _min_run(e)
    chunk = _noise(1024)
    chunk[0:m] = 0.0
    chunk[300:300 + m] = 0.0
    chunk[700:700 + m] = 0.0
    e.ingest_chunk(chunk)
    assert e.zero_run_count == 3


def test_run_spanning_chunks_counts_once():
    e = _entity()
    m = _min_run(e)
    # Trailing zeros of chunk1 + leading zeros of chunk2 together pass
    # the threshold; neither part alone does.
    c1 = _noise(1024)
    c1[-(m // 2):] = 0.0
    c2 = _noise(1024, seed=2)
    c2[:m - m // 2] = 0.0
    e.ingest_chunk(c1)
    assert e.zero_run_count == 0        # not yet at threshold
    e.ingest_chunk(c2)
    assert e.zero_run_count == 1
    assert e.zero_run_longest == m


def test_long_run_over_many_chunks_counts_once():
    e = _entity()
    # 4 fully-zero chunks = one long run; must count exactly once.
    for _ in range(4):
        e.ingest_chunk(np.zeros(1024, dtype=np.float32))
    e.ingest_chunk(_noise(1024, seed=3))   # run ends
    assert e.zero_run_count == 1
    assert e.zero_run_longest == 4096


def test_stereo_requires_all_channels_zero():
    e = _entity()
    e.channel_mode = 'Stereo'
    m = _min_run(e)
    chunk = np.stack([_noise(1024), _noise(1024, seed=4)], axis=1)
    chunk[100:100 + m, 0] = 0.0          # left only — not inserted silence
    e.ingest_chunk(chunk)
    assert e.zero_run_count == 0
    chunk2 = np.stack([_noise(1024, seed=5), _noise(1024, seed=6)], axis=1)
    chunk2[100:100 + m, :] = 0.0         # both channels — counts
    e.ingest_chunk(chunk2)
    assert e.zero_run_count == 1


def test_wav_input_source_exempt():
    e = _entity()
    e.input_source = 'wav_file'
    chunk = _noise(1024)
    chunk[: _min_run(e) * 2] = 0.0
    e.ingest_chunk(chunk)
    assert e.zero_run_count == 0
    assert e.has_ever_zero_run is False


def test_consume_and_clear_contract():
    e = _entity()
    m = _min_run(e)
    chunk = _noise(1024)
    chunk[0:m] = 0.0
    e.ingest_chunk(chunk)
    assert e.consume_zero_run_count() == 1
    assert e.zero_run_count == 0           # transient drained
    assert e.zero_run_count_total == 1     # sticky total kept
    assert e.has_ever_zero_run is True
    assert e.consume_zero_run_count() == 0
    e.clear_error_flag()
    assert e.zero_run_count_total == 0
    assert e.has_ever_zero_run is False
    assert e.zero_run_longest == 0


# ── Start-of-acquisition warm-up ─────────────────────────────────────────

def test_warmup_ignores_zeros_then_detects_after_it_elapses():
    """A priming capture pin delivers silence in its first buffers —
    that must not light the badge on every acquisition start."""
    e = _entity()
    e._zero_warmup_left = 2048
    e.ingest_chunk(np.zeros(1024, dtype=np.float32))
    e.ingest_chunk(np.zeros(1024, dtype=np.float32))
    assert e.zero_run_count == 0
    assert e.has_ever_zero_run is False
    # Warm-up exhausted — the next silent chunk counts.
    e.ingest_chunk(np.zeros(1024, dtype=np.float32))
    e.ingest_chunk(_noise(1024))          # run ends
    assert e.zero_run_count == 1


def test_warmup_run_does_not_carry_into_detection():
    """A zero run straddling the warm-up boundary is judged only on the
    samples after it — no credit for the warm-up half."""
    e = _entity()
    m = _min_run(e)
    e._zero_warmup_left = 1024
    e.ingest_chunk(np.zeros(1024, dtype=np.float32))   # all warm-up
    tail = _noise(1024)
    tail[:m - 1] = 0.0                    # just under threshold on its own
    e.ingest_chunk(tail)
    assert e.zero_run_count == 0


def test_start_acq_arms_the_warmup():
    from tests.test_acq_restart import _FakeCapture
    e = _entity()
    e.capture.close()
    e.capture = _FakeCapture(valid=True)
    e.input_source = 'device'
    try:
        e.start_acq()
        assert e._zero_warmup_left == int(e.sample_rate * 1.0)
    finally:
        e.stop_acq()
        e.close()


def test_badge_composition_includes_zero_runs():
    from chirp.ui.status_util import compose_error_state
    e = _entity()
    m = _min_run(e)
    chunk = _noise(1024)
    chunk[0:m] = 0.0
    e.ingest_chunk(chunk)
    any_err, tip = compose_error_state(e)
    assert any_err is True
    assert 'inserted-silence' in tip
