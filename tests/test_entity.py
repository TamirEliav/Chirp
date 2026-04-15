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


# ── #13 / c15: AudioCapture drop counter ───────────────────────────────────

def test_capture_drop_counter_increments_and_consumes():
    """Simulating queue-full drops should increment `drop_count`, and
    `consume_drop_count` should return-and-reset atomically."""
    import queue as _q
    from chirp.audio.capture import AudioCapture
    q = _q.Queue(maxsize=1)
    cap = AudioCapture(q, device=None)  # opens nothing in headless test
    # Manually emulate three full-queue drops in the callback path.
    cap.drop_count = 3
    assert cap.consume_drop_count() == 3
    assert cap.drop_count == 0
    assert cap.consume_drop_count() == 0


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


# ── #20 / c14: ring-buffer cursors stay coherent across chunk sizes ────────

def test_cursors_advance_in_lockstep_with_default_chunk_size():
    """Stock CHUNK_FRAMES chunks: write_head advances by CHUNK_FRAMES
    and col_head advances by 1, exactly as in the legacy monolith.
    """
    from chirp.constants import CHUNK_FRAMES
    e = _entity()
    chunk = np.zeros(CHUNK_FRAMES, dtype=np.float32)
    for i in range(5):
        e.ingest_chunk(chunk)
        assert e._samples_total == (i + 1) * CHUNK_FRAMES
        assert e.col_head == (i + 1) % e._n_cols
        assert e.write_head == ((i + 1) * CHUNK_FRAMES) % e._total_samples


def test_cursors_stay_coherent_with_irregular_chunk_size():
    """A non-CHUNK_FRAMES chunk must not desync the two cursors.

    Regression for #20: pre-fix, col_head incremented by 1 per chunk
    independent of chunk length, while write_head advanced by `n`. A
    chunk of 2*CHUNK_FRAMES would advance write_head by two columns'
    worth of samples but col_head by only one, losing display sync.
    """
    from chirp.constants import CHUNK_FRAMES
    e = _entity()
    big = np.zeros(2 * CHUNK_FRAMES, dtype=np.float32)
    e.ingest_chunk(big)
    # After one 2-chunk-sized ingest the sample clock moved by
    # 2*CHUNK_FRAMES; both cursors must reflect that.
    assert e._samples_total == 2 * CHUNK_FRAMES
    assert e.col_head == 2 % e._n_cols
    assert e.write_head == (2 * CHUNK_FRAMES) % e._total_samples
    # A standard chunk after the big one keeps the lockstep going.
    e.ingest_chunk(np.zeros(CHUNK_FRAMES, dtype=np.float32))
    assert e._samples_total == 3 * CHUNK_FRAMES
    assert e.col_head == 3 % e._n_cols
    assert e.write_head == (3 * CHUNK_FRAMES) % e._total_samples


def test_oversize_chunk_raises():
    """A single chunk longer than the entire ring buffer is rejected
    rather than silently smearing across the buffer multiple times."""
    e = _entity()
    huge = np.zeros(e._total_samples + 1, dtype=np.float32)
    try:
        e.ingest_chunk(huge)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for oversize chunk")


def test_saturation_true_on_unfiltered_clip():
    """Sanity check: clipping with the band filter off still trips."""
    e = _entity()
    e.freq_filter_enabled = False
    chunk = np.full(1024, 0.999, dtype=np.float32)
    e.ingest_chunk(chunk)
    assert e.saturated is True
