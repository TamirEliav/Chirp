"""Full test suite for `chirp.recording.trigger.ThresholdRecorder`.

Upgraded from the Phase 0 characterization test (plan: c04). These
tests pin the pre-refactor behavior in detail so that later commits
can rewrite the state machine for per-sample accuracy (#15) and safe
shutdown (#17) without regressing anything.

The tests still operate at chunk granularity — a deliberate choice,
because the monolith does too. Issue #15 will land a sample-accurate
rewrite and these tests will be updated at that point.

All flushes are captured in memory by monkeypatching
`ThresholdRecorder._start_flush`; no WAV files are written to disk.
"""

import numpy as np
import pytest

from chirp.constants import CHUNK_FRAMES
from chirp.recording.trigger import ThresholdRecorder


# ─────────────────────────────────────────────────────────────────────────────
# Fixture + helpers
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def captured_flushes(monkeypatch):
    """Replace `_start_flush` with an in-memory capture.

    Returns a list of dicts with the concatenated audio buffer, chunk
    count, and metadata. This avoids the daemon-thread WAV writer
    entirely (that is a target for issue #17).
    """
    flushes: list[dict] = []

    def _capture(buf_snapshot, output_dir, prefix='', suffix='',
                 sample_rate=44100, onset_time=None):
        flushes.append({
            "audio": np.concatenate(list(buf_snapshot)),
            "n_chunks": len(buf_snapshot),
            "output_dir": output_dir,
            "prefix": prefix,
            "suffix": suffix,
            "sample_rate": sample_rate,
            "onset_time": onset_time,
        })

    monkeypatch.setattr(
        ThresholdRecorder, "_start_flush", staticmethod(_capture),
    )
    yield flushes


def _silent():
    return np.zeros(CHUNK_FRAMES, dtype=np.float32)


def _loud(level: float = 0.5):
    return np.full(CHUNK_FRAMES, level, dtype=np.float32)


def _params(**overrides):
    p = dict(
        threshold=0.1,
        min_cross_sec=0.0,
        hold_sec=0.0,
        post_trig_sec=0.0,
        max_rec_sec=10.0,
        pre_trig_sec=0.0,
        output_dir="/tmp/chirp_test",
        enabled=True,
        filename_prefix="",
        filename_suffix="",
        sample_rate=44100,
    )
    p.update(overrides)
    return p


def _drive(rec, chunks, peaks, params):
    """Drive the recorder with a sequence of (chunk, peak) pairs."""
    for chunk, peak in zip(chunks, peaks):
        rec.process_chunk(chunk, trigger_peak=peak, **params)


# ─────────────────────────────────────────────────────────────────────────────
# Basic lifecycle
# ─────────────────────────────────────────────────────────────────────────────

def test_single_event_minimal(captured_flushes):
    """3 loud chunks between silences → exactly one flush of 3 chunks."""
    rec = ThresholdRecorder()
    p = _params()
    _drive(rec, [_silent()] * 3, [0.0] * 3, p)
    assert not rec.is_recording
    _drive(rec, [_loud(0.5)] * 3, [0.5] * 3, p)
    assert rec.is_recording
    assert not captured_flushes
    _drive(rec, [_silent()], [0.0], p)
    assert not rec.is_recording
    assert len(captured_flushes) == 1
    assert captured_flushes[0]["audio"].shape[0] == 3 * CHUNK_FRAMES


def test_disabled_never_triggers(captured_flushes):
    rec = ThresholdRecorder()
    p = _params(enabled=False)
    _drive(rec, [_loud(0.9)] * 5, [0.9] * 5, p)
    assert not rec.is_recording
    assert not captured_flushes


def test_disable_mid_event_flushes(captured_flushes):
    rec = ThresholdRecorder()
    on = _params(enabled=True)
    off = _params(enabled=False)
    _drive(rec, [_loud(0.5)] * 2, [0.5] * 2, on)
    assert rec.is_recording
    rec.process_chunk(_silent(), trigger_peak=0.0, **off)
    assert not rec.is_recording
    assert len(captured_flushes) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Pre-trigger window
# ─────────────────────────────────────────────────────────────────────────────

def test_pre_trigger_includes_silent_lookback(captured_flushes):
    """pre_trig_sec large enough to capture 2 prior silent chunks."""
    rec = ThresholdRecorder()
    pre_sec = 2.0 * CHUNK_FRAMES / 44100
    p = _params(pre_trig_sec=pre_sec)

    # 2 silent chunks (the pre-roll), 1 loud, then 1 silent to end the event
    rec.process_chunk(_silent(), trigger_peak=0.0, **p)
    rec.process_chunk(_silent(), trigger_peak=0.0, **p)
    rec.process_chunk(_loud(0.5), trigger_peak=0.5, **p)
    rec.process_chunk(_silent(), trigger_peak=0.0, **p)

    assert len(captured_flushes) == 1
    audio = captured_flushes[0]["audio"]
    # Expect: 2 silent pre-roll chunks + 1 loud chunk = 3 chunks total
    # (post_trig=0 so no tail after the loud chunk).
    assert audio.shape[0] == 3 * CHUNK_FRAMES
    # First 2*CHUNK_FRAMES samples should be silent, next chunk loud.
    assert np.all(audio[:2 * CHUNK_FRAMES] == 0.0)
    assert np.allclose(audio[2 * CHUNK_FRAMES:], 0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Post-trigger tail
# ─────────────────────────────────────────────────────────────────────────────

def test_post_trigger_extends_tail(captured_flushes):
    """post_trig_sec appends silent chunks after the last loud one."""
    rec = ThresholdRecorder()
    post_sec = 2.0 * CHUNK_FRAMES / 44100  # 2 chunks of tail
    p = _params(post_trig_sec=post_sec)

    rec.process_chunk(_loud(0.5), trigger_peak=0.5, **p)
    # Silent chunks after: first ends event (hold=0), then tail-draining
    for _ in range(5):
        rec.process_chunk(_silent(), trigger_peak=0.0, **p)

    assert len(captured_flushes) == 1
    audio = captured_flushes[0]["audio"]
    # 1 loud chunk + 2-chunk tail = 3 chunks (2 silent tail chunks are appended)
    assert audio.shape[0] == 3 * CHUNK_FRAMES
    assert np.allclose(audio[:CHUNK_FRAMES], 0.5)
    assert np.all(audio[CHUNK_FRAMES:] == 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Hold
# ─────────────────────────────────────────────────────────────────────────────

def test_hold_delays_event_end(captured_flushes):
    """hold_sec >= 2 chunks keeps the event alive across 1 silent chunk."""
    rec = ThresholdRecorder()
    hold_sec = 2.0 * CHUNK_FRAMES / 44100
    p = _params(hold_sec=hold_sec)

    rec.process_chunk(_loud(0.5), trigger_peak=0.5, **p)
    assert rec.is_recording
    # One silent chunk → still recording because hold not elapsed
    rec.process_chunk(_silent(), trigger_peak=0.0, **p)
    assert rec.is_recording
    assert not captured_flushes
    # Second silent chunk hits the hold threshold → ends + flushes
    rec.process_chunk(_silent(), trigger_peak=0.0, **p)
    assert not rec.is_recording
    assert len(captured_flushes) == 1


# ─────────────────────────────────────────────────────────────────────────────
# min_cross
# ─────────────────────────────────────────────────────────────────────────────

def test_min_cross_requires_sustained_above(captured_flushes):
    """A single loud chunk doesn't trigger if min_cross requires 2 chunks."""
    rec = ThresholdRecorder()
    min_cross_sec = 2.0 * CHUNK_FRAMES / 44100
    p = _params(min_cross_sec=min_cross_sec)

    rec.process_chunk(_loud(0.5), trigger_peak=0.5, **p)
    assert not rec.is_recording
    rec.process_chunk(_silent(), trigger_peak=0.0, **p)
    assert not rec.is_recording
    assert not captured_flushes


def test_min_cross_fires_after_enough_chunks(captured_flushes):
    rec = ThresholdRecorder()
    min_cross_sec = 2.0 * CHUNK_FRAMES / 44100
    p = _params(min_cross_sec=min_cross_sec)

    rec.process_chunk(_loud(0.5), trigger_peak=0.5, **p)
    assert not rec.is_recording
    rec.process_chunk(_loud(0.5), trigger_peak=0.5, **p)
    assert rec.is_recording


# ─────────────────────────────────────────────────────────────────────────────
# max_rec force-split
# ─────────────────────────────────────────────────────────────────────────────

def test_max_rec_splits_long_events(captured_flushes):
    """An event longer than max_rec_sec is force-flushed and split."""
    rec = ThresholdRecorder()
    # 3-chunk cap
    max_rec = 3.0 * CHUNK_FRAMES / 44100
    p = _params(max_rec_sec=max_rec)

    # 5 continuous loud chunks — should force at least one split
    for _ in range(5):
        rec.process_chunk(_loud(0.5), trigger_peak=0.5, **p)

    # First event must have flushed when buf reached 3 chunks.
    assert len(captured_flushes) >= 1
    first = captured_flushes[0]
    assert first["n_chunks"] == 3


# ─────────────────────────────────────────────────────────────────────────────
# Sub-threshold → no flush
# ─────────────────────────────────────────────────────────────────────────────

def test_sub_threshold_never_triggers(captured_flushes):
    rec = ThresholdRecorder()
    p = _params(threshold=0.5)
    _drive(rec, [_loud(0.1)] * 5, [0.1] * 5, p)
    assert not rec.is_recording
    assert not captured_flushes


# ─────────────────────────────────────────────────────────────────────────────
# Onset time
# ─────────────────────────────────────────────────────────────────────────────

def test_onset_time_is_populated(captured_flushes):
    """Every flush carries a datetime onset_time. #23 will make it
    monotonic-based; for now this pins that it is at least set.
    """
    import datetime
    rec = ThresholdRecorder()
    p = _params()
    rec.process_chunk(_loud(0.5), trigger_peak=0.5, **p)
    rec.process_chunk(_silent(), trigger_peak=0.0, **p)
    assert len(captured_flushes) == 1
    onset = captured_flushes[0]["onset_time"]
    assert isinstance(onset, datetime.datetime)
