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
                 sample_rate=44100, onset_time=None, filename_stream=''):
        flushes.append({
            "audio": np.concatenate(list(buf_snapshot)),
            "n_chunks": len(buf_snapshot),
            "output_dir": output_dir,
            "prefix": prefix,
            "suffix": suffix,
            "sample_rate": sample_rate,
            "onset_time": onset_time,
            "filename_stream": filename_stream,
        })

    monkeypatch.setattr(
        ThresholdRecorder, "_start_flush", staticmethod(_capture),
    )
    yield flushes


def _silent():
    return np.zeros(CHUNK_FRAMES, dtype=np.float32)


def _loud(level: float = 0.5):
    return np.full(CHUNK_FRAMES, level, dtype=np.float32)


# ── #16 / c12: should_trigger override ──────────────────────────────────────

def test_should_trigger_true_overrides_subthreshold_peak(captured_flushes):
    """A quiet chunk with `should_trigger=True` should still fire."""
    rec = ThresholdRecorder()
    p = dict(
        threshold=0.5, min_cross_sec=0.0, hold_sec=0.0,
        post_trig_sec=0.0, max_rec_sec=10.0, pre_trig_sec=0.0,
        output_dir="/tmp/x", enabled=True,
        filename_prefix="", filename_suffix="", sample_rate=44100,
    )
    # Loud burst (above amplitude threshold) but caller forces False.
    rec.process_chunk(np.full(1024, 0.9, dtype=np.float32),
                      trigger_peak=0.9, should_trigger=False, **p)
    assert captured_flushes == []
    # Quiet burst with should_trigger=True → opens an event.
    rec.process_chunk(np.zeros(1024, dtype=np.float32),
                      trigger_peak=0.0, should_trigger=True, **p)
    # Sub-threshold to end it
    rec.process_chunk(np.zeros(1024, dtype=np.float32),
                      trigger_peak=0.0, should_trigger=False, **p)
    assert len(captured_flushes) == 1


def test_should_trigger_none_falls_back_to_peak_compare(captured_flushes):
    """When `should_trigger` is omitted the legacy compare path is used."""
    rec = ThresholdRecorder()
    p = dict(
        threshold=0.5, min_cross_sec=0.0, hold_sec=0.0,
        post_trig_sec=0.0, max_rec_sec=10.0, pre_trig_sec=0.0,
        output_dir="/tmp/x", enabled=True,
        filename_prefix="", filename_suffix="", sample_rate=44100,
    )
    rec.process_chunk(np.zeros(1024, dtype=np.float32),
                      trigger_peak=0.9, **p)
    rec.process_chunk(np.zeros(1024, dtype=np.float32),
                      trigger_peak=0.0, **p)
    assert len(captured_flushes) == 1


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
    """Every flush carries a datetime onset_time, derived from the
    monotonic clock anchor (#23 / c13)."""
    import datetime
    rec = ThresholdRecorder()
    p = _params()
    rec.process_chunk(_loud(0.5), trigger_peak=0.5, **p)
    rec.process_chunk(_silent(), trigger_peak=0.0, **p)
    assert len(captured_flushes) == 1
    onset = captured_flushes[0]["onset_time"]
    assert isinstance(onset, datetime.datetime)


def test_monotonic_anchor_resets_on_disable(captured_flushes):
    """Disabling clears the anchor so the next enable re-anchors fresh."""
    rec = ThresholdRecorder()
    on = _params(enabled=True)
    off = _params(enabled=False)
    rec.process_chunk(_loud(0.5), trigger_peak=0.5, **on)
    assert rec._mono_anchor is not None
    rec.process_chunk(_silent(), trigger_peak=0.0, **off)
    assert rec._mono_anchor is None


def test_flush_all_drains_active_events(captured_flushes):
    """`flush_all` writes every pending event and resets state (#17)."""
    rec = ThresholdRecorder()
    p = _params(hold_sec=10.0)  # huge hold so the event stays open
    rec.process_chunk(_loud(0.5), trigger_peak=0.5, **p)
    rec.process_chunk(_loud(0.5), trigger_peak=0.5, **p)
    assert rec.is_recording
    assert len(captured_flushes) == 0
    n = rec.flush_all(output_dir='/tmp/x', filename_stream='Mic_A',
                      reason='test')
    assert n == 1
    assert len(captured_flushes) == 1
    assert captured_flushes[0]["filename_stream"] == 'Mic_A'
    assert not rec.is_recording
    assert rec._mono_anchor is None


def test_filename_stream_kwarg_forwarded(captured_flushes):
    """The `filename_stream` kwarg flows through to the flush callback."""
    rec = ThresholdRecorder()
    p = _params()
    rec.process_chunk(_loud(0.5), trigger_peak=0.5,
                      filename_stream='Mic_A', **p)
    rec.process_chunk(_silent(), trigger_peak=0.0,
                      filename_stream='Mic_A', **p)
    assert len(captured_flushes) == 1
    assert captured_flushes[0]["filename_stream"] == 'Mic_A'


# ── #15 / c18: sample-accurate trigger evaluation ────────────────────────────

def test_sample_accurate_min_cross_5ms_burst(captured_flushes):
    """A 5 ms tone burst with min_cross_sec=0.003 should trigger.

    Regression gate from the plan: feed a 5 ms tone burst, verify
    `min_cross_sec=0.003` triggers correctly. Pre-c18 this would fail
    because min_cross was chunk-quantized to ~23 ms at 44.1 kHz.
    """
    sr = 44100
    rec = ThresholdRecorder()
    p = dict(
        threshold=0.1, min_cross_sec=0.003, hold_sec=0.0,
        post_trig_sec=0.0, max_rec_sec=10.0, pre_trig_sec=0.0,
        output_dir="/tmp/x", enabled=True,
        filename_prefix="", filename_suffix="", sample_rate=sr,
    )
    burst_samps = int(0.005 * sr)  # ~221 samples
    chunk = np.zeros(1024, dtype=np.float32)
    chunk[100:100 + burst_samps] = 0.5  # 5 ms tone embedded in a 1024-frame chunk
    mask = np.abs(chunk) >= 0.1
    rec.process_chunk(chunk, trigger_peak=0.5, trigger_mask=mask, **p)
    # The next chunk is silent → ends the event and flushes.
    silent = np.zeros(1024, dtype=np.float32)
    rec.process_chunk(silent, trigger_peak=0.0,
                      trigger_mask=np.zeros(1024, dtype=bool), **p)
    assert len(captured_flushes) == 1, (
        "5 ms burst should trigger with min_cross_sec=0.003")
    audio = captured_flushes[0]["audio"]
    # Trim should land exactly on the last above-threshold sample
    # (last_above_kept + 1 from event start). Since the burst is
    # 221 samples starting at offset 100 with no pre-trig, the kept
    # audio length is exactly burst_samps.
    assert audio.shape[0] == burst_samps, (
        f"Expected {burst_samps} samples sample-accurate, got {audio.shape[0]}")


def test_sample_accurate_min_cross_too_short_does_not_trigger(captured_flushes):
    """A 2 ms burst with min_cross_sec=0.003 must NOT trigger."""
    sr = 44100
    rec = ThresholdRecorder()
    p = dict(
        threshold=0.1, min_cross_sec=0.003, hold_sec=0.0,
        post_trig_sec=0.0, max_rec_sec=10.0, pre_trig_sec=0.0,
        output_dir="/tmp/x", enabled=True,
        filename_prefix="", filename_suffix="", sample_rate=sr,
    )
    short = int(0.002 * sr)  # ~88 samples, below the 132-sample threshold
    chunk = np.zeros(1024, dtype=np.float32)
    chunk[100:100 + short] = 0.5
    mask = np.abs(chunk) >= 0.1
    rec.process_chunk(chunk, trigger_peak=0.5, trigger_mask=mask, **p)
    rec.process_chunk(np.zeros(1024, dtype=np.float32),
                      trigger_peak=0.0,
                      trigger_mask=np.zeros(1024, dtype=bool), **p)
    assert captured_flushes == []


def test_sample_accurate_post_trig_tail_samples(captured_flushes):
    """post_trig should be sample-accurate, not chunk-quantized."""
    sr = 44100
    rec = ThresholdRecorder()
    post_samps = 500  # arbitrary non-chunk-aligned tail
    p = dict(
        threshold=0.1, min_cross_sec=0.0, hold_sec=0.0,
        post_trig_sec=post_samps / sr, max_rec_sec=10.0, pre_trig_sec=0.0,
        output_dir="/tmp/x", enabled=True,
        filename_prefix="", filename_suffix="", sample_rate=sr,
    )
    burst = 200
    chunk = np.zeros(1024, dtype=np.float32)
    chunk[0:burst] = 0.5
    mask = np.abs(chunk) >= 0.1
    rec.process_chunk(chunk, trigger_peak=0.5, trigger_mask=mask, **p)
    # Need another chunk to drain the post-trig tail.
    rec.process_chunk(np.zeros(1024, dtype=np.float32),
                      trigger_peak=0.0,
                      trigger_mask=np.zeros(1024, dtype=bool), **p)
    assert len(captured_flushes) == 1
    # Kept = burst (last_above_kept=burst-1, +1) + post_samps
    assert captured_flushes[0]["audio"].shape[0] == burst + post_samps
