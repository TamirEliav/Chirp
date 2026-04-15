"""Characterization test for ThresholdRecorder.

This test pins the *current, pre-refactor* behavior of the trigger state
machine. It is intentionally **descriptive, not prescriptive** — it
records what the monolith does today so that the Phase 1 module
extraction (and the Phase 0 → Phase 1 transition more broadly) cannot
silently change behavior without producing a visible test failure.

Once Phase 1 lands and `ThresholdRecorder` moves into its own module,
this test is upgraded into a full suite (#15, #16 will then rewrite the
state machine itself, at which point these characterizations will be
updated to reflect the new, sample-accurate semantics).

Known quirks pinned by this test (to be addressed in later commits):
  - `min_cross` and `hold` operate at chunk granularity even though their
    sample-counting math looks fine-grained (issue #15).
  - Flushed WAVs are trimmed to chunk boundaries (issue #15).
  - The event onset uses wall-clock `datetime.now()` (issue #23).

The test avoids touching the disk by monkeypatching `_start_flush` to
capture flush calls in a list. This sidesteps the daemon-thread WAV
writer entirely (which is itself a target for issue #17).
"""

import numpy as np
import pytest


@pytest.fixture
def captured_flushes(monkeypatch):
    """Replace `_start_flush` with an in-memory capture.

    Yields a list that accumulates one entry per flush call; each entry
    is a dict with the buffer (as a single concatenated np.ndarray for
    easy length assertions) and the metadata kwargs.
    """
    import chirp

    flushes: list[dict] = []

    def _capture(buf_snapshot, output_dir, prefix='', suffix='',
                 sample_rate=chirp.SAMPLE_RATE, onset_time=None):
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
        chirp.ThresholdRecorder,
        "_start_flush",
        staticmethod(_capture),
    )
    yield flushes


def _silent_chunk():
    import chirp
    return np.zeros(chirp.CHUNK_FRAMES, dtype=np.float32)


def _loud_chunk(level: float = 0.5):
    import chirp
    return np.full(chirp.CHUNK_FRAMES, level, dtype=np.float32)


def _common_params(**overrides):
    """Default parameters chosen to make the basic case predictable.

    All time-based knobs are zero so the state machine's chunk-quantized
    quirks (issue #15) don't affect the count of samples in the flushed
    buffer — every transition happens on a chunk boundary anyway.
    """
    params = dict(
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
    params.update(overrides)
    return params


def test_single_event_lifecycle(captured_flushes):
    """A 3-chunk loud burst between silent chunks produces exactly one
    flush containing the 3 loud chunks (no pre-trigger, no post-trigger).
    """
    import chirp

    rec = chirp.ThresholdRecorder()
    params = _common_params()

    silent = _silent_chunk()
    loud = _loud_chunk(0.5)

    # Pre-roll: 3 silent chunks → no event
    for _ in range(3):
        rec.process_chunk(silent, trigger_peak=0.0, **params)
    assert not rec.is_recording
    assert len(captured_flushes) == 0

    # Burst: 3 loud chunks → event starts on the first one
    for _ in range(3):
        rec.process_chunk(loud, trigger_peak=0.5, **params)
    assert rec.is_recording, "Event should be active during the burst"
    assert len(captured_flushes) == 0, "No flush yet — the event is still open"

    # Post-roll: 1 silent chunk → ends the event (hold=0) and flushes
    rec.process_chunk(silent, trigger_peak=0.0, **params)
    assert not rec.is_recording, "Event should have ended on the first silent chunk"
    assert len(captured_flushes) == 1, "Exactly one event should have been flushed"

    # The flushed audio contains the 3 loud chunks (no tail since post_trig=0).
    flush = captured_flushes[0]
    expected_samples = 3 * chirp.CHUNK_FRAMES
    assert flush["audio"].shape[0] == expected_samples, (
        f"Expected {expected_samples} samples in the flushed event, "
        f"got {flush['audio'].shape[0]}"
    )
    # All flushed audio should be at the loud level (no silent contamination
    # at this knob configuration).
    assert np.allclose(flush["audio"], 0.5)


def test_disabled_recorder_does_nothing(captured_flushes):
    """When `enabled=False`, no event is created regardless of the signal."""
    import chirp

    rec = chirp.ThresholdRecorder()
    params = _common_params(enabled=False)
    loud = _loud_chunk(0.9)

    for _ in range(5):
        rec.process_chunk(loud, trigger_peak=0.9, **params)

    assert not rec.is_recording
    assert len(captured_flushes) == 0


def test_disable_mid_event_flushes(captured_flushes):
    """Disabling the recorder while an event is open flushes it.

    This pins the current `_was_enabled and not enabled` transition
    behavior in `process_chunk` (chirp.py:349-355).
    """
    import chirp

    rec = chirp.ThresholdRecorder()
    enabled_params = _common_params(enabled=True)
    disabled_params = _common_params(enabled=False)
    loud = _loud_chunk(0.5)
    silent = _silent_chunk()

    # Open an event
    for _ in range(2):
        rec.process_chunk(loud, trigger_peak=0.5, **enabled_params)
    assert rec.is_recording
    assert len(captured_flushes) == 0

    # Disable mid-event → should flush the in-progress event
    rec.process_chunk(silent, trigger_peak=0.0, **disabled_params)
    assert not rec.is_recording
    assert len(captured_flushes) == 1, (
        "Disabling mid-event should flush the open event"
    )
