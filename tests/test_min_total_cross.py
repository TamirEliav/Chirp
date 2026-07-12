"""Min-total-crossing file filter.

An event whose ACCUMULATED above-threshold duration (summed over the
whole event, across separate bursts) stays below ``min_total_cross_sec``
is discarded at finalize time instead of written:

* buffered events simply never reach ``_start_flush``;
* streamed events abort their StreamingWavWriter, deleting the ``.tmp``
  in-progress file — nothing is ever published;
* force-split part flushes are EXEMPT (the event's total is still
  growing mid-split) and continuations inherit the parent's running
  total, so the criterion is judged once per event, on its final file;
* 0 (the default) disables the filter entirely.
"""

import os

import numpy as np
import pytest

from chirp.constants import CHUNK_FRAMES
from chirp.recording.trigger import ThresholdRecorder

SR = 44100


@pytest.fixture
def captured_flushes(monkeypatch):
    flushes: list[dict] = []

    def _capture(buf_snapshot, output_dir, prefix='', suffix='',
                 sample_rate=SR, onset_time=None, filename_stream=''):
        flushes.append({
            "audio": np.concatenate(list(buf_snapshot)),
            "suffix": suffix,
        })

    monkeypatch.setattr(
        ThresholdRecorder, "_start_flush", staticmethod(_capture),
    )
    yield flushes


def _params(min_total: float = 0.0, hold: float = 0.0, **over):
    p = dict(
        trigger_peak=0.9, threshold=0.5, min_cross_sec=0.0,
        hold_sec=hold, post_trig_sec=0.0, max_rec_sec=100.0,
        pre_trig_sec=0.0, output_dir=".", enabled=True,
        min_total_cross_sec=min_total, sample_rate=SR,
    )
    p.update(over)
    return p


def _chunk_with_mask(n_above_start: int):
    """One chunk whose trigger_mask is True for the first N samples."""
    chunk = np.full(CHUNK_FRAMES, 0.9, dtype=np.float32)
    mask = np.zeros(CHUNK_FRAMES, dtype=bool)
    mask[:n_above_start] = True
    return chunk, mask


def test_short_crossing_discarded(captured_flushes):
    rec = ThresholdRecorder()
    chunk, mask = _chunk_with_mask(100)          # 100 above samples total
    rec.process_chunk(chunk, trigger_mask=mask,
                      **_params(min_total=200 / SR))
    assert captured_flushes == []                # event ended + discarded


def test_crossing_at_minimum_is_kept(captured_flushes):
    rec = ThresholdRecorder()
    chunk, mask = _chunk_with_mask(100)
    rec.process_chunk(chunk, trigger_mask=mask,
                      **_params(min_total=100 / SR))
    assert len(captured_flushes) == 1            # exactly at min → kept


def test_zero_minimum_keeps_everything(captured_flushes):
    rec = ThresholdRecorder()
    chunk, mask = _chunk_with_mask(3)
    rec.process_chunk(chunk, trigger_mask=mask, **_params(min_total=0.0))
    assert len(captured_flushes) == 1


def test_crossing_accumulates_across_bursts(captured_flushes):
    """Two 60-sample bursts separated by a sub-hold gap: no single burst
    reaches 100 samples but the ACCUMULATED total (120) does."""
    rec = ThresholdRecorder()
    chunk = np.full(CHUNK_FRAMES, 0.9, dtype=np.float32)
    mask = np.zeros(CHUNK_FRAMES, dtype=bool)
    mask[0:60] = True
    mask[65:125] = True                          # 5-sample gap < hold
    rec.process_chunk(chunk, trigger_mask=mask,
                      **_params(min_total=100 / SR, hold=100 / SR))
    assert len(captured_flushes) == 1

    rec2 = ThresholdRecorder()
    rec2.process_chunk(chunk, trigger_mask=mask,
                       **_params(min_total=150 / SR, hold=100 / SR))
    assert len(captured_flushes) == 1            # 120 < 150 → discarded


def test_disable_flush_applies_criterion(captured_flushes):
    """REC toggled off mid-event: the file is final, so the criterion
    applies to whatever crossing had accumulated by then."""
    rec = ThresholdRecorder()
    loud = np.full(CHUNK_FRAMES, 0.9, dtype=np.float32)
    all_above = np.ones(CHUNK_FRAMES, dtype=bool)
    p = _params(min_total=2 * CHUNK_FRAMES / SR, hold=1.0)
    rec.process_chunk(loud, trigger_mask=all_above, **p)
    assert rec.is_recording
    p_off = dict(p, enabled=False)
    rec.process_chunk(np.zeros(CHUNK_FRAMES, dtype=np.float32),
                      trigger_mask=np.zeros(CHUNK_FRAMES, dtype=bool),
                      **p_off)
    assert captured_flushes == []                # 1024 < 2048 → discarded


def test_flush_all_applies_criterion(captured_flushes):
    rec = ThresholdRecorder()
    loud = np.full(CHUNK_FRAMES, 0.9, dtype=np.float32)
    all_above = np.ones(CHUNK_FRAMES, dtype=bool)
    rec.process_chunk(loud, trigger_mask=all_above,
                      **_params(min_total=2 * CHUNK_FRAMES / SR, hold=1.0))
    assert rec.flush_all(".") == 1               # one event was pending
    assert captured_flushes == []                # …but it was discarded


def test_force_split_parts_exempt_and_total_inherited(captured_flushes):
    """A force-split part flush happens mid-event (total still growing)
    so it must NOT be filtered even under an absurdly large minimum;
    the continuation carries the running total."""
    rec = ThresholdRecorder()
    loud = np.full(CHUNK_FRAMES, 0.9, dtype=np.float32)
    all_above = np.ones(CHUNK_FRAMES, dtype=bool)
    p = _params(min_total=10 * CHUNK_FRAMES / SR, hold=1.0,
                max_rec_sec=CHUNK_FRAMES / SR)
    for _ in range(3):
        rec.process_chunk(loud, trigger_mask=all_above, **p)
    part_suffixes = [f["suffix"] for f in captured_flushes]
    assert part_suffixes == ["part01", "part02", "part03"]


def test_streaming_discard_deletes_tmp(tmp_path):
    """Streamed events already have bytes on disk in a ``.tmp`` file —
    a discard must delete it and never publish the canonical WAV."""
    out = str(tmp_path)
    rec = ThresholdRecorder(streaming=True)
    loud = np.full(CHUNK_FRAMES, 0.9, dtype=np.float32)
    all_above = np.ones(CHUNK_FRAMES, dtype=bool)
    none_above = np.zeros(CHUNK_FRAMES, dtype=bool)
    p = _params(min_total=5 * CHUNK_FRAMES / SR, hold=10 / SR,
                output_dir=out)
    rec.process_chunk(loud, trigger_mask=all_above, **p)   # opens + streams
    rec.process_chunk(np.zeros(CHUNK_FRAMES, dtype=np.float32),
                      trigger_mask=none_above, **p)        # ends → discard
    assert os.listdir(out) == []                           # no wav, no tmp

    # Control: same drive with the filter off → the WAV is published.
    rec2 = ThresholdRecorder(streaming=True)
    p2 = _params(min_total=0.0, hold=10 / SR, output_dir=out)
    rec2.process_chunk(loud, trigger_mask=all_above, **p2)
    rec2.process_chunk(np.zeros(CHUNK_FRAMES, dtype=np.float32),
                       trigger_mask=none_above, **p2)
    names = os.listdir(out)
    assert len(names) == 1 and names[0].endswith(".wav")


def test_param_serialization_roundtrip():
    from chirp.recording.entity import RecordingEntity
    e = RecordingEntity(name="rt", device_id=None)
    try:
        e.min_total_cross_sec = 0.75
        d = e.to_dict()
        assert d["min_total_cross_sec"] == 0.75
        e2, _warn = RecordingEntity.from_dict(d)
        try:
            assert e2.min_total_cross_sec == 0.75
        finally:
            e2.close()
    finally:
        e.close()
