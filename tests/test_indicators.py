"""Tests for detection / recorded-region indicators (#32).

Pins the contract of:
  * ``ThresholdRecorder.process_chunk`` returning a report dict with
    ``detect_mask``, ``active_spans``, and ``flushed_spans``.
  * ``RecordingEntity.detect_mask_buffer`` / ``record_mask_buffer``
    being maintained in lockstep with the amplitude ring buffer,
    including retroactive pre-trigger marking.

All tests avoid real WAV writes via the standard ``_start_flush``
monkeypatch, and avoid opening real audio devices by passing
``device_id=None`` (the capture falls back to invalid silently).
"""

from __future__ import annotations

import numpy as np
import pytest

from chirp.constants import CHUNK_FRAMES
from chirp.recording.entity import RecordingEntity
from chirp.recording.trigger import ThresholdRecorder


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def captured_flushes(monkeypatch):
    flushes: list[dict] = []

    def _capture(buf_snapshot, output_dir, prefix='', suffix='',
                 sample_rate=44100, onset_time=None, filename_stream=''):
        flushes.append({
            'audio': np.concatenate(list(buf_snapshot)),
            'n_chunks': len(buf_snapshot),
        })

    monkeypatch.setattr(ThresholdRecorder, "_start_flush",
                        staticmethod(_capture))
    yield flushes


def _p(**over):
    base = dict(
        threshold=0.5, min_cross_sec=0.0, hold_sec=0.0,
        post_trig_sec=0.0, max_rec_sec=10.0, pre_trig_sec=0.0,
        output_dir='/tmp/x', enabled=True,
        filename_prefix='', filename_suffix='', sample_rate=44100,
    )
    base.update(over)
    return base


# ── ThresholdRecorder report shape ─────────────────────────────────────

def test_report_shape_quiet_chunk(captured_flushes):
    rec = ThresholdRecorder()
    report = rec.process_chunk(np.zeros(1024, dtype=np.float32),
                               trigger_peak=0.0,
                               global_chunk_end=1024, **_p())
    assert set(report) == {'detect_mask', 'active_spans', 'flushed_spans'}
    assert report['detect_mask'].dtype == bool
    assert report['detect_mask'].shape == (1024,)
    assert not report['detect_mask'].any()
    assert report['active_spans'] == []
    assert report['flushed_spans'] == []


def test_report_detect_mask_follows_trigger_mask_input(captured_flushes):
    """When caller supplies a per-sample ``trigger_mask``, the
    returned ``detect_mask`` matches it byte-for-byte."""
    rec = ThresholdRecorder()
    chunk = np.zeros(1024, dtype=np.float32)
    chunk[100:300] = 0.9
    tmask = np.abs(chunk) >= 0.5
    report = rec.process_chunk(chunk, trigger_peak=0.0,
                               trigger_mask=tmask,
                               global_chunk_end=1024, **_p())
    np.testing.assert_array_equal(report['detect_mask'], tmask)


def test_report_detect_mask_broadcasts_should_trigger(captured_flushes):
    """Without trigger_mask, should_trigger broadcasts chunk-uniformly
    (the entity's normal driving path)."""
    rec = ThresholdRecorder()
    chunk = np.zeros(1024, dtype=np.float32)
    report = rec.process_chunk(chunk, trigger_peak=0.0,
                               should_trigger=True,
                               global_chunk_end=1024, **_p())
    assert report['detect_mask'].all()


def test_report_active_span_opens_on_trigger(captured_flushes):
    rec = ThresholdRecorder()
    # Long post_trig so the event stays open.
    params = _p(post_trig_sec=1.0, hold_sec=0.01)
    report = rec.process_chunk(np.full(1024, 0.9, dtype=np.float32),
                               trigger_peak=0.9,
                               global_chunk_end=1024, **params)
    # One event opened, none flushed yet.
    assert len(report['active_spans']) == 1
    assert report['flushed_spans'] == []
    g_lo, g_hi = report['active_spans'][0]
    # Event covers this chunk entirely (no pre-trigger configured).
    assert g_lo == 0
    assert g_hi == 1024


def test_report_flushed_span_on_event_close(captured_flushes):
    """A burst-then-silence sequence flushes on hold; flushed_spans
    carries the final WAV range."""
    rec = ThresholdRecorder()
    params = _p(hold_sec=0.0, post_trig_sec=0.0)
    # Chunk 1: above threshold.
    r1 = rec.process_chunk(np.full(1024, 0.9, dtype=np.float32),
                           trigger_peak=0.9,
                           global_chunk_end=1024, **params)
    assert r1['flushed_spans'] == []
    assert len(r1['active_spans']) == 1
    # Chunk 2: silence — hold=0 means the event ends on the first
    # below-threshold sample, and flushes this chunk.
    r2 = rec.process_chunk(np.zeros(1024, dtype=np.float32),
                           trigger_peak=0.0,
                           global_chunk_end=2048, **params)
    assert len(r2['flushed_spans']) == 1
    assert r2['active_spans'] == []
    g_lo, g_hi = r2['flushed_spans'][0]
    assert g_lo == 0
    # WAV covers chunk 1 only: last_above = sample 1023 (end of chunk
    # 1, event-coord), target = last_above + 1 + post_trig = 1024.
    assert g_hi == 1024


def test_report_spans_include_pre_trigger(captured_flushes):
    """pre_trig_sec pushes the span start into prior history."""
    rec = ThresholdRecorder()
    sr = 44100
    pre = CHUNK_FRAMES / sr  # exactly one chunk of pre-trigger
    params = _p(pre_trig_sec=pre, post_trig_sec=1.0, hold_sec=0.1)
    # Feed one silent chunk so the pre-trigger deque has history.
    rec.process_chunk(np.zeros(1024, dtype=np.float32),
                      trigger_peak=0.0,
                      global_chunk_end=1024, **params)
    # Now a loud chunk opens an event; pre_trig should reach back
    # one chunk.
    r = rec.process_chunk(np.full(1024, 0.9, dtype=np.float32),
                          trigger_peak=0.9,
                          global_chunk_end=2048, **params)
    assert len(r['active_spans']) == 1
    g_lo, g_hi = r['active_spans'][0]
    # Span starts at ~0 (one chunk before the loud chunk) and ends at
    # the chunk end = 2048.
    assert g_lo == 0
    assert g_hi == 2048


def test_report_spans_empty_without_global_chunk_end(captured_flushes):
    """Legacy callers that don't pass global_chunk_end get empty lists."""
    rec = ThresholdRecorder()
    r = rec.process_chunk(np.full(1024, 0.9, dtype=np.float32),
                          trigger_peak=0.9, **_p(post_trig_sec=1.0))
    assert r['detect_mask'].shape == (1024,)
    assert r['active_spans'] == []
    assert r['flushed_spans'] == []


def test_report_disabled_returns_empty_spans_with_mask(captured_flushes):
    """When enabled=False, still return the raw detect_mask."""
    rec = ThresholdRecorder()
    chunk = np.full(1024, 0.9, dtype=np.float32)
    r = rec.process_chunk(chunk, trigger_peak=0.9,
                          global_chunk_end=1024, **_p(enabled=False))
    assert r['detect_mask'].all()
    assert r['active_spans'] == []
    assert r['flushed_spans'] == []


# ── RecordingEntity indicator buffers ──────────────────────────────────

def _make_entity():
    e = RecordingEntity(name='ind-test', device_id=None, sample_rate=44100)
    e.threshold = 0.5
    e.min_cross_sec = 0.0
    e.hold_sec = 0.0
    e.pre_trig_sec = 0.0
    e.post_trig_sec = 0.0
    e.max_rec_sec = 10.0
    e.rec_enabled = True
    return e


def test_entity_has_indicator_buffers_matching_amp_size():
    e = _make_entity()
    try:
        assert e.detect_mask_buffer.shape == e.amp_buffer.shape
        assert e.record_mask_buffer.shape == e.amp_buffer.shape
        assert e.detect_mask_buffer.dtype == bool
        assert e.record_mask_buffer.dtype == bool
        assert not e.detect_mask_buffer.any()
        assert not e.record_mask_buffer.any()
    finally:
        e.close()


def test_entity_detect_buffer_mirrors_threshold_crossings(captured_flushes):
    e = _make_entity()
    try:
        chunk = np.zeros(CHUNK_FRAMES, dtype=np.float32)
        chunk[100:500] = 0.9
        e.ingest_chunk(chunk)
        assert e.detect_mask_buffer[100:500].all()
        assert not e.detect_mask_buffer[:100].any()
        assert not e.detect_mask_buffer[500:CHUNK_FRAMES].any()
    finally:
        e.close()


def test_entity_record_buffer_lights_up_on_active_event(captured_flushes):
    e = _make_entity()
    e.post_trig_sec = 1.0  # keep event open
    try:
        loud = np.full(CHUNK_FRAMES, 0.9, dtype=np.float32)
        e.ingest_chunk(loud)
        assert e.record_mask_buffer[:CHUNK_FRAMES].all()
    finally:
        e.close()


def test_entity_record_buffer_pre_trigger_retroactive(captured_flushes):
    """Pre-trigger samples get retroactively marked True when an
    event opens one chunk later."""
    e = _make_entity()
    sr = e.sample_rate
    e.pre_trig_sec = CHUNK_FRAMES / sr  # one chunk of lookback
    e.post_trig_sec = 1.0                # keep event open
    try:
        silent = np.zeros(CHUNK_FRAMES, dtype=np.float32)
        e.ingest_chunk(silent)
        # Before trigger: record_mask_buffer is all False.
        assert not e.record_mask_buffer[:CHUNK_FRAMES].any()
        loud = np.full(CHUNK_FRAMES, 0.9, dtype=np.float32)
        e.ingest_chunk(loud)
        # After trigger: both chunks (pre-trig + current) marked True.
        assert e.record_mask_buffer[:2 * CHUNK_FRAMES].all()
    finally:
        e.close()


def test_entity_record_buffer_clears_when_no_event(captured_flushes):
    """After an event flushes and silence follows, newly-overwritten
    samples go back to False."""
    e = _make_entity()
    try:
        loud = np.full(CHUNK_FRAMES, 0.9, dtype=np.float32)
        e.ingest_chunk(loud)
        # Event opens + flushes on the next silent chunk (hold=0).
        silent = np.zeros(CHUNK_FRAMES, dtype=np.float32)
        e.ingest_chunk(silent)
        e.ingest_chunk(silent)
        # After a full silent chunk past the flush, the ring region
        # covering that silence is False.
        assert not e.record_mask_buffer[2 * CHUNK_FRAMES:3 * CHUNK_FRAMES].any()
    finally:
        e.close()


def test_entity_reset_display_clears_indicator_buffers(captured_flushes):
    e = _make_entity()
    e.post_trig_sec = 1.0
    try:
        loud = np.full(CHUNK_FRAMES, 0.9, dtype=np.float32)
        e.ingest_chunk(loud)
        assert e.detect_mask_buffer.any()
        assert e.record_mask_buffer.any()
        e.reset_display()
        assert not e.detect_mask_buffer.any()
        assert not e.record_mask_buffer.any()
    finally:
        e.close()


def test_entity_change_sample_rate_rebuilds_indicator_buffers(captured_flushes):
    e = _make_entity()
    try:
        e.change_sample_rate(22050)
        assert e.detect_mask_buffer.shape == e.amp_buffer.shape
        assert e.record_mask_buffer.shape == e.amp_buffer.shape
        assert not e.detect_mask_buffer.any()
        assert not e.record_mask_buffer.any()
    finally:
        e.close()


def test_entity_change_display_seconds_rebuilds_indicator_buffers(captured_flushes):
    e = _make_entity()
    try:
        e.change_display_seconds(30.0)
        assert e.detect_mask_buffer.shape == e.amp_buffer.shape
        assert e.record_mask_buffer.shape == e.amp_buffer.shape
    finally:
        e.close()


def test_entity_record_buffer_clears_after_full_cycle(captured_flushes):
    """After an event ends, the marked samples must clear once the
    ring has rolled past them — i.e. the bars should not persist
    forever."""
    e = _make_entity()
    e.pre_trig_sec = 0.0
    e.hold_sec = 0.0
    e.post_trig_sec = 0.0
    try:
        # One loud chunk, then silence for enough chunks to wrap fully.
        loud = np.full(CHUNK_FRAMES, 0.9, dtype=np.float32)
        silent = np.zeros(CHUNK_FRAMES, dtype=np.float32)
        e.ingest_chunk(loud)
        # After the single burst, record_mask has ~1 chunk of True.
        assert e.record_mask_buffer.any()
        # Feed silence past a full cycle (n_cols chunks).
        for _ in range(e._n_cols + 5):
            e.ingest_chunk(silent)
        # After one full cycle plus, every ring position should have
        # been cleared at least once. No active event. No stale True.
        assert not e.record_mask_buffer.any(), (
            f'{int(e.record_mask_buffer.sum())} stale True samples '
            f'after wrapping past the event')
    finally:
        e.close()


def test_entity_detect_buffer_clears_after_full_cycle(captured_flushes):
    """detect_mask_buffer is overwritten per-chunk, so after one cycle
    of silence, any prior crossings must be gone."""
    e = _make_entity()
    try:
        chunk = np.zeros(CHUNK_FRAMES, dtype=np.float32)
        chunk[100:500] = 0.9
        e.ingest_chunk(chunk)
        assert e.detect_mask_buffer.any()
        silent = np.zeros(CHUNK_FRAMES, dtype=np.float32)
        for _ in range(e._n_cols + 5):
            e.ingest_chunk(silent)
        assert not e.detect_mask_buffer.any()
    finally:
        e.close()
