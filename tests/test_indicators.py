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
    """Detect mask lights up in the region where a burst is present,
    stays dark elsewhere. Uses a 1 kHz sine burst because the mask is
    now built from the analytic-signal envelope (see envelope.py) —
    the envelope of a rectangular DC pulse has Hilbert sidelobes that
    bleed into the silent regions, while a sine burst gives a clean
    step transition.
    """
    e = _make_entity()
    try:
        sr = e.sample_rate
        chunk = np.zeros(CHUNK_FRAMES, dtype=np.float32)
        # 1 kHz sine at amp 0.9 over samples [200:800]. Margins of
        # ~200 samples on each side give the Hilbert transient enough
        # room to decay below threshold (0.5).
        burst_lo, burst_hi = 200, 800
        n_burst = burst_hi - burst_lo
        t = np.arange(n_burst, dtype=np.float64) / sr
        chunk[burst_lo:burst_hi] = (
            0.9 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
        e.ingest_chunk(chunk)
        # Interior of the burst is solidly True.
        assert e.detect_mask_buffer[burst_lo + 50:burst_hi - 50].all()
        # Far from the burst, mask is False (skip a guard band around
        # each edge for the Hilbert transient).
        guard = 100
        assert not e.detect_mask_buffer[:burst_lo - guard].any()
        assert not e.detect_mask_buffer[
            burst_hi + guard:CHUNK_FRAMES].any()
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


def test_entity_detect_mask_is_same_array_state_machine_sees(
        captured_flushes, monkeypatch):
    """Pins the 1:1 invariant: the bool array written into
    ``detect_mask_buffer`` for a chunk is literally the same
    ``trigger_mask`` that ``ThresholdRecorder.process_chunk`` walks
    sample-by-sample. Not two parallel computations.
    """
    e = _make_entity()
    seen: dict = {}

    real_process = e.recorder.process_chunk

    def _spy(chunk, *args, **kwargs):
        seen['trigger_mask'] = kwargs.get('trigger_mask')
        return real_process(chunk, *args, **kwargs)

    monkeypatch.setattr(e.recorder, 'process_chunk', _spy)
    try:
        sr = e.sample_rate
        # Two 1 kHz sine bursts inside one chunk, separated by silence.
        # The distinction between chunk-uniform broadcast and per-sample
        # walking only matters for signals whose mask changes mid-chunk.
        # Using sine bursts (not DC) because the mask is built from the
        # analytic envelope.
        chunk = np.zeros(CHUNK_FRAMES, dtype=np.float32)
        b1_lo, b1_hi = 150, 400   # first burst (250 samples wide)
        b2_lo, b2_hi = 550, 800   # second burst (250 samples wide)
        for lo, hi in ((b1_lo, b1_hi), (b2_lo, b2_hi)):
            n = hi - lo
            t = np.arange(n, dtype=np.float64) / sr
            chunk[lo:hi] = (
                0.9 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
        e.ingest_chunk(chunk)
        tm = seen['trigger_mask']
        assert tm is not None, 'entity must pass trigger_mask to recorder'
        assert tm.dtype == bool
        assert tm.shape == (CHUNK_FRAMES,)
        # detect_mask_buffer for this chunk's ring region must equal
        # the recorder's trigger_mask, element for element — the
        # single-source-of-truth invariant.
        np.testing.assert_array_equal(
            e.detect_mask_buffer[:CHUNK_FRAMES], tm)
        # Per-sample granularity: interior of each burst is True,
        # interior of each silent region is False (skip guard bands
        # around each boundary for the Hilbert transient). This
        # proves the mask is NOT a chunk-uniform broadcast.
        guard = 50
        assert tm[b1_lo + guard:b1_hi - guard].all()
        assert tm[b2_lo + guard:b2_hi - guard].all()
        # Gap between bursts has a True→False→True transition in the
        # middle — check the center of the gap, away from edges.
        gap_center = (b1_hi + b2_lo) // 2
        assert not tm[gap_center - 30:gap_center + 30].any()
        # Leading silence is False, well away from the first burst.
        assert not tm[:b1_lo - guard].any()
        # Trailing silence likewise.
        assert not tm[b2_hi + guard:].any()
    finally:
        e.close()


def test_entity_detect_mask_includes_spectral_gate(captured_flushes):
    """In 'Amp AND Spectral' mode, the detect strip must reflect the
    spectral gate too — not just amplitude crossings. This is part
    of the 1:1 contract (strip == state-machine input)."""
    e = _make_entity()
    e.spectral_trigger_mode = 'Amp AND Spectral'
    # Force spec_primed = True by priming the analysis FFT, and force
    # the gate CLOSED by setting spectral_threshold = 0 (entropy can
    # never go below zero → spec_triggered is always False).
    e.spectral_threshold = 0.0
    try:
        loud = np.full(CHUNK_FRAMES, 0.9, dtype=np.float32)
        # Ingest enough chunks to prime the analysis accumulator.
        for _ in range(8):
            e.ingest_chunk(loud)
        # Amplitude trigger would fire every sample (0.9 > 0.5), but
        # the closed spectral gate AND's everything to False — no
        # samples in the detect strip.
        assert not e.detect_mask_buffer.any(), (
            f'{int(e.detect_mask_buffer.sum())} samples still lit '
            f'despite closed spectral gate')
    finally:
        e.close()


def test_entity_detect_buffer_respects_bandpass_filter(captured_flushes):
    """When the bandpass filter is enabled, the detect-strip mask must
    be computed from the *filtered* signal, not the raw input. A loud
    out-of-band tone should NOT light up the detect strip.

    Pins the regression where the strip used `record` (raw) and lit
    up on tones the filter was supposed to suppress, making it look
    like the trigger was ignoring the bandpass.
    """
    e = _make_entity()
    e.freq_filter_enabled = True
    # Pass-band well away from the out-of-band tone we'll feed in.
    e.freq_lo = 5000.0
    e.freq_hi = 10000.0
    try:
        sr = e.sample_rate
        # 500 Hz tone, 0.9 amplitude — well below the 5 kHz high-pass.
        # Raw peak is 0.9 (> threshold 0.5). Filtered peak → ~0.
        t = np.arange(CHUNK_FRAMES, dtype=np.float32) / sr
        out_of_band = (0.9 * np.sin(2 * np.pi * 500.0 * t)).astype(np.float32)
        # Warm up the IIR state a little so the first chunk's transient
        # isn't what we measure — feed one chunk of the same signal
        # first, then assert on a fresh chunk.
        e.ingest_chunk(out_of_band)
        e.ingest_chunk(out_of_band)
        # After warm-up the filter output is ≪ 0.5 — no crossings.
        # Check the second chunk's ring region specifically.
        region = e.detect_mask_buffer[CHUNK_FRAMES:2 * CHUNK_FRAMES]
        assert not region.any(), (
            f'out-of-band tone lit up detect strip '
            f'({int(region.sum())} samples) — filter not honored')

        # Sanity: an IN-band tone of the same amplitude DOES light up.
        in_band = (0.9 * np.sin(2 * np.pi * 7000.0 * t)).astype(np.float32)
        e.ingest_chunk(in_band)
        e.ingest_chunk(in_band)
        region = e.detect_mask_buffer[3 * CHUNK_FRAMES:4 * CHUNK_FRAMES]
        assert region.any(), 'in-band tone failed to light up detect strip'
    finally:
        e.close()
