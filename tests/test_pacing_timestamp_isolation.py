"""Display pacing / A/V sync must not perturb WAV filename timestamps.

The pacing work (paced cursor, paced display buffers, monitor-delay
servo) touches the same entity that owns the timestamp clock, and it
reads the same counters the clock is built on. Filename timestamps are
the one thing in Chirp that cannot be checked after the fact from the
data itself — a wrong one silently mislabels a recording — so this file
pins the separation directly rather than by inspection:

* the same audio, ingested with the display being paced and without,
  must produce bit-identical capture timestamps and identical filenames;
* pacing must never move ``_samples_total`` or the ring cursors (the
  documented invariant the disciplined clock depends on);
* the A/V sync servo, driven hard, must not perturb them either;
* bursty delivery (WASAPI exclusive / WDM-KS hand over one device buffer
  at a time, which is what the recent capture work enabled) must not
  bias the clock: several callbacks arrive back-to-back with the same
  wall time, and the windowed-minimum filter has to pick the last one.
"""

from __future__ import annotations

import datetime

import numpy as np
import pytest

from chirp.audio.clock import DisciplinedClock
from chirp.constants import CHUNK_FRAMES
from chirp.recording.entity import RecordingEntity

T0 = 1786000000.0
ANCHOR = datetime.datetime(2026, 8, 10, 9, 0, 0)


def _spy(e):
    """Record every chunk_end_wall the recorder is handed."""
    seen = []
    orig = e.recorder.process_chunk

    def wrapped(*a, **kw):
        seen.append(kw.get('chunk_end_wall'))
        return orig(*a, **kw)

    e.recorder.process_chunk = wrapped
    return seen


def _audio(k):
    t = (np.arange(CHUNK_FRAMES) + k * CHUNK_FRAMES) / 44100.0
    return (0.3 * np.sin(2 * np.pi * 700 * t)).astype(np.float32)


def _run(paced: bool, *, use_clock: bool, monitor_delay=0.4, n=60):
    """Ingest n chunks, optionally driving the display pacing between
    them exactly as the UI tick does. Returns the capture timestamps."""
    e = RecordingEntity(name='ts', device_id=None)
    try:
        seen = _spy(e)
        if not use_clock:
            e._wall_anchor_time = ANCHOR
            e._wall_anchor_samples = 0
        wall = T0
        for k in range(n):
            abs_end = (k + 1) * CHUNK_FRAMES
            if use_clock:
                # One observation per callback, as AudioCapture does.
                e.clock.observe(abs_end, wall)
            e.ingest_chunk(_audio(k), abs_end=abs_end)
            wall += CHUNK_FRAMES / e.sample_rate
            if paced:
                # The UI tick, interleaved with ingestion.
                e.advance_display(now=k * 0.05,
                                  monitor_delay_sec=monitor_delay)
                e.publish_display()
        return [None if c is None else c.timestamp() for c in seen], e
    finally:
        e.close()


@pytest.mark.parametrize('use_clock', [True, False],
                         ids=['disciplined-clock', 'anchor-fallback'])
def test_pacing_does_not_change_capture_timestamps(use_clock):
    """The decisive check: identical audio in, identical timestamps out,
    whether or not the display is being paced."""
    unpaced, _ = _run(False, use_clock=use_clock)
    paced, _ = _run(True, use_clock=use_clock)
    assert unpaced == paced
    assert any(t is not None for t in paced), 'test drove no timestamps at all'


def test_pacing_does_not_move_the_sample_counter_or_cursors():
    """The invariant the timestamp clock is built on: display-only work
    never touches ``_samples_total`` or the ring cursors."""
    e = RecordingEntity(name='inv', device_id=None)
    try:
        for k in range(20):
            e.ingest_chunk(_audio(k), abs_end=(k + 1) * CHUNK_FRAMES)
        before = (e._samples_total, e.write_head, e.col_head,
                  e._wall_anchor_samples)
        for i in range(400):
            e.advance_display(now=i * 0.05, monitor_delay_sec=0.6)
            e.publish_display()
        after = (e._samples_total, e.write_head, e.col_head,
                 e._wall_anchor_samples)
        assert before == after
    finally:
        e.close()


def test_sync_servo_does_not_perturb_timestamps_under_stress():
    """Drive the servo across its whole correction range (monitor delay
    swinging every tick) while ingesting, and require the timestamps to
    match a run with no pacing at all."""
    plain, _ = _run(False, use_clock=True)

    e = RecordingEntity(name='stress', device_id=None)
    try:
        seen = _spy(e)
        wall = T0
        for k in range(60):
            abs_end = (k + 1) * CHUNK_FRAMES
            e.clock.observe(abs_end, wall)
            e.ingest_chunk(_audio(k), abs_end=abs_end)
            wall += CHUNK_FRAMES / e.sample_rate
            # A monitor delay that lurches between extremes — far worse
            # than anything a real device does.
            e.advance_display(now=k * 0.05,
                              monitor_delay_sec=(0.05 if k % 2 else 1.9))
            e.publish_display()
        stressed = [None if c is None else c.timestamp() for c in seen]
    finally:
        e.close()
    assert plain == stressed


def test_paced_view_never_feeds_back_into_the_recorded_signal():
    """``view()`` hands out copies; nothing the display does may reach
    the buffers the trigger and writer read."""
    e = RecordingEntity(name='feedback', device_id=None)
    try:
        for k in range(30):
            e.ingest_chunk(_audio(k), abs_end=(k + 1) * CHUNK_FRAMES)
        e.advance_display(now=0.0, monitor_delay_sec=0.4)
        e.publish_display()
        for name in ('amp_buffer', 'abs_amp_buffer', 'spec_buffer'):
            v = e.view(name)
            assert v is not getattr(e, name), f'{name} view aliases the live buffer'
            live_before = getattr(e, name).copy()
            v[...] = -12345.0                     # scribble on the view
            np.testing.assert_array_equal(getattr(e, name), live_before)
    finally:
        e.close()


def test_filenames_match_between_paced_and_unpaced_runs():
    """End to end: the composed WAV names, not just the epochs."""
    from chirp.recording.writer import _compose_filename

    unpaced, _ = _run(False, use_clock=True)
    paced, _ = _run(True, use_clock=True)
    def name(t):
        return _compose_filename('pre', 'suf', datetime.datetime.fromtimestamp(
            t, tz=datetime.timezone.utc))

    names_u = [name(t) for t in unpaced if t is not None]
    names_p = [name(t) for t in paced if t is not None]
    assert names_u and names_u == names_p


# ── Bursty delivery (WASAPI exclusive / WDM-KS) ──────────────────────────

def test_clock_picks_the_last_callback_of_a_burst():
    """Exclusive mode and WDM-KS hand over a whole device buffer at
    once, so PortAudio fires several blocksize callbacks back-to-back
    with (almost) the same wall time. Only the LAST one is honest — its
    newest sample really was captured just now; the earlier ones claim
    'now' for audio that is one or two blocks old. The windowed-minimum
    filter must therefore settle on the last, or every filename would be
    stamped early by up to one device buffer.
    """
    sr = 44100
    blk = 8192
    clk = DisciplinedClock(sr)
    # Six 8192-frame callbacks per 1.1 s burst, delivered together.
    burst_blocks = 6
    burst_dur = burst_blocks * blk / sr
    wall = T0
    total = 0
    for b in range(20):
        # A buffer is handed over only once it has been filled.
        wall = T0 + (b + 1) * burst_dur
        for j in range(burst_blocks):
            total += blk
            clk.observe(total, wall + j * 0.0005)   # ~0.5 ms apart
    got = clk.wall_at(total)
    assert got is not None
    # The last sample of the burst was captured at `wall`; a filter that
    # latched onto the FIRST callback would be ~5 blocks (0.93 s) early.
    assert abs(got - wall) < 0.05, \
        f'{got - wall:+.3f}s off — filter did not track the burst tail'


def test_bursty_delivery_does_not_drift_over_a_long_run():
    """A whole hour of burst-delivered callbacks must not accumulate
    error: the filenames of a long unattended run depend on it."""
    sr = 44100
    blk = 8192
    clk = DisciplinedClock(sr)
    burst_blocks = 6
    burst_dur = burst_blocks * blk / sr
    total = 0
    wall = T0
    for b in range(int(3600 / burst_dur)):
        wall = T0 + (b + 1) * burst_dur
        for j in range(burst_blocks):
            total += blk
            clk.observe(total, wall + j * 0.0005)
        clk.wall_at(total)          # as the ingest thread would
    got = clk.wall_at(total)
    assert abs(got - wall) < 0.05, f'drifted {got - wall:+.3f}s over an hour'


def test_first_timestamp_is_right_immediately_under_burst_delivery():
    """Regression: the correction used to start at zero and slew.

    The anchor observation is the FIRST callback of the first delivery,
    which under burst delivery is the most-late observation of its
    group — so the filtered target starts a whole device buffer behind
    it. At 1 ms of correction per second of audio that took 10-30
    minutes to walk out, and every filename written in the meantime was
    stamped late by up to that buffer (measured: +900 ms at start, still
    +329 ms ten minutes in, on a 1 s WDM-KS buffer). Nothing had been
    timestamped yet at the first call, so there was nothing for the slew
    limiter to protect: the filtered offset is now adopted outright.
    """
    sr, blk = 44100, 8192
    for burst_blocks in (3, 6, 11):        # 0.5 s, 1 s, 2 s device buffers
        clk = DisciplinedClock(sr)
        dur = burst_blocks * blk / sr
        total = 0
        first_err = None
        for b in range(int(300 / dur)):    # five minutes
            wall = T0 + (b + 1) * dur
            for j in range(burst_blocks):
                total += blk
                clk.observe(total, wall + j * 0.0005)
            err = clk.wall_at(total) - wall
            if first_err is None:
                first_err = err
            assert abs(err) < 0.05, (
                f'{burst_blocks} blocks/burst: {err * 1000:+.0f} ms at '
                f't={(b + 1) * dur:.0f}s')
        assert abs(first_err) < 0.05, \
            f'{burst_blocks} blocks/burst: first stamp {first_err * 1000:+.0f} ms out'


def test_initial_adoption_does_not_become_a_licence_to_jump():
    """Only the FIRST call adopts. After that the slew limiter must
    still bound every correction, or adjacent and split files would stop
    being sample-aligned."""
    sr = 44100
    clk = DisciplinedClock(sr)
    clk.observe(sr, T0)
    first = clk.wall_at(sr)
    assert first == pytest.approx(T0, abs=1e-6)

    # A wildly late observation arrives; the clock must crawl, not jump.
    clk.observe(2 * sr, T0 + 1.0 + 0.5)
    second = clk.wall_at(2 * sr)
    assert second - first == pytest.approx(1.0, abs=2e-3), \
        'a post-init correction escaped the slew limiter'


def test_capture_restart_re_adopts_rather_than_inheriting_a_slew():
    """Stop/Start Acq, a sample-rate change and the zero-run recovery all
    build a fresh clock — each must land on the correct offset at once,
    not inherit a 15-minute convergence."""
    sr, blk = 44100, 8192
    for _ in range(3):
        clk = DisciplinedClock(sr)
        total = 0
        wall = T0
        for b in range(3):
            wall = T0 + (b + 1) * (6 * blk / sr)
            for j in range(6):
                total += blk
                clk.observe(total, wall)
        assert abs(clk.wall_at(total) - wall) < 0.01
