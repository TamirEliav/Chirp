"""Display pacing: a real-time cursor over burst-delivered capture.

The plots draw the ring buffer with a cursor at the write head, so the
view advances exactly as audio is *delivered*. Under WASAPI exclusive
and WDM-KS that is one device buffer at a time — at a 1 s buffer the
spectrogram advances once per second, which cannot be followed next to
the (smooth) audio monitor.

``RecordingEntity.advance_display`` decouples the two: a second cursor
moves at real time, held back from the write head by a self-tuning
cushion. These tests pin the properties that make that safe to watch:

* the cursor never runs backwards, and never past data we don't have;
* it FREEZES rather than jumping when it catches up, and the cushion
  doubles so the next burst is covered;
* an idle stream doesn't starve its way to the maximum cushion;
* a counter reset (sample-rate change) and a long UI stall resync
  instead of replaying history in slow motion;
* rendering never modifies the entity's buffers;
* HISTORY stays visible ahead of the cursor until the cursor reaches it.

That last one is why ``view``/``publish_display`` exist. The ingest
thread has already overwritten the region between the paced cursor and
the write head, so a panel reading the live buffers sees audio the
monitor has not played yet — the view and the sound disagree. Blanking
that strip was tried first and was worse: its far edge is the write
head, which still jumps by a whole device buffer, so the blank pulsed
at the burst rate. The paced copy keeps the previous sweep instead,
which is what a scrolling display is supposed to show.
"""

import numpy as np
import pytest

from chirp.constants import CHUNK_FRAMES
from chirp.recording.entity import RecordingEntity


@pytest.fixture
def ent():
    e = RecordingEntity(name='pace', device_id=None)
    yield e
    e.close()


def _ingest(e, n_chunks):
    """Advance the write head without going through the DSP pipeline."""
    n = n_chunks * CHUNK_FRAMES
    e._samples_total += n
    e.write_head = e._samples_total % e._total_samples
    return n


# ── Pacing ───────────────────────────────────────────────────────────────

def test_first_tick_syncs_and_primes(ent):
    _ingest(ent, 40)
    ent.advance_display(now=100.0)
    assert ent.display_head == ent.write_head
    assert ent._disp_priming is True


def test_cursor_advances_at_real_time_not_delivery_time(ent):
    sr = ent.sample_rate
    _ingest(ent, 40)
    ent.advance_display(now=100.0)
    # Build the cushion: data arrives, cursor holds.
    _ingest(ent, 40)
    ent.advance_display(now=100.05)
    assert ent._disp_priming is False

    # Now a burst lands and nothing arrives for a while. The cursor must
    # keep moving smoothly through the gap rather than stepping.
    before = ent._disp_abs
    for k in range(1, 6):
        ent.advance_display(now=100.05 + 0.05 * k)
    moved = ent._disp_abs - before
    assert moved == pytest.approx(0.25 * sr, rel=0.02), \
        'cursor must advance by wall-clock time, not by what arrived'


def test_cursor_never_passes_the_data_and_freezes_instead(ent):
    _ingest(ent, 4)
    ent.advance_display(now=0.0)
    _ingest(ent, 4)
    ent.advance_display(now=0.05)       # cushion built, now playing
    assert ent._disp_priming is False
    lag_before = ent._disp_lag
    # A whole second of wall time, but only one chunk arrived.
    _ingest(ent, 1)
    ent.advance_display(now=1.0)
    assert ent._disp_abs <= ent._samples_total, 'must never show absent audio'
    assert ent._disp_priming is True, 'must freeze, not race ahead'
    assert ent._disp_lag > lag_before, 'cushion must grow after a starve'


def test_cursor_never_moves_backwards(ent):
    sr = ent.sample_rate
    heads = []
    t = 0.0
    for cycle in range(30):
        # 0.5 s bursts every 0.5 s — the exclusive/WDM-KS pattern.
        _ingest(ent, int(0.5 * sr) // CHUNK_FRAMES)
        for _ in range(10):          # 10 ticks of 50 ms = the 0.5 s gap
            t += 0.05
            ent.advance_display(now=t)
            heads.append(ent._disp_abs)
    assert all(b >= a for a, b in zip(heads, heads[1:])), \
        'a display that jumps back reads as broken'
    assert ent._disp_abs <= ent._samples_total


def test_cushion_converges_and_stops_starving(ent):
    """After a few bursts the cushion must cover the delivery cadence,
    so the cursor stops freezing and simply runs."""
    sr = ent.sample_rate
    t = 0.0
    for cycle in range(40):
        _ingest(ent, int(0.5 * sr) // CHUNK_FRAMES)
        for _ in range(10):
            t += 0.05
            ent.advance_display(now=t)
    # Settled: keep driving and require no further freezes.
    froze = 0
    for cycle in range(20):
        _ingest(ent, int(0.5 * sr) // CHUNK_FRAMES)
        for _ in range(10):
            t += 0.05
            ent.advance_display(now=t)
            if ent._disp_priming:
                froze += 1
    assert froze == 0, f'cushion never converged ({froze} freezes)'
    assert ent.display_lag_sec > 0.0


def test_idle_stream_does_not_inflate_its_cushion(ent):
    """Acquisition stopped: nothing arrives for minutes. The cushion
    must not ratchet up to the maximum and be inherited by the next
    run."""
    _ingest(ent, 40)
    ent.advance_display(now=0.0)
    _ingest(ent, 40)
    ent.advance_display(now=0.05)
    lag = ent._disp_lag
    for k in range(1, 600):
        ent.advance_display(now=0.05 + 0.05 * k)
    assert ent._disp_lag <= lag * 2, 'idle must not grow the cushion'


def test_counter_reset_resyncs(ent):
    """A sample-rate change rebuilds the pipeline and zeroes the
    counter; the paced cursor must not sit in the future."""
    _ingest(ent, 100)
    ent.advance_display(now=0.0)
    ent.advance_display(now=0.05)
    ent._samples_total = 0
    ent.write_head = 0
    ent.advance_display(now=0.10)
    assert ent._disp_abs == 0.0
    assert ent._disp_priming is True


def test_long_ui_stall_skips_forward(ent):
    """After a blocked UI thread the cursor must rejoin the present, not
    replay minutes of history at real-time speed."""
    sr = ent.sample_rate
    _ingest(ent, 40)
    ent.advance_display(now=0.0)
    _ingest(ent, 40)
    ent.advance_display(now=0.05)
    ent._disp_priming = False
    _ingest(ent, int(120 * sr) // CHUNK_FRAMES)     # 2 minutes arrived
    ent.advance_display(now=0.10)
    assert ent.display_lag_sec <= 2 * ent.DISPLAY_LAG_MAX_SEC + 0.1


def test_cushion_is_capped_by_the_visible_window(ent):
    """A cushion is a blanked strip; it must never eat the window."""
    for _ in range(200):
        _ingest(ent, 1)
        ent.advance_display(now=0.0)
        ent.advance_display(now=10.0)      # force a starve every round
    assert ent.display_lag_sec <= ent.DISPLAY_LAG_MAX_SEC + 1e-6
    assert (ent._disp_lag
            <= ent.DISPLAY_LAG_MAX_FRACTION * ent._total_samples + CHUNK_FRAMES)


# ── Panels ───────────────────────────────────────────────────────────────

def test_panels_render_paced_without_touching_live_buffers():
    """Rendering is a read: the display must never write into buffers
    the ingest thread and recorder still own."""
    pytest.importorskip('pyqtgraph')
    from PyQt5.QtWidgets import QApplication

    from chirp.ui.pg_panel import StreamPlotPanel
    QApplication.instance() or QApplication([])

    e = RecordingEntity(name='paced', device_id=None)
    try:
        sr = e.sample_rate
        for k in range(40):
            t = (np.arange(CHUNK_FRAMES) + k * CHUNK_FRAMES) / sr
            e.ingest_chunk((0.4 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32))
        e.advance_display(now=0.0)
        e.advance_display(now=0.05)
        e._disp_priming = False
        e._disp_abs -= 4 * CHUNK_FRAMES         # cursor behind the data

        amp_before = e.amp_buffer.copy()
        abs_before = e.abs_amp_buffer.copy()
        spec_before = e.spec_buffer.copy()
        ent_before = e.entropy_buffer.copy()

        panel = StreamPlotPanel(use_opengl=False, show_waveform=True)
        panel.update_from_entity(e)

        np.testing.assert_array_equal(e.amp_buffer, amp_before)
        np.testing.assert_array_equal(e.abs_amp_buffer, abs_before)
        np.testing.assert_array_equal(e.spec_buffer, spec_before)
        np.testing.assert_array_equal(e.entropy_buffer, ent_before)
    finally:
        e.close()


def test_panel_cursor_follows_the_paced_head_not_the_write_head():
    pytest.importorskip('pyqtgraph')
    from PyQt5.QtWidgets import QApplication

    from chirp.ui.pg_panel import StreamPlotPanel
    QApplication.instance() or QApplication([])

    e = RecordingEntity(name='paced2', device_id=None)
    try:
        _ingest(e, 40)
        e.advance_display(now=0.0)
        e._disp_priming = False
        e._disp_abs -= 8 * CHUNK_FRAMES
        panel = StreamPlotPanel(use_opengl=False)
        panel.update_from_entity(e)
        expected = (e.display_head / e.sample_rate) % float(e.display_seconds)
        assert panel._spec_cursor.value() == pytest.approx(expected)
        assert panel._spec_cursor.value() != pytest.approx(
            (e.write_head / e.sample_rate) % float(e.display_seconds))
    finally:
        e.close()


def test_unpaced_entity_renders_exactly_as_before(ent):
    """Nothing must change for a caller that never drives the pacing:
    the cursor follows the write head, exactly as before pacing."""
    _ingest(ent, 40)
    assert ent._disp_wall is None
    assert ent.display_head == ent.write_head


# ── Paced display buffers ────────────────────────────────────────────────

def _fill_live(e, value, start, n):
    """Write ``value`` into the live per-sample display buffers as the
    ingest thread would, wrapping, and move the write head there."""
    total = e._total_samples
    idx = (np.arange(start, start + n) % total)
    e.amp_buffer[idx] = value
    e.abs_amp_buffer[idx] = abs(value)
    e._samples_total = start + n
    e.write_head = e._samples_total % total


def test_history_stays_visible_until_the_cursor_reaches_it(ent):
    """The property the whole feature exists for: what is drawn ahead of
    the red line is the PREVIOUS sweep, not audio the monitor has yet to
    play."""
    ent.amp_buffer[:] = 1.0                 # sweep 1 = history
    ent.advance_display(now=0.0)            # engage pacing
    assert ent.view('amp_buffer') is not ent.amp_buffer
    np.testing.assert_array_equal(ent.view('amp_buffer'), 1.0)

    # A burst of sweep-2 audio lands; the cursor has only reached 2 of
    # its 5 chunks.
    _fill_live(ent, 2.0, 0, 5 * CHUNK_FRAMES)
    ent._disp_abs = float(2 * CHUNK_FRAMES)
    ent.publish_display()

    view = ent.view('amp_buffer')
    np.testing.assert_array_equal(view[:2 * CHUNK_FRAMES], 2.0), 'revealed'
    np.testing.assert_array_equal(
        view[2 * CHUNK_FRAMES:5 * CHUNK_FRAMES], 1.0), 'history preserved'
    # ...while the live buffer has already been overwritten.
    np.testing.assert_array_equal(
        ent.amp_buffer[2 * CHUNK_FRAMES:5 * CHUNK_FRAMES], 2.0)

    # The cursor arrives: history is replaced, in order.
    ent._disp_abs = float(5 * CHUNK_FRAMES)
    ent.publish_display()
    np.testing.assert_array_equal(ent.view('amp_buffer')[:5 * CHUNK_FRAMES], 2.0)


def test_published_region_wraps(ent):
    ent.amp_buffer[:] = 1.0
    ent.advance_display(now=0.0)
    ent.view('amp_buffer')
    total = ent._total_samples
    start = total - 2 * CHUNK_FRAMES         # straddle the end of the ring
    ent._view_pos = start
    ent._view_col_pos = start // CHUNK_FRAMES
    _fill_live(ent, 3.0, start, 4 * CHUNK_FRAMES)
    ent._disp_abs = float(start + 4 * CHUNK_FRAMES)
    ent.publish_display()
    view = ent.view('amp_buffer')
    np.testing.assert_array_equal(view[start:], 3.0)          # tail
    np.testing.assert_array_equal(view[:2 * CHUNK_FRAMES], 3.0)  # wrapped head


def test_column_buffers_advance_in_whole_columns(ent):
    """Spectrogram columns move in CHUNK_FRAMES steps; a sample cursor
    that lands mid-column must not skip or re-copy one."""
    ent.spec_buffer[:] = -99.0
    ent.advance_display(now=0.0)
    ent.view('spec_buffer')
    ent.spec_buffer[:] = -5.0                # a whole new sweep arrives
    ent._samples_total = 10 * CHUNK_FRAMES
    # Cursor lands mid-column: 2 whole columns are revealed.
    ent._disp_abs = 2.5 * CHUNK_FRAMES
    ent.publish_display()
    view = ent.view('spec_buffer')
    np.testing.assert_array_equal(view[:, :2], -5.0)
    np.testing.assert_array_equal(view[:, 2:], -99.0)
    # The rest of that column is published once the cursor clears it.
    ent._disp_abs = 3.0 * CHUNK_FRAMES
    ent.publish_display()
    np.testing.assert_array_equal(ent.view('spec_buffer')[:, :3], -5.0)


def test_unpaced_entity_draws_the_live_buffers(ent):
    """No pacing driven → no mirror allocated, and panels see live data
    exactly as before this feature."""
    assert ent.view('amp_buffer') is ent.amp_buffer
    assert ent.view('spec_buffer') is ent.spec_buffer
    assert ent._view_bufs == {}


def test_buffer_reallocation_resyncs_the_paced_copy(ent):
    """Changing display seconds / sample rate / FFT size replaces the
    buffers; the paced copy must follow rather than render a stale
    shape."""
    ent.advance_display(now=0.0)
    ent.view('amp_buffer')
    ent.change_display_seconds(5.0)
    v = ent.view('amp_buffer')
    assert v.shape == ent.amp_buffer.shape
    ent.publish_display()                    # must not raise
    assert ent.view('spec_buffer').shape == ent.spec_buffer.shape


def test_gap_larger_than_the_window_falls_back_to_a_full_copy(ent):
    ent.advance_display(now=0.0)
    ent.view('amp_buffer')
    ent.amp_buffer[:] = 7.0
    ent._samples_total = 3 * ent._total_samples
    ent._disp_abs = float(ent._samples_total)
    ent.publish_display()
    np.testing.assert_array_equal(ent.view('amp_buffer'), 7.0)


def test_publish_after_a_counter_reset_does_not_copy_backwards(ent):
    ent.advance_display(now=0.0)
    ent.view('amp_buffer')
    ent._view_pos = 10 * CHUNK_FRAMES
    ent._disp_abs = 0.0
    ent.publish_display()                    # must not raise or copy
    assert ent._view_pos == 0
    assert ent._view_col_pos == 0
