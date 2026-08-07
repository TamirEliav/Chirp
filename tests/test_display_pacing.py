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
* rendering never modifies the entity's buffers.

Nothing is hidden between the paced cursor and the write head: those
samples are already drawn at their correct position on the wrapping
window, simply revealed before the cursor reaches them. Blanking them
was tried and removed — the gap's far edge is the write head, which
still jumps by a whole device buffer, so the blank strip pulsed at the
burst rate and was more distracting than the early reveal.
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
