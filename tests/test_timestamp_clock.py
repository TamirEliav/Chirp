"""Regression tests for the filename-timestamp sample clock (M5).

WAV filename onsets are derived from a wall-clock/sample-counter anchor
pair stamped at ``start_acq``::

    chunk_end_wall = _wall_anchor_time
                     + (_samples_total - _wall_anchor_samples) / sample_rate

Field bug: ``reset_display()`` and ``change_display_seconds()`` zeroed
``_samples_total`` mid-acquisition WITHOUT re-stamping the anchor, so
every filename timestamp after the reset snapped back to the anchor
time — observed as a consistent ~1-day-backwards jump after clicking
Start Acq (which reset_displays every entity) on a session that had
been running since the previous day.

These tests pin the contract: display-only operations must never move
the timestamp clock, and any operation that legitimately resets the
counter (sample-rate change) must invalidate the wall anchor with it.
"""

import datetime

import numpy as np

from chirp.constants import CHUNK_FRAMES
from chirp.recording.entity import RecordingEntity


ANCHOR = datetime.datetime(2026, 7, 15, 13, 0, 0)


def _entity_with_anchor():
    e = RecordingEntity(name="TsTest", device_id=None)
    e._wall_anchor_time = ANCHOR
    e._wall_anchor_samples = 0
    return e


def _spy_chunk_end_wall(e):
    """Wrap the recorder's process_chunk to capture the chunk_end_wall
    the entity derives, without altering behavior."""
    captured = []
    orig = e.recorder.process_chunk

    def spy(chunk, **kw):
        captured.append(kw.get('chunk_end_wall'))
        return orig(chunk, **kw)

    e.recorder.process_chunk = spy
    return captured


def _expected(k, sr):
    """Capture-time wall clock after k chunks, same math as the entity."""
    return ANCHOR + datetime.timedelta(seconds=(k * CHUNK_FRAMES) / sr)


def _assert_close(a, b, tol_us=100):
    assert abs((a - b).total_seconds()) * 1e6 <= tol_us, f'{a} != {b}'


def test_chunk_end_wall_tracks_sample_clock():
    e = _entity_with_anchor()
    captured = _spy_chunk_end_wall(e)
    chunk = np.zeros(CHUNK_FRAMES, dtype=np.float32)
    for k in range(1, 4):
        e.ingest_chunk(chunk)
        _assert_close(captured[-1], _expected(k, e.sample_rate))


def test_reset_display_does_not_move_timestamp_clock():
    """The field bug: reset_display mid-acquisition must not snap
    chunk_end_wall back to the anchor."""
    e = _entity_with_anchor()
    captured = _spy_chunk_end_wall(e)
    chunk = np.zeros(CHUNK_FRAMES, dtype=np.float32)
    for _ in range(3):
        e.ingest_chunk(chunk)

    e.reset_display()
    assert e._samples_total == 3 * CHUNK_FRAMES

    e.ingest_chunk(chunk)
    _assert_close(captured[-1], _expected(4, e.sample_rate))
    # Monotonic across the reset — no backwards jump.
    assert captured[-1] > captured[-2]


def test_change_display_seconds_does_not_move_timestamp_clock():
    e = _entity_with_anchor()
    captured = _spy_chunk_end_wall(e)
    chunk = np.zeros(CHUNK_FRAMES, dtype=np.float32)
    for _ in range(3):
        e.ingest_chunk(chunk)

    e.change_display_seconds(30.0)
    assert e._samples_total == 3 * CHUNK_FRAMES

    e.ingest_chunk(chunk)
    _assert_close(captured[-1], _expected(4, e.sample_rate))
    assert captured[-1] > captured[-2]


def test_change_display_seconds_rederives_cursors_for_new_geometry():
    """Cursors must match ingest_chunk's modulo derivation against the
    NEW ring size immediately after the change (readers may hit the
    buffers before the next chunk lands)."""
    e = _entity_with_anchor()
    chunk = np.zeros(CHUNK_FRAMES, dtype=np.float32)
    for _ in range(5):
        e.ingest_chunk(chunk)

    e.change_display_seconds(30.0)
    assert e.write_head == e._samples_total % e._total_samples
    assert e.col_head == (e._samples_total // CHUNK_FRAMES) % e._n_cols
    assert e.write_head < e._total_samples
    assert e.col_head < e._n_cols

    # Lockstep continues on the next ingest.
    e.ingest_chunk(chunk)
    assert e.write_head == e._samples_total % e._total_samples
    assert e.col_head == (e._samples_total // CHUNK_FRAMES) % e._n_cols


def test_change_sample_rate_invalidates_wall_anchor():
    """A sample-rate change resets the counter (old-rate counts are
    meaningless at the new rate) — the wall anchor must die with it so
    a stale anchor can never pair with the fresh counter. start_acq
    stamps a new pair before any chunk is ingested."""
    e = _entity_with_anchor()
    chunk = np.zeros(CHUNK_FRAMES, dtype=np.float32)
    for _ in range(3):
        e.ingest_chunk(chunk)

    new_rate = 22050 if e.sample_rate != 22050 else 44100
    e.change_sample_rate(new_rate)
    assert e._wall_anchor_time is None
    assert e._wall_anchor_samples == 0
    e.close()


def test_reset_display_still_clears_buffer_contents():
    """The display-clearing contract of reset_display is unchanged."""
    e = _entity_with_anchor()
    e.ingest_chunk(np.full(CHUNK_FRAMES, 0.5, dtype=np.float32))
    assert e.amp_buffer.any()
    e.reset_display()
    assert not e.amp_buffer.any()
    assert not e.detect_mask_buffer.any()
    assert not e.record_mask_buffer.any()
