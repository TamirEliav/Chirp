"""Tests for safe-teardown flush semantics (#45).

Five teardown paths used to drop in-flight trigger events on the floor:

  1. ``stop_acq``             — joined the ingest thread but never flushed
  2. ``change_device``        — closed the capture mid-event
  3. ``change_sample_rate``   — closed the capture mid-event
  4. ``use_wav_file``         — closed the capture mid-event
  5. ``close``                — destroyed the entity mid-event

Every one of those now funnels through ``_stop_ingest_and_flush``, which
(a) joins the ingest thread so no concurrent mutation of
``_active_events`` races the flush, (b) drains queue stragglers, and
(c) calls ``recorder.flush_all`` with the entity's current output-dir
semantics (ref_date day-subfolder included).

We exercise the recorder directly — the ingest thread isn't running in
this test because the entity never calls ``start_acq``. That's
deliberate: it isolates the flush contract from the sounddevice /
device-ID side of the pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

from chirp.constants import CHUNK_FRAMES, SAMPLE_RATE
from chirp.recording.entity import RecordingEntity
from chirp.recording.trigger import ThresholdRecorder


# ── Fixture: capture flushes in memory ───────────────────────────────

@pytest.fixture
def captured_flushes(monkeypatch):
    """Replace ``ThresholdRecorder._start_flush`` with an in-memory
    capture so nothing hits disk. Mirrors the pattern used in
    test_trigger.py."""
    flushes: list[dict] = []

    def _capture(buf_snapshot, output_dir, prefix='', suffix='',
                 sample_rate=SAMPLE_RATE, onset_time=None,
                 filename_stream=''):
        flushes.append({
            'audio':           np.concatenate(list(buf_snapshot)),
            'output_dir':      output_dir,
            'prefix':          prefix,
            'suffix':          suffix,
            'sample_rate':     sample_rate,
            'filename_stream': filename_stream,
        })

    monkeypatch.setattr(
        ThresholdRecorder, '_start_flush', staticmethod(_capture),
    )
    return flushes


# ── Helpers ──────────────────────────────────────────────────────────

def _loud_chunk(level: float = 0.5) -> np.ndarray:
    return np.full(CHUNK_FRAMES, level, dtype=np.float32)


def _drive_trigger_to_open(e: RecordingEntity, n_loud: int = 3) -> None:
    """Feed ``n_loud`` loud chunks directly to the recorder so an
    event opens and stays open — no ingest thread, no capture. The
    trigger state machine keeps the event mid-recording because we
    never send silence long enough to close it.
    """
    e.rec_enabled = True
    for i in range(n_loud):
        e.recorder.process_chunk(
            _loud_chunk(),
            trigger_peak     = 0.5,
            threshold        = e.threshold,
            min_cross_sec    = e.min_cross_sec,
            hold_sec         = e.hold_sec,
            post_trig_sec    = e.post_trig_sec,
            max_rec_sec      = e.max_rec_sec,
            pre_trig_sec     = e.pre_trig_sec,
            output_dir       = e.output_dir,
            enabled          = True,
            filename_prefix  = e.filename_prefix,
            filename_suffix  = e.filename_suffix,
            sample_rate      = e.sample_rate,
            filename_stream  = e.name,
            global_chunk_end = (i + 1) * CHUNK_FRAMES,
        )


# ── _flush_active_events helper semantics ────────────────────────────

def test_flush_active_events_returns_count_and_clears_state(captured_flushes):
    e = RecordingEntity(name='t', device_id=None)
    _drive_trigger_to_open(e, n_loud=3)
    assert e.recorder.is_recording is True

    n = e._flush_active_events(reason='unit-test')

    assert n == 1
    assert len(captured_flushes) == 1
    assert captured_flushes[0]['filename_stream'] == 't'
    assert captured_flushes[0]['sample_rate'] == e.sample_rate
    # The recorder's internal state must be clean.
    assert e.recorder._active_events == []
    assert e.recorder.is_recording is False


def test_flush_active_events_is_zero_when_nothing_open(captured_flushes):
    e = RecordingEntity(name='t', device_id=None)
    n = e._flush_active_events(reason='noop')
    assert n == 0
    assert captured_flushes == []


def test_flush_active_events_uses_ref_date_subfolder(captured_flushes, tmp_path):
    """When ref_date is set, flushes should land in the same
    day-count subfolder that ingest_chunk would use — otherwise a
    teardown flush ends up in a different directory from the
    recorded-through-ingest events for the same session."""
    import datetime
    e = RecordingEntity(name='t', device_id=None)
    e.output_dir = str(tmp_path)
    e.ref_date = datetime.date.today() - datetime.timedelta(days=5)
    e.dph_folder_prefix = 'dph'
    _drive_trigger_to_open(e)

    e._flush_active_events(reason='unit-test')

    assert len(captured_flushes) == 1
    # e.g. <tmp>/dph5
    assert captured_flushes[0]['output_dir'].endswith('dph5')


# ── stop_acq flushes ─────────────────────────────────────────────────

def test_stop_acq_flushes_in_flight_event(captured_flushes):
    """The headline bug from #45: Stop Acq while a tone is still
    playing discarded the captured samples. Now it must flush."""
    e = RecordingEntity(name='t', device_id=None)
    _drive_trigger_to_open(e, n_loud=3)
    # Simulate the running state: stop_acq only flushes when
    # ``acq_running`` is True. The ingest thread isn't alive here —
    # ``_stop_ingest_and_flush`` handles that case (``_ingest_thread
    # is None``).
    e.acq_running = True

    e.stop_acq()

    assert e.acq_running is False
    assert e.rec_enabled is False
    assert len(captured_flushes) == 1
    assert captured_flushes[0]['filename_stream'] == 't'


def test_stop_acq_when_idle_is_still_noop(captured_flushes):
    """stop_acq guards on ``self.acq_running`` — if acquisition is
    already stopped, nothing changes (including no flush)."""
    e = RecordingEntity(name='t', device_id=None)
    _drive_trigger_to_open(e, n_loud=3)
    # Note: acq_running is False by default on a fresh entity.
    assert e.acq_running is False

    e.stop_acq()

    # Event still open — stop_acq did not run its body.
    assert e.recorder.is_recording is True
    assert captured_flushes == []


# ── change_device flushes ────────────────────────────────────────────

def test_change_device_flushes_in_flight_event(captured_flushes, monkeypatch):
    """Swapping the input device used to close the capture mid-event
    and silently lose samples. Must now flush first."""
    e = RecordingEntity(name='t', device_id=None)
    _drive_trigger_to_open(e)

    # Avoid touching the real sounddevice layer — stub the factory.
    from chirp.audio.capture import AudioCapture as _RealCap
    class _FakeCap:
        valid = True
        def __init__(self, *a, **kw): pass
        def close(self): pass
        def pause(self): pass
        def resume(self): pass
        def consume_drop_count(self): return 0
        def set_monitor(self, *a, **kw): pass
    monkeypatch.setattr(e, '_make_capture', lambda channels=1: _FakeCap())
    # Don't re-spin the ingest thread in tests — it would race the
    # flush capture (and we're stubbing the capture anyway).
    monkeypatch.setattr(e, 'start_acq', lambda: None)

    e.change_device(device_id=3, channels=1)

    assert len(captured_flushes) == 1


# ── change_sample_rate flushes ───────────────────────────────────────

def test_change_sample_rate_flushes_at_old_rate(captured_flushes, monkeypatch):
    """The flushed event must be tagged with the OLD sample rate —
    reconstructing it at the new rate would render a file that plays
    back at the wrong speed."""
    e = RecordingEntity(name='t', device_id=None, sample_rate=44100)
    _drive_trigger_to_open(e)

    class _FakeCap:
        valid = True
        def __init__(self, *a, **kw): pass
        def close(self): pass
        def pause(self): pass
        def resume(self): pass
        def consume_drop_count(self): return 0
        def set_monitor(self, *a, **kw): pass
    monkeypatch.setattr(e, '_make_capture', lambda channels=1: _FakeCap())
    monkeypatch.setattr(e, 'start_acq', lambda: None)

    e.change_sample_rate(48000)

    assert len(captured_flushes) == 1
    # Flush happened BEFORE the SR was swapped — the saved file must
    # be labelled 44100 or it plays back wrong.
    assert captured_flushes[0]['sample_rate'] == 44100
    # And the entity is now at the new rate.
    assert e.sample_rate == 48000


def test_change_sample_rate_noop_when_same_rate(captured_flushes):
    """Passing the current rate is a no-op — no flush should fire."""
    e = RecordingEntity(name='t', device_id=None, sample_rate=44100)
    _drive_trigger_to_open(e)

    e.change_sample_rate(44100)

    # Event still open, no flush.
    assert e.recorder.is_recording is True
    assert captured_flushes == []


# ── use_wav_file flushes ─────────────────────────────────────────────

def test_use_wav_file_flushes_in_flight_event(captured_flushes, monkeypatch, tmp_path):
    """Swapping from live capture to WAV-file replay must not discard
    a recording that's currently open."""
    e = RecordingEntity(name='t', device_id=None)
    _drive_trigger_to_open(e)

    class _FakeWavCap:
        valid = True
        file_sample_rate = e.sample_rate  # same rate → no SR rebuild path
        def __init__(self, *a, **kw): pass
        def close(self): pass
        def pause(self): pass
        def resume(self): pass
        def consume_drop_count(self): return 0
        def set_monitor(self, *a, **kw): pass
    monkeypatch.setattr(e, '_make_capture', lambda channels=1: _FakeWavCap())
    monkeypatch.setattr(e, 'start_acq', lambda: None)

    ok, warning = e.use_wav_file(str(tmp_path / 'not-real.wav'))

    assert ok is True
    assert len(captured_flushes) == 1


# ── close flushes ────────────────────────────────────────────────────

def test_close_flushes_in_flight_event(captured_flushes):
    """Removing a stream (or app shutdown via closeEvent) must flush
    any event that's still mid-recording."""
    e = RecordingEntity(name='t', device_id=None)
    _drive_trigger_to_open(e)

    e.close()

    assert len(captured_flushes) == 1


def test_close_is_safe_when_nothing_active(captured_flushes):
    """Close on a quiet entity is just a capture teardown — no flush."""
    e = RecordingEntity(name='t', device_id=None)
    e.close()
    assert captured_flushes == []


# ── _stop_ingest_and_flush idempotence ───────────────────────────────

def test_stop_ingest_and_flush_is_idempotent(captured_flushes):
    """Teardown paths sometimes chain (e.g. ``use_wav_file`` → SR
    change). Calling the helper twice must not re-flush events that
    were already flushed on the first call."""
    e = RecordingEntity(name='t', device_id=None)
    _drive_trigger_to_open(e)

    e._stop_ingest_and_flush(reason='first')
    e._stop_ingest_and_flush(reason='second')

    assert len(captured_flushes) == 1
