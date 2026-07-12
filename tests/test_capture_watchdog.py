"""TODO#1 (RDP): capture-stall watchdog + auto-reconnect.

An RDP connect/disconnect makes Windows tear down / re-route audio
endpoints; the PortAudio stream silently stops delivering frames while
``acq_running`` still claims otherwise. These tests pin:

* stall detection latches after CAPTURE_STALL_SECONDS without a single
  new ring frame (and only for live-device captures);
* recovery flushes in-flight trigger events BEFORE rebuilding, reopens
  the capture, restarts acquisition, and preserves ``rec_enabled``;
* a failed reopen leaves the entity stalled for the next retry round.
"""

import time

import numpy as np
import pytest

from chirp.constants import CHUNK_FRAMES, SAMPLE_RATE
from chirp.recording.entity import RecordingEntity
from chirp.recording.trigger import ThresholdRecorder


class _FakeCapture:
    def __init__(self, valid=True):
        self.valid = valid
        self.resumed = False
        self.closed = False
        self.drop_count = 0
        self.drop_count_total = 0
        self.has_ever_dropped = False
        self.os_drop_count = 0
        self.os_drop_count_total = 0
        self.has_ever_os_dropped = False
        self.open_error = None if valid else 'fake open failure'

    def resume(self):
        self.resumed = True

    def pause(self):
        pass

    def close(self):
        self.closed = True

    def consume_drop_count(self):
        return 0

    def consume_os_drop_count(self):
        return 0

    def set_monitor(self, *a):
        pass


@pytest.fixture
def fast_stall(monkeypatch):
    monkeypatch.setattr(RecordingEntity, 'CAPTURE_STALL_SECONDS', 0.05)


def _make_stalled_entity(monkeypatch) -> RecordingEntity:
    e = RecordingEntity(name='wdt', device_id=None)
    e.capture.close()
    e.capture = _FakeCapture(valid=True)
    e.input_source = 'device'
    e.acq_running = True
    return e


def test_stall_latches_after_timeout(monkeypatch, fast_stall):
    e = _make_stalled_entity(monkeypatch)
    try:
        assert e.check_capture_stalled() is False   # baseline sample
        time.sleep(0.1)
        assert e.check_capture_stalled() is True    # no frames arrived
        assert e.capture_stalled is True
        assert e.has_ever_ingest_errored is True
        assert 'reconnect' in (e.last_ingest_error or '')
    finally:
        e.acq_running = False
        e.close()


def test_no_stall_while_frames_flow(monkeypatch, fast_stall):
    e = _make_stalled_entity(monkeypatch)
    try:
        assert e.check_capture_stalled() is False
        for _ in range(3):
            time.sleep(0.03)
            e.ring.write(np.zeros(CHUNK_FRAMES, dtype=np.float32))
            assert e.check_capture_stalled() is False
    finally:
        e.acq_running = False
        e.close()


def test_wav_source_never_stalls(monkeypatch, fast_stall):
    e = _make_stalled_entity(monkeypatch)
    try:
        e.input_source = 'wav_file'
        assert e.check_capture_stalled() is False
        time.sleep(0.1)
        assert e.check_capture_stalled() is False
    finally:
        e.acq_running = False
        e.close()


def test_recovery_reopens_and_preserves_rec_enabled(monkeypatch, fast_stall):
    e = _make_stalled_entity(monkeypatch)
    try:
        e.rec_enabled = True
        e.check_capture_stalled()
        time.sleep(0.1)
        assert e.check_capture_stalled() is True

        new_cap = _FakeCapture(valid=True)
        monkeypatch.setattr(e, '_make_capture', lambda channels: new_cap)
        assert e.attempt_capture_recovery() is True
        assert e.capture is new_cap
        assert new_cap.resumed is True          # start_acq ran
        assert e.acq_running is True
        assert e.rec_enabled is True            # recording resumes
        assert e.capture_stalled is False
        assert e.recovery_count == 1
    finally:
        e.stop_acq()
        e.close()


def test_failed_recovery_stays_stalled_for_retry(monkeypatch, fast_stall):
    e = _make_stalled_entity(monkeypatch)
    try:
        e.check_capture_stalled()
        time.sleep(0.1)
        assert e.check_capture_stalled() is True

        monkeypatch.setattr(e, '_make_capture',
                            lambda channels: _FakeCapture(valid=False))
        assert e.attempt_capture_recovery() is False
        assert e.capture_stalled is True
        assert e.acq_running is False           # dead pipeline torn down

        # Device comes back on a later round → recovery succeeds.
        good = _FakeCapture(valid=True)
        monkeypatch.setattr(e, '_make_capture', lambda channels: good)
        assert e.attempt_capture_recovery() is True
        assert e.acq_running is True
    finally:
        e.stop_acq()
        e.close()


def test_transient_stall_unlatches_on_resume(monkeypatch, fast_stall):
    """Frames resuming on their own (transient endpoint churn during a
    remote-desktop attach) must cancel the pending recovery — tearing
    down a stream that is healthy again is exactly the regression that
    froze the app under AnyDesk."""
    e = _make_stalled_entity(monkeypatch)
    try:
        assert e.check_capture_stalled() is False
        time.sleep(0.1)
        assert e.check_capture_stalled() is True
        # Frames resume before the recovery worker gets to this entity.
        e.ring.write(np.zeros(CHUNK_FRAMES, dtype=np.float32))
        assert e.check_capture_stalled() is False
        assert e.capture_stalled is False
        # Recovery is now a no-op: nothing torn down, no reconnect.
        assert e.attempt_capture_recovery() is True
        assert e.recovery_count == 0
        assert e.acq_running is True
    finally:
        e.acq_running = False
        e.close()


def test_recovery_flushes_open_events_first(monkeypatch, fast_stall):
    """An event mid-recording when the device dies must land on disk
    (via the flush path) before the pipeline is rebuilt."""
    flushes = []

    def _capture_flush(buf_snapshot, *args, **kwargs):
        flushes.append(np.concatenate(buf_snapshot))

    monkeypatch.setattr(ThresholdRecorder, '_start_flush',
                        staticmethod(_capture_flush))
    e = _make_stalled_entity(monkeypatch)
    try:
        e.rec_enabled = True
        # Drive an event open through the recorder directly.
        loud = np.full(CHUNK_FRAMES, 0.5, dtype=np.float32)
        for i in range(3):
            e.recorder.process_chunk(
                loud, trigger_peak=0.5, threshold=0.1,
                min_cross_sec=0.0, hold_sec=1.0, post_trig_sec=1.0,
                max_rec_sec=60.0, pre_trig_sec=0.0,
                output_dir='.', enabled=True,
                sample_rate=SAMPLE_RATE,
                global_chunk_end=(i + 1) * CHUNK_FRAMES)
        assert e.recorder.is_recording

        e.check_capture_stalled()
        time.sleep(0.1)
        assert e.check_capture_stalled() is True
        monkeypatch.setattr(e, '_make_capture',
                            lambda channels: _FakeCapture(valid=True))
        assert e.attempt_capture_recovery() is True
        assert len(flushes) == 1                 # event hit the flush path
        assert flushes[0].size == 3 * CHUNK_FRAMES
    finally:
        e.stop_acq()
        e.close()
