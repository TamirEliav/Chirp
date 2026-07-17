"""L6 (RDP / WDM-KS): Stop Acq → Start Acq must not resume a paused
live-device stream.

A stopped-but-open stream still holds the device (exclusively so under
WDM-KS), and its kernel state can be invalidated while idle by an RDP
session switch or device power management — ``stream.start()`` on the
stale handle then fails, and before this fix the failure was silent
(no badge, no log, a dead Start Acq button until the user reselected
the device). These tests pin the new contract:

* ``stop_acq`` CLOSES a live-device capture (releasing the device);
  WAV-playback captures stay open (pause preserves position).
* ``start_acq`` opens a fresh capture when the current one is closed,
  and falls back to one close + reopen when ``resume()`` raises.
* a failed (re)open latches ``last_ingest_error`` + the `!` badge
  instead of silently doing nothing.
* callers that just built a valid capture (recovery, device change)
  keep it — no redundant second open.
"""

import numpy as np
import pytest

from chirp.recording.entity import RecordingEntity


class _FakeCapture:
    """Mirrors the AudioCapture contract; close() invalidates like the
    real one (``_stream = None`` → ``valid`` False)."""

    def __init__(self, valid=True):
        self.valid = valid
        self.resumed = False
        self.paused = False
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
        self.paused = True

    def close(self):
        self.closed = True
        self.valid = False

    def consume_drop_count(self):
        return 0

    def consume_os_drop_count(self):
        return 0

    def set_monitor(self, *a):
        pass


class _RefusingCapture(_FakeCapture):
    """A paused stream whose restart fails — the WDM-KS 'stale pin
    after RDP churn' signature."""

    def resume(self):
        raise RuntimeError('KS pin refused to restart')


def _make_device_entity(name='rst') -> RecordingEntity:
    e = RecordingEntity(name=name, device_id=None)
    e.capture.close()
    e.capture = _FakeCapture(valid=True)
    e.input_source = 'device'
    return e


def test_stop_acq_closes_device_capture():
    e = _make_device_entity()
    try:
        cap = e.capture
        e.start_acq()
        assert e.acq_running is True
        assert cap.resumed is True
        assert cap.closed is False      # start kept the fresh capture
        e.stop_acq()
        assert e.acq_running is False
        assert cap.closed is True       # device released while idle
    finally:
        e.close()


def test_start_acq_reopens_after_stop(monkeypatch):
    e = _make_device_entity()
    try:
        e.start_acq()
        e.stop_acq()
        fresh = _FakeCapture(valid=True)
        monkeypatch.setattr(e, '_make_capture', lambda channels: fresh)
        e.start_acq()
        assert e.capture is fresh       # fresh open, not a resume
        assert fresh.resumed is True
        assert e.acq_running is True
    finally:
        e.stop_acq()
        e.close()


def test_wav_capture_survives_stop(tmp_path):
    """WAV playback keeps pause/resume semantics — closing would lose
    the playback position."""
    import scipy.io.wavfile as wavfile
    from chirp.constants import SAMPLE_RATE
    wav = tmp_path / 'tone.wav'
    wavfile.write(str(wav), SAMPLE_RATE,
                  np.zeros(SAMPLE_RATE // 10, dtype=np.int16))
    e = RecordingEntity(name='rst-wav', device_id=None)
    try:
        e.capture.close()
        ok, _warn = e.use_wav_file(str(wav), loop=True)
        assert ok
        cap = e.capture
        e.start_acq()
        assert e.acq_running is True
        e.stop_acq()
        assert e.capture is cap         # same capture object kept open
        e.start_acq()
        assert e.acq_running is True
        assert e.capture is cap
    finally:
        e.stop_acq()
        e.close()


def test_resume_failure_falls_back_to_reopen(monkeypatch):
    e = _make_device_entity()
    try:
        stale = _RefusingCapture(valid=True)
        e.capture = stale
        fresh = _FakeCapture(valid=True)
        monkeypatch.setattr(e, '_make_capture', lambda channels: fresh)
        e.start_acq()
        assert stale.closed is True     # stale stream torn down
        assert e.capture is fresh
        assert fresh.resumed is True
        assert e.acq_running is True
    finally:
        e.stop_acq()
        e.close()


def test_failed_reopen_latches_error(monkeypatch):
    e = _make_device_entity()
    try:
        e.start_acq()
        e.stop_acq()
        monkeypatch.setattr(e, '_make_capture',
                            lambda channels: _FakeCapture(valid=False))
        e.start_acq()
        assert e.acq_running is False
        assert e.has_ever_ingest_errored is True
        assert 'could not start acquisition' in (e.last_ingest_error or '')
    finally:
        e.close()


def test_recovery_capture_not_reopened_twice(monkeypatch):
    """attempt_capture_recovery builds the new capture itself, then
    calls start_acq — start_acq must keep that valid capture instead
    of closing it and opening a second one."""
    monkeypatch.setattr(RecordingEntity, 'CAPTURE_STALL_SECONDS', 0.0)
    e = _make_device_entity()
    try:
        e.acq_running = True
        e.check_capture_stalled()       # baseline
        e.check_capture_stalled()       # 0 s threshold → stalled
        assert e.capture_stalled is True
        opened = []

        def _make(channels):
            cap = _FakeCapture(valid=True)
            opened.append(cap)
            return cap

        monkeypatch.setattr(e, '_make_capture', _make)
        assert e.attempt_capture_recovery() is True
        assert len(opened) == 1         # exactly one open for the round
        assert e.capture is opened[0]
        assert e.acq_running is True
    finally:
        e.stop_acq()
        e.close()
