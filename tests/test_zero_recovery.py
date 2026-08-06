"""Inserted-silence auto-recovery.

A capture session can latch into a state where the driver/engine
periodically zero-fills the audio. The only reliable reset is stopping
acquisition on EVERY stream holding that endpoint and starting again —
stopping one is not enough, because the OS keeps the session alive for
the remaining client. An unattended overnight episode ran 96 minutes.

These tests pin the watchdog's decision logic and the reset itself
without building a full ChirpWindow (which is known to segfault at
pytest teardown — see test_pg_window_integration): the methods are
called unbound against a lightweight stub carrying the same attributes.
"""

import time

import pytest

pytest.importorskip('pyqtgraph')

from chirp.config.schema import DEFAULT_AUDIO          # noqa: E402
from chirp.ui.window import ChirpWindow                # noqa: E402

# Module-level so the QApplication outlives every widget built below.
# Letting it be collected while widgets are still alive crashes the
# interpreter at teardown when this file runs on its own.
_APP = None


def _qt_app():
    global _APP
    from PyQt5.QtWidgets import QApplication
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP


class _Ent:
    def __init__(self, name, frac=0.0, device_id=1, sample_rate=44100,
                 acq=True, source='device'):
        self.name = name
        self.zero_sample_frac = frac
        self.device_id = device_id
        self.sample_rate = sample_rate
        self.acq_running = acq
        self.input_source = source
        self.rec_enabled = False
        self.calls = []

    def stop_acq(self):
        self.calls.append('stop')
        self.acq_running = False
        self.rec_enabled = False

    def start_acq(self):
        self.calls.append('start')
        self.acq_running = True


class _Win:
    """Minimal stand-in carrying only what the watchdog touches."""

    def __init__(self, entities, **cfg):
        self._entities = entities
        self._audio_cfg = dict(DEFAULT_AUDIO)
        self._audio_cfg.update(cfg)
        self._zero_high_since = {}
        self._zero_recover_last = {}
        self._zero_recover_thread = None
        self.zero_recovery_count = 0
        self.fired = []

    # Capture the decision instead of spawning the worker thread.
    def _start_zero_recovery(self, group, trigger):
        self.fired.append(([g.name for g in group], trigger.name))

    tick = ChirpWindow._zero_recovery_tick
    worker = ChirpWindow._zero_recovery_worker


def _age(win, ent, seconds):
    """Pretend this entity's duty cycle has been high for `seconds`."""
    win._zero_high_since[id(ent)] = time.monotonic() - seconds


# ── Triggering ───────────────────────────────────────────────────────────

def test_no_trigger_below_threshold():
    e = _Ent('a', frac=0.02)                    # 2% < 5% default
    w = _Win([e])
    w.tick()
    _age(w, e, 60)
    w.tick()
    assert w.fired == []


def test_no_trigger_until_sustained():
    e = _Ent('a', frac=0.20)
    w = _Win([e], zero_recover_seconds=15.0)
    w.tick()                                    # starts the clock
    assert w.fired == []
    _age(w, e, 14)
    w.tick()
    assert w.fired == [], 'fired before the hold time elapsed'
    _age(w, e, 16)
    w.tick()
    assert len(w.fired) == 1


def test_recovery_covers_every_stream_on_the_device():
    """The whole point: one stream's reset does not release the endpoint."""
    a = _Ent('a', frac=0.20, device_id=7)
    b = _Ent('b', frac=0.00, device_id=7)       # sibling, looks clean
    other = _Ent('c', frac=0.00, device_id=9)   # different endpoint
    w = _Win([a, b, other])
    w.tick()
    _age(w, a, 60)
    w.tick()
    assert len(w.fired) == 1
    group, trigger = w.fired[0]
    assert trigger == 'a'
    assert sorted(group) == ['a', 'b']          # sibling included
    assert 'c' not in group                     # other endpoint untouched


def test_clean_window_resets_the_timer():
    e = _Ent('a', frac=0.20)
    w = _Win([e])
    w.tick()
    _age(w, e, 14)
    e.zero_sample_frac = 0.0                    # fault stopped
    w.tick()
    assert id(e) not in w._zero_high_since
    e.zero_sample_frac = 0.20                   # and comes back
    w.tick()
    assert w.fired == [], 'timer must restart, not resume'


def test_disabled_setting_never_fires():
    e = _Ent('a', frac=0.90)
    w = _Win([e], auto_recover_zero_runs=False)
    w.tick()
    _age(w, e, 600)
    w.tick()
    assert w.fired == []


def test_cooldown_blocks_a_restart_loop():
    e = _Ent('a', frac=0.50)
    w = _Win([e], zero_recover_cooldown_sec=120.0)
    w.tick()
    _age(w, e, 60)
    w.tick()
    assert len(w.fired) == 1
    e.zero_sample_frac = 0.50                   # fault persists
    w.tick()
    _age(w, e, 60)
    w.tick()
    assert len(w.fired) == 1, 'cooldown must suppress the second attempt'


def test_stopped_or_wav_streams_are_ignored():
    stopped = _Ent('stopped', frac=0.9, acq=False)
    wav = _Ent('wav', frac=0.9, source='wav_file')
    w = _Win([stopped, wav])
    w.tick()
    _age(w, stopped, 60)
    _age(w, wav, 60)
    w.tick()
    assert w.fired == []


def test_no_second_round_while_one_is_running():
    class _Alive:
        def is_alive(self):
            return True

    e = _Ent('a', frac=0.50)
    w = _Win([e])
    w._zero_recover_thread = _Alive()
    _age(w, e, 60)
    w.tick()
    assert w.fired == []


# ── The reset itself ─────────────────────────────────────────────────────

def test_worker_stops_all_before_starting_any():
    """If a stream restarted before its sibling stopped, the OS session
    would never be released and the fault would survive the reset."""
    a = _Ent('a')
    b = _Ent('b')
    order = []
    for e in (a, b):
        e.stop_acq = (lambda n=e.name, o=order: o.append(('stop', n)))
        e.start_acq = (lambda n=e.name, o=order: o.append(('start', n)))
    w = _Win([a, b])
    w.worker([a, b])
    kinds = [k for k, _n in order]
    assert kinds == ['stop', 'stop', 'start', 'start'], order


def test_worker_preserves_recording_state():
    a = _Ent('a')
    a.rec_enabled = True
    b = _Ent('b')
    b.rec_enabled = False
    w = _Win([a, b])
    w.worker([a, b])
    assert a.calls == ['stop', 'start']
    assert a.rec_enabled is True, 'recording must resume after the reset'
    assert b.rec_enabled is False


def test_worker_survives_a_failing_stream():
    bad = _Ent('bad')

    def _boom():
        raise RuntimeError('device gone')

    bad.start_acq = _boom
    good = _Ent('good')
    w = _Win([bad, good])
    w.worker([bad, good])                       # must not raise
    assert good.calls == ['stop', 'start']


# ── Advanced Settings dialog ─────────────────────────────────────────────

def test_advanced_dialog_builds_and_applies(monkeypatch):
    """The dialog is the only way most users will reach these settings,
    so exercise the real build + apply path (not a full ChirpWindow —
    that segfaults at teardown; a bare QWidget is enough of a parent)."""
    from PyQt5.QtWidgets import QDialog, QWidget

    import chirp.audio.shared_stream as shared

    _qt_app()

    class _Host(QWidget):
        def __init__(self):
            super().__init__()
            self._audio_cfg = dict(DEFAULT_AUDIO)
            self._zero_high_since = {'stale': 1.0}
            self.dirty = 0

        def _mark_dirty(self):
            self.dirty += 1

        open_advanced = ChirpWindow._open_advanced_settings

    before = shared.current_params()
    host = _Host()
    try:
        shared.configure(blocksize=4096, latency=0.25)
        monkeypatch.setattr(QDialog, 'exec_', lambda self: QDialog.Accepted)
        host.open_advanced()
        # Values round-trip out of the widgets unchanged.
        bs, lat = shared.current_params()
        assert bs == 4096
        assert lat == 0.25
        assert host._audio_cfg['auto_recover_zero_runs'] is True
        assert host._audio_cfg['zero_recover_percent'] == pytest.approx(5.0)
        assert host.dirty == 1
        assert host._zero_high_since == {}, 'stale timers must be cleared'
    finally:
        shared.configure(*before)


def test_advanced_dialog_cancel_changes_nothing(monkeypatch):
    from PyQt5.QtWidgets import QDialog, QWidget

    import chirp.audio.shared_stream as shared

    _qt_app()

    class _Host(QWidget):
        def __init__(self):
            super().__init__()
            self._audio_cfg = dict(DEFAULT_AUDIO)
            self._audio_cfg['zero_recover_percent'] = 42.0
            self._zero_high_since = {}
            self.dirty = 0

        def _mark_dirty(self):
            self.dirty += 1

        open_advanced = ChirpWindow._open_advanced_settings

    before = shared.current_params()
    host = _Host()
    try:
        monkeypatch.setattr(QDialog, 'exec_', lambda self: QDialog.Rejected)
        host.open_advanced()
        assert host._audio_cfg['zero_recover_percent'] == 42.0
        assert host.dirty == 0
        assert shared.current_params() == before
    finally:
        shared.configure(*before)
