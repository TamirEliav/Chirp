"""Tests for resilient closeEvent teardown (#56).

``ChirpWindow.closeEvent`` used to run ``e.stop_acq()``, ``e.close()``
and the writer drain as bare calls. One exception anywhere in the
chain would skip every remaining step — ingest threads orphaned,
in-flight recordings lost, queued WAVs abandoned. The user saw the
app close "cleanly" with zero warning.

These tests inject a failure on entity #0 and pin the post-teardown
state: entities #1 and #2 still had their ``stop_acq`` + ``close`` +
``flush_all`` called, the writer pool still shut down, and a warning
modal was shown.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch, MagicMock

import pytest
from PyQt5.QtWidgets import QApplication, QMainWindow


@contextmanager
def _patched_close():
    """Patch super().closeEvent so we don't need a real QMainWindow
    __init__. Also patch QMessageBox + QApplication so the modal
    code path is observable + non-blocking."""
    with patch.object(QMainWindow, 'closeEvent', lambda self, ev: None), \
         patch('chirp.ui.window.QApplication'):
        yield


@pytest.fixture(scope='module')
def qapp():
    app = QApplication.instance() or QApplication([])
    return app


def _make_window_with_entities(n: int):
    """Build a ``ChirpWindow`` via ``__new__`` + stubs so closeEvent
    can run without a full UI / sounddevice stack."""
    from chirp.ui.window import ChirpWindow

    win = ChirpWindow.__new__(ChirpWindow)
    # Stub Qt-side attrs touched by closeEvent.
    win._timer = MagicMock()
    win._monitor = MagicMock()

    # Build n fake entities. closeEvent reads: acq_running, rec_enabled,
    # name, output_dir, filename_prefix/suffix, sample_rate, recorder,
    # and calls stop_acq() + close().
    entities = []
    call_log: list[tuple[str, str]] = []
    for i in range(n):
        e = MagicMock()
        e.name            = f'ent{i}'
        e.acq_running     = False
        e.rec_enabled     = False
        e.output_dir      = '/tmp'
        e.filename_prefix = 'p'
        e.filename_suffix = ''
        e.sample_rate     = 44100
        # Wire the mock methods to record call order.
        def _mk(name, idx):
            def _inner(*a, **kw):
                call_log.append((name, f'ent{idx}'))
            return _inner
        e.stop_acq = _mk('stop_acq', i)
        e.close    = _mk('close',    i)
        e.recorder = MagicMock()
        e.recorder.flush_all = _mk('flush_all', i)
        entities.append(e)
    win._entities = entities
    return win, call_log


# ── Happy path ───────────────────────────────────────────────────────

def test_close_event_calls_all_entity_teardown_steps(qapp):
    """Sanity: with no injected failures, every entity's stop_acq /
    flush_all / close must be called exactly once."""
    from PyQt5.QtGui import QCloseEvent

    win, log = _make_window_with_entities(3)
    with patch('chirp.ui.window.QMessageBox'), _patched_close():
        win.closeEvent(QCloseEvent())

    stop_acqs = [l for l in log if l[0] == 'stop_acq']
    flushes   = [l for l in log if l[0] == 'flush_all']
    closes    = [l for l in log if l[0] == 'close']
    assert len(stop_acqs) == 3
    assert len(flushes)   == 3
    assert len(closes)    == 3


# ── stop_acq failure does not skip subsequent entities ──────────────

def test_close_event_stop_acq_failure_does_not_skip(qapp):
    """If entity #0.stop_acq raises, entities #1 and #2 must still
    have their stop_acq called — and their flush_all + close must
    still run as well."""
    from PyQt5.QtGui import QCloseEvent

    win, log = _make_window_with_entities(3)
    # Poison entity #0's stop_acq.
    def _bad_stop(): raise RuntimeError('simulated stop_acq failure')
    win._entities[0].stop_acq = _bad_stop

    with patch('chirp.ui.window.QMessageBox') as _mb, _patched_close():
        win.closeEvent(QCloseEvent())

    # Entities #1 and #2 had their stop_acq called.
    stop_acq_entities = {l[1] for l in log if l[0] == 'stop_acq'}
    assert 'ent1' in stop_acq_entities
    assert 'ent2' in stop_acq_entities
    # All 3 entities still had flush_all and close called — the
    # teardown loop did not abort on entity #0's failure.
    flush_entities = {l[1] for l in log if l[0] == 'flush_all'}
    close_entities = {l[1] for l in log if l[0] == 'close'}
    assert flush_entities == {'ent0', 'ent1', 'ent2'}
    assert close_entities == {'ent0', 'ent1', 'ent2'}
    # And a warning modal was shown summarizing the failure.
    _mb.warning.assert_called()


# ── close() failure on early entity does not skip later close()s ────

def test_close_event_close_failure_does_not_skip(qapp):
    """If entity #0.close raises, entities #1 and #2 must still have
    ``close`` called. Without the fix, one exception ate the rest."""
    from PyQt5.QtGui import QCloseEvent

    win, log = _make_window_with_entities(3)
    def _bad_close(): raise RuntimeError('simulated close failure')
    win._entities[0].close = _bad_close

    with patch('chirp.ui.window.QMessageBox') as _mb, _patched_close():
        win.closeEvent(QCloseEvent())

    close_entities = {l[1] for l in log if l[0] == 'close'}
    assert 'ent1' in close_entities
    assert 'ent2' in close_entities
    _mb.warning.assert_called()


# ── flush_all failure on one entity does not skip the rest ──────────

def test_close_event_flush_failure_does_not_skip(qapp):
    """flush_all already had its own try/except — make sure the
    refactor didn't regress: one bad flush must not skip the others."""
    from PyQt5.QtGui import QCloseEvent

    win, log = _make_window_with_entities(3)
    def _bad_flush(*a, **kw): raise RuntimeError('simulated flush failure')
    win._entities[1].recorder.flush_all = _bad_flush

    with patch('chirp.ui.window.QMessageBox'), _patched_close():
        win.closeEvent(QCloseEvent())

    flush_entities = {l[1] for l in log if l[0] == 'flush_all'}
    # ent1 failed, but ent0 and ent2 still flushed.
    assert 'ent0' in flush_entities
    assert 'ent2' in flush_entities


# ── writer.drain failure does not skip shutdown + close ─────────────

def test_close_event_writer_drain_failure_does_not_skip(qapp, monkeypatch):
    """A broken writer.drain must not skip writer.shutdown, monitor
    teardown, or per-entity close()."""
    from PyQt5.QtGui import QCloseEvent
    from chirp.recording import writer as _writer

    win, log = _make_window_with_entities(2)
    # Force pending() > 0 so the modal+drain branch fires.
    monkeypatch.setattr(_writer, 'pending', lambda: 3)
    def _bad_drain(timeout=None): raise RuntimeError('simulated drain failure')
    monkeypatch.setattr(_writer, 'drain', _bad_drain)

    shutdown_called = {'n': 0}
    def _count_shutdown(timeout=None): shutdown_called['n'] += 1
    monkeypatch.setattr(_writer, 'shutdown', _count_shutdown)

    with patch('chirp.ui.window.QMessageBox') as _mb, _patched_close():
        win.closeEvent(QCloseEvent())

    # Shutdown still called despite the drain failure.
    assert shutdown_called['n'] == 1
    # Monitor still closed.
    win._monitor.close.assert_called()
    # Both entities still had close().
    close_entities = {l[1] for l in log if l[0] == 'close'}
    assert close_entities == {'ent0', 'ent1'}
    # User was warned.
    _mb.warning.assert_called()


# ── Drain timeout surfaces as a warning ──────────────────────────────

def test_close_event_drain_timeout_is_surfaced(qapp, monkeypatch):
    """When drain returns False (timed out), the user must be warned
    that queued recordings are about to be abandoned — the pre-fix
    behaviour discarded them silently."""
    from PyQt5.QtGui import QCloseEvent
    from chirp.recording import writer as _writer

    win, _log = _make_window_with_entities(1)
    # Simulate a backlog + a drain that times out without finishing.
    pending_sequence = iter([5, 2, 2])
    monkeypatch.setattr(_writer, 'pending', lambda: next(pending_sequence))
    monkeypatch.setattr(_writer, 'drain', lambda timeout=None: False)
    monkeypatch.setattr(_writer, 'shutdown', lambda timeout=None: None)

    with patch('chirp.ui.window.QMessageBox') as _mb, _patched_close():
        win.closeEvent(QCloseEvent())

    # A warning modal was shown.
    _mb.warning.assert_called()
    # The warning text mentions the remaining count.
    args, kwargs = _mb.warning.call_args
    combined = ' '.join(str(a) for a in args) + ' ' + ' '.join(
        str(v) for v in kwargs.values())
    assert 'timed out' in combined.lower() or 'lost' in combined.lower()
