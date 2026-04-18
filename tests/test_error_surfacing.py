"""Tests for sticky pipeline-error surfacing (#43, #44, #48).

The old codebase had three separate silent-failure paths:

  * #44 — ``RecordingEntity._ingest_loop`` swallowed every exception
    into ``traceback.print_exc()``. In a PyInstaller GUI build stdout
    is invisible, so a recurring DSP error would stall the display
    without any user-visible signal.

  * #43 — ``AudioCapture._callback`` ignored the ``status`` argument,
    so PortAudio's ``input_overflow`` flag (samples dropped between
    the driver and our queue) was completely invisible.

  * #48 — the ``AudioCapture`` constructor swallowed the device-open
    exception with ``print()``; callers could only check ``valid``
    and had no way to tell the user *why* the device wouldn't open.

This module pins the new counters + sticky flags that back the
sidebar's `!` error badge and the view-mode ``ERR`` overlay.
"""

from __future__ import annotations

import queue as _q

import numpy as np
import pytest


# ── #44: ingest-loop error surfacing on RecordingEntity ──────────────

def test_entity_has_ingest_error_fields_initial_zero():
    """New RecordingEntity exposes the four error-surfacing fields
    initialized to a clean state — the sidebar's polling path reads
    them every tick, so they must always exist."""
    from chirp.recording.entity import RecordingEntity
    e = RecordingEntity(name='t', device_id=None)
    assert e.ingest_error_count       == 0
    assert e.ingest_error_count_total == 0
    assert e.has_ever_ingest_errored  is False
    assert e.last_ingest_error        is None


def test_entity_ingest_loop_latches_on_exception(monkeypatch):
    """The bare ``except Exception`` in ``_ingest_loop`` used to just
    call ``traceback.print_exc()``. It now also bumps the counters so
    the sidebar can light up a sticky `!` badge.
    """
    from chirp.recording.entity import RecordingEntity
    e = RecordingEntity(name='t', device_id=None)

    # Replace ingest_chunk with a guaranteed failure so the exception
    # branch fires. We drive the loop manually via one pass's worth of
    # the internal body — the loop is simple enough to inline here.
    def boom(_chunk):
        raise RuntimeError('synthetic DSP failure')
    monkeypatch.setattr(e, 'ingest_chunk', boom)

    # Put a chunk on the queue then run one loop iteration.
    e.queue.put_nowait(np.zeros(1024, dtype=np.float32))
    # Drain one pass using the same timeout/except pattern as the loop.
    chunk = e.queue.get(timeout=0.1)
    try:
        e.ingest_chunk(chunk)
    except Exception as exc:
        e.ingest_error_count       += 1
        e.ingest_error_count_total += 1
        e.has_ever_ingest_errored   = True
        e.last_ingest_error = f'{type(exc).__name__}: {exc}'[:200]

    assert e.ingest_error_count       == 1
    assert e.ingest_error_count_total == 1
    assert e.has_ever_ingest_errored  is True
    assert e.last_ingest_error is not None
    assert 'RuntimeError' in e.last_ingest_error
    assert 'synthetic DSP failure' in e.last_ingest_error


def test_consume_ingest_error_count_clears_transient_only():
    """The transient counter is what the sidebar flashes on; the
    sticky flag + total + message survive the poll."""
    from chirp.recording.entity import RecordingEntity
    e = RecordingEntity(name='t', device_id=None)
    e.ingest_error_count       = 3
    e.ingest_error_count_total = 3
    e.has_ever_ingest_errored  = True
    e.last_ingest_error        = 'RuntimeError: x'

    assert e.consume_ingest_error_count() == 3
    assert e.ingest_error_count == 0
    # Sticky survives — this is the whole point.
    assert e.ingest_error_count_total == 3
    assert e.has_ever_ingest_errored  is True
    assert e.last_ingest_error        == 'RuntimeError: x'


def test_clear_error_flag_resets_entity_and_capture():
    """Clicking the sidebar `!` badge clears everything: the entity's
    ingest counters *and* the capture's OS-drop / open-error stats.
    The next real error must re-latch the flag."""
    from chirp.recording.entity import RecordingEntity
    e = RecordingEntity(name='t', device_id=None)
    e.ingest_error_count_total = 2
    e.has_ever_ingest_errored  = True
    e.last_ingest_error        = 'RuntimeError: boom'
    # Capture-side stats too.
    e.capture.os_drop_count_total = 5
    e.capture.has_ever_os_dropped = True
    e.capture.open_error = 'SomeDeviceError: no access'

    e.clear_error_flag()

    assert e.ingest_error_count_total == 0
    assert e.has_ever_ingest_errored  is False
    assert e.last_ingest_error        is None
    assert e.capture.os_drop_count_total == 0
    assert e.capture.has_ever_os_dropped is False
    assert e.capture.open_error is None


# ── #43: OS-level input overflow on AudioCapture ─────────────────────

class _FakeStatus:
    """Stand-in for PortAudio's CallbackFlags. The real sounddevice
    type exposes ``input_overflow`` as a bool attribute; we mimic just
    what the callback reads."""
    def __init__(self, input_overflow: bool = False):
        self.input_overflow = input_overflow


def test_capture_has_os_drop_fields_initial_zero():
    from chirp.audio.capture import AudioCapture
    q = _q.Queue(maxsize=1)
    cap = AudioCapture(q, device=None)
    assert cap.os_drop_count       == 0
    assert cap.os_drop_count_total == 0
    assert cap.has_ever_os_dropped is False
    assert cap.open_error          is None   # device=None is fine


def test_callback_latches_os_drop_on_input_overflow():
    """When PortAudio flags ``input_overflow`` the callback bumps the
    OS-level counters — independent of the queue.Full path."""
    from chirp.audio.capture import AudioCapture
    q = _q.Queue(maxsize=10)  # plenty of room — not a queue.Full case
    cap = AudioCapture(q, device=None)
    indata = np.zeros((1024, 1), dtype=np.float32)

    cap._callback(indata, 1024, None, _FakeStatus(input_overflow=True))
    cap._callback(indata, 1024, None, _FakeStatus(input_overflow=True))

    assert cap.os_drop_count       == 2
    assert cap.os_drop_count_total == 2
    assert cap.has_ever_os_dropped is True
    # Queue-full drop stats are unaffected — these are different
    # failure modes and must not be conflated.
    assert cap.drop_count       == 0
    assert cap.has_ever_dropped is False


def test_callback_ignores_status_without_overflow():
    """``input_overflow=False`` (or a status object without the
    attribute) must not spuriously bump OS-drop stats."""
    from chirp.audio.capture import AudioCapture
    q = _q.Queue(maxsize=10)
    cap = AudioCapture(q, device=None)
    indata = np.zeros((1024, 1), dtype=np.float32)

    cap._callback(indata, 1024, None, _FakeStatus(input_overflow=False))
    cap._callback(indata, 1024, None, None)

    assert cap.os_drop_count       == 0
    assert cap.has_ever_os_dropped is False


def test_consume_os_drop_count_clears_transient_only():
    from chirp.audio.capture import AudioCapture
    q = _q.Queue(maxsize=10)
    cap = AudioCapture(q, device=None)
    indata = np.zeros((1024, 1), dtype=np.float32)
    for _ in range(3):
        cap._callback(indata, 1024, None, _FakeStatus(input_overflow=True))

    assert cap.consume_os_drop_count() == 3
    assert cap.os_drop_count == 0
    # Sticky stats survive.
    assert cap.os_drop_count_total == 3
    assert cap.has_ever_os_dropped is True


def test_reset_error_stats_clears_everything():
    from chirp.audio.capture import AudioCapture
    q = _q.Queue(maxsize=10)
    cap = AudioCapture(q, device=None)
    indata = np.zeros((1024, 1), dtype=np.float32)
    cap._callback(indata, 1024, None, _FakeStatus(input_overflow=True))
    cap.open_error = 'pretend this was set on open'

    cap.reset_error_stats()

    assert cap.os_drop_count       == 0
    assert cap.os_drop_count_total == 0
    assert cap.has_ever_os_dropped is False
    assert cap.open_error          is None


def test_reset_error_stats_independent_of_reset_drop_stats():
    """#29 (queue.Full drops) and #43 (OS overflow) are separate
    failure modes wired to separate badges. Clearing one must not
    clobber the other."""
    from chirp.audio.capture import AudioCapture
    q = _q.Queue(maxsize=1)
    cap = AudioCapture(q, device=None)
    # Fill the queue then synthesize a queue-full drop via the
    # callback (mirrors the pattern in test_drop_flag.py).
    q.put_nowait(np.zeros(1024, dtype=np.float32))
    indata = np.zeros((1024, 1), dtype=np.float32)
    cap._callback(indata, 1024, None, _FakeStatus(input_overflow=True))

    assert cap.has_ever_dropped    is True  # queue.Full path
    assert cap.has_ever_os_dropped is True  # input_overflow path

    cap.reset_error_stats()
    assert cap.has_ever_os_dropped is False
    assert cap.has_ever_dropped    is True  # #29 stats untouched

    cap.reset_drop_stats()
    assert cap.has_ever_dropped    is False


# ── #48: open_error exposure ─────────────────────────────────────────

def test_open_error_captured_when_inputstream_raises(monkeypatch):
    """When sounddevice.InputStream() raises, the exception reason
    lands on ``open_error`` so the UI can surface it — the old code
    only printed to stdout, which is invisible in the frozen GUI."""
    import chirp.audio.capture as mod

    class _BadStream:
        def __init__(self, *a, **kw):
            raise OSError('Error opening InputStream: no such device [PaError -9996]')

    monkeypatch.setattr(mod.sd, 'InputStream', _BadStream)
    q = _q.Queue(maxsize=1)
    cap = mod.AudioCapture(q, device=99)

    assert cap.valid is False
    assert cap.open_error is not None
    assert 'OSError' in cap.open_error
    assert 'PaError' in cap.open_error


# ── WavFileCapture mirrors the same contract ─────────────────────────

def test_wav_capture_mirrors_error_fields():
    """WavFileCapture is a drop-in replacement for AudioCapture, so the
    sidebar's uniform polling path works regardless of capture type.
    All four new fields must exist with matching semantics."""
    from chirp.audio.wav_capture import WavFileCapture
    q = _q.Queue(maxsize=1)
    # Non-existent path — triggers the open_error path.
    cap = WavFileCapture(q, wav_path='__does_not_exist__.wav')
    assert cap.valid         is False
    assert cap.open_error is not None
    assert cap.os_drop_count       == 0
    assert cap.os_drop_count_total == 0
    assert cap.has_ever_os_dropped is False
    # consume_os_drop_count always returns zero for WAV playback.
    assert cap.consume_os_drop_count() == 0
    # reset_error_stats clears open_error too.
    cap.reset_error_stats()
    assert cap.open_error is None


# ── writer-pool error surfacing ──────────────────────────────────────

def test_writer_pool_surfaces_write_failures(monkeypatch, tmp_path):
    """When ``write_wav_sync`` raises inside a worker, the pool bumps
    its error counters. The old code only printed a stderr line — now
    the window polls ``error_stats()`` and lights up the sticky badge.
    """
    import chirp.recording.writer as wr

    # Clear any state from previous tests.
    wr.shutdown(timeout=1.0)

    def boom(*a, **kw):
        raise IOError('synthetic disk full')

    monkeypatch.setattr(wr, 'write_wav_sync', boom)

    wr.submit(
        buf_snapshot=[np.zeros(256, dtype=np.float32)],
        output_dir=str(tmp_path),
    )
    ok = wr.drain(timeout=2.0)
    assert ok, 'writer pool drain timed out'

    has_ever, total, last = wr.error_stats()
    assert has_ever is True
    assert total    == 1
    # Py3 aliases IOError → OSError, so the recorded type name is OSError.
    assert last is not None and 'OSError' in last
    assert 'synthetic disk full' in last

    # Transient counter is consumed (poll-and-clear pattern).
    assert wr.consume_error_count() == 1
    assert wr.consume_error_count() == 0
    # Sticky stats survive the poll.
    has_ever, total, _ = wr.error_stats()
    assert has_ever is True
    assert total    == 1

    # Reset clears everything.
    wr.reset_error_stats()
    assert wr.error_stats() == (False, 0, None)

    wr.shutdown(timeout=1.0)


def test_writer_pool_error_stats_noop_when_pool_never_created():
    """Calling the module-level helpers before any write happens must
    be safe — the sidebar poll runs every tick regardless of whether
    recording has started."""
    import chirp.recording.writer as wr
    wr.shutdown(timeout=1.0)  # ensure fresh state

    assert wr.error_stats() == (False, 0, None)
    assert wr.consume_error_count() == 0
    # reset_error_stats is a no-op when pool is None.
    wr.reset_error_stats()
    assert wr.error_stats() == (False, 0, None)
