"""Append-only error log written to ``chirp_errors.log`` in the
current working directory.

Surfaces the same error categories tracked for the sidebar `!` badge
(audio ring overruns, OS-level audio overflows, ingest-thread
exceptions, capture open failures, WAV-writer failures) so the user
can trace any indicator back to a precise timestamp + stream + (for
writer / WAV-playback failures) the file path involved.

Realtime safety
---------------
The redesign moves *all* disk I/O off the realtime and DSP threads. A
call to :func:`log` does nothing but stamp a wall-clock time and drop a
small tuple onto a lock-free ``queue.SimpleQueue``; a single daemon
writer thread drains that queue, applies throttling, formats the line,
and writes it to disk. The audio callback and the per-stream DSP
threads therefore never block on the filesystem — the root cause of the
cascading overflows in the previous design, where ``log`` opened and
wrote ``chirp_errors.log`` from inside the PortAudio callback.

Categories
----------
``ring_overrun`` — The capture ring buffer wrapped before the DSP thread
                  consumed it (consumer fell far behind). Throttled.
``queue_full``  — Legacy alias for ``ring_overrun`` (kept for any caller
                  still using it). Throttled.
``os_drop``     — PortAudio ``input_overflow`` flag (driver / OS lost
                  samples upstream of our ring). Throttled.
``underflow``   — PortAudio ``input_underflow`` flag: zero samples were
                  INSERTED into the capture buffer to cover missing
                  data — silent zero runs are entering the recorded
                  audio itself (spectrogram / monitor / WAVs).
                  Throttled.
``zero_run``    — Signal-level inserted-silence detector: exact-zero
                  runs >= ~1 ms found in the raw captured audio (the
                  Windows engine / driver zero-fills without raising
                  any PortAudio flag; restart acquisition on ALL
                  streams of the affected input device to clear the
                  latched endpoint state). Throttled.
``zero_run_recovery`` — The inserted-silence watchdog restarted
                  acquisition on every stream of a device because its
                  zero-sample duty cycle stayed above the configured
                  threshold. Logged per intervention (and per failure to
                  restart). Not throttled.
``ingest``      — Exception raised inside the per-entity DSP loop
                  (DSP / FFT / trigger). Logged every event.
``open``        — Capture failed to open the device or the WAV input
                  file. Logged every event.
``capture_dead``— Live-device capture stopped delivering frames (RDP
                  session change / device removal); also logs the
                  matching auto-reconnect. Logged every event.
``wav_writer``  — Worker failed while writing a triggered WAV. Logged
                  every event; ``wav_path`` carries the target folder
                  so the user can see which output was affected.
``saturation``  — A successfully-written WAV contained clipped samples
                  (peak >= 0.99 of full scale). Logged once per file;
                  ``wav_path`` carries the full path of the recording.
``clock_step``  — The disciplined timestamp clock detected a capture
                  hole (device stall / drop burst) and stepped forward
                  across it between recording events. Logged per step.
``timestamp_divergence`` — A published WAV's ``onset + duration``
                  disagrees with the wall clock beyond the sanity
                  threshold (writer.TIMESTAMP_DIVERGENCE_SEC) — the
                  filename timestamps may be wrong. Logged per file;
                  ``wav_path`` carries the recording's full path.

Throttling
----------
``ring_overrun``, ``queue_full``, ``os_drop`` and ``underflow`` can fire
on every audio chunk. To keep the log bounded, those categories are
limited to one
entry per (stream, category) per ``_THROTTLE_SECONDS``. Throttling is
applied on the writer thread (keyed on the event's stamped wall time),
so the cheap, non-blocking enqueue on the producer side stays uniform.

The logger never raises — any I/O failure (path locked, disk full,
permission error) is swallowed silently. Losing log lines is strictly
preferable to crashing the audio pipeline.
"""

import datetime
import os
import queue
import sys
import threading
import time

_LOG_FILENAME = 'chirp_errors.log'
_THROTTLE_SECONDS = 1.0
_THROTTLED_CATEGORIES = frozenset({'ring_overrun', 'queue_full', 'os_drop',
                                   'underflow', 'zero_run'})

# Lock-free, unbounded, thread-safe producer→writer channel. Producers
# (realtime callback, DSP threads, writer threads, UI poll) only ever
# call ``.put`` on this; the single writer thread calls ``.get``.
_records: "queue.SimpleQueue" = queue.SimpleQueue()

_STOP = object()
_writer_thread: threading.Thread | None = None
_writer_start_lock = threading.Lock()

# Owned exclusively by the writer thread — no lock required.
_last_log_at: dict[tuple[str, str], float] = {}


def _path() -> str:
    # L5: in a frozen (PyInstaller) build the CWD of a shortcut-launched
    # app is often system32 or the user profile — anchor the log next to
    # the executable so the user can actually find it. Dev runs keep the
    # CWD (the repo root when launched via ``python -m chirp``).
    if getattr(sys, 'frozen', False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.getcwd()
    return os.path.join(base, _LOG_FILENAME)


def _ensure_writer() -> None:
    """Lazily start the background writer thread (idempotent)."""
    global _writer_thread
    t = _writer_thread
    if t is not None and t.is_alive():
        return
    with _writer_start_lock:
        t = _writer_thread
        if t is not None and t.is_alive():
            return
        t = threading.Thread(target=_writer_loop, name='chirp-errlog',
                             daemon=True)
        _writer_thread = t
        t.start()


def _writer_loop() -> None:
    while True:
        rec = _records.get()
        if rec is _STOP:
            return
        try:
            _write_record(rec)
        except Exception:
            pass


def _write_record(rec: tuple) -> None:
    category, stream, message, wav_path, wall = rec
    if category in _THROTTLED_CATEGORIES:
        key = (stream or '', category)
        last = _last_log_at.get(key, 0.0)
        if wall - last < _THROTTLE_SECONDS:
            return
        _last_log_at[key] = wall

    ts = datetime.datetime.fromtimestamp(wall).isoformat(timespec='milliseconds')
    fields = [ts, category, f'stream={stream or "?"}']
    if wav_path:
        fields.append(f'file={wav_path}')
    fields.append((message or '').replace('\n', ' | ').strip())
    line = '\t'.join(fields) + '\n'
    try:
        with open(_path(), 'a', encoding='utf-8') as f:
            f.write(line)
    except Exception:
        pass


def log(category: str, stream: str, message: str,
        wav_path: str | None = None) -> None:
    """Enqueue one error entry for the background writer.

    Non-blocking and realtime-safe: stamps a wall-clock time and drops a
    tuple on a lock-free queue. Never touches disk on the caller's
    thread, never raises.
    """
    try:
        _ensure_writer()
        _records.put((category, stream, message, wav_path, time.time()))
    except Exception:
        pass


def flush(timeout: float = 2.0) -> bool:
    """Best-effort wait until the writer has drained the queue.

    Intended for tests and for the app's clean-shutdown path. Returns
    ``True`` if the queue emptied within ``timeout``.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _records.empty():
            return True
        time.sleep(0.005)
    return _records.empty()


def shutdown(timeout: float = 2.0) -> None:
    """Flush pending records and stop the writer thread.

    Called from the window's close path so a frozen GUI build doesn't
    lose the tail of the log. Safe to call when no writer was started.
    """
    global _writer_thread
    t = _writer_thread
    if t is None or not t.is_alive():
        _writer_thread = None
        return
    flush(timeout)
    _records.put(_STOP)
    t.join(timeout)
    _writer_thread = None
