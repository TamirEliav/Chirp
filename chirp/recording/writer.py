"""WAV writer — synchronous helper + bounded worker pool.

Extracted from `ThresholdRecorder._write_wav` in the Phase 1 refactor
(plan: c06) and upgraded in c16 (#17) from a fire-and-forget
daemon-thread launcher to a proper writer pool that can be drained
on application shutdown.

The pool uses *non-daemon* worker threads — once the application
calls `drain()` (from `ChirpWindow.closeEvent`), pending writes
finish before the interpreter exits. Daemon threads, by contrast,
get killed mid-write at interpreter teardown and the WAV is left
truncated.

API:

  - `write_wav_sync(...)`  — synchronous helper, used by the worker
    threads and directly by tests that want to assert on disk.
  - `submit(...)`          — enqueue a write on the pool; returns
    immediately after queuing.
  - `start_flush_thread(...)` — back-compat shim. Now delegates to
    `submit` so existing callers (ThresholdRecorder._start_flush)
    transparently route through the pool.
  - `drain(timeout)`       — wait for the pool to finish all
    in-flight writes. Called from `closeEvent`.
  - `pending()`            — number of writes still queued or in
    progress; used by the UI to decide whether to show a modal.
"""

import datetime
import os
import queue
import threading

import numpy as np
import scipy.io.wavfile

from chirp.constants import SAMPLE_RATE


_FILENAME_SAFE = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._")


def _sanitize_token(s: str) -> str:
    """Strip filename-hostile characters from a stream-name token."""
    return ''.join(c if c in _FILENAME_SAFE else '_' for c in s).strip('_')


def write_wav_sync(buf_snapshot: list, output_dir: str,
                   prefix: str = '', suffix: str = '',
                   sample_rate: int = SAMPLE_RATE,
                   onset_time=None,
                   filename_stream: str = '') -> str:
    """Concatenate chunks and write a 16-bit PCM WAV synchronously.

    Returns the output path. Raises on I/O failure — the worker
    thread in the pool catches and logs but does not propagate.

    #52: the write is atomic — ``scipy.io.wavfile.write`` lands the
    bytes at ``<target>.tmp``, the tmp is fsynced, and
    ``os.replace(tmp, target)`` publishes it in one step. A crash
    mid-write (power loss, OOM-kill, force-close during drain)
    leaves either the old file untouched or the new file complete —
    never a truncated RIFF header with a wrong sample count.
    """
    audio = np.concatenate(buf_snapshot)
    if audio.ndim == 1:
        audio = audio.flatten()
    pcm16 = (audio * 32767.0).clip(-32768, 32767).astype(np.int16)
    os.makedirs(output_dir, exist_ok=True)
    n_samples = audio.shape[0]
    audio_dur = n_samples / sample_rate
    if onset_time is not None:
        onset = onset_time
    else:
        onset = datetime.datetime.now() - datetime.timedelta(seconds=audio_dur)
    epoch_ms = int(onset.timestamp() * 1000)
    local_ts = onset.strftime('%Y%m%d_%H%M%S_%f')[:-3]
    parts = [p for p in [prefix.rstrip('_'), str(epoch_ms), local_ts,
                         suffix.lstrip('_')] if p]
    fname = '_'.join(parts) + '.wav'
    path  = os.path.join(output_dir, fname)
    # #52: write to a sibling tmp file then rename atomically. Keep the
    # tmp file on the SAME directory as the target so ``os.replace``
    # stays an in-filesystem atomic rename (cross-filesystem would
    # fall back to a non-atomic copy).
    tmp_path = path + '.tmp'
    scipy.io.wavfile.write(tmp_path, sample_rate, pcm16)
    # Best-effort fsync so the bytes are durable before the rename
    # publishes the file. A missing / unsupported fsync (e.g.
    # certain FUSE filesystems) must not fail the write.
    try:
        fd = os.open(tmp_path, os.O_RDWR)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass
    os.replace(tmp_path, path)
    ch_str = 'stereo' if audio.ndim == 2 else 'mono'
    print(f'[REC] saved {path}  ({n_samples/sample_rate:.2f} s, {ch_str})')
    return path


# ── Writer pool ──────────────────────────────────────────────────────────────

class _WriterPool:
    """Bounded non-daemon worker pool for WAV writes (#17 / c16).

    Workers are non-daemon so the interpreter cannot exit while a WAV
    is mid-write. `drain()` blocks until the queue is empty and all
    workers are idle; it is called from `ChirpWindow.closeEvent`.
    """

    def __init__(self, n_workers: int = 2):
        self._queue: queue.Queue = queue.Queue()
        self._stop = object()  # sentinel
        self._inflight = 0
        self._lock = threading.Lock()
        self._idle = threading.Condition(self._lock)
        # #44: surface write failures. The worker loop used to swallow
        # every exception into a stdout print() — in a GUI build the
        # user has no way to tell that a recording never made it to
        # disk. The window polls the transient counter each tick and
        # latches the sticky flag for the sidebar error badge.
        self._err_count       = 0     # transient per-tick
        self._err_count_total = 0     # session-wide monotonic
        self._has_ever_errored = False
        self._last_error: str | None = None
        self._workers: list[threading.Thread] = []
        for i in range(n_workers):
            t = threading.Thread(
                target=self._worker_loop,
                name=f'chirp-wav-writer-{i}',
                daemon=False,
            )
            t.start()
            self._workers.append(t)

    def _worker_loop(self) -> None:
        while True:
            job = self._queue.get()
            if job is self._stop:
                self._queue.task_done()
                return
            try:
                write_wav_sync(*job[0], **job[1])
            except Exception as exc:
                # #44: bump the session-wide counters so the window
                # surfaces a sticky error badge. Still log to stderr
                # for post-mortem diagnosis.
                with self._lock:
                    self._err_count       += 1
                    self._err_count_total += 1
                    self._has_ever_errored = True
                    self._last_error = f'{type(exc).__name__}: {exc}'[:200]
                print(f'[REC] WAV write failed: {exc}')
            finally:
                with self._lock:
                    self._inflight -= 1
                    if self._inflight == 0:
                        self._idle.notify_all()
                self._queue.task_done()

    def submit(self, args: tuple, kwargs: dict) -> None:
        with self._lock:
            self._inflight += 1
        self._queue.put((args, kwargs))

    def pending(self) -> int:
        with self._lock:
            return self._inflight

    def consume_error_count(self) -> int:
        """#44: return & clear the transient error counter. Polled
        once per UI tick."""
        with self._lock:
            n = self._err_count
            self._err_count = 0
            return n

    def error_stats(self) -> tuple[bool, int, str | None]:
        """#44: return (has_ever_errored, total_count, last_message).
        Read-only snapshot — caller does not need the lock."""
        with self._lock:
            return (self._has_ever_errored,
                    self._err_count_total,
                    self._last_error)

    def reset_error_stats(self) -> None:
        """#44: clear all write-error stats (triggered by the user
        clicking the sticky error badge)."""
        with self._lock:
            self._err_count        = 0
            self._err_count_total  = 0
            self._has_ever_errored = False
            self._last_error       = None

    def drain(self, timeout: float | None = None) -> bool:
        """Block until all queued + in-flight writes finish.

        Returns True if drained within `timeout`, False on timeout.
        Does NOT shut the pool down — call `shutdown()` separately
        if you want to join the worker threads.
        """
        with self._lock:
            if self._inflight == 0:
                return True
            return self._idle.wait_for(lambda: self._inflight == 0,
                                       timeout=timeout)

    def shutdown(self, timeout: float | None = None) -> None:
        """Drain + send stop sentinels + join the worker threads."""
        self.drain(timeout=timeout)
        for _ in self._workers:
            self._queue.put(self._stop)
        for t in self._workers:
            t.join(timeout=timeout)


_pool: _WriterPool | None = None
_pool_lock = threading.Lock()


def _get_pool() -> _WriterPool:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = _WriterPool(n_workers=2)
        return _pool


def submit(buf_snapshot: list, output_dir: str,
           prefix: str = '', suffix: str = '',
           sample_rate: int = SAMPLE_RATE,
           onset_time=None,
           filename_stream: str = '') -> None:
    """Enqueue a WAV write on the singleton pool."""
    _get_pool().submit(
        args=(list(buf_snapshot), output_dir, prefix, suffix),
        kwargs=dict(sample_rate=sample_rate, onset_time=onset_time,
                    filename_stream=filename_stream),
    )


def pending() -> int:
    """Number of writes still in-flight (queued + executing)."""
    with _pool_lock:
        if _pool is None:
            return 0
        return _pool.pending()


def consume_error_count() -> int:
    """#44: return & clear the transient write-error count on the
    singleton pool. Safe to call when the pool was never used."""
    with _pool_lock:
        p = _pool
    if p is None:
        return 0
    return p.consume_error_count()


def error_stats() -> tuple[bool, int, str | None]:
    """#44: (has_ever_errored, total, last_message) on the singleton
    pool. Returns the "no errors" tuple when the pool was never used."""
    with _pool_lock:
        p = _pool
    if p is None:
        return (False, 0, None)
    return p.error_stats()


def reset_error_stats() -> None:
    """#44: clear write-error stats on the singleton pool. No-op
    when the pool hasn't been created yet."""
    with _pool_lock:
        p = _pool
    if p is not None:
        p.reset_error_stats()


def drain(timeout: float | None = None) -> bool:
    """Block until the pool finishes all in-flight writes.

    Returns True if drained within `timeout`, False on timeout. Safe
    to call when the pool was never used (returns True immediately).
    """
    with _pool_lock:
        p = _pool
    if p is None:
        return True
    return p.drain(timeout=timeout)


def shutdown(timeout: float | None = None) -> None:
    """Drain + tear down the pool. Idempotent."""
    global _pool
    with _pool_lock:
        p = _pool
        _pool = None
    if p is not None:
        p.shutdown(timeout=timeout)


def start_flush_thread(buf_snapshot: list, output_dir: str,
                       prefix: str = '', suffix: str = '',
                       sample_rate: int = SAMPLE_RATE,
                       onset_time=None,
                       filename_stream: str = '') -> None:
    """Back-compat shim: route through the pool (#17 / c16).

    Kept so existing callers (ThresholdRecorder._start_flush) and any
    out-of-tree code continue to work without modification. The old
    fire-and-forget daemon-thread implementation is gone — writes
    now survive interpreter shutdown when `drain()` is called first.
    """
    submit(buf_snapshot, output_dir, prefix, suffix,
           sample_rate=sample_rate, onset_time=onset_time,
           filename_stream=filename_stream)
