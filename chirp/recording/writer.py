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

# #51: Windows reserved device names. Case-insensitive match on the
# token minus any extension — ``CON.wav`` is still reserved. If a user
# sets ``filename_prefix = 'CON'`` on Windows, ``os.path.join(out,
# 'CON_...wav')`` will open the console device, not a file. Reject
# these outright — the sanitized form gets an ``_r`` suffix so the name
# remains stable + human-readable.
_WIN_RESERVED = {
    'CON', 'PRN', 'AUX', 'NUL',
    *(f'COM{i}' for i in range(1, 10)),
    *(f'LPT{i}' for i in range(1, 10)),
}

# #51: hard cap on any single token written into a filename. Windows'
# MAX_PATH is 260; leaving ~64 chars per token keeps a four-token
# filename under that even with a long output folder.
_TOKEN_MAX_LEN = 64


def _sanitize_token(s: str) -> str:
    """#51: Strip filename-hostile characters from a filename token.

    Guarantees:
      - The return value contains only chars from ``_FILENAME_SAFE``
        (ASCII alnum + ``-._``). Everything else — path separators,
        drive letters, Unicode, whitespace, control bytes — becomes
        ``_``.
      - Reserved Windows device names (``CON``, ``PRN``, ``AUX``,
        ``NUL``, ``COM1..9``, ``LPT1..9``) are never returned as-is;
        they get an ``_r`` suffix so the rename is stable.
      - Length is capped at ``_TOKEN_MAX_LEN`` so a pathological prefix
        doesn't blow past ``MAX_PATH`` on Windows.
      - Pure-dot inputs (``.``, ``..``) map to ``''`` so they can't
        participate in path traversal.
    """
    if not s:
        return ''
    cleaned = ''.join(c if c in _FILENAME_SAFE else '_' for c in s).strip('_')
    # A run of dots (``..`` or ``.``) sanitizes to itself under the
    # char filter above — the ``.`` is in _FILENAME_SAFE. Strip those
    # explicitly so they can't walk the path.
    if cleaned.strip('.') == '':
        return ''
    if cleaned.upper() in _WIN_RESERVED:
        cleaned = cleaned + '_r'
    if len(cleaned) > _TOKEN_MAX_LEN:
        cleaned = cleaned[:_TOKEN_MAX_LEN].rstrip('_')
    return cleaned


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
    # #50 / #51: reject obviously-invalid ``output_dir`` early. A blank
    # string would turn ``os.path.join`` into a relative path next to
    # the executable, silently stashing WAVs where the user can't find
    # them. A non-str would crash later inside ``os.makedirs``; better
    # to fail loudly now so the writer-pool error counter picks it up
    # and the sidebar badge lights.
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ValueError(f'write_wav_sync: output_dir must be a non-empty '
                         f'string, got {output_dir!r}')

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
    # #51: every user-controlled token that lands in the filename MUST
    # be sanitized — a ``filename_prefix`` of ``../../escape`` would
    # otherwise let a WAV land outside ``output_dir``. The
    # ``filename_stream`` token is now ALSO included in ``parts`` so
    # two streams that trigger on the same physical event (same
    # ms-precision timestamp) do not clobber each other's files.
    prefix_s = _sanitize_token(prefix)
    suffix_s = _sanitize_token(suffix)
    stream_s = _sanitize_token(filename_stream)
    parts = [p for p in [prefix_s, str(epoch_ms), local_ts,
                         stream_s, suffix_s] if p]
    fname = '_'.join(parts) + '.wav'
    path  = os.path.join(output_dir, fname)
    # #51 belt: verify the final path is still inside output_dir. A
    # sanitizer bug or a future refactor must not allow ``os.path.join``
    # to escape the directory.
    real_out = os.path.realpath(output_dir)
    real_path = os.path.realpath(path)
    if os.path.commonpath([real_out, real_path]) != real_out:
        raise ValueError(
            f'write_wav_sync: refusing to write outside output_dir '
            f'(target={real_path!r}, output_dir={real_out!r})')
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
        # #47: worker supervisor — if a worker dies from an
        # unexpected BaseException (MemoryError, a bug in
        # write_wav_sync, etc), the pool respawns it so the queue
        # keeps draining. ``_respawn_count`` is exposed for tests.
        self._shutting_down = False
        self._respawn_count = 0
        # #47: queue-backlog high watermark. Tracks the largest size
        # the queue has ever reached — the UI can surface a warning
        # when this exceeds a sane threshold on slow output targets.
        self._queue_high_watermark = 0
        self._workers: list[threading.Thread | None] = [None] * n_workers
        for i in range(n_workers):
            self._spawn_worker(i)

    def _spawn_worker(self, worker_id: int) -> None:
        """#47: create + start a worker thread at slot ``worker_id``.
        Used both at pool startup and for supervisor respawns."""
        t = threading.Thread(
            target=self._worker_loop,
            args=(worker_id,),
            name=f'chirp-wav-writer-{worker_id}',
            daemon=False,
        )
        with self._lock:
            self._workers[worker_id] = t
        t.start()

    def _worker_loop(self, worker_id: int) -> None:
        while True:
            # Dequeue — if this fails the thread hasn't consumed a
            # job yet, so there's no accounting to unwind.
            try:
                job = self._queue.get()
            except BaseException as dequeue_exc:
                self._on_worker_death(worker_id, dequeue_exc,
                                      decrement_inflight=False,
                                      task_done_needed=False)
                return

            if job is self._stop:
                self._queue.task_done()
                return

            # Process the job. The ``finally`` block runs on both
            # regular returns and on BaseException propagation, so
            # accounting is always consistent — the outer supervisor
            # only needs to arrange the respawn.
            try:
                try:
                    write_wav_sync(*job[0], **job[1])
                except Exception as exc:
                    # #44: ordinary Exception path — log + bump counters,
                    # keep the worker alive for the next job.
                    with self._lock:
                        self._err_count       += 1
                        self._err_count_total += 1
                        self._has_ever_errored = True
                        self._last_error = f'{type(exc).__name__}: {exc}'[:200]
                    print(f'[REC] WAV write failed: {exc}')
                except BaseException as base_exc:
                    # #47: a BaseException subclass escaped the inner
                    # ``except Exception`` — e.g. a bug raising
                    # SystemExit from inside scipy, or something
                    # similarly unusual. Log + arrange respawn; the
                    # ``finally`` below still runs and keeps accounting
                    # consistent.
                    print(f'[REC] writer worker {worker_id} died during write: '
                          f'{type(base_exc).__name__}: {base_exc!r}; respawning')
                    with self._lock:
                        self._err_count_total += 1
                        self._has_ever_errored = True
                        self._last_error = (
                            f'worker died: {type(base_exc).__name__}'[:200])
                        self._respawn_count += 1
                        shutting_down = self._shutting_down
                    if not shutting_down:
                        self._spawn_worker(worker_id)
                    return
            finally:
                with self._lock:
                    self._inflight -= 1
                    if self._inflight == 0:
                        self._idle.notify_all()
                self._queue.task_done()

    def _on_worker_death(self, worker_id: int, exc: BaseException,
                         decrement_inflight: bool,
                         task_done_needed: bool) -> None:
        """#47: shared cleanup for the "worker died before finishing
        its job" paths. Logs, bumps the sticky error flag, respawns a
        fresh worker at ``worker_id`` unless the pool is shutting
        down, and optionally unwinds inflight / task_done accounting
        so ``drain()`` doesn't hang forever."""
        print(f'[REC] writer worker {worker_id} died: '
              f'{type(exc).__name__}: {exc!r}; respawning')
        with self._lock:
            self._err_count_total += 1
            self._has_ever_errored = True
            self._last_error = f'worker died: {type(exc).__name__}'[:200]
            if decrement_inflight:
                self._inflight -= 1
                if self._inflight == 0:
                    self._idle.notify_all()
            self._respawn_count += 1
            shutting_down = self._shutting_down
        if task_done_needed:
            try:
                self._queue.task_done()
            except ValueError:
                pass
        if not shutting_down:
            self._spawn_worker(worker_id)

    def submit(self, args: tuple, kwargs: dict) -> None:
        with self._lock:
            self._inflight += 1
            # #47: track queue-backlog high watermark before we put so
            # the reading is "after this submit, how deep can it get".
            depth = self._inflight
            if depth > self._queue_high_watermark:
                self._queue_high_watermark = depth
        self._queue.put((args, kwargs))

    def pending(self) -> int:
        with self._lock:
            return self._inflight

    def queue_stats(self) -> tuple[int, int, int]:
        """#47: return (inflight, high_watermark, respawn_count). UI
        uses this to surface queue-backlog + worker-death telemetry."""
        with self._lock:
            return (self._inflight,
                    self._queue_high_watermark,
                    self._respawn_count)

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
        # #47: mark the pool as shutting down so any worker that dies
        # AFTER drain returns doesn't get respawned into a zombie that
        # would keep the interpreter alive past closeEvent.
        with self._lock:
            self._shutting_down = True
            workers = [w for w in self._workers if w is not None]
        for _ in workers:
            self._queue.put(self._stop)
        for t in workers:
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


def queue_stats() -> tuple[int, int, int]:
    """#47: (inflight, high_watermark, respawn_count) snapshot of the
    singleton pool. Zero/zero/zero when the pool was never created."""
    with _pool_lock:
        p = _pool
    if p is None:
        return (0, 0, 0)
    return p.queue_stats()


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
