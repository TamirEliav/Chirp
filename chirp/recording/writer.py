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
import itertools
import os
import queue
import threading
import time

import numpy as np
import scipy.io.wavfile
import soundfile as sf

from chirp.constants import SAMPLE_RATE
from chirp.error_log import log as _err_log


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


def _compose_filename(prefix: str, suffix: str, onset) -> str:
    """Compose the WAV filename from sanitized tokens (#51 / #23).

    Layout: ``<prefix>_<epoch_ms>_<localts>_<suffix>.wav`` with blank
    tokens dropped (no stray ``__``). ``epoch_ms`` + the ms-precision
    local timestamp come from ``onset``. Only the user-controlled
    ``prefix`` / ``suffix`` fields decorate the name — the stream name is
    deliberately NOT included (the user asked for prefix/suffix to be the
    sole naming controls). Same-millisecond collisions between two
    streams are handled downstream by ``_dedup_target`` (the ``_rNN``
    token). Shared by ``write_wav_sync`` and ``StreamingWavWriter`` so
    the naming contract can't drift between the two write paths.
    """
    # Timezone handling: aware onsets (the disciplined-clock path,
    # internally UTC) are converted to the system local zone HERE, at
    # composition time — so the local token is DST-correct even when a
    # transition happened mid-session. Naive onsets (legacy fallbacks,
    # tests) keep their historical treatment as local time.
    epoch_ms = int(onset.timestamp() * 1000)
    local = onset.astimezone() if onset.tzinfo is not None else onset
    local_ts = local.strftime('%Y%m%d_%H%M%S_%f')[:-3]
    parts = [p for p in [_sanitize_token(prefix), str(epoch_ms), local_ts,
                         _sanitize_token(suffix)] if p]
    return '_'.join(parts) + '.wav'


def _resolve_safe_path(output_dir: str, fname: str) -> str:
    """Join ``fname`` onto ``output_dir`` and verify the result stays
    inside it (#50 / #51). Rejects a blank/non-str ``output_dir`` and any
    composed path that escapes the directory (sanitizer-bug belt)."""
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ValueError(f'output_dir must be a non-empty string, '
                         f'got {output_dir!r}')
    path = os.path.join(output_dir, fname)
    real_out = os.path.realpath(output_dir)
    real_path = os.path.realpath(path)
    if os.path.commonpath([real_out, real_path]) != real_out:
        raise ValueError(
            f'refusing to write outside output_dir '
            f'(target={real_path!r}, output_dir={real_out!r})')
    return path


def _dedup_target(path: str) -> str:
    """L4: if ``path`` already exists, insert a ``_rNN`` token before
    the extension so a same-millisecond filename collision (two events
    flushed with identical onset tokens) can't silently overwrite an
    earlier recording via ``os.replace``."""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    for i in range(2, 100):
        cand = f'{base}_r{i:02d}{ext}'
        if not os.path.exists(cand):
            return cand
    return path  # 98 collisions — give up and overwrite


# Publish-time timestamp sanity check: when a finalized file's
# ``onset + duration`` disagrees with the wall clock by more than this
# many seconds, flag it via the pool error stats (sidebar badge) and
# ``chirp_errors.log`` — an end-of-pipeline watchdog for the class of
# bug where filename timestamps silently detach from reality (the
# 2026-07 ~1-day-backwards jump). Generous enough to tolerate the
# legitimate lag between a file's last sample and its publish: hold +
# post-trigger tail, writer-pool backlog on a slow disk, the fsync of
# a long part. ``None`` disables the check (the test-suite default —
# many tests write WAVs with fixed historical onsets; see
# tests/conftest.py).
TIMESTAMP_DIVERGENCE_SEC: float | None = 10.0


def _check_timestamp_divergence(onset, duration_sec: float, path: str,
                                stream: str) -> None:
    """Flag a published file whose derived end time (onset + duration)
    disagrees with the wall clock beyond ``TIMESTAMP_DIVERGENCE_SEC``.

    Naive onsets are interpreted as local time (the convention under
    which every naive onset in this codebase is created); aware onsets
    (the disciplined-clock path) are exact. Never raises — a watchdog
    must not be able to fail a write.
    """
    limit = TIMESTAMP_DIVERGENCE_SEC
    if limit is None or onset is None:
        return
    try:
        delta = time.time() - (onset.timestamp() + float(duration_sec))
        if abs(delta) <= limit:
            return
        msg = (f'published file timestamp diverges from wall clock by '
               f'{delta:+.1f}s (onset + duration vs now) — filename '
               f'timestamps may be wrong')
        p = _get_pool()
        with p._lock:
            p._err_count       += 1
            p._err_count_total += 1
            p._has_ever_errored = True
            p._last_error = msg[:200]
        _err_log('timestamp_divergence', stream or 'global', msg,
                 wav_path=path)
    except Exception:
        pass


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
    # Saturation detection: matches the per-chunk threshold used in
    # RecordingEntity (raw_peak >= 0.99). Logged once per saturated
    # file (after the path is finalised below) so the user can find
    # exactly which recordings had clipping without per-sample noise.
    saturated = bool(np.abs(audio).max() >= 0.99)
    pcm16 = (audio * 32767.0).clip(-32768, 32767).astype(np.int16)
    os.makedirs(output_dir, exist_ok=True)
    n_samples = audio.shape[0]
    audio_dur = n_samples / sample_rate
    if onset_time is not None:
        onset = onset_time
    else:
        onset = datetime.datetime.now() - datetime.timedelta(seconds=audio_dur)
    # #51: every user-controlled token that lands in the filename is
    # sanitized; the final path is verified to stay inside output_dir.
    # Shared with StreamingWavWriter via the module helpers.
    fname = _compose_filename(prefix, suffix, onset)
    path  = _resolve_safe_path(output_dir, fname)
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
    path = _dedup_target(path)
    os.replace(tmp_path, path)
    # Only when the caller supplied the onset — the fallback above is
    # now-derived, so its divergence is zero by construction.
    if onset_time is not None:
        _check_timestamp_divergence(onset_time, audio_dur, path,
                                    filename_stream)
    ch_str = 'stereo' if audio.ndim == 2 else 'mono'
    print(f'[REC] saved {path}  ({n_samples/sample_rate:.2f} s, {ch_str})')
    if saturated:
        peak = float(np.abs(audio).max())
        _err_log('saturation', filename_stream or 'global',
                 f'recording contains clipped samples (peak={peak:.4f})',
                 wav_path=path)
    return path


# ── Streaming WAV writer ───────────────────────────────────────────────────────

# Uniquifier for streaming tmp files: two events on the same stream can
# be open concurrently (post-trigger tail draining while a new burst
# opens) and could compose identical target filenames down to the
# millisecond — the tmp paths must never collide or the two writers
# would corrupt each other's file.
_tmp_counter = itertools.count(1)

# Async-append plumbing for StreamingWavWriter. The bound is in queued
# buffers, not bytes: at the default chunk size it is well over a minute
# of audio per open event, so ``put`` only ever blocks if the disk has
# stopped keeping up entirely — at which point stalling beats losing
# samples.
_APPEND_QUEUE_MAX = 4096
_APPEND_STOP = object()
# Upper bound on how long close()/abort() wait for the queue to drain.
# Runs on a writer-pool worker, so a slow flush delays publication, not
# audio capture.
_APPEND_DRAIN_TIMEOUT = 120.0


class StreamingWavWriter:
    """Incremental, atomic WAV writer backed by soundfile / libsndfile.

    Replaces the buffer-the-whole-event-then-write model: the file is
    opened at event onset and audio is appended as it is produced, so a
    long event never holds its whole self in RAM. One instance per active
    recording event, owned by the per-stream DSP thread — writes are
    therefore naturally parallel and isolated across streams (a failure
    on one stream's file cannot corrupt another's).

    Atomicity is preserved from the previous design: data lands in
    ``<path>.tmp`` and is published with fsync + ``os.replace`` only on
    :meth:`close`. A crash mid-event leaves an orphan ``.tmp`` (cleanable)
    and never a truncated canonical WAV. :meth:`abort` discards the tmp.

    Frames are float32 in [-1, 1]; they are converted to 16-bit PCM with
    the same ``* 32767`` scaling ``write_wav_sync`` uses, so a streamed
    file is byte-identical to a whole-buffer write of the same audio.
    """

    def __init__(self, output_dir: str, *, prefix: str = '', suffix: str = '',
                 sample_rate: int = SAMPLE_RATE, onset_time=None,
                 channels: int = 1, filename_stream: str = ''):
        if onset_time is None:
            onset_time = datetime.datetime.now()
        # Kept for the publish-time timestamp sanity check in close().
        self._onset = onset_time
        os.makedirs(output_dir, exist_ok=True)
        fname = _compose_filename(prefix, suffix, onset_time)
        self.path = _resolve_safe_path(output_dir, fname)
        self._tmp = f'{self.path}.{next(_tmp_counter)}.tmp'
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.filename_stream = filename_stream
        self.frames_written = 0
        self.peak = 0.0
        self._closed = False
        # Async append plumbing (see ``append``). The queue is created
        # lazily on the first append so a writer that never receives
        # audio costs nothing.
        self._q: "queue.Queue | None" = None
        self._thread: threading.Thread | None = None
        self._async_err: BaseException | None = None
        self._sf = sf.SoundFile(self._tmp, mode='w',
                                samplerate=self.sample_rate,
                                channels=self.channels,
                                format='WAV', subtype='PCM_16')

    @property
    def saturated(self) -> bool:
        return self.peak >= 0.99

    def retarget(self, output_dir: str, *, prefix: str = '', suffix: str = '',
                 onset_time=None, filename_stream: str = '') -> None:
        """Recompute the canonical publish path from fresh tokens.

        Used by the streaming recorder at flush time: the day-subfolder
        output dir may have rolled over mid-event and the suffix may
        have changed since open. Only ``self.path`` changes — the tmp file
        keeps accumulating where it is and :meth:`close` publishes to the
        new target via the same fsync + ``os.replace``."""
        if self._closed:
            return
        if onset_time is None:
            onset_time = datetime.datetime.now()
        else:
            # An explicit onset is the event's real one — remember it
            # for the close()-time timestamp sanity check. (The
            # now-default above is only a path-composition fallback and
            # must not overwrite a real onset from open time.)
            self._onset = onset_time
        os.makedirs(output_dir, exist_ok=True)
        fname = _compose_filename(prefix, suffix, onset_time)
        self.path = _resolve_safe_path(output_dir, fname)

    # ── Async append path ────────────────────────────────────────────

    def _ensure_worker(self) -> None:
        if self._thread is not None:
            return
        self._q = queue.Queue(maxsize=_APPEND_QUEUE_MAX)
        t = threading.Thread(
            target=self._drain_loop, daemon=True,
            name=f'chirp-wav-{os.path.basename(self.path)[:24]}')
        self._thread = t
        t.start()

    def _drain_loop(self) -> None:
        """Own the file handle: every byte of this WAV is written here,
        on this one thread, in queue order."""
        while True:
            item = self._q.get()
            if item is _APPEND_STOP:
                return
            if self._async_err is not None:
                continue          # already failed; drain and discard
            try:
                self._write_frames(item)
            except BaseException as exc:      # noqa: BLE001
                self._async_err = exc

    def _write_frames(self, frames: np.ndarray) -> None:
        pcm16 = (frames * 32767.0).clip(-32768, 32767).astype(np.int16)
        self._sf.write(pcm16)

    def _stop_worker(self) -> None:
        """Drain and retire the append worker. Safe to call repeatedly."""
        t, self._thread = self._thread, None
        if t is None:
            return
        try:
            self._q.put(_APPEND_STOP)
            t.join(timeout=_APPEND_DRAIN_TIMEOUT)
        except Exception:
            pass

    def append(self, frames: np.ndarray) -> None:
        """Queue float32 frames (mono ``(n,)`` or ``(n, channels)``).

        Realtime-path note: this is called from the per-stream ingest
        thread, which also runs the FFTs, the envelope and the trigger
        state machine. Doing the PCM conversion and the actual
        ``SoundFile.write`` here meant a slow or stalled disk (a busy
        SSD, an SMB share, a writeback burst) blocked audio consumption;
        the ingest thread then processed its backlog in a burst, holding
        the GIL long enough to delay the PortAudio callback — and a
        capture callback that misses its deadline is exactly what makes
        the driver zero-fill or drop samples. So the caller now only
        hands the buffer over; conversion and I/O happen on this
        writer's own thread.

        The queue is FIFO with a single consumer, so bytes land in the
        file in call order. It is bounded: if the disk falls
        catastrophically behind, ``put`` blocks rather than dropping
        audio (losing samples is worse than stalling) or growing without
        limit.

        The peak (saturation) scan stays on the caller: it is a single
        cheap pass next to the conversion and syscall being moved, and
        keeping it here preserves ``saturated`` as readable immediately
        after ``append`` rather than making it race the worker.
        """
        if self._closed or frames is None:
            return
        if frames.shape[0] == 0:
            return
        local_peak = float(np.abs(frames).max())
        if local_peak > self.peak:
            self.peak = local_peak
        err = self._async_err
        if err is not None:
            # Surface the background failure on the caller's thread so
            # the recorder aborts this event exactly as it did when the
            # write was synchronous.
            self._async_err = None
            raise err
        self._ensure_worker()
        self._q.put(frames)
        self.frames_written += int(frames.shape[0])

    def close(self) -> str:
        """Finalize: fsync the tmp then atomically publish the canonical
        path. Idempotent — returns the published path."""
        if self._closed:
            return self.path
        self._closed = True
        # Flush every queued buffer to disk before the file is finalized
        # — this runs on a writer-pool worker, never on the ingest
        # thread, so waiting here costs no audio.
        self._stop_worker()
        if self._async_err is not None:
            err, self._async_err = self._async_err, None
            try:
                self._sf.close()
            except Exception:
                pass
            try:
                os.remove(self._tmp)
            except OSError:
                pass
            raise err
        try:
            self._sf.close()
        except Exception:
            pass
        try:
            fd = os.open(self._tmp, os.O_RDWR)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass
        self.path = _dedup_target(self.path)
        os.replace(self._tmp, self.path)
        dur = self.frames_written / self.sample_rate if self.sample_rate else 0.0
        _check_timestamp_divergence(self._onset, dur, self.path,
                                    self.filename_stream)
        ch_str = 'stereo' if self.channels == 2 else 'mono'
        print(f'[REC] saved {self.path}  ({dur:.2f} s, {ch_str}, streamed)')
        if self.saturated:
            _err_log('saturation', self.filename_stream or 'global',
                     f'recording contains clipped samples (peak={self.peak:.4f})',
                     wav_path=self.path)
        return self.path

    def abort(self) -> None:
        """Discard the in-progress file without publishing it."""
        if self._closed:
            return
        self._closed = True
        self._stop_worker()
        self._async_err = None
        try:
            self._sf.close()
        except Exception:
            pass
        try:
            os.remove(self._tmp)
        except OSError:
            pass


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
                    # Two job shapes share the pool: the classic
                    # buffered write ``((buf, out_dir, prefix, suffix),
                    # kwargs)`` and a finalize call ``((callable,
                    # out_dir), kwargs)`` used to fsync + publish a
                    # StreamingWavWriter off the ingest thread (the
                    # synchronous close of a multi-MB part file at a
                    # force-split boundary stalled ingestion mid-event
                    # — the zero-sample corruption seen in the field).
                    if callable(job[0][0]):
                        job[0][0]()
                    else:
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
                    try:
                        out_dir = job[0][1] if len(job[0]) > 1 else ''
                        stream = job[1].get('filename_stream', '') or 'global'
                    except Exception:
                        out_dir, stream = '', 'global'
                    _err_log('wav_writer', stream,
                             f'{type(exc).__name__}: {exc}',
                             wav_path=out_dir or None)
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
                    try:
                        out_dir = job[0][1] if len(job[0]) > 1 else ''
                        stream = job[1].get('filename_stream', '') or 'global'
                    except Exception:
                        out_dir, stream = '', 'global'
                    _err_log('wav_writer', stream,
                             f'worker died: {type(base_exc).__name__}: '
                             f'{base_exc!r}',
                             wav_path=out_dir or None)
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


def submit_close(writer: 'StreamingWavWriter', filename_stream: str = '',
                 out_dir: str = '') -> None:
    """Enqueue the finalize/publish of a StreamingWavWriter on the pool.

    ``close()`` fsyncs the whole tmp file and atomically renames it —
    potentially seconds of blocking I/O for a long high-sample-rate
    part. Running it on a pool worker keeps the ingest thread realtime:
    a force-split boundary must never stall audio consumption mid-event
    (the capture pipeline glitches under that stall, corrupting the
    beginning of the next part). Failures surface through the same
    pool error stats / ``chirp_errors.log`` path as buffered writes;
    ``drain()``/``shutdown()`` wait for queued closes like any write.
    """
    _get_pool().submit(args=(writer.close, out_dir),
                       kwargs=dict(filename_stream=filename_stream))


def submit_call(fn, filename_stream: str = '', out_dir: str = '') -> None:
    """Enqueue a small arbitrary callable on the writer pool.

    Same error accounting / ``drain()`` semantics as WAV jobs (the pool
    already dispatches callable-shaped jobs — see the worker loop).
    Used for off-thread bookkeeping I/O such as the sidecar clock log,
    so the ingest thread never touches disk.
    """
    _get_pool().submit(args=(fn, out_dir),
                       kwargs=dict(filename_stream=filename_stream))


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


def note_stream_error(exc: BaseException, stream: str = '',
                      out_dir: str | None = None) -> None:
    """Record a streaming-write failure (StreamingWavWriter open/append
    raised on the DSP thread) in the same error stats the pool workers
    use, so the existing sidebar `!` badge and ``chirp_errors.log``
    surface it without a parallel accounting path."""
    p = _get_pool()
    with p._lock:
        p._err_count += 1
        p._err_count_total += 1
        p._has_ever_errored = True
        p._last_error = f'{type(exc).__name__}: {exc}'[:200]
    _err_log('wav_writer', stream or 'global',
             f'streaming write failed: {type(exc).__name__}: {exc}',
             wav_path=out_dir or None)


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
