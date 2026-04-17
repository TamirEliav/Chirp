"""WavFileCapture — feed a WAV file through the same queue as AudioCapture.

Used for reproducible testing and offline analysis (#<new issue>). Reads
the chosen WAV into memory once, then emits ``CHUNK_FRAMES``-sized
chunks onto the shared audio queue at real-time pace so the rest of the
pipeline (FFT, entropy, trigger, writer) behaves exactly as it would
with live input. Mirrors the ``AudioCapture`` contract — ``valid``,
``drop_count``, ``consume_drop_count``, ``resume``, ``pause``, ``close``
— so ``RecordingEntity`` can swap one for the other without any
further plumbing.
"""

from __future__ import annotations

import queue
import threading
import time

import numpy as np
import scipy.io.wavfile

from chirp.constants import CHUNK_FRAMES


def _load_wav(path: str) -> tuple[int, np.ndarray]:
    """Read a WAV file into a float32 numpy array in [-1, 1].

    Returns ``(sample_rate, samples)``. ``samples`` is 1-D for mono or
    ``(N, C)`` for multi-channel. Raises on I/O failure.
    """
    sr, data = scipy.io.wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    elif data.dtype == np.float32:
        data = data.astype(np.float32, copy=False)
    else:
        data = data.astype(np.float32)
    return int(sr), data


class WavFileCapture:
    """Drop-in replacement for AudioCapture that reads from a WAV file.

    The internal producer thread emits chunks to the queue at
    wall-clock pace (one ``CHUNK_FRAMES``/``sample_rate``-second chunk
    per tick) so the UI and the trigger state machine see timing
    equivalent to live capture. When the file is shorter than the
    session, it loops by default (pass ``loop=False`` for one-shot).
    """

    def __init__(self, audio_queue: queue.Queue, wav_path: str,
                 channels: int = 1, loop: bool = True):
        self._queue    = audio_queue
        self._channels = channels
        self._loop     = loop
        self._stop_evt  = threading.Event()
        self._pause_evt = threading.Event()
        self._pause_evt.set()  # start paused; resume() starts playback
        self._thread: threading.Thread | None = None
        self.drop_count = 0
        # Playback cursor (frame index). Written by the producer thread,
        # read from the UI thread — a plain int assignment is atomic in
        # CPython so no lock is needed.
        self._pos = 0
        self._reset_requested = False

        self._samples: np.ndarray | None = None
        self._file_sample_rate: int | None = None
        self._file_channels: int = 1
        self.wav_path = wav_path
        try:
            sr, samples = _load_wav(wav_path)
            self._samples = samples
            self._file_sample_rate = sr
            self._file_channels = 1 if samples.ndim == 1 else samples.shape[1]
        except Exception as exc:
            print(f"[WavFileCapture] Failed to open {wav_path}: {exc}")

    @property
    def valid(self) -> bool:
        return self._samples is not None

    @property
    def file_sample_rate(self) -> int | None:
        return self._file_sample_rate

    @property
    def file_channels(self) -> int:
        return self._file_channels

    @property
    def duration_sec(self) -> float:
        if self._samples is None or not self._file_sample_rate:
            return 0.0
        return len(self._samples) / self._file_sample_rate

    @property
    def position_sec(self) -> float:
        if self._samples is None or not self._file_sample_rate:
            return 0.0
        return self._pos / self._file_sample_rate

    def set_loop(self, loop: bool) -> None:
        self._loop = bool(loop)

    def reset_position(self) -> None:
        """Rewind playback to the start of the file.

        Thread-safe: raises a flag the producer thread observes on its
        next iteration, which avoids tearing the playback cursor.
        """
        self._reset_requested = True

    def consume_drop_count(self) -> int:
        n = self.drop_count
        self.drop_count = 0
        return n

    def resume(self) -> None:
        if self._samples is None:
            return
        self._pause_evt.clear()
        if self._thread is None or not self._thread.is_alive():
            self._stop_evt.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="chirp-wav-capture",
                daemon=True,
            )
            self._thread.start()

    def pause(self) -> None:
        self._pause_evt.set()

    def close(self) -> None:
        self._stop_evt.set()
        self._pause_evt.clear()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    # ── Internals ────────────────────────────────────────────────────

    def _build_chunk(self, start: int) -> tuple[np.ndarray, int]:
        """Return ``(chunk, next_pos)``.

        ``chunk`` is always ``CHUNK_FRAMES`` long; when looping, it
        wraps. When not looping and the file ends mid-chunk, the tail
        is zero-padded and ``next_pos`` is set to ``len(samples)`` so
        the producer stops.
        """
        samples = self._samples
        total   = len(samples)
        end     = start + CHUNK_FRAMES
        if end <= total:
            return samples[start:end], end % total if self._loop else end
        head = samples[start:]
        if self._loop:
            wrap = end - total
            tail = samples[:wrap]
            return np.concatenate([head, tail]), wrap
        pad_shape = list(head.shape)
        pad_shape[0] = CHUNK_FRAMES - len(head)
        pad = np.zeros(pad_shape, dtype=head.dtype)
        return np.concatenate([head, pad]), total

    def _format_for_queue(self, chunk: np.ndarray) -> np.ndarray:
        """Shape the chunk to match ``AudioCapture``'s queue contract."""
        if self._channels == 1:
            if chunk.ndim == 2:
                return chunk[:, 0].astype(np.float32, copy=False)
            return chunk.astype(np.float32, copy=False)
        if chunk.ndim == 1:
            # Duplicate mono to both channels so stereo configs work.
            return np.stack([chunk, chunk], axis=1).astype(np.float32, copy=False)
        return chunk[:, :2].astype(np.float32, copy=False)

    def _run(self) -> None:
        total = len(self._samples)
        sr = self._file_sample_rate or 44100
        chunk_dt = CHUNK_FRAMES / sr
        next_t = time.monotonic()
        # Emit chunks in a catch-up loop so we don't lose samples to the
        # Windows scheduler's coarse timer resolution (~15.6 ms by
        # default) — a naive ``wait(chunk_dt)`` per chunk rounds up to
        # the next tick and plays back noticeably slower than real time.
        # Whenever the wall clock has advanced past `next_t`, we emit
        # however many chunks are due in one go, then sleep for the
        # remainder. On average this matches the file's sample rate.
        while not self._stop_evt.is_set():
            if self._pause_evt.is_set():
                if self._stop_evt.wait(0.05):
                    return
                next_t = time.monotonic()
                continue

            if self._reset_requested:
                self._pos = 0
                self._reset_requested = False
                next_t = time.monotonic()

            now = time.monotonic()
            # If we've fallen more than ~1 s behind (long GC pause, slow
            # consumer, etc.), resync rather than bursting a flood of
            # back-to-back chunks.
            if now - next_t > 1.0:
                next_t = now

            emitted = False
            while now >= next_t and not self._stop_evt.is_set():
                chunk, next_pos = self._build_chunk(self._pos)
                emit = self._format_for_queue(chunk)
                try:
                    self._queue.put_nowait(emit.copy())
                except queue.Full:
                    self.drop_count += 1

                if not self._loop and next_pos >= total:
                    self._pos = total
                    return
                self._pos = next_pos
                next_t += chunk_dt
                emitted = True
                # Re-read wall clock after the put to keep catch-up tight.
                now = time.monotonic()

            if not emitted:
                # Nothing due yet — sleep until the next scheduled chunk.
                delay = next_t - now
                if delay > 0 and self._stop_evt.wait(delay):
                    return
