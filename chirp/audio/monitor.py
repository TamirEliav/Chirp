"""AudioMonitor — real-time loopback to a shared output device (#7).

A single, app-wide object that owns one ``sounddevice.OutputStream``
pointed at the user-chosen monitor device. Any number of capture
sources (live ``AudioCapture``, ``WavFileCapture``) push raw chunks via
``feed(source_id, chunk)``; only the chunks whose ``source_id`` matches
the currently selected monitor source survive — everything else is
dropped, which is how the "radio-style, only one stream at a time"
constraint is enforced.

Latency is kept low by bypassing Chirp's DSP pipeline entirely: the
capture callback writes raw samples straight into a small ring buffer,
and the PortAudio output callback pops them out a blocksize at a time.
No filtering, no FFT, no main-thread involvement.

Monitoring is independent of acquisition/recording state — the caller
simply chooses a source and an output device, then hits the toggle.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
import sounddevice as sd

from chirp.constants import CHUNK_FRAMES, DTYPE, SAMPLE_RATE


# Default ring-buffer capacity: ~8 chunks (~185 ms at 44.1 kHz). Large
# enough to absorb Windows scheduler jitter, small enough that a stall
# drops recent audio rather than piling up lag.
_DEFAULT_RING_CHUNKS = 8


class _RingBuffer:
    """Thread-safe mono/stereo sample ring buffer.

    Writers push frames with :meth:`write`, the output callback pops
    frames with :meth:`read`. When a write would overflow, the oldest
    samples are discarded so the buffer always holds the most recent
    ``capacity`` frames — this keeps monitor latency bounded when the
    consumer stalls.
    """

    def __init__(self, capacity_frames: int, channels: int):
        self._cap = int(capacity_frames)
        self._channels = int(channels)
        shape = (self._cap, self._channels) if self._channels > 1 else (self._cap,)
        self._buf = np.zeros(shape, dtype=np.float32)
        self._read = 0
        self._write = 0
        self._size = 0
        self._lock = threading.Lock()

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def capacity(self) -> int:
        return self._cap

    def size(self) -> int:
        with self._lock:
            return self._size

    def clear(self) -> None:
        with self._lock:
            self._read = 0
            self._write = 0
            self._size = 0

    def write(self, data: np.ndarray) -> int:
        """Append ``data`` (shape ``(N,)`` or ``(N, C)``). Returns the
        number of samples actually kept after any drop-oldest trim."""
        if data.ndim == 1 and self._channels > 1:
            # Broadcast mono to all channels.
            data = np.repeat(data[:, None], self._channels, axis=1)
        elif data.ndim == 2 and self._channels == 1:
            # Downmix to mono.
            data = data.mean(axis=1)
        elif data.ndim == 2 and data.shape[1] != self._channels:
            # Channel-count mismatch — best effort: truncate or widen.
            if data.shape[1] > self._channels:
                data = data[:, :self._channels]
            else:
                pad = np.repeat(data[:, -1:], self._channels - data.shape[1], axis=1)
                data = np.concatenate([data, pad], axis=1)

        n = int(data.shape[0])
        if n == 0:
            return 0
        if n >= self._cap:
            # Only the most recent `cap` samples fit.
            data = data[-self._cap:]
            n = int(data.shape[0])
        with self._lock:
            overflow = (self._size + n) - self._cap
            if overflow > 0:
                self._read = (self._read + overflow) % self._cap
                self._size -= overflow
            end = self._write + n
            if end <= self._cap:
                self._buf[self._write:end] = data
            else:
                first = self._cap - self._write
                self._buf[self._write:] = data[:first]
                self._buf[:end - self._cap] = data[first:]
            self._write = end % self._cap
            self._size += n
            return n

    def read(self, n: int, out: np.ndarray) -> int:
        """Copy up to ``n`` samples into ``out``; return the count."""
        with self._lock:
            take = min(self._size, int(n))
            if take == 0:
                return 0
            end = self._read + take
            if end <= self._cap:
                out[:take] = self._buf[self._read:end]
            else:
                first = self._cap - self._read
                out[:first] = self._buf[self._read:]
                out[first:take] = self._buf[:end - self._cap]
            self._read = end % self._cap
            self._size -= take
            return take


class AudioMonitor:
    """Global audio-monitor loopback.

    Usage::

        monitor = AudioMonitor()
        monitor.set_output_device(device_id, samplerate=44100, channels=1)
        monitor.set_source(id(entity))     # enable
        # (capture threads call monitor.feed(id(entity), chunk) on every tick)
        monitor.set_source(None)           # disable
        monitor.close()

    ``source_id`` can be any hashable token; Chirp uses ``id(entity)``
    because it is stable for the lifetime of the entity and unique
    across concurrently-live entities.
    """

    def __init__(self):
        self._stream: sd.OutputStream | None = None
        self._device: Any = None
        self._samplerate: int = SAMPLE_RATE
        self._channels: int = 1
        self._ring = _RingBuffer(
            capacity_frames=CHUNK_FRAMES * _DEFAULT_RING_CHUNKS,
            channels=1,
        )
        self._source_id: Any = None
        self._last_error: str | None = None

    # ── Public API ────────────────────────────────────────────────────

    @property
    def output_device(self) -> Any:
        return self._device

    @property
    def source_id(self) -> Any:
        return self._source_id

    @property
    def running(self) -> bool:
        return self._stream is not None

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def samplerate(self) -> int:
        return self._samplerate

    @property
    def channels(self) -> int:
        return self._channels

    def set_output_device(self, device: Any,
                          samplerate: int | None = None,
                          channels: int = 1) -> bool:
        """Open (or reopen) the output stream on ``device``.

        Pass ``device=None`` to disable the monitor entirely (stops the
        stream without opening a new one). Returns True on success,
        False on failure (the error message is stored in
        :attr:`last_error`).
        """
        self._close_stream()
        if device is None:
            self._device = None
            return True
        try:
            sr = int(samplerate or self._samplerate or SAMPLE_RATE)
            ch = max(1, int(channels))
            # Probe the device to clamp channel count to what it supports.
            try:
                info = sd.query_devices(device)
                max_out = int(info.get('max_output_channels', ch))
                if max_out > 0:
                    ch = min(ch, max_out)
            except Exception:
                pass
            self._samplerate = sr
            self._channels = ch
            # Re-size the ring buffer for the new channel count / SR.
            self._ring = _RingBuffer(
                capacity_frames=CHUNK_FRAMES * _DEFAULT_RING_CHUNKS,
                channels=ch,
            )
            self._stream = sd.OutputStream(
                samplerate=sr,
                channels=ch,
                dtype=DTYPE,
                blocksize=CHUNK_FRAMES,
                device=device,
                callback=self._callback,
            )
            self._stream.start()
            self._device = device
            self._last_error = None
            return True
        except Exception as exc:
            self._last_error = str(exc)
            print(f'[AudioMonitor] Failed to open output device {device}: {exc}')
            self._stream = None
            self._device = None
            return False

    def set_source(self, source_id: Any) -> None:
        """Switch which capture is allowed to feed the monitor.

        Flushes any pending samples from the previous source so the
        changeover is crisp rather than playing the old stream's tail.
        """
        if source_id != self._source_id:
            self._source_id = source_id
            self._ring.clear()

    def feed(self, source_id: Any, chunk: np.ndarray) -> None:
        """Called from capture threads — no-op unless this is the active source."""
        if source_id != self._source_id:
            return
        if self._stream is None:
            # No output device selected — drop silently.
            return
        if chunk is None or chunk.size == 0:
            return
        self._ring.write(chunk)

    def close(self) -> None:
        """Stop and release the output stream."""
        self._close_stream()
        self._source_id = None
        self._ring.clear()

    # ── Internals ─────────────────────────────────────────────────────

    def _close_stream(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self._device = None

    def _callback(self, outdata, frames, time_info, status):
        if self._channels == 1:
            # ``outdata`` is (frames, 1); read into a flat view.
            n = self._ring.read(frames, outdata[:, 0])
            if n < frames:
                outdata[n:, 0] = 0.0
        else:
            n = self._ring.read(frames, outdata)
            if n < frames:
                outdata[n:] = 0.0
