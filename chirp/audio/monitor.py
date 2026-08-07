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

from typing import Any

import numpy as np
import sounddevice as sd

from chirp.constants import CAPTURE_BLOCKSIZE, CHUNK_FRAMES, DTYPE, SAMPLE_RATE


# Default ring-buffer capacity: ~8 chunks (~185 ms at 44.1 kHz). Large
# enough to absorb Windows scheduler jitter, small enough that a stall
# drops recent audio rather than piling up lag.
_DEFAULT_RING_CHUNKS = 8
# The capture callback feeds the monitor one CAPTURE_BLOCKSIZE burst at a
# time while the output callback drains CHUNK_FRAMES at a time, so the
# ring must comfortably hold several capture bursts or every burst would
# partially overwrite itself before playback caught up.
_RING_FRAMES = max(CHUNK_FRAMES * _DEFAULT_RING_CHUNKS, CAPTURE_BLOCKSIZE * 4)

# ── Jitter buffer ────────────────────────────────────────────────────────
#
# WASAPI *shared* mode delivers capture on the engine's own steady period
# no matter how much latency was requested, so audio trickles in evenly
# and the output callback always finds samples waiting. WASAPI
# *exclusive* and WDM-KS do not: the requested latency becomes the
# device buffer, and the driver hands over that whole buffer at once —
# 0.5 s of audio arriving in a few milliseconds, then nothing for 0.5 s.
# Draining that as it arrives empties the ring before the next burst, so
# the output callback zero-fills the gap and the monitor stutters once
# per burst. That is the "chunky" monitor, and it scales with the
# latency setting exactly as observed in the field.
#
# The cure is the standard one: hold back a target level before playing,
# so playback rides across the gap between bursts. The cost is monitor
# delay equal to that level — unavoidable, since a source that speaks
# only every 0.5 s cannot be monitored continuously on less than 0.5 s
# of buffer. Nothing here touches acquisition or recording: the capture
# ring is 10 s deep and absorbs bursts without noticing.
#
# The target self-tunes rather than being configured: it starts at one
# capture block and DOUBLES on each underrun until playback is
# continuous, so it converges on whatever cadence the driver actually
# uses without the user having to know what the driver does. It is not
# shrunk again while the device stays open — lowering it could not
# reclaim latency already sitting in the ring, and re-priming on purpose
# would create the very gap it exists to avoid.
_PREFILL_MAX_SEC = 1.0
_PREFILL_RING_FACTOR = 3


class _RingBuffer:
    """Lock-free mono/stereo sample ring buffer (SPSC + clear-floor).

    Writers push frames with :meth:`write`, the output callback pops
    frames with :meth:`read`. When a write would overflow, the oldest
    samples are discarded so the buffer always holds the most recent
    ``capacity`` frames — this keeps monitor latency bounded when the
    consumer stalls.

    Realtime safety: the previous implementation took a
    ``threading.Lock`` inside :meth:`write` — i.e. inside the PortAudio
    *input* callback (via ``AudioMonitor.feed``) — and the same lock was
    held by the UI thread in ``clear()``, a textbook priority inversion
    that could block the capture callback and lose samples upstream.
    This version uses the same monotonic-cursor SPSC scheme as
    :class:`chirp.audio.ringbuffer.AudioRing`:

    * the producer (input callback) only writes ``_write_total``;
    * the consumer (output callback) only writes ``_read_total``,
      clamps itself to the resident window, and re-validates after
      copying so a producer lapping it mid-copy yields silence for one
      block instead of torn audio;
    * ``clear()`` (UI thread, on source switch) only writes
      ``_clear_floor`` — a single int assignment, atomic under the GIL —
      which both sides treat as an additional lower bound on the read
      cursor. No thread ever blocks.

    During a source switch two input callbacks can briefly interleave
    writes; worst case is one glitched monitor block — audible only,
    never touching the acquisition/recording path.
    """

    def __init__(self, capacity_frames: int, channels: int):
        self._cap = int(capacity_frames)
        self._channels = int(channels)
        # Always 2-D storage; mono readers index column 0.
        self._buf = np.zeros((self._cap, self._channels), dtype=np.float32)
        self._write_total = 0
        self._read_total = 0
        self._clear_floor = 0

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def capacity(self) -> int:
        return self._cap

    def _floor(self, write_total: int) -> int:
        """Effective read start: consumer cursor, raised by the UI's
        clear-floor and by eviction (resident window)."""
        f = self._read_total
        cf = self._clear_floor
        if cf > f:
            f = cf
        rs = write_total - self._cap
        if rs > f:
            f = rs
        return f

    def size(self) -> int:
        wt = self._write_total
        return max(0, wt - self._floor(wt))

    def clear(self) -> None:
        # UI-thread flush on source switch: raise the floor past all
        # currently-buffered samples. Single int store — no lock.
        self._clear_floor = self._write_total

    def write(self, data: np.ndarray) -> int:
        """Append ``data`` (shape ``(N,)`` or ``(N, C)``). Returns the
        number of samples kept. Runs on the PortAudio input callback —
        no locks; channel adaptation prefers views/broadcasts (the only
        allocation is the small per-chunk downmix mean)."""
        ch = self._channels
        if data.ndim == 1:
            # (n,) → (n,1) view; numpy broadcasts it across all ring
            # channels at assignment time (no np.repeat allocation).
            src = data.reshape(-1, 1)
        elif data.shape[1] == ch:
            src = data
        elif ch == 1:
            # Downmix to mono (small per-chunk allocation).
            src = data.mean(axis=1).reshape(-1, 1)
        elif data.shape[1] > ch:
            src = data[:, :ch]          # truncate — a view
        else:
            src = data[:, :1]           # (n,1) broadcasts to wider rings
        n = int(src.shape[0])
        if n == 0:
            return 0
        cap = self._cap
        new_write = self._write_total + n
        if n >= cap:
            src = src[n - cap:]
            place_start = new_write - cap
            m = cap
        else:
            place_start = self._write_total
            m = n
        head = place_start % cap
        end = head + m
        if end <= cap:
            self._buf[head:end] = src
        else:
            first = cap - head
            self._buf[head:] = src[:first]
            self._buf[:end - cap] = src[first:]
        self._write_total = new_write
        return n

    def read(self, n: int, out: np.ndarray) -> int:
        """Copy up to ``n`` samples into ``out``; return the count.
        Consumer-side (output callback). Clamps to the resident window
        and re-validates after the copy, mirroring ``AudioRing.read``."""
        cap = self._cap
        for _ in range(3):
            wt = self._write_total
            start = self._floor(wt)
            take = min(int(n), wt - start)
            if take <= 0:
                self._read_total = start
                return 0
            head = start % cap
            end = head + take
            if out.ndim == 1:
                col = self._buf[:, 0]
                if end <= cap:
                    out[:take] = col[head:end]
                else:
                    first = cap - head
                    out[:first] = col[head:]
                    out[first:take] = col[:end - cap]
            else:
                if end <= cap:
                    out[:take] = self._buf[head:end]
                else:
                    first = cap - head
                    out[:first] = self._buf[head:]
                    out[first:take] = self._buf[:end - cap]
            # Tear check: valid iff the producer didn't lap the copied
            # region while we copied it.
            if self._write_total - start <= cap:
                self._read_total = start + take
                return take
            self._read_total = self._write_total - cap
        self._read_total = self._write_total - cap
        return 0


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
        # Output gain, 0.0–2.0 (1.0 = unity / 100%). Applied in the
        # output callback; survives device reopen and mute toggles.
        self._gain: float = 1.0
        # Jitter buffer: play only once this many frames are resident,
        # so bursty capture (exclusive / WDM-KS) doesn't stutter.
        self._prefill_floor: int = CAPTURE_BLOCKSIZE
        self._prefill_frames: int = CAPTURE_BLOCKSIZE
        self._prefill_max: int = int(_PREFILL_MAX_SEC * SAMPLE_RATE)
        self._priming: bool = True
        self._underrun_count: int = 0

    # ── Public API ────────────────────────────────────────────────────

    @property
    def gain(self) -> float:
        return self._gain

    def set_gain(self, gain: float) -> None:
        """Set the monitor output gain (clamped to 0.0–2.0; 1.0 =
        unity). A single float store, atomic under the GIL — safe to
        call from the UI thread while the output callback runs."""
        self._gain = float(min(2.0, max(0.0, gain)))

    @property
    def prefill_frames(self) -> int:
        """Frames the jitter buffer currently holds back before playing.
        Grows on underrun; the monitor's delay tracks it."""
        return self._prefill_frames

    @property
    def underrun_count(self) -> int:
        """Times the output callback ran dry since the device opened.
        A handful right after a source switch is the buffer finding the
        driver's cadence; a number that keeps climbing means the capture
        is arriving in bursts longer than ``_PREFILL_MAX_SEC``."""
        return self._underrun_count

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
            # It must hold several times the largest jitter buffer the
            # monitor may grow to, or a burst would overwrite its own
            # tail before playback reached it.
            self._prefill_max = int(_PREFILL_MAX_SEC * sr)
            self._prefill_floor = min(CAPTURE_BLOCKSIZE, self._prefill_max)
            self._prefill_frames = self._prefill_floor
            self._priming = True
            self._underrun_count = 0
            self._ring = _RingBuffer(
                capacity_frames=max(_RING_FRAMES,
                                    _PREFILL_RING_FACTOR * self._prefill_max),
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
            # The ring is empty again — re-prime rather than play the
            # first arriving block into an empty buffer.
            self._priming = True

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
        self._priming = True

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
        # Jitter buffer. While priming, output silence until the target
        # level is resident — playing early is what makes a bursty
        # capture stutter once per burst.
        if self._priming:
            if self._ring.size() < self._prefill_frames + frames:
                outdata[:] = 0.0
                return
            self._priming = False
        if self._channels == 1:
            # ``outdata`` is (frames, 1); read into a flat view.
            n = self._ring.read(frames, outdata[:, 0])
            if n < frames:
                outdata[n:, 0] = 0.0
        else:
            n = self._ring.read(frames, outdata)
            if n < frames:
                outdata[n:] = 0.0
        if n < frames:
            # Ran dry: the source delivers in bursts longer than the
            # current target. Double it (once per underrun, capped) and
            # re-prime, so the stutter converges to a single gap instead
            # of repeating on every burst.
            self._underrun_count += 1
            self._prefill_frames = min(self._prefill_max,
                                       max(self._prefill_floor,
                                           self._prefill_frames * 2))
            self._priming = True
        # Output gain (0–200%). In-place scale on the filled region;
        # boosted output is clipped to full scale so a >100% gain can't
        # wrap when the driver converts to fixed point.
        g = self._gain
        if g != 1.0 and n > 0:
            outdata[:n] *= g
            if g > 1.0:
                np.clip(outdata[:n], -1.0, 1.0, out=outdata[:n])
