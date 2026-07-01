"""AudioRing — preallocated single-producer / single-consumer circular
buffer for raw audio frames.

This replaces the per-chunk ``queue.Queue`` of freshly-copied numpy
arrays used by the pre-redesign data path. The PortAudio callback (the
single producer) does nothing but copy incoming frames into preallocated
memory and advance an integer write cursor — no per-chunk object
allocation, no locks, no logging — so it can never stall the realtime
thread. The per-stream DSP thread (the single consumer) drains
newly-written frames in batches and can also pull an arbitrary
still-resident absolute range for pre-trigger lookback.

Design
------
* Coordinates are **absolute frame counts** since construction
  (``write_total`` / ``read_total``); the physical slot for absolute
  index ``i`` is ``i % capacity``.
* Exactly one producer thread calls :meth:`write`; exactly one consumer
  thread calls :meth:`read` / :meth:`read_range`. CPython's GIL makes the
  single-writer/single-reader integer cursor updates atomic, so the hot
  path needs no lock.
* Sizing is decoupled from both the display window and ``max_rec``: the
  ring only has to hold the pre-trigger lookback plus enough slack to
  absorb consumer jitter. With a multi-second ring an overrun means the
  consumer stalled for seconds — it should never happen in practice, but
  it is counted and surfaced if it does (overwrite-oldest policy).
"""

import numpy as np


class AudioRing:
    def __init__(self, capacity_frames: int, channels: int = 1,
                 dtype=np.float32):
        if capacity_frames < 1:
            raise ValueError('capacity_frames must be >= 1')
        if channels < 1:
            raise ValueError('channels must be >= 1')
        self._cap = int(capacity_frames)
        self._channels = int(channels)
        self._dtype = dtype
        self._buf = np.zeros((self._cap, self._channels), dtype=dtype)
        # Absolute monotonic frame cursors.
        self._write_total = 0
        self._read_total = 0
        # Overrun stats — mirror the capture drop-stat contract so the
        # sidebar badge logic is uniform across capture types.
        self.overrun_count = 0          # transient (per UI tick)
        self.overrun_count_total = 0    # session-wide monotonic
        self.has_ever_overrun = False   # sticky latch
        self.dropped_frames_total = 0   # total frames lost to overrun
        self._empty = (np.zeros(0, dtype=dtype) if channels == 1
                       else np.zeros((0, channels), dtype=dtype))

    # ── Introspection ────────────────────────────────────────────────
    @property
    def capacity(self) -> int:
        return self._cap

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def write_total(self) -> int:
        return self._write_total

    @property
    def read_total(self) -> int:
        return self._read_total

    @property
    def available(self) -> int:
        """Frames written but not yet consumed by :meth:`read`."""
        return self._write_total - self._read_total

    # ── Producer side (audio callback) ───────────────────────────────
    def write(self, data: np.ndarray) -> int:
        """Append ``data`` to the ring. Returns the number of frames
        written. Realtime-safe: only array copies into preallocated
        memory and integer arithmetic — no allocation, no locks.

        ``data`` may be 1-D ``(n,)`` (mono) or 2-D ``(n, channels)``.
        On overrun (consumer fell more than ``capacity`` frames behind)
        the oldest unread frames are overwritten and counted.
        """
        n = data.shape[0]
        if n == 0:
            return 0
        cap = self._cap
        new_write = self._write_total + n

        # Overrun: if the new data would overwrite frames the consumer
        # has not read, advance the read cursor past the lost region and
        # account for it. The resident window is always the last ``cap``
        # frames ending at ``new_write``.
        min_read = new_write - cap
        if min_read > self._read_total:
            lost = min_read - self._read_total
            self._read_total = min_read
            self.dropped_frames_total += int(lost)
            self.overrun_count += 1
            self.overrun_count_total += 1
            self.has_ever_overrun = True

        src = data.reshape(n, 1) if data.ndim == 1 else data
        if n >= cap:
            # The write is at least a full buffer; only the final ``cap``
            # frames survive. Place them aligned to absolute addressing.
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

    # ── Consumer side (DSP thread) ───────────────────────────────────
    def read(self, max_frames: int | None = None):
        """Return ``(start_abs, frames)`` for up to ``max_frames`` of the
        oldest unread data and advance the read cursor past it.

        ``frames`` is a fresh copy shaped ``(m,)`` for mono or
        ``(m, channels)`` for multi-channel; ``start_abs`` is the
        absolute frame index of its first sample. When nothing is
        available, ``frames`` is empty.
        """
        avail = self._write_total - self._read_total
        if avail <= 0:
            return self._read_total, self._empty
        m = avail if max_frames is None else min(avail, int(max_frames))
        start = self._read_total
        cap = self._cap
        head = start % cap
        end = head + m
        if end <= cap:
            out = self._buf[head:end].copy()
        else:
            first = cap - head
            out = np.concatenate([self._buf[head:], self._buf[:end - cap]])
        self._read_total = start + m
        if self._channels == 1:
            out = out[:, 0]
        return start, out

    def read_range(self, start_abs: int, end_abs: int) -> np.ndarray:
        """Return a copy of the absolute range ``[start_abs, end_abs)``,
        clipped to what is still resident in the ring. Does NOT advance
        the read cursor — used to pull pre-trigger lookback at event
        onset. Returns empty if the range is fully evicted.
        """
        resident_start = max(0, self._write_total - self._cap)
        s = max(int(start_abs), resident_start)
        e = min(int(end_abs), self._write_total)
        if e <= s:
            return self._empty
        cap = self._cap
        head = s % cap
        m = e - s
        end = head + m
        if end <= cap:
            out = self._buf[head:end].copy()
        else:
            first = cap - head
            out = np.concatenate([self._buf[head:], self._buf[:end - cap]])
        if self._channels == 1:
            out = out[:, 0]
        return out

    def drain_unread(self) -> int:
        """Discard all currently-unread frames (advance the read cursor to
        the write cursor). Used on stop/teardown to drop stragglers so a
        restart doesn't replay stale audio. Returns frames discarded."""
        n = self._write_total - self._read_total
        self._read_total = self._write_total
        return n

    # ── Stats helpers (mirror AudioCapture drop-stat contract) ───────
    def consume_overrun_count(self) -> int:
        n = self.overrun_count
        self.overrun_count = 0
        return n

    def reset_overrun_stats(self) -> None:
        self.overrun_count = 0
        self.overrun_count_total = 0
        self.has_ever_overrun = False
        self.dropped_frames_total = 0
