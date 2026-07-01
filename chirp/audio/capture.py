"""AudioCapture — sounddevice.InputStream wrapper.

Redesign: the PortAudio callback now writes raw frames into a
preallocated single-producer/single-consumer ring buffer
(:class:`chirp.audio.ringbuffer.AudioRing`) instead of allocating a
fresh numpy array per chunk and pushing it onto a ``queue.Queue``. The
callback does only a memcpy into preallocated memory plus integer
arithmetic — no allocation, no locks, no disk I/O — so it can never
stall the realtime thread. Drop accounting maps to ring *overruns*
(consumer fell behind by more than the ring's capacity), which with a
multi-second ring should never happen in practice. Logging is emitted
off the realtime path, when the UI polls ``consume_drop_count`` /
``consume_os_drop_count``.
"""

import sounddevice as sd

from chirp.audio.ringbuffer import AudioRing
from chirp.constants import CHUNK_FRAMES, DTYPE, SAMPLE_RATE
from chirp.error_log import log as _err_log


class AudioCapture:
    def __init__(self, audio_ring: AudioRing, device=None, channels=1,
                 samplerate=SAMPLE_RATE, name: str = ''):
        self._ring     = audio_ring
        self._channels = channels
        self._stream   = None
        # Stream label included in error-log entries so the user can
        # tell which entity dropped chunks / overflowed.
        self._name     = name
        # #13 / c15: count of audio chunks the PortAudio callback had
        # to drop because the queue was full. The UI samples this on
        # each plot tick to surface a drop-indicator badge in the
        # sidebar so silent loss is no longer invisible.
        self.drop_count = 0
        # #29: session-wide persistent stats. `drop_count_total` only
        # increases; `has_ever_dropped` latches True on the first drop
        # and can only be cleared by `reset_drop_stats()`. The sidebar
        # uses these to keep a sticky "drops happened at some point"
        # badge visible until the user explicitly clears it.
        self.drop_count_total = 0
        self.has_ever_dropped = False
        # #43: PortAudio-level drops. When the audio interface or the
        # OS input ring buffer loses samples between the driver and our
        # callback, sounddevice raises the ``input_overflow`` flag on
        # the ``status`` argument. The previous implementation ignored
        # ``status`` entirely, so upstream dropouts were completely
        # invisible. These counters feed the sidebar `!` error badge.
        self.os_drop_count        = 0      # transient per-tick
        self.os_drop_count_total  = 0      # session-wide monotonic
        self.has_ever_os_dropped  = False
        # #48: stream-open failure reason. The constructor used to
        # swallow the exception with a print() call — in a GUI build
        # nothing reaches the user. Callers can now check ``valid`` and
        # read ``open_error`` to surface a message.
        self.open_error: str | None = None
        # #7: optional monitor loopback. When wired by the owning
        # RecordingEntity, the callback also forwards raw samples to
        # the shared AudioMonitor — the monitor itself gates on
        # source_id so only the selected stream is actually played.
        self._monitor = None
        self._monitor_source_id = None
        try:
            self._stream = sd.InputStream(
                samplerate=samplerate, channels=channels,
                dtype=DTYPE, blocksize=CHUNK_FRAMES,
                device=device,
                callback=self._callback,
            )
        except Exception as exc:
            self.open_error = f'{type(exc).__name__}: {exc}'[:200]
            print(f"[AudioCapture] Failed to open device {device}: {exc}")
            _err_log('open', self._name,
                     f'failed to open device {device}: '
                     f'{type(exc).__name__}: {exc}')

    def set_monitor(self, monitor, source_id) -> None:
        """Wire the shared audio monitor. Safe to call at any time."""
        self._monitor = monitor
        self._monitor_source_id = source_id

    @property
    def valid(self):
        return self._stream is not None

    def _callback(self, indata, frames, time_info, status):
        # Realtime-safety: this runs on the PortAudio thread and must not
        # block. It does *only* counter increments and an array copy into
        # the queue — no disk I/O and no logging. Drop / overflow events
        # are turned into log lines off the realtime path, when the UI
        # polls ``consume_drop_count`` / ``consume_os_drop_count``.
        #
        # #43: ``input_overflow`` means the driver's input ring buffer
        # wrapped before we serviced it — samples were lost upstream of
        # our queue. A separate failure mode from our own queue.Full.
        if status is not None:
            try:
                overflow = bool(getattr(status, 'input_overflow', False))
            except Exception:
                overflow = False
            if overflow:
                self.os_drop_count       += 1
                self.os_drop_count_total += 1
                self.has_ever_os_dropped  = True
        # Feed the monitor first — it's the lowest-latency path and
        # doesn't care whether the DSP ring is full.
        mon = self._monitor
        if mon is not None:
            try:
                if self._channels == 1:
                    mon.feed(self._monitor_source_id, indata[:, 0])
                else:
                    mon.feed(self._monitor_source_id, indata[:, :2])
            except Exception:
                # Monitor must never break acquisition.
                pass
        # Write into the ring (memcpy into preallocated memory — no
        # allocation, no lock). An overrun (consumer fell more than the
        # ring's capacity behind) overwrites the oldest unread frames;
        # mirror that into the capture's drop stats for the sidebar.
        before = self._ring.overrun_count_total
        if self._channels == 1:
            self._ring.write(indata[:, 0])
        else:
            self._ring.write(indata[:, :2])
        if self._ring.overrun_count_total > before:
            self.drop_count += 1
            self.drop_count_total += 1
            self.has_ever_dropped = True

    def consume_drop_count(self) -> int:
        """Return the drop count and reset it to zero. Intended to be
        polled once per UI tick — the sidebar latches a transient
        drop indicator whenever this returns > 0.

        Does NOT touch ``drop_count_total`` / ``has_ever_dropped`` —
        those are the sticky session stats, cleared only by
        ``reset_drop_stats()``.

        Logging happens here (off the realtime callback) rather than in
        ``_callback``: when this poll observes new drops it emits one
        throttled log line so the realtime thread never touches disk.
        """
        n = self.drop_count
        self.drop_count = 0
        if n:
            _err_log('ring_overrun', self._name,
                     f'capture ring overrun — {n} block(s) overwrote unread '
                     f'audio (cumulative={self.drop_count_total})')
        return n

    def reset_drop_stats(self) -> None:
        """Clear both the transient and persistent drop stats (#29).
        Triggered by the user clicking the sticky drop badge.
        """
        self.drop_count = 0
        self.drop_count_total = 0
        self.has_ever_dropped = False

    def consume_os_drop_count(self) -> int:
        """#43: return and clear the transient OS-level drop counter.
        Polled once per UI tick so the sidebar error badge can flash.
        Does NOT touch the sticky session stats. Emits a throttled log
        line here (off the realtime callback) when new overflows appear.
        """
        n = self.os_drop_count
        self.os_drop_count = 0
        if n:
            _err_log('os_drop', self._name,
                     f'PortAudio input_overflow x{n} '
                     f'(cumulative={self.os_drop_count_total})')
        return n

    def reset_error_stats(self) -> None:
        """#43 / #48: clear OS-drop stats and any cached open-error
        message. Triggered by the user clicking the sticky error
        badge. Kept separate from ``reset_drop_stats`` because the
        two have distinct badges.
        """
        self.os_drop_count       = 0
        self.os_drop_count_total = 0
        self.has_ever_os_dropped = False
        self.open_error          = None

    def resume(self):
        if self._stream is not None:
            self._stream.start()

    def pause(self):
        if self._stream is not None:
            self._stream.stop()

    def close(self):
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:
                pass
            self._stream.close()
            self._stream = None
