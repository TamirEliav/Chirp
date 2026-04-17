"""AudioCapture — sounddevice.InputStream wrapper.

Extracted from the monolith in the Phase 1 refactor (plan: c05).
Behavior unchanged: opens an InputStream on construction, routes the
PortAudio callback into a thread-safe queue, and silently drops on
`queue.Full`. The silent-drop behavior is flagged by #13; c15 will
add a drop counter here and a visible badge in the sidebar.
"""

import queue

import sounddevice as sd

from chirp.constants import CHUNK_FRAMES, DTYPE, SAMPLE_RATE


class AudioCapture:
    def __init__(self, audio_queue: queue.Queue, device=None, channels=1,
                 samplerate=SAMPLE_RATE):
        self._queue    = audio_queue
        self._channels = channels
        self._stream   = None
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
            print(f"[AudioCapture] Failed to open device {device}: {exc}")

    def set_monitor(self, monitor, source_id) -> None:
        """Wire the shared audio monitor. Safe to call at any time."""
        self._monitor = monitor
        self._monitor_source_id = source_id

    @property
    def valid(self):
        return self._stream is not None

    def _callback(self, indata, frames, time_info, status):
        # Feed the monitor first — it's the lowest-latency path and
        # doesn't care whether the DSP queue is full.
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
        try:
            if self._channels == 1:
                self._queue.put_nowait(indata[:, 0].copy())
            else:
                self._queue.put_nowait(indata[:, :2].copy())
        except queue.Full:
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
        """
        n = self.drop_count
        self.drop_count = 0
        return n

    def reset_drop_stats(self) -> None:
        """Clear both the transient and persistent drop stats (#29).
        Triggered by the user clicking the sticky drop badge.
        """
        self.drop_count = 0
        self.drop_count_total = 0
        self.has_ever_dropped = False

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
