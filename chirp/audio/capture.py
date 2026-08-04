"""AudioCapture — per-stream view of a (possibly shared) input stream.

The PortAudio callback writes raw frames into a preallocated
single-producer/single-consumer ring buffer
(:class:`chirp.audio.ringbuffer.AudioRing`) instead of allocating a
fresh numpy array per chunk and pushing it onto a ``queue.Queue``. The
callback does only a memcpy into preallocated memory plus integer
arithmetic — no allocation, no locks, no disk I/O — so it can never
stall the realtime thread. Drop accounting maps to ring *overruns*
(consumer fell behind by more than the ring's capacity), which with a
multi-second ring should never happen in practice. Logging is emitted
off the realtime path, when the UI polls ``consume_drop_count`` /
``consume_os_drop_count``.

Stream sharing: an AudioCapture no longer owns a PortAudio stream. It is
a *sink* of a :class:`~chirp.audio.shared_stream.SharedInputStream`,
which opens exactly one ``sd.InputStream`` per (device, samplerate) and
fans each buffer out to every attached sink. Two Chirp streams splitting
one stereo input therefore present a single client to the Windows
capture session — see ``chirp/audio/shared_stream.py`` for the field
evidence that made this necessary. Everything else about the contract is
unchanged: ``valid`` / ``open_error`` / ``resume`` / ``pause`` /
``close`` / ``set_monitor`` / the drop, overflow and underflow counters
all behave exactly as before, and ``_callback`` keeps its signature so
it can still be driven directly in tests.
"""

import time

from chirp.audio import shared_stream as _shared
from chirp.audio.ringbuffer import AudioRing
from chirp.constants import SAMPLE_RATE
from chirp.error_log import log as _err_log


class AudioCapture:
    def __init__(self, audio_ring: AudioRing, device=None, channels=1,
                 samplerate=SAMPLE_RATE, name: str = '', clock=None):
        self._ring     = audio_ring
        self._channels = channels
        # Shared PortAudio stream this capture is a sink of (None until
        # attached / after close). ``_active`` gates the fan-out: a
        # paused sink stays attached — keeping the endpoint open for its
        # siblings — but receives no buffers, so its ring doesn't fill.
        self._shared   = None
        self._active   = False
        # Disciplined timestamp clock (chirp.audio.clock). When wired,
        # the callback records one (write_total, time.time()) pair per
        # buffer — the raw observations the ingest thread's clock servo
        # filters into filename timestamps. Optional so tests and the
        # WAV-playback capture (whose pacing is synthetic) skip it.
        self._clock    = clock
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
        # Zero-insertion detection: PortAudio's ``input_underflow``
        # status flag means zero samples were INSERTED into ``indata``
        # to compensate for missing capture data (portaudio.h, fixed
        # blocksize: "one or more zero samples have been inserted").
        # Distinct from ``input_overflow`` (data discarded): underflow
        # corrupts the audio *content* with silent zero runs that flow
        # into the spectrogram, the monitor, and every recorded WAV.
        # Previously ignored entirely — periodic zero runs in long
        # triggered recordings were invisible to every badge and log.
        self.underflow_count       = 0     # transient per-tick
        self.underflow_count_total = 0     # session-wide monotonic
        self.has_ever_underflowed  = False
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
        # Which input column feeds the monitor (honors channel_mode):
        # 0 = left/mono, 1 = right, None = both (stereo). Without this a
        # 'Right' stream opened with 2 channels would feed both columns.
        self._monitor_channel = None
        # Attach to the shared stream for this (device, samplerate),
        # opening one if this is the first stream on the endpoint. The
        # open-failure contract is unchanged: ``valid`` False +
        # ``open_error`` set + one ``open`` log line.
        shared, err = _shared.acquire(self, device, samplerate, channels,
                                      name=name)
        if err is not None:
            self.open_error = err[:200]
        else:
            self._shared = shared

    # ── Shared-stream plumbing ───────────────────────────────────────

    def mark_stream_dead(self) -> None:
        """Declare the underlying shared stream unusable so the next
        capture built for this device opens a fresh one.

        Called from the capture-recovery path: when a device stalls,
        every sink on that endpoint is stalled, so the whole shared
        stream has to go rather than being inherited by the replacement
        captures (which would silently reuse the dead session)."""
        shared = self._shared
        if shared is not None:
            shared.mark_dead()

    def set_monitor(self, monitor, source_id, channel=None) -> None:
        """Wire the shared audio monitor. Safe to call at any time.

        ``channel`` selects which input column the monitor loopback
        plays so it matches the stream's ``channel_mode``: 0 = left /
        mono, 1 = right, None = feed both columns (stereo). A 'Right'
        stream is opened with 2 input channels, so without this the
        callback would feed both columns and the mono monitor would
        average the left channel back in — audibly a *different* stream
        when two streams split one stereo input device.
        """
        self._monitor = monitor
        self._monitor_source_id = source_id
        self._monitor_channel = channel

    @property
    def valid(self):
        shared = self._shared
        return shared is not None and shared.stream is not None

    def _callback(self, indata, frames, time_info, status):
        # Invoked by SharedInputStream's fan-out, on the PortAudio
        # thread, with ``indata`` already sliced to this sink's channel
        # count. Realtime-safety: must not block. It does *only* counter
        # increments and an array copy into the ring — no disk I/O and no
        # logging. Drop / overflow events are turned into log lines off
        # the realtime path, when the UI polls ``consume_drop_count`` /
        # ``consume_os_drop_count``. The status flags are shared by every
        # sink on the endpoint (they describe the endpoint, not us), so
        # each sink counts them independently.
        #
        # #43: ``input_overflow`` means the driver's input ring buffer
        # wrapped before we serviced it — samples were lost upstream of
        # our queue. A separate failure mode from our own queue.Full.
        if status is not None:
            try:
                overflow  = bool(getattr(status, 'input_overflow', False))
                underflow = bool(getattr(status, 'input_underflow', False))
            except Exception:
                overflow = underflow = False
            if overflow:
                self.os_drop_count       += 1
                self.os_drop_count_total += 1
                self.has_ever_os_dropped  = True
            # ``input_underflow``: PortAudio inserted zero samples into
            # this buffer (missing capture data zero-filled in place).
            # Counter increments only — logging happens off the
            # realtime path in ``consume_underflow_count``.
            if underflow:
                self.underflow_count       += 1
                self.underflow_count_total += 1
                self.has_ever_underflowed   = True
        # Feed the monitor first — it's the lowest-latency path and
        # doesn't care whether the DSP ring is full.
        mon = self._monitor
        if mon is not None:
            try:
                if self._channels == 1:
                    mon.feed(self._monitor_source_id, indata[:, 0])
                else:
                    chan = self._monitor_channel
                    if chan == 1:
                        mon.feed(self._monitor_source_id, indata[:, 1])
                    elif chan == 0:
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
        # Timestamp-clock observation: the only point where a sample
        # index and the wall clock meet with no queue backlog between
        # them. A bounded-deque append — realtime-safe.
        clk = self._clock
        if clk is not None:
            clk.observe(self._ring.write_total, time.time())

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

    def consume_underflow_count(self) -> int:
        """Return and clear the transient input-underflow counter.
        Polled once per UI tick alongside ``consume_os_drop_count``.
        Does NOT touch the sticky session stats. Emits a throttled log
        line here (off the realtime callback) when new underflows
        appeared — each flagged buffer contained zero samples PortAudio
        inserted to cover missing capture data, i.e. silent zero runs
        are being recorded into the audio content itself.
        """
        n = self.underflow_count
        self.underflow_count = 0
        if n:
            _err_log('underflow', self._name,
                     f'PortAudio input_underflow x{n} — zero samples '
                     f'inserted into captured audio '
                     f'(cumulative={self.underflow_count_total})')
        return n

    def reset_error_stats(self) -> None:
        """#43 / #48: clear OS-drop + underflow stats and any cached
        open-error message. Triggered by the user clicking the sticky
        error badge. Kept separate from ``reset_drop_stats`` because
        the two have distinct badges.
        """
        self.os_drop_count       = 0
        self.os_drop_count_total = 0
        self.has_ever_os_dropped = False
        self.underflow_count       = 0
        self.underflow_count_total = 0
        self.has_ever_underflowed  = False
        self.open_error          = None

    def resume(self):
        """Begin receiving buffers. Starts the shared PortAudio stream
        if this is the first active sink on the endpoint."""
        shared = self._shared
        if shared is None:
            return
        self._active = True
        try:
            shared.start()
        except Exception:
            # Roll back so a failed start doesn't leave a sink marked
            # active on a stopped stream; the caller (start_acq) latches
            # the failure and may rebuild the capture.
            self._active = False
            raise

    def pause(self):
        """Stop receiving buffers. The shared stream keeps running while
        any sibling sink is still active; it stops when none are."""
        shared = self._shared
        self._active = False
        if shared is not None:
            shared.stop_if_idle()

    def close(self):
        """Detach from the shared stream. The PortAudio stream — and
        with it the OS capture session — is closed once the last sink on
        the endpoint detaches. Idempotent."""
        shared, self._shared = self._shared, None
        self._active = False
        if shared is not None:
            shared.detach(self)
