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
        try:
            self._stream = sd.InputStream(
                samplerate=samplerate, channels=channels,
                dtype=DTYPE, blocksize=CHUNK_FRAMES,
                device=device,
                callback=self._callback,
            )
        except Exception as exc:
            print(f"[AudioCapture] Failed to open device {device}: {exc}")

    @property
    def valid(self):
        return self._stream is not None

    def _callback(self, indata, frames, time_info, status):
        try:
            if self._channels == 1:
                self._queue.put_nowait(indata[:, 0].copy())
            else:
                self._queue.put_nowait(indata[:, :2].copy())
        except queue.Full:
            pass

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
