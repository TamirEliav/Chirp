"""Chirp — audio subpackage.

Owns the sounddevice-facing layer: `SharedInputStream` (one PortAudio
stream per device, fanned out to every Chirp stream using it),
`AudioCapture` (one stream's sink — ring buffer, clock, monitor wiring,
drop/error stats) and `devices` (enumeration + name matching). Keeping
this separate from `chirp.recording` makes it easy to mock the I/O layer
in tests and keeps PortAudio out of pure-numpy code paths.
"""

from chirp.audio.capture import AudioCapture
from chirp.audio.clock import DisciplinedClock
from chirp.audio.monitor import AudioMonitor
from chirp.audio.shared_stream import SharedInputStream
from chirp.audio.wav_capture import WavFileCapture

__all__ = ["AudioCapture", "AudioMonitor", "DisciplinedClock",
           "SharedInputStream", "WavFileCapture"]
