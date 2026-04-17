"""Chirp — audio subpackage.

Owns the sounddevice-facing layer: `AudioCapture` (the InputStream
wrapper) and `devices` (enumeration + name matching). Keeping this
separate from `chirp.recording` makes it easy to mock the I/O layer
in tests and keeps PortAudio out of pure-numpy code paths.
"""

from chirp.audio.capture import AudioCapture
from chirp.audio.monitor import AudioMonitor
from chirp.audio.wav_capture import WavFileCapture

__all__ = ["AudioCapture", "AudioMonitor", "WavFileCapture"]
