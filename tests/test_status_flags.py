"""PortAudio callback status-flag instrumentation.

``input_underflow`` means PortAudio INSERTED zero samples into the
capture buffer to compensate for missing data (fixed-blocksize
semantics, portaudio.h) — silent zero runs flow into the spectrogram,
the monitor, and every recorded WAV. The callback previously inspected
only ``input_overflow``, so zero-insertion was invisible to every badge
and log (the "periodic zero runs in long recordings" bug). These tests
pin the new counter contract:

* ``input_underflow`` increments the transient + total counters and
  latches the sticky flag; audio in the same buffer still reaches the
  ring (the zeros are real captured content as far as the pipeline is
  concerned — the point is to *know*, not to drop).
* ``consume_underflow_count`` drains the transient counter only.
* ``reset_error_stats`` clears the underflow stats (badge-click path).
* ``input_overflow`` and ``input_underflow`` count independently.
* WavFileCapture mirrors the fields/methods as inert zeros.
"""

import numpy as np

from chirp.audio.capture import AudioCapture
from chirp.audio.ringbuffer import AudioRing


class _Status:
    """Stand-in for sounddevice.CallbackFlags."""

    def __init__(self, underflow=False, overflow=False):
        self.input_underflow = underflow
        self.input_overflow = overflow


def _make_capture():
    ring = AudioRing(8192, channels=1)
    # Bogus device id → InputStream never opens (open_error latches),
    # but the callback and counters are fully exercisable.
    cap = AudioCapture(ring, device=-12345, channels=1,
                       samplerate=44100, name='ut')
    assert cap.open_error is not None
    return ring, cap


def _chunk(n=256):
    return np.ones((n, 1), dtype=np.float32)


def test_underflow_flag_counts_and_latches():
    ring, cap = _make_capture()
    cap._callback(_chunk(), 256, None, _Status(underflow=True))
    assert cap.underflow_count == 1
    assert cap.underflow_count_total == 1
    assert cap.has_ever_underflowed is True
    # Overflow stats untouched.
    assert cap.os_drop_count == 0
    assert cap.has_ever_os_dropped is False
    # The buffer's audio still reached the ring.
    assert ring.write_total == 256


def test_no_status_counts_nothing():
    ring, cap = _make_capture()
    cap._callback(_chunk(), 256, None, None)
    cap._callback(_chunk(), 256, None, _Status())
    assert cap.underflow_count == 0
    assert cap.underflow_count_total == 0
    assert cap.has_ever_underflowed is False
    assert ring.write_total == 512


def test_consume_drains_transient_keeps_totals():
    _, cap = _make_capture()
    for _ in range(3):
        cap._callback(_chunk(), 256, None, _Status(underflow=True))
    assert cap.consume_underflow_count() == 3
    assert cap.underflow_count == 0            # transient drained
    assert cap.underflow_count_total == 3      # sticky total kept
    assert cap.has_ever_underflowed is True
    assert cap.consume_underflow_count() == 0  # idempotent when empty


def test_overflow_and_underflow_independent():
    _, cap = _make_capture()
    cap._callback(_chunk(), 256, None,
                  _Status(underflow=True, overflow=True))
    cap._callback(_chunk(), 256, None, _Status(overflow=True))
    assert cap.underflow_count_total == 1
    assert cap.os_drop_count_total == 2


def test_reset_error_stats_clears_underflow():
    _, cap = _make_capture()
    cap._callback(_chunk(), 256, None, _Status(underflow=True))
    cap.reset_error_stats()
    assert cap.underflow_count == 0
    assert cap.underflow_count_total == 0
    assert cap.has_ever_underflowed is False


def test_wav_capture_mirror_is_inert(tmp_path):
    import scipy.io.wavfile

    from chirp.audio.wav_capture import WavFileCapture

    wav = tmp_path / 't.wav'
    scipy.io.wavfile.write(
        str(wav), 44100, np.zeros(1024, dtype=np.int16))
    ring = AudioRing(8192, channels=1)
    cap = WavFileCapture(ring, str(wav), channels=1)
    assert cap.underflow_count == 0
    assert cap.has_ever_underflowed is False
    assert cap.consume_underflow_count() == 0
    cap.reset_error_stats()
    assert cap.underflow_count_total == 0
