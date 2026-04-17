"""Tests for the audio-monitor loopback (#7).

Pins the contract of ``AudioMonitor`` and its integration with
``AudioCapture`` / ``WavFileCapture`` / ``RecordingEntity``:

* Source-ID gating: only the selected source's chunks reach the ring
  buffer, everything else is dropped silently.
* Switching the source flushes stale samples so the changeover is
  crisp rather than playing the previous stream's tail.
* Ring buffer drops oldest on overflow so latency stays bounded when
  the consumer stalls.
* Channel-shape coercion (mono→stereo, stereo→mono, widen/narrow).
* ``AudioCapture.set_monitor`` forwards raw chunks to the monitor on
  every PortAudio callback tick, without disturbing the normal queue.
* ``WavFileCapture`` forwards chunks through the same hook.
* ``RecordingEntity.set_monitor`` persists across ``_make_capture``
  rebuilds so a SR / device / WAV change doesn't drop the loopback.

Tests avoid opening real output devices — ``set_output_device`` is
never called with a real ID, so the ring buffer is exercised directly
through ``feed`` / internal ``_ring``.
"""

from __future__ import annotations

import queue
import time

import numpy as np
import pytest
import scipy.io.wavfile

from chirp.audio.capture import AudioCapture
from chirp.audio.monitor import AudioMonitor, _RingBuffer
from chirp.audio.wav_capture import WavFileCapture
from chirp.constants import CHUNK_FRAMES
from chirp.recording.entity import RecordingEntity


# ── _RingBuffer -----------------------------------------------------------

def test_ring_buffer_basic_write_read_mono():
    rb = _RingBuffer(capacity_frames=8, channels=1)
    rb.write(np.array([1, 2, 3, 4], dtype=np.float32))
    out = np.zeros(4, dtype=np.float32)
    n = rb.read(4, out)
    assert n == 4
    np.testing.assert_array_equal(out, [1, 2, 3, 4])
    assert rb.size() == 0


def test_ring_buffer_overflow_drops_oldest():
    rb = _RingBuffer(capacity_frames=4, channels=1)
    rb.write(np.array([1, 2, 3, 4], dtype=np.float32))
    rb.write(np.array([5, 6], dtype=np.float32))  # overflow by 2
    out = np.zeros(4, dtype=np.float32)
    n = rb.read(4, out)
    assert n == 4
    # Oldest two (1, 2) are dropped; most-recent 4 remain: 3,4,5,6
    np.testing.assert_array_equal(out, [3, 4, 5, 6])


def test_ring_buffer_huge_write_truncates_to_capacity():
    rb = _RingBuffer(capacity_frames=4, channels=1)
    rb.write(np.arange(20, dtype=np.float32))
    out = np.zeros(4, dtype=np.float32)
    n = rb.read(4, out)
    assert n == 4
    np.testing.assert_array_equal(out, [16, 17, 18, 19])


def test_ring_buffer_read_underrun_returns_available():
    rb = _RingBuffer(capacity_frames=8, channels=1)
    rb.write(np.array([1, 2], dtype=np.float32))
    out = np.zeros(4, dtype=np.float32)
    n = rb.read(4, out)
    assert n == 2
    np.testing.assert_array_equal(out[:2], [1, 2])


def test_ring_buffer_clear_resets_size():
    rb = _RingBuffer(capacity_frames=8, channels=1)
    rb.write(np.ones(4, dtype=np.float32))
    assert rb.size() == 4
    rb.clear()
    assert rb.size() == 0
    out = np.zeros(4, dtype=np.float32)
    assert rb.read(4, out) == 0


def test_ring_buffer_mono_broadcasts_to_stereo():
    rb = _RingBuffer(capacity_frames=4, channels=2)
    rb.write(np.array([1, 2, 3, 4], dtype=np.float32))
    out = np.zeros((4, 2), dtype=np.float32)
    n = rb.read(4, out)
    assert n == 4
    np.testing.assert_array_equal(out[:, 0], [1, 2, 3, 4])
    np.testing.assert_array_equal(out[:, 1], [1, 2, 3, 4])


def test_ring_buffer_stereo_downmixes_to_mono():
    rb = _RingBuffer(capacity_frames=4, channels=1)
    rb.write(np.array([[1, 3], [2, 4]], dtype=np.float32))
    out = np.zeros(2, dtype=np.float32)
    n = rb.read(2, out)
    assert n == 2
    np.testing.assert_array_equal(out, [2.0, 3.0])  # means of (1,3), (2,4)


# ── AudioMonitor source gating -------------------------------------------

def test_monitor_feed_drops_when_no_source_selected():
    m = AudioMonitor()
    # No output device, no source → feed is a silent no-op.
    m.feed(123, np.ones(16, dtype=np.float32))
    assert m._ring.size() == 0


def test_monitor_feed_drops_when_stream_closed_even_if_source_set():
    m = AudioMonitor()
    m.set_source(42)
    # Stream not open → early return, nothing buffered.
    m.feed(42, np.ones(16, dtype=np.float32))
    assert m._ring.size() == 0


def test_monitor_feed_accepts_only_matching_source():
    m = AudioMonitor()
    # Simulate an open stream by poking the internal field so feed
    # reaches the ring buffer (tests don't open real audio devices).
    m._stream = object()
    m.set_source(7)
    m.feed(7, np.ones(16, dtype=np.float32))
    assert m._ring.size() == 16
    m.feed(99, np.ones(32, dtype=np.float32))  # wrong source → ignored
    assert m._ring.size() == 16


def test_monitor_set_source_flushes_buffer():
    m = AudioMonitor()
    m._stream = object()
    m.set_source(1)
    m.feed(1, np.ones(16, dtype=np.float32))
    assert m._ring.size() == 16
    m.set_source(2)
    assert m._ring.size() == 0  # switchover flushes


def test_monitor_set_source_none_disables():
    m = AudioMonitor()
    m._stream = object()
    m.set_source(1)
    m.feed(1, np.ones(16, dtype=np.float32))
    m.set_source(None)
    m.feed(1, np.ones(16, dtype=np.float32))
    assert m._ring.size() == 0


def test_monitor_close_clears_source_and_buffer():
    m = AudioMonitor()
    m._stream = object()
    m.set_source(1)
    m.feed(1, np.ones(16, dtype=np.float32))
    m.close()
    assert m.source_id is None
    assert m._ring.size() == 0
    assert not m.running


# ── AudioCapture / WavFileCapture wiring ---------------------------------

def test_audio_capture_forwards_to_monitor_via_set_monitor():
    """AudioCapture._callback should call monitor.feed() even when the
    device failed to open (test never opens a real InputStream)."""
    m = AudioMonitor()
    m._stream = object()
    m.set_source('cap-a')
    q: queue.Queue = queue.Queue(maxsize=10)
    cap = AudioCapture(q, device=None, channels=1, samplerate=44100)
    cap.set_monitor(m, 'cap-a')
    # Drive the callback manually with a synthetic indata matrix.
    frames = 1024
    indata = np.full((frames, 1), 0.25, dtype=np.float32)
    cap._callback(indata, frames, None, None)
    # Queue receives a copy, monitor ring receives the samples.
    assert q.qsize() == 1
    assert m._ring.size() == frames


def test_audio_capture_monitor_ignored_when_source_mismatch():
    m = AudioMonitor()
    m._stream = object()
    m.set_source('other')
    q: queue.Queue = queue.Queue(maxsize=10)
    cap = AudioCapture(q, device=None, channels=1, samplerate=44100)
    cap.set_monitor(m, 'me')
    cap._callback(np.zeros((CHUNK_FRAMES, 1), dtype=np.float32),
                  CHUNK_FRAMES, None, None)
    assert m._ring.size() == 0


def test_wav_capture_forwards_to_monitor(tmp_path):
    # Build a short WAV and point a WavFileCapture at it.
    wav = tmp_path / 'tone.wav'
    sr = 44100
    n = int(sr * 0.5)
    tone = (0.25 * np.sin(2 * np.pi * 440 * np.arange(n) / sr)).astype(np.float32)
    pcm = (tone * 32767.0).astype(np.int16)
    scipy.io.wavfile.write(str(wav), sr, pcm)

    m = AudioMonitor()
    m._stream = object()  # pretend stream is open
    m.set_source('wav-a')

    q: queue.Queue = queue.Queue(maxsize=500)
    cap = WavFileCapture(q, str(wav))
    cap.set_monitor(m, 'wav-a')
    try:
        cap.resume()
        # Let a few chunks play out.
        deadline = time.monotonic() + 1.0
        while q.qsize() < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        cap.pause()
        assert q.qsize() >= 3
        assert m._ring.size() > 0
    finally:
        cap.close()


# ── RecordingEntity integration ------------------------------------------

def test_entity_set_monitor_persists_across_capture_rebuilds():
    m = AudioMonitor()
    m._stream = object()
    e = RecordingEntity(name='e1', device_id=None)
    try:
        e.set_monitor(m)
        # Baseline wiring.
        assert e.capture._monitor is m
        assert e.capture._monitor_source_id == id(e)
        # Force a capture rebuild (SR change). The new capture must
        # still carry the monitor reference.
        e.change_sample_rate(22050)
        assert e.capture._monitor is m
        assert e.capture._monitor_source_id == id(e)
    finally:
        e.close()


def test_entity_set_monitor_none_detaches_current_capture():
    m = AudioMonitor()
    e = RecordingEntity(name='e2', device_id=None)
    try:
        e.set_monitor(m)
        assert e.capture._monitor is m
        e.set_monitor(None)
        assert e.capture._monitor is None
        assert e.capture._monitor_source_id is None
    finally:
        e.close()


def test_entity_monitor_only_fires_when_entity_selected():
    """End-to-end: the monitor routes exactly one entity at a time."""
    m = AudioMonitor()
    m._stream = object()
    e1 = RecordingEntity(name='e1', device_id=None)
    e2 = RecordingEntity(name='e2', device_id=None)
    try:
        e1.set_monitor(m)
        e2.set_monitor(m)
        # Nothing selected yet → both captures feed a no-op.
        e1.capture._callback(np.ones((CHUNK_FRAMES, 1), dtype=np.float32),
                             CHUNK_FRAMES, None, None)
        e2.capture._callback(np.ones((CHUNK_FRAMES, 1), dtype=np.float32),
                             CHUNK_FRAMES, None, None)
        assert m._ring.size() == 0
        # Select e2 → e1 feeds are dropped, e2 feeds accumulate.
        m.set_source(id(e2))
        e1.capture._callback(np.ones((CHUNK_FRAMES, 1), dtype=np.float32),
                             CHUNK_FRAMES, None, None)
        assert m._ring.size() == 0
        e2.capture._callback(np.ones((CHUNK_FRAMES, 1), dtype=np.float32),
                             CHUNK_FRAMES, None, None)
        assert m._ring.size() == CHUNK_FRAMES
        # Flip to e1 → ring flushes, then fills from e1.
        m.set_source(id(e1))
        assert m._ring.size() == 0
        e1.capture._callback(np.ones((CHUNK_FRAMES, 1), dtype=np.float32),
                             CHUNK_FRAMES, None, None)
        assert m._ring.size() == CHUNK_FRAMES
    finally:
        e1.close()
        e2.close()


def test_monitor_set_source_is_hashable_token():
    """Entity ids and arbitrary hashable tokens all work as source_id."""
    m = AudioMonitor()
    m._stream = object()
    for token in (42, 'stream-a', ('tuple', 1), id(object())):
        m.set_source(token)
        assert m.source_id == token
    m.set_source(None)
    assert m.source_id is None
