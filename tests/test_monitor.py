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
from chirp.audio.ringbuffer import AudioRing
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
    ring = AudioRing(CHUNK_FRAMES * 8, channels=1)
    cap = AudioCapture(ring, device=None, channels=1, samplerate=44100)
    cap.set_monitor(m, 'cap-a')
    # Drive the callback manually with a synthetic indata matrix.
    frames = 1024
    indata = np.full((frames, 1), 0.25, dtype=np.float32)
    cap._callback(indata, frames, None, None)
    # Capture ring receives the frames, monitor ring receives the samples.
    assert ring.available == frames
    assert m._ring.size() == frames


def test_audio_capture_monitor_ignored_when_source_mismatch():
    m = AudioMonitor()
    m._stream = object()
    m.set_source('other')
    ring = AudioRing(CHUNK_FRAMES * 8, channels=1)
    cap = AudioCapture(ring, device=None, channels=1, samplerate=44100)
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

    ring = AudioRing(int(44100 * 3), channels=1)
    cap = WavFileCapture(ring, str(wav))
    cap.set_monitor(m, 'wav-a')
    try:
        cap.resume()
        # Let a few chunks play out.
        deadline = time.monotonic() + 1.0
        while ring.available < 3 * CHUNK_FRAMES and time.monotonic() < deadline:
            time.sleep(0.01)
        cap.pause()
        assert ring.available >= 3 * CHUNK_FRAMES
        assert m._ring.size() > 0
    finally:
        cap.close()


# ── Channel selection (stereo-split streams) -----------------------------

def _stereo_indata(frames=CHUNK_FRAMES):
    """Distinct left/right so channel leakage is detectable: L=+0.5, R=-0.5."""
    d = np.empty((frames, 2), dtype=np.float32)
    d[:, 0] = 0.5
    d[:, 1] = -0.5
    return d


def test_capture_monitor_right_channel_only():
    """A 'Right' stream (2 input channels) must feed only column 1 — the
    left column belongs to a *different* stream when two streams split a
    stereo device (regression: left leaked in via the mono downmix)."""
    m = AudioMonitor()
    m._stream = object()
    m.set_source('r')
    ring = AudioRing(CHUNK_FRAMES * 8, channels=2)
    cap = AudioCapture(ring, device=None, channels=2, samplerate=44100)
    cap.set_monitor(m, 'r', channel=1)
    cap._callback(_stereo_indata(), CHUNK_FRAMES, None, None)
    out = np.zeros(CHUNK_FRAMES, dtype=np.float32)
    m._ring.read(CHUNK_FRAMES, out)
    np.testing.assert_allclose(out, -0.5, atol=1e-7)  # right only, no left mix


def test_capture_monitor_left_channel_only():
    m = AudioMonitor()
    m._stream = object()
    m.set_source('l')
    ring = AudioRing(CHUNK_FRAMES * 8, channels=2)
    cap = AudioCapture(ring, device=None, channels=2, samplerate=44100)
    cap.set_monitor(m, 'l', channel=0)
    cap._callback(_stereo_indata(), CHUNK_FRAMES, None, None)
    out = np.zeros(CHUNK_FRAMES, dtype=np.float32)
    m._ring.read(CHUNK_FRAMES, out)
    np.testing.assert_allclose(out, 0.5, atol=1e-7)  # left only, no right mix


def test_capture_monitor_stereo_feeds_both():
    m = AudioMonitor()
    m._stream = object()
    # Stereo mode opens the output at 2 channels; mirror that so the
    # monitor ring keeps both columns instead of downmixing to mono.
    m._channels = 2
    m._ring = _RingBuffer(capacity_frames=CHUNK_FRAMES * 8, channels=2)
    m.set_source('s')
    ring = AudioRing(CHUNK_FRAMES * 8, channels=2)
    cap = AudioCapture(ring, device=None, channels=2, samplerate=44100)
    cap.set_monitor(m, 's', channel=None)  # stereo → both columns
    assert m._ring.channels == 2
    cap._callback(_stereo_indata(), CHUNK_FRAMES, None, None)
    out = np.zeros((CHUNK_FRAMES, 2), dtype=np.float32)
    m._ring.read(CHUNK_FRAMES, out)
    np.testing.assert_allclose(out[:, 0], 0.5, atol=1e-7)
    np.testing.assert_allclose(out[:, 1], -0.5, atol=1e-7)


def test_entity_monitor_channel_tracks_channel_mode():
    """The entity derives the monitor channel from channel_mode and
    re-wires the capture so 'Right'/'Left'/'Stereo' play the right thing."""
    m = AudioMonitor()
    e = RecordingEntity(name='e', device_id=None)
    try:
        e.set_monitor(m)
        e.channel_mode = 'Right'
        assert e._monitor_channel() == 1
        e.channel_mode = 'Left'
        assert e._monitor_channel() == 0
        e.channel_mode = 'Mono'
        assert e._monitor_channel() == 0
        e.channel_mode = 'Stereo'
        assert e._monitor_channel() is None
        # Rebuilding the capture propagates the current channel to it.
        e.channel_mode = 'Right'
        e.change_device(None, 2)
        assert e.capture._monitor_channel == 1
    finally:
        e.close()


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


# ── Output gain (0–200%) ---------------------------------------------------

def test_gain_default_and_clamping():
    m = AudioMonitor()
    assert m.gain == 1.0
    m.set_gain(0.5)
    assert m.gain == 0.5
    m.set_gain(-1.0)
    assert m.gain == 0.0          # clamped low
    m.set_gain(5.0)
    assert m.gain == 2.0          # clamped high (200%)


def test_callback_applies_gain():
    m = AudioMonitor()
    m.set_gain(0.5)
    m._ring.write(np.full(CHUNK_FRAMES, 0.8, dtype=np.float32))
    out = np.zeros((CHUNK_FRAMES, 1), dtype=np.float32)
    m._callback(out, CHUNK_FRAMES, None, None)
    np.testing.assert_allclose(out[:, 0], 0.4, atol=1e-6)


def test_callback_boost_clips_to_full_scale():
    m = AudioMonitor()
    m.set_gain(2.0)
    m._ring.write(np.full(CHUNK_FRAMES, 0.8, dtype=np.float32))
    out = np.zeros((CHUNK_FRAMES, 1), dtype=np.float32)
    m._callback(out, CHUNK_FRAMES, None, None)
    # 0.8 * 2.0 = 1.6 → clipped to full scale, never wrapped.
    np.testing.assert_allclose(out[:, 0], 1.0, atol=1e-6)


def test_callback_unity_gain_leaves_samples_untouched():
    m = AudioMonitor()
    m._ring.write(np.full(4, 0.25, dtype=np.float32))
    out = np.zeros((CHUNK_FRAMES, 1), dtype=np.float32)
    m._callback(out, CHUNK_FRAMES, None, None)
    np.testing.assert_allclose(out[:4, 0], 0.25, atol=1e-7)
    np.testing.assert_allclose(out[4:, 0], 0.0)
