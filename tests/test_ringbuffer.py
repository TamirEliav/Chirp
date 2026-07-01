"""Tests for AudioRing — the SPSC capture ring buffer (redesign).

Pins the contract the new realtime-safe capture and DSP consumer rely
on: lossless round-trip while the consumer keeps up, correct wrap-around,
overwrite-oldest overrun accounting, still-resident random access for
pre-trigger lookback, and the drop-stat surface that mirrors
AudioCapture so the sidebar badge logic stays uniform.
"""

import numpy as np

from chirp.audio.ringbuffer import AudioRing


def test_basic_roundtrip_mono():
    r = AudioRing(capacity_frames=100, channels=1)
    data = np.arange(30, dtype=np.float32)
    assert r.write(data) == 30
    assert r.available == 30
    start, out = r.read()
    assert start == 0
    assert out.ndim == 1
    np.testing.assert_array_equal(out, data)
    assert r.available == 0
    # Subsequent read with nothing available is empty.
    start2, out2 = r.read()
    assert start2 == 30
    assert out2.size == 0


def test_partial_read_advances_cursor():
    r = AudioRing(capacity_frames=100, channels=1)
    r.write(np.arange(50, dtype=np.float32))
    start, out = r.read(max_frames=20)
    assert start == 0
    np.testing.assert_array_equal(out, np.arange(20))
    start, out = r.read(max_frames=20)
    assert start == 20
    np.testing.assert_array_equal(out, np.arange(20, 40))
    assert r.available == 10


def test_wraparound_roundtrip():
    r = AudioRing(capacity_frames=16, channels=1)
    # Write then read repeatedly so the physical head wraps the buffer.
    val = 0
    expected_next = 0
    for _ in range(20):
        block = np.arange(val, val + 10, dtype=np.float32)
        r.write(block)
        val += 10
        start, out = r.read()
        assert start == expected_next
        np.testing.assert_array_equal(out, np.arange(expected_next, val))
        expected_next = val


def test_stereo_roundtrip_and_shape():
    r = AudioRing(capacity_frames=64, channels=2)
    data = np.stack([np.arange(20), np.arange(100, 120)], axis=1).astype(np.float32)
    r.write(data)
    start, out = r.read()
    assert out.shape == (20, 2)
    np.testing.assert_array_equal(out, data)


def test_overrun_overwrites_oldest_and_counts():
    r = AudioRing(capacity_frames=10, channels=1)
    r.write(np.arange(8, dtype=np.float32))      # resident [0,8)
    # Now write 6 more without reading → total 14 > cap 10.
    # Oldest 4 frames (0..3) are evicted; resident window is [4,14).
    r.write(np.arange(8, 14, dtype=np.float32))
    assert r.has_ever_overrun is True
    assert r.overrun_count_total == 1
    assert r.dropped_frames_total == 4
    # read() returns only what's still resident, starting at abs index 4.
    start, out = r.read()
    assert start == 4
    np.testing.assert_array_equal(out, np.arange(4, 14))


def test_overrun_stats_consume_and_reset():
    r = AudioRing(capacity_frames=4, channels=1)
    r.write(np.zeros(4, dtype=np.float32))
    r.write(np.zeros(4, dtype=np.float32))   # overrun #1
    r.write(np.zeros(4, dtype=np.float32))   # overrun #2
    assert r.consume_overrun_count() == 2
    assert r.consume_overrun_count() == 0
    # Sticky stats survive the transient consume.
    assert r.overrun_count_total == 2
    assert r.has_ever_overrun is True
    r.reset_overrun_stats()
    assert r.overrun_count_total == 0
    assert r.has_ever_overrun is False
    assert r.dropped_frames_total == 0


def test_read_range_resident_and_evicted():
    r = AudioRing(capacity_frames=20, channels=1)
    r.write(np.arange(50, dtype=np.float32))   # resident window [30,50)
    cursor_before = r.read_total            # advanced to 30 by the overrun
    # Fully resident sub-range.
    out = r.read_range(35, 45)
    np.testing.assert_array_equal(out, np.arange(35, 45))
    # Partially evicted range is clipped to the resident start (30).
    out = r.read_range(10, 40)
    np.testing.assert_array_equal(out, np.arange(30, 40))
    # Fully evicted range returns empty.
    assert r.read_range(0, 10).size == 0
    # read_range must not move the consumer cursor.
    assert r.read_total == cursor_before


def test_read_range_wraps():
    r = AudioRing(capacity_frames=16, channels=1)
    r.write(np.arange(40, dtype=np.float32))   # resident [24,40), head wrapped
    out = r.read_range(28, 36)
    np.testing.assert_array_equal(out, np.arange(28, 36))
