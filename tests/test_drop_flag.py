"""Tests for sticky drop statistics on AudioCapture / WavFileCapture (#29).

Redesign note: the drop *mechanism* changed from a full ``queue.Queue``
to a capture ``AudioRing`` overrun (consumer fell more than the ring's
capacity behind, overwriting unread audio). The sticky-stat *contract*
the sidebar badge relies on is unchanged:

    drop_count         — transient per-tick count
    drop_count_total   — monotonic session-wide drop count
    has_ever_dropped   — latches True on first drop, cleared only by
                         ``reset_drop_stats()``

To force the overrun path deterministically we use a one-chunk ring and
pre-fill it, so every subsequent callback write overwrites unread audio.
"""

import numpy as np

from chirp.audio.capture import AudioCapture
from chirp.audio.ringbuffer import AudioRing
from chirp.constants import CHUNK_FRAMES


def _full_ring_capture():
    """AudioCapture over a one-chunk ring, pre-filled so the ring is at
    capacity — every following callback write triggers an overrun."""
    ring = AudioRing(CHUNK_FRAMES, channels=1)
    cap = AudioCapture(ring, device=None)
    ring.write(np.zeros(CHUNK_FRAMES, dtype=np.float32))  # ring now full
    return cap, ring


def _force_drop(cap: AudioCapture, frames: int = CHUNK_FRAMES) -> None:
    """Drive the callback once against a full ring so the overrun branch
    (which bumps the drop counters in production) fires."""
    indata = np.zeros((frames, 1), dtype=np.float32)
    cap._callback(indata, frames, None, None)


def test_drop_stats_initial_state_is_zero():
    cap = AudioCapture(AudioRing(CHUNK_FRAMES, channels=1), device=None)
    assert cap.drop_count == 0
    assert cap.drop_count_total == 0
    assert cap.has_ever_dropped is False


def test_drop_increments_transient_and_sticky_together():
    cap, _ = _full_ring_capture()
    _force_drop(cap)
    _force_drop(cap)
    assert cap.drop_count == 2
    assert cap.drop_count_total == 2
    assert cap.has_ever_dropped is True


def test_consume_drop_count_does_not_touch_sticky_stats():
    """The per-tick poller clears ``drop_count`` but the sticky stats
    must survive so the badge stays lit across ticks."""
    cap, _ = _full_ring_capture()
    _force_drop(cap)
    _force_drop(cap)
    _force_drop(cap)
    # First tick: poller reads and clears the transient.
    assert cap.consume_drop_count() == 3
    assert cap.drop_count == 0
    # Sticky stats are untouched — this is the whole point of #29.
    assert cap.drop_count_total == 3
    assert cap.has_ever_dropped is True
    # Second tick with no new drops: transient stays zero, sticky survives.
    assert cap.consume_drop_count() == 0
    assert cap.drop_count_total == 3
    assert cap.has_ever_dropped is True


def test_reset_drop_stats_clears_everything():
    """Triggered by the user clicking the sticky sidebar badge."""
    cap, _ = _full_ring_capture()
    _force_drop(cap)
    _force_drop(cap)
    assert cap.has_ever_dropped is True
    cap.reset_drop_stats()
    assert cap.drop_count == 0
    assert cap.drop_count_total == 0
    assert cap.has_ever_dropped is False


def test_drop_stats_relatches_after_reset():
    """After the user clears the badge, a fresh drop must re-latch the
    sticky flag — otherwise the reset would permanently disable the
    session-wide indicator."""
    cap, _ = _full_ring_capture()
    _force_drop(cap)
    cap.reset_drop_stats()
    assert cap.has_ever_dropped is False
    _force_drop(cap)
    assert cap.has_ever_dropped is True
    assert cap.drop_count_total == 1


# ── WavFileCapture mirrors the same contract ─────────────────────────

def test_wav_capture_exposes_same_drop_stats_fields():
    """WavFileCapture is a drop-in replacement for AudioCapture — the
    sidebar polls the same three fields regardless of capture type, so
    they must exist with matching semantics on both classes."""
    from chirp.audio.wav_capture import WavFileCapture
    # Non-existent path — valid=False is fine; we only need the object.
    cap = WavFileCapture(AudioRing(CHUNK_FRAMES, channels=1),
                         wav_path="__does_not_exist__.wav")
    assert cap.drop_count == 0
    assert cap.drop_count_total == 0
    assert cap.has_ever_dropped is False
    # Simulate three drops (same fields the producer thread bumps).
    cap.drop_count += 1
    cap.drop_count_total += 1
    cap.has_ever_dropped = True
    cap.drop_count += 1
    cap.drop_count_total += 1
    cap.drop_count += 1
    cap.drop_count_total += 1
    assert cap.consume_drop_count() == 3
    assert cap.drop_count_total == 3
    assert cap.has_ever_dropped is True
    cap.reset_drop_stats()
    assert cap.drop_count_total == 0
    assert cap.has_ever_dropped is False
