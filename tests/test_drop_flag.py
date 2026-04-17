"""Tests for sticky drop statistics on AudioCapture / WavFileCapture (#29).

#29 adds two session-wide persistent fields alongside the existing
transient ``drop_count``:

    drop_count_total   — monotonic session-wide drop count
    has_ever_dropped   — latches True on first drop, cleared only by
                         ``reset_drop_stats()``

The sidebar uses these to keep a sticky "drops happened at some point"
badge visible until the user clicks it — surviving across the
``consume_drop_count()`` tick calls that clear the transient count.
"""

import queue as _q

import numpy as np

from chirp.audio.capture import AudioCapture


def _force_drop(cap: AudioCapture, q: _q.Queue, frames: int = 1024) -> None:
    """Drive the AudioCapture PortAudio callback with a full queue so
    the ``queue.Full`` branch fires — exactly the path that increments
    the drop counters in production."""
    indata = np.zeros((frames, 1), dtype=np.float32)
    # Fill the queue so the next put_nowait raises queue.Full.
    while not q.full():
        q.put_nowait(indata[:, 0].copy())
    cap._callback(indata, frames, None, None)


def test_drop_stats_initial_state_is_zero():
    q = _q.Queue(maxsize=1)
    cap = AudioCapture(q, device=None)
    assert cap.drop_count == 0
    assert cap.drop_count_total == 0
    assert cap.has_ever_dropped is False


def test_drop_increments_transient_and_sticky_together():
    q = _q.Queue(maxsize=1)
    cap = AudioCapture(q, device=None)
    _force_drop(cap, q)
    _force_drop(cap, q)
    assert cap.drop_count == 2
    assert cap.drop_count_total == 2
    assert cap.has_ever_dropped is True


def test_consume_drop_count_does_not_touch_sticky_stats():
    """The per-tick poller clears ``drop_count`` but the sticky stats
    must survive so the badge stays lit across ticks."""
    q = _q.Queue(maxsize=1)
    cap = AudioCapture(q, device=None)
    _force_drop(cap, q)
    _force_drop(cap, q)
    _force_drop(cap, q)
    # First tick: poller reads and clears the transient.
    assert cap.consume_drop_count() == 3
    assert cap.drop_count == 0
    # Sticky stats are untouched — this is the whole point of #29.
    assert cap.drop_count_total == 3
    assert cap.has_ever_dropped is True
    # Second tick on a quiet queue: transient stays zero, sticky survives.
    assert cap.consume_drop_count() == 0
    assert cap.drop_count_total == 3
    assert cap.has_ever_dropped is True


def test_reset_drop_stats_clears_everything():
    """Triggered by the user clicking the sticky sidebar badge."""
    q = _q.Queue(maxsize=1)
    cap = AudioCapture(q, device=None)
    _force_drop(cap, q)
    _force_drop(cap, q)
    assert cap.has_ever_dropped is True
    cap.reset_drop_stats()
    assert cap.drop_count == 0
    assert cap.drop_count_total == 0
    assert cap.has_ever_dropped is False


def test_drop_stats_relatches_after_reset():
    """After the user clears the badge, a fresh drop must re-latch the
    sticky flag — otherwise the reset would permanently disable the
    session-wide indicator."""
    q = _q.Queue(maxsize=1)
    cap = AudioCapture(q, device=None)
    _force_drop(cap, q)
    cap.reset_drop_stats()
    assert cap.has_ever_dropped is False
    _force_drop(cap, q)
    assert cap.has_ever_dropped is True
    assert cap.drop_count_total == 1


# ── WavFileCapture mirrors the same contract ─────────────────────────

def test_wav_capture_exposes_same_drop_stats_fields():
    """WavFileCapture is a drop-in replacement for AudioCapture — the
    sidebar polls the same three fields regardless of capture type, so
    they must exist with matching semantics on both classes."""
    from chirp.audio.wav_capture import WavFileCapture
    q = _q.Queue(maxsize=1)
    # Non-existent path — valid=False is fine; we only need the object.
    cap = WavFileCapture(q, wav_path="__does_not_exist__.wav")
    assert cap.drop_count == 0
    assert cap.drop_count_total == 0
    assert cap.has_ever_dropped is False
    # Simulate three drops (same pattern as the AudioCapture callback).
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
