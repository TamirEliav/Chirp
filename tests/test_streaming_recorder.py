"""Streaming-mode ThresholdRecorder (H1) — on-disk equivalence with the
buffered path, bounded pending RAM, force-split part naming, and
flush-on-disable behavior.

The buffered path is the long-pinned reference (see
test_trigger_characterization.py); these tests feed identical chunk
sequences through a buffered recorder and a streaming recorder and
require byte-identical WAV output.
"""

import glob
import os

import numpy as np
import pytest
import scipy.io.wavfile

from chirp.constants import CHUNK_FRAMES
from chirp.recording.trigger import ThresholdRecorder
from chirp.recording import writer as writer_mod


SR = 44100


@pytest.fixture(autouse=True)
def _shutdown_writer_pool():
    """These tests exercise the real singleton writer pool (buffered
    mode submits to it). Its workers are non-daemon by design, so the
    pool must be shut down or the pytest process never exits."""
    yield
    writer_mod.shutdown(timeout=10.0)


def _drive(rec: ThresholdRecorder, chunks, out_dir, *, enabled=True,
           threshold=0.5, min_cross=0.0, hold=0.0,
           post=0.0, pre=0.0, max_rec=1000.0, suffix=''):
    total = 0
    for ch in chunks:
        total += len(ch)
        rec.process_chunk(
            ch, trigger_peak=float(np.max(np.abs(ch))),
            threshold=threshold, min_cross_sec=min_cross, hold_sec=hold,
            post_trig_sec=post, max_rec_sec=max_rec, pre_trig_sec=pre,
            output_dir=str(out_dir), enabled=enabled,
            filename_suffix=suffix,
            sample_rate=SR, global_chunk_end=total,
        )


def _read_wavs(out_dir):
    """Return the PCM payloads of all WAVs in out_dir sorted by name."""
    out = []
    for p in sorted(glob.glob(os.path.join(str(out_dir), '*.wav'))):
        _, data = scipy.io.wavfile.read(p)
        out.append((os.path.basename(p), data))
    return out


def _burst_chunks(n_loud_chunks=3, n_quiet_chunks=3, amp=0.8):
    """Quiet lead-in, loud burst, quiet tail — one clean event."""
    rng = np.random.default_rng(42)
    quiet = [np.full(CHUNK_FRAMES, 0.01, dtype=np.float32)
             for _ in range(n_quiet_chunks)]
    loud = [(amp * (0.5 + 0.5 * rng.random(CHUNK_FRAMES))).astype(np.float32)
            for _ in range(n_loud_chunks)]
    tail = [np.full(CHUNK_FRAMES, 0.01, dtype=np.float32)
            for _ in range(n_quiet_chunks)]
    return quiet + loud + tail


def test_streaming_matches_buffered_single_event(tmp_path):
    chunks = _burst_chunks()
    buf_dir = tmp_path / 'buffered'
    str_dir = tmp_path / 'streamed'

    _drive(ThresholdRecorder(), chunks, buf_dir,
           pre=0.01, post=0.01, hold=0.005)
    writer_mod.drain(timeout=10.0)
    _drive(ThresholdRecorder(streaming=True), chunks, str_dir,
           pre=0.01, post=0.01, hold=0.005)
    writer_mod.drain(timeout=10.0)

    buffered = _read_wavs(buf_dir)
    streamed = _read_wavs(str_dir)
    assert len(buffered) == len(streamed) == 1
    np.testing.assert_array_equal(buffered[0][1], streamed[0][1])
    # No orphan tmp files left behind.
    assert not glob.glob(os.path.join(str(str_dir), '*.tmp'))


def test_streaming_matches_buffered_force_split_with_part_names(tmp_path):
    # Continuous loud signal long enough to force one split.
    rng = np.random.default_rng(7)
    n_chunks = 12
    chunks = [(0.8 * (0.5 + 0.5 * rng.random(CHUNK_FRAMES))).astype(np.float32)
              for _ in range(n_chunks)]
    max_rec = (5 * CHUNK_FRAMES) / SR  # split after 5 chunks

    buf_dir = tmp_path / 'buffered'
    str_dir = tmp_path / 'streamed'
    for rec, d in ((ThresholdRecorder(), buf_dir),
                   (ThresholdRecorder(streaming=True), str_dir)):
        _drive(rec, chunks, d, hold=0.005, post=0.0, max_rec=max_rec)
        rec.flush_all(output_dir=str(d), sample_rate=SR)
    writer_mod.drain(timeout=10.0)

    buffered = _read_wavs(buf_dir)
    streamed = _read_wavs(str_dir)
    assert len(buffered) == len(streamed) >= 2
    for (bn, bd), (sn, sd) in zip(buffered, streamed):
        np.testing.assert_array_equal(bd, sd)
    # Split parts carry partNN tokens in both modes.
    assert any('part01' in n for n, _ in streamed)
    assert any('part02' in n for n, _ in streamed)


def test_streaming_flush_on_disable_matches_buffered(tmp_path):
    rng = np.random.default_rng(3)
    loud = [(0.8 * (0.5 + 0.5 * rng.random(CHUNK_FRAMES))).astype(np.float32)
            for _ in range(4)]

    buf_dir = tmp_path / 'buffered'
    str_dir = tmp_path / 'streamed'
    for rec, d in ((ThresholdRecorder(), buf_dir),
                   (ThresholdRecorder(streaming=True), str_dir)):
        _drive(rec, loud, d, hold=1.0, post=1.0)
        # Disable mid-event → flush with whatever was kept.
        _drive(rec, [np.zeros(CHUNK_FRAMES, dtype=np.float32)], d,
               enabled=False, hold=1.0, post=1.0)
    writer_mod.drain(timeout=10.0)

    buffered = _read_wavs(buf_dir)
    streamed = _read_wavs(str_dir)
    assert len(buffered) == len(streamed) == 1
    np.testing.assert_array_equal(buffered[0][1], streamed[0][1])


def test_streaming_pending_ram_is_bounded(tmp_path):
    """While a long event records, the in-RAM pending buffer must stay
    bounded by hold + post_trig + one chunk — not grow with max_rec."""
    rng = np.random.default_rng(11)
    rec = ThresholdRecorder(streaming=True)
    hold = post = 0.01
    bound = int((hold + post) * SR) + 2 * CHUNK_FRAMES
    total = 0
    for _ in range(40):  # ~1 s of continuous loud signal
        ch = (0.8 * (0.5 + 0.5 * rng.random(CHUNK_FRAMES))).astype(np.float32)
        total += len(ch)
        rec.process_chunk(
            ch, trigger_peak=0.8, threshold=0.5,
            min_cross_sec=0.0, hold_sec=hold, post_trig_sec=post,
            max_rec_sec=1000.0, pre_trig_sec=0.0,
            output_dir=str(tmp_path), enabled=True,
            sample_rate=SR, global_chunk_end=total)
        for ev in rec._active_events:
            pending = sum(len(c) for c in ev['buf']) - ev['start_offset']
            assert pending <= bound
    rec.flush_all(output_dir=str(tmp_path), sample_rate=SR)
    writer_mod.drain(timeout=10.0)
    assert len(_read_wavs(tmp_path)) == 1


def test_streaming_steps_aside_when_start_flush_monkeypatched(tmp_path,
                                                              monkeypatch):
    """The test seam contract: a monkeypatched _start_flush receives the
    full in-memory buffer even on a streaming=True recorder."""
    captured = []

    def _capture(buf_snapshot, *args, **kwargs):
        captured.append(np.concatenate(buf_snapshot))

    monkeypatch.setattr(ThresholdRecorder, '_start_flush',
                        staticmethod(_capture))
    rec = ThresholdRecorder(streaming=True)
    chunks = _burst_chunks()
    _drive(rec, chunks, tmp_path, hold=0.005, post=0.0)
    rec.flush_all(output_dir=str(tmp_path), sample_rate=SR)
    assert len(captured) == 1
    assert captured[0].size > 0
    assert not glob.glob(os.path.join(str(tmp_path), '*.wav'))


def test_streaming_stereo_event_roundtrip(tmp_path):
    rng = np.random.default_rng(5)
    chunks = [np.stack([0.8 * rng.random(CHUNK_FRAMES) + 0.1,
                        0.6 * rng.random(CHUNK_FRAMES) + 0.1],
                       axis=1).astype(np.float32) for _ in range(4)]
    buf_dir = tmp_path / 'buffered'
    str_dir = tmp_path / 'streamed'
    for rec, d in ((ThresholdRecorder(), buf_dir),
                   (ThresholdRecorder(streaming=True), str_dir)):
        _drive(rec, chunks, d, threshold=0.05, hold=0.005, post=0.0)
        rec.flush_all(output_dir=str(d), sample_rate=SR)
    writer_mod.drain(timeout=10.0)
    buffered = _read_wavs(buf_dir)
    streamed = _read_wavs(str_dir)
    assert len(buffered) == len(streamed) == 1
    assert streamed[0][1].ndim == 2 and streamed[0][1].shape[1] == 2
    np.testing.assert_array_equal(buffered[0][1], streamed[0][1])
