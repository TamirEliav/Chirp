"""8-stream lossless soak — the redesign's headline guarantee.

Drives eight RecordingEntities concurrently from real-time WAV playback
(WavFileCapture → AudioRing → ingest thread → DSP → trigger) and asserts
that NOTHING is dropped: no capture-ring overruns, no PortAudio-style
os_drops, no ingest-thread exceptions. A second variant turns recording
on (full trigger + WAV-writer path) and additionally asserts the writer
pool finished cleanly and produced files.

These are intentionally a few seconds long — they exercise the real
threading at real-time pace, which is the only way to prove the capture
path keeps up across many streams. With the vectorized DSP (~32us FFT
per chunk) and a 10s ring, eight streams have enormous headroom on
common multi-core hardware, so the expected drop count is exactly zero.
"""

from __future__ import annotations

import time

import numpy as np
import scipy.io.wavfile

from chirp.recording.entity import RecordingEntity
from chirp.recording import writer as _writer


N_STREAMS = 8
RUN_SECONDS = 2.5


def _make_wav(path: str, sample_rate: int = 44100, duration: float = 2.0,
              freq: float = 1200.0) -> None:
    n = int(sample_rate * duration)
    t = np.arange(n, dtype=np.float32) / sample_rate
    # 0.5 amplitude tone — comfortably above the default 0.05 threshold so
    # the recording variant triggers immediately.
    tone = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    pcm16 = (tone * 32767.0).astype(np.int16)
    scipy.io.wavfile.write(path, sample_rate, pcm16)


def _assert_lossless(entities) -> None:
    for i, e in enumerate(entities):
        assert e.ring.overrun_count_total == 0, (
            f'stream {i}: {e.ring.overrun_count_total} ring overrun(s), '
            f'{e.ring.dropped_frames_total} frames lost')
        assert e.capture.drop_count_total == 0, (
            f'stream {i}: capture reported {e.capture.drop_count_total} drops')
        assert e.capture.os_drop_count_total == 0, (
            f'stream {i}: {e.capture.os_drop_count_total} os_drop(s)')
        assert e.ingest_error_count_total == 0, (
            f'stream {i}: ingest error: {e.last_ingest_error}')


def test_eight_stream_acquisition_is_lossless(tmp_path):
    wav = tmp_path / 'tone.wav'
    _make_wav(str(wav), sample_rate=44100, duration=2.0)

    entities = [RecordingEntity(name=f'S{i}', device_id=None)
                for i in range(N_STREAMS)]
    try:
        for e in entities:
            ok, _ = e.use_wav_file(str(wav), loop=True)
            assert ok
            e.start_acq()
        # Let all eight run at real-time pace.
        time.sleep(RUN_SECONDS)
        for e in entities:
            assert e.acq_running
        _assert_lossless(entities)
    finally:
        for e in entities:
            try:
                e.stop_acq()
                e.close()
            except Exception:
                pass


def test_eight_stream_recording_is_lossless(tmp_path):
    wav = tmp_path / 'tone.wav'
    _make_wav(str(wav), sample_rate=44100, duration=2.0)
    out_dir = tmp_path / 'rec'

    # Start from a clean writer pool so the assertions below see only
    # this test's writes.
    _writer.shutdown(timeout=2.0)

    entities = [RecordingEntity(name=f'R{i}', device_id=None)
                for i in range(N_STREAMS)]
    try:
        for i, e in enumerate(entities):
            ok, _ = e.use_wav_file(str(wav), loop=True)
            assert ok
            e.output_dir = str(out_dir / f'stream{i}')
            e.ref_date = None
            e.start_rec()  # enables recording + starts acquisition
        time.sleep(RUN_SECONDS)
        _assert_lossless(entities)
    finally:
        for e in entities:
            try:
                e.stop_acq()   # flushes any open event to the writer pool
                e.close()
            except Exception:
                pass

    # The continuous tone keeps one event open per stream; stop_acq flushes
    # them. Drain the pool and confirm it finished without errors.
    assert _writer.drain(timeout=30.0) is True
    has_ever, total, last = _writer.error_stats()
    assert has_ever is False, f'writer errors: total={total} last={last}'
    files = list(out_dir.rglob('*.wav'))
    assert len(files) >= N_STREAMS, (
        f'expected >= {N_STREAMS} recordings, got {len(files)}')
    _writer.shutdown(timeout=5.0)
