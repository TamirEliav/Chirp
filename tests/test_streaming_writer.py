"""Tests for StreamingWavWriter — incremental, atomic soundfile writer
(redesign Phase 3).

Pins the contract the Phase 5 trigger rewrite relies on:
  * incremental append produces a valid WAV with the right sr/channels/
    sample count;
  * a streamed file is byte-identical to a whole-buffer write of the
    same audio (so output is unchanged by the streaming switch);
  * atomicity — only ``.tmp`` exists mid-write, only the canonical path
    after close, no ``.tmp`` leak; abort discards everything;
  * saturation detection mirrors write_wav_sync;
  * filename composition matches the shared helper.
"""

import datetime
import os

import numpy as np
import scipy.io.wavfile

from chirp.recording.writer import StreamingWavWriter, write_wav_sync


def test_stream_basic_mono(tmp_path):
    onset = datetime.datetime(2024, 1, 2, 3, 4, 5, 678000)
    w = StreamingWavWriter(str(tmp_path), prefix='bird', sample_rate=44100,
                           onset_time=onset, channels=1,
                           filename_stream='Channel 1')
    # Mid-write: only the tmp file exists. (The tmp name carries a
    # uniquifier so two concurrent events on one stream can never
    # collide — assert via the writer's own handle.)
    assert os.path.exists(w._tmp)
    assert not os.path.exists(w.path)
    rng = np.random.default_rng(1)
    blocks = [rng.uniform(-0.5, 0.5, 1000).astype(np.float32) for _ in range(5)]
    for b in blocks:
        w.append(b)
    path = w.close()
    # After close: canonical exists, no tmp leak.
    assert os.path.exists(path)
    assert not os.path.exists(path + '.tmp')
    assert 'bird_' in os.path.basename(path)
    # The stream name is deliberately NOT part of the filename.
    assert 'Channel' not in os.path.basename(path)
    sr, data = scipy.io.wavfile.read(path)
    assert sr == 44100
    assert data.dtype == np.int16
    assert data.shape[0] == 5000
    assert w.frames_written == 5000


def test_stream_equivalent_to_whole_buffer(tmp_path):
    """Streaming in chunks must yield the exact same PCM bytes as
    write_wav_sync on the concatenated buffer."""
    onset = datetime.datetime(2024, 5, 6, 7, 8, 9, 123000)
    rng = np.random.default_rng(42)
    blocks = [rng.uniform(-0.9, 0.9, 777).astype(np.float32) for _ in range(7)]

    # Whole-buffer reference.
    ref_dir = tmp_path / 'ref'
    ref_path = write_wav_sync(blocks, str(ref_dir), prefix='x',
                              sample_rate=22050, onset_time=onset,
                              filename_stream='s')
    _, ref = scipy.io.wavfile.read(ref_path)

    # Streamed.
    str_dir = tmp_path / 'str'
    w = StreamingWavWriter(str(str_dir), prefix='x', sample_rate=22050,
                           onset_time=onset, channels=1, filename_stream='s')
    for b in blocks:
        w.append(b)
    str_path = w.close()
    _, streamed = scipy.io.wavfile.read(str_path)

    np.testing.assert_array_equal(streamed, ref)
    # Same filename composition too.
    assert os.path.basename(str_path) == os.path.basename(ref_path)


def test_stream_stereo(tmp_path):
    onset = datetime.datetime(2024, 1, 1, 0, 0, 0)
    w = StreamingWavWriter(str(tmp_path), sample_rate=48000, onset_time=onset,
                           channels=2)
    rng = np.random.default_rng(3)
    for _ in range(4):
        block = rng.uniform(-0.3, 0.3, (500, 2)).astype(np.float32)
        w.append(block)
    path = w.close()
    sr, data = scipy.io.wavfile.read(path)
    assert sr == 48000
    assert data.shape == (2000, 2)


def test_stream_saturation_flag(tmp_path):
    onset = datetime.datetime(2024, 1, 1, 0, 0, 0)
    w = StreamingWavWriter(str(tmp_path), sample_rate=44100, onset_time=onset)
    w.append(np.array([0.1, -0.2, 0.5], dtype=np.float32))
    assert w.saturated is False
    w.append(np.array([0.999, -1.0], dtype=np.float32))
    assert w.saturated is True
    w.close()


def test_stream_abort_discards(tmp_path):
    onset = datetime.datetime(2024, 1, 1, 0, 0, 0)
    w = StreamingWavWriter(str(tmp_path), sample_rate=44100, onset_time=onset)
    w.append(np.zeros(256, dtype=np.float32))
    tmp = w._tmp
    assert os.path.exists(tmp)
    w.abort()
    assert not os.path.exists(tmp)
    assert not os.path.exists(w.path)


def test_stream_close_idempotent(tmp_path):
    onset = datetime.datetime(2024, 1, 1, 0, 0, 0)
    w = StreamingWavWriter(str(tmp_path), sample_rate=44100, onset_time=onset)
    w.append(np.zeros(128, dtype=np.float32))
    p1 = w.close()
    p2 = w.close()
    assert p1 == p2
    assert os.path.exists(p1)
    # append after close is a no-op (no crash).
    w.append(np.ones(64, dtype=np.float32))
    sr, data = scipy.io.wavfile.read(p1)
    assert data.shape[0] == 128


def test_stream_atomic_no_tmp_after_close(tmp_path):
    onset = datetime.datetime(2024, 1, 1, 0, 0, 0)
    w = StreamingWavWriter(str(tmp_path), prefix='p', sample_rate=44100,
                           onset_time=onset)
    w.append(np.zeros(512, dtype=np.float32))
    out = w.close()
    assert os.path.exists(out)
    assert not os.path.exists(out + '.tmp')
    assert out.endswith('.wav')
