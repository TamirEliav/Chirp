"""Tests for `chirp.recording.writer`.

Pin the file naming contract — particularly the c13 (#23) addition
of a sanitized stream-name token, which prevents two streams with
events at the same millisecond from clobbering each other's WAVs.
"""

import datetime
import os

import numpy as np

from chirp.recording.writer import write_wav_sync, _sanitize_token


def test_sanitize_token_strips_separators():
    assert _sanitize_token("Mic A / 1") == "Mic_A___1"
    assert _sanitize_token("ok-_.name") == "ok-_.name"
    assert _sanitize_token("///") == ""


def test_filename_includes_sanitized_stream_name(tmp_path):
    """#51: ``filename_stream`` must appear in the filename — it is
    the ONLY disambiguator between two streams that trigger on the
    same physical event at the same ms timestamp. The old
    test_filename_excludes_stream_name pinned the buggy behavior
    (clobber!) — this replaces it."""
    onset = datetime.datetime(2024, 1, 2, 3, 4, 5, 678000)
    audio = np.zeros(1024, dtype=np.float32)
    path = write_wav_sync([audio], str(tmp_path),
                          prefix='bird', suffix='',
                          sample_rate=44100,
                          onset_time=onset,
                          filename_stream='Channel 1')
    name = os.path.basename(path)
    # Sanitized stream name is present and disambiguates streams.
    assert "Channel_1" in name
    assert name.startswith("bird_")
    assert name.endswith(".wav")


def test_writer_pool_drains_pending_writes(tmp_path):
    """`submit` followed by `drain` should leave every WAV on disk
    before drain returns (#17 / c16). Worker threads are non-daemon
    so this guarantees survival across interpreter shutdown.
    """
    from chirp.recording import writer

    # Use a fresh pool to avoid interference with module-level state.
    writer.shutdown(timeout=1.0)
    n = 5
    for i in range(n):
        onset = datetime.datetime(2024, 1, 1, 0, 0, i)
        writer.submit([np.zeros(512, dtype=np.float32)], str(tmp_path),
                      sample_rate=44100, onset_time=onset)
    assert writer.drain(timeout=10.0) is True
    assert writer.pending() == 0
    files = list(tmp_path.glob('*.wav'))
    assert len(files) == n
    writer.shutdown(timeout=1.0)


def test_filename_omits_token_when_blank(tmp_path):
    onset = datetime.datetime(2024, 1, 2, 3, 4, 5, 678000)
    audio = np.zeros(1024, dtype=np.float32)
    path = write_wav_sync([audio], str(tmp_path),
                          sample_rate=44100,
                          onset_time=onset)
    name = os.path.basename(path)
    # Should still be valid (epoch_ms + local timestamp present)
    assert name.endswith(".wav")
    # Should not have stray double-underscores from a missing token
    assert "__" not in name
