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


def test_filename_includes_stream_token(tmp_path):
    onset = datetime.datetime(2024, 1, 2, 3, 4, 5, 678000)
    audio = np.zeros(1024, dtype=np.float32)
    path = write_wav_sync([audio], str(tmp_path),
                          prefix='', suffix='',
                          sample_rate=44100,
                          onset_time=onset,
                          filename_stream='Channel 1')
    name = os.path.basename(path)
    assert "Channel_1" in name
    assert name.endswith(".wav")


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
