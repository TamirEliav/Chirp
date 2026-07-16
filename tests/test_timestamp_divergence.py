"""Tests for the publish-time timestamp sanity check (writer.py).

When a finalized WAV's ``onset + duration`` disagrees with the wall
clock by more than ``TIMESTAMP_DIVERGENCE_SEC``, the writer flags it
through the pool error stats (sidebar badge) and ``chirp_errors.log``
— an end-of-pipeline watchdog for the backwards-timestamp bug class.

The check is disabled suite-wide by a conftest autouse fixture (many
tests write files with fixed historical onsets); each test here
re-enables it explicitly.
"""

import datetime

import numpy as np
import pytest

from chirp.recording import writer
from chirp.recording.writer import StreamingWavWriter


@pytest.fixture()
def divergence_enabled(monkeypatch):
    monkeypatch.setattr(writer, 'TIMESTAMP_DIVERGENCE_SEC', 10.0)
    writer.reset_error_stats()
    yield
    writer.reset_error_stats()


def _buf(n=1024):
    return [np.zeros(n, dtype=np.float32)]


def test_stale_onset_flags_error_stats(tmp_path, divergence_enabled):
    onset = datetime.datetime.now() - datetime.timedelta(hours=1)
    writer.write_wav_sync(_buf(), str(tmp_path), sample_rate=44100,
                          onset_time=onset, filename_stream='s1')
    has_ever, total, last = writer.error_stats()
    assert has_ever and total == 1
    assert 'diverges' in last


def test_future_onset_flags_too(tmp_path, divergence_enabled):
    onset = datetime.datetime.now() + datetime.timedelta(minutes=5)
    writer.write_wav_sync(_buf(), str(tmp_path), sample_rate=44100,
                          onset_time=onset)
    assert writer.error_stats()[0] is True


def test_fresh_onset_does_not_flag(tmp_path, divergence_enabled):
    dur = 1024 / 44100
    onset = datetime.datetime.now() - datetime.timedelta(seconds=dur)
    writer.write_wav_sync(_buf(), str(tmp_path), sample_rate=44100,
                          onset_time=onset)
    assert writer.error_stats() == (False, 0, None)


def test_fallback_now_onset_never_flags(tmp_path, divergence_enabled):
    """onset_time=None derives the onset from now() — zero divergence
    by construction, and the check is skipped entirely."""
    writer.write_wav_sync(_buf(), str(tmp_path), sample_rate=44100)
    assert writer.error_stats() == (False, 0, None)


def test_aware_utc_onset_handled(tmp_path, divergence_enabled):
    """The disciplined-clock path produces aware-UTC onsets — both the
    fresh (no flag) and stale (flag) cases must work on them."""
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    dur = 1024 / 44100
    writer.write_wav_sync(_buf(), str(tmp_path), sample_rate=44100,
                          onset_time=now_utc - datetime.timedelta(seconds=dur))
    assert writer.error_stats() == (False, 0, None)
    writer.write_wav_sync(_buf(), str(tmp_path), sample_rate=44100,
                          onset_time=now_utc - datetime.timedelta(hours=1))
    assert writer.error_stats()[0] is True


def test_streaming_close_flags_stale_onset(tmp_path, divergence_enabled):
    onset = datetime.datetime.now() - datetime.timedelta(hours=2)
    w = StreamingWavWriter(str(tmp_path), sample_rate=44100,
                           onset_time=onset, filename_stream='s2')
    w.append(np.zeros(1024, dtype=np.float32))
    w.close()
    has_ever, total, last = writer.error_stats()
    assert has_ever and 'diverges' in last


def test_streaming_close_fresh_onset_clean(tmp_path, divergence_enabled):
    w = StreamingWavWriter(str(tmp_path), sample_rate=44100,
                           onset_time=datetime.datetime.now())
    w.append(np.zeros(1024, dtype=np.float32))
    w.close()
    assert writer.error_stats() == (False, 0, None)


def test_retarget_with_explicit_onset_updates_check(tmp_path,
                                                    divergence_enabled):
    """retarget() with a real onset must inform the close()-time check;
    a retarget WITHOUT an onset (path-recompute only) must not
    overwrite the remembered one."""
    stale = datetime.datetime.now() - datetime.timedelta(hours=1)
    w = StreamingWavWriter(str(tmp_path), sample_rate=44100,
                           onset_time=datetime.datetime.now())
    w.append(np.zeros(1024, dtype=np.float32))
    w.retarget(str(tmp_path), onset_time=stale)
    w.close()
    assert writer.error_stats()[0] is True

    writer.reset_error_stats()
    w2 = StreamingWavWriter(str(tmp_path), sample_rate=44100,
                            onset_time=datetime.datetime.now(),
                            filename_stream='keep')
    w2.append(np.zeros(1024, dtype=np.float32))
    w2.retarget(str(tmp_path), filename_stream='keep2')  # no onset given
    w2.close()
    assert writer.error_stats() == (False, 0, None)


def test_disabled_when_threshold_none(tmp_path):
    """The conftest autouse fixture sets the threshold to None — a
    wildly stale onset must not flag anything."""
    writer.reset_error_stats()
    onset = datetime.datetime(2024, 1, 1)
    writer.write_wav_sync(_buf(), str(tmp_path), sample_rate=44100,
                          onset_time=onset)
    assert writer.error_stats() == (False, 0, None)
