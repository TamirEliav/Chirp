"""Tests for WAV-replay correctness (#49, #54).

Two pre-fix data-corruption modes:

  1. ``RecordingEntity.use_wav_file`` silently fell back to a live
     device when the requested WAV file was missing. Combined with
     ``from_dict``, a saved session that was explicitly configured to
     replay a WAV started recording from the user's default microphone
     instead. Researchers analysed what they thought was the canned
     clip but were actually looking at mic hiss from an unrelated
     input.

  2. ``WavFileCapture._format_for_queue`` truncated multi-channel WAVs
     to the session's channel count without warning. A 4-channel
     ambisonic recording loaded into a stereo session silently lost
     channels 3–4. The target use-case for WAV replay is offline
     analysis on canned data — the silent drop was specifically bad
     because there was no live operator to notice.

This file pins the post-fix behaviour:

  - Missing WAV: capture is left invalid, ``input_source`` stays
    ``'wav_file'``, the open_error / sticky error flag are set, and
    ``start_acq`` is a no-op (would have started the live mic
    pre-fix).
  - Multi-channel WAV: the capture latches ``channels_truncated`` +
    ``channels_truncated_msg`` so the sidebar can surface it, and
    ``use_wav_file`` returns a non-empty warning.
  - ``from_dict`` propagates the warning string for either case.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import scipy.io.wavfile

from chirp.recording.entity import RecordingEntity


# ── Helpers ──────────────────────────────────────────────────────────

def _write_test_wav(path: str, sr: int = 44100, duration: float = 0.1,
                    channels: int = 1) -> str:
    """Write a small WAV file for tests. Float32 sine, just enough
    samples to make a valid file."""
    n = int(sr * duration)
    if channels == 1:
        data = np.zeros(n, dtype=np.float32)
    else:
        data = np.zeros((n, channels), dtype=np.float32)
    pcm16 = (data * 32767.0).astype(np.int16)
    scipy.io.wavfile.write(path, sr, pcm16)
    return path


# ── #49: missing WAV must NOT fall back to live device ───────────────

def test_use_wav_file_does_not_fall_back_to_live_device(tmp_path):
    """The pre-fix code called ``self.capture = self._make_capture(...)``
    after ``probe.valid is False``, switching to the default mic.
    Post-fix: the invalid WavFileCapture is kept in place so
    ``capture.valid is False`` and ``open_error`` carries the reason."""
    e = RecordingEntity(name='t', device_id=None)
    bad_path = str(tmp_path / 'definitely-does-not-exist.wav')

    ok, warn = e.use_wav_file(bad_path)

    assert ok is False
    assert warn is not None
    assert 'Could not open' in warn
    # The capture must NOT be a fresh live-device AudioCapture — its
    # ``valid`` is False (the failed WavFileCapture probe).
    assert e.capture.valid is False
    # And the entity remembers we WANTED a WAV — input_source stays
    # 'wav_file', wav_file_path is preserved, so the user can fix the
    # path and retry without losing context.
    assert e.input_source == 'wav_file'
    assert e.wav_file_path == bad_path
    # The error is also latched for the sidebar `!` badge.
    assert e.has_ever_ingest_errored is True
    assert e.last_ingest_error is not None


def test_start_acq_refuses_when_wav_invalid(tmp_path):
    """``start_acq`` must be a no-op when the requested WAV failed to
    open — pre-fix it would silently start the live-device fallback."""
    e = RecordingEntity(name='t', device_id=None)
    bad_path = str(tmp_path / 'missing.wav')
    e.use_wav_file(bad_path)
    assert e.capture.valid is False

    e.start_acq()
    assert e.acq_running is False


# ── #49: from_dict propagates the warning ────────────────────────────

def test_from_dict_propagates_missing_wav_warning(tmp_path):
    """A saved session pointing at a now-missing WAV must produce a
    warning from ``from_dict`` so the load-modal surfaces it."""
    bad_path = str(tmp_path / 'gone.wav')
    d = {
        'name':            'replay',
        'input_source':    'wav_file',
        'wav_file_path':   bad_path,
        'wav_loop':        True,
        'channel_mode':    'Mono',
    }
    e, warning = RecordingEntity.from_dict(d)
    assert warning is not None
    assert 'gone.wav' in warning or 'Could not open' in warning
    assert e.input_source == 'wav_file'
    assert e.capture.valid is False


# ── #54: multi-channel WAV truncation is surfaced ────────────────────

def test_multichannel_wav_latches_truncation_flag(tmp_path):
    """A 4-channel WAV loaded into a Mono session must latch
    ``channels_truncated`` on the capture so the sidebar `!` badge
    can surface it."""
    wav = _write_test_wav(str(tmp_path / 'four.wav'),
                          sr=44100, channels=4)
    e = RecordingEntity(name='t', device_id=None)
    # Mono session (need_ch=1) but the WAV has 4 channels.
    ok, warn = e.use_wav_file(wav)
    assert ok is True
    assert warn is not None
    assert '4 channels' in warn
    assert 'ignored' in warn or 'dropped' in warn
    # Capture latches the flag.
    assert e.capture.channels_truncated is True
    assert '4' in e.capture.channels_truncated_msg
    # And the sticky error signal is set so the sidebar lights up.
    assert e.has_ever_ingest_errored is True


def test_stereo_session_with_4ch_wav_still_truncates(tmp_path):
    wav = _write_test_wav(str(tmp_path / 'four.wav'),
                          sr=44100, channels=4)
    e = RecordingEntity(name='t', device_id=None)
    e.channel_mode = 'Stereo'  # need_ch=2
    ok, warn = e.use_wav_file(wav)
    assert ok is True
    assert warn is not None
    assert e.capture.channels_truncated is True
    # Channels 3 and 4 are dropped — message should mention the count.
    assert '4 channels' in warn
    assert e.capture.file_channels == 4


def test_matching_channels_does_not_latch_truncation(tmp_path):
    """A mono WAV in a mono session must NOT trigger the truncation
    flag — false positives would spam the badge."""
    wav = _write_test_wav(str(tmp_path / 'mono.wav'),
                          sr=44100, channels=1)
    e = RecordingEntity(name='t', device_id=None)
    ok, warn = e.use_wav_file(wav)
    assert ok is True
    # warn may be present for SR change, but truncation flag must NOT.
    assert e.capture.channels_truncated is False


def test_stereo_wav_in_stereo_session_no_truncation(tmp_path):
    wav = _write_test_wav(str(tmp_path / 'stereo.wav'),
                          sr=44100, channels=2)
    e = RecordingEntity(name='t', device_id=None)
    e.channel_mode = 'Stereo'
    ok, warn = e.use_wav_file(wav)
    assert ok is True
    assert e.capture.channels_truncated is False


# ── Wav-Capture direct construction (#54 lower-level) ────────────────

def test_wav_file_capture_latches_truncation_at_open(tmp_path):
    """Direct construction (no entity) — the capture itself must do
    the latching at __init__ time, not only when use_wav_file is the
    entry point."""
    from chirp.audio import WavFileCapture
    import queue as _q
    wav = _write_test_wav(str(tmp_path / 'four.wav'), channels=4)
    cap = WavFileCapture(_q.Queue(), wav, channels=2, loop=False)
    try:
        assert cap.valid is True
        assert cap.channels_truncated is True
        assert cap.file_channels == 4
        assert '4' in cap.channels_truncated_msg
    finally:
        cap.close()
