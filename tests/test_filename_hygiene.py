"""Tests for filename + output-folder hygiene (#51, #50).

Three pre-fix failure modes:

  1. ``_sanitize_token`` was defined but ``filename_prefix`` /
     ``filename_suffix`` bypassed it — a user prefix of ``../../escape``
     landed a WAV outside ``output_dir``; a prefix of ``CON`` opened
     the Windows console device instead of a file.

  2. ``filename_stream`` was plumbed through but dropped on the
     filename side. Two streams that triggered on the same physical
     event produced identical ms-precision timestamps, so
     ``scipy.io.wavfile.write`` silently overwrote the first with the
     second.

  3. ``_on_folder_changed`` / ``_on_browse`` didn't validate the
     folder. The first sanity check was inside the writer worker —
     which prints to stdout (invisible in the packaged build) and drops
     the WAV. The transport UI showed "REC RUNNING" while every
     triggered event vanished.

Covered here:

  - Sanitizer hardening: path separators, ``..``, Windows reserved
    names, Unicode, length cap.
  - ``write_wav_sync`` sanitizes prefix + suffix + filename_stream and
    refuses to write outside ``output_dir``.
  - Two simultaneous events with the same prefix + timestamp produce
    DIFFERENT filenames thanks to ``filename_stream``.
  - ``dph_folder_prefix`` is sanitized in
    ``RecordingEntity._effective_output_dir``.
  - ``ChirpWindow._probe_output_dir`` returns the right ``(ok, reason)``
    tuple for valid / missing / non-writable paths.
"""

from __future__ import annotations

import datetime
import os
from unittest.mock import MagicMock

import numpy as np
import pytest

from chirp.recording.writer import (_sanitize_token, _TOKEN_MAX_LEN,
                                    write_wav_sync)


# ── Sanitizer ────────────────────────────────────────────────────────

def test_sanitize_rejects_path_separators():
    assert '/' not in _sanitize_token('a/b')
    assert '\\' not in _sanitize_token(r'a\b')
    # Drive letters — the colon must go.
    assert ':' not in _sanitize_token('C:\\Users\\foo')


def test_sanitize_maps_dot_dot_to_empty():
    """A pure-dot token ('.', '..', '....') must not be usable as a
    path component."""
    assert _sanitize_token('.') == ''
    assert _sanitize_token('..') == ''
    assert _sanitize_token('....') == ''


def test_sanitize_escapes_windows_reserved():
    """Windows reserved device names (CON, PRN, AUX, NUL, COM1-9,
    LPT1-9) must not be returned verbatim — they'd open the device
    instead of creating a file."""
    for reserved in ('CON', 'PRN', 'AUX', 'NUL',
                     'COM1', 'COM9', 'LPT1', 'LPT9',
                     'con', 'Con'):  # case-insensitive
        out = _sanitize_token(reserved)
        assert out.upper() != reserved.upper(), (
            f'{reserved!r} returned verbatim — would open Windows device')


def test_sanitize_length_capped():
    long = 'a' * (_TOKEN_MAX_LEN + 200)
    out = _sanitize_token(long)
    assert len(out) <= _TOKEN_MAX_LEN


def test_sanitize_empty_is_empty():
    assert _sanitize_token('') == ''
    assert _sanitize_token('___') == ''


# ── write_wav_sync refuses to escape output_dir ──────────────────────

def test_write_wav_sync_refuses_path_traversal_prefix(tmp_path):
    """A malicious prefix like ``../../escape`` must not land a WAV
    outside the output folder — the sanitizer strips the slashes AND
    the final realpath check is a belt-and-suspenders."""
    onset = datetime.datetime(2024, 1, 2, 3, 4, 5, 678000)
    audio = np.zeros(512, dtype=np.float32)
    path = write_wav_sync([audio], str(tmp_path),
                          prefix='../../escape',
                          sample_rate=44100,
                          onset_time=onset)
    # The WAV must be inside tmp_path.
    real_path = os.path.realpath(path)
    real_out  = os.path.realpath(str(tmp_path))
    assert os.path.commonpath([real_out, real_path]) == real_out


def test_write_wav_sync_rejects_blank_output_dir():
    audio = np.zeros(512, dtype=np.float32)
    with pytest.raises(ValueError):
        write_wav_sync([audio], '', sample_rate=44100,
                       onset_time=datetime.datetime(2024, 1, 1))
    with pytest.raises(ValueError):
        write_wav_sync([audio], '   ', sample_rate=44100,
                       onset_time=datetime.datetime(2024, 1, 1))


# ── filename_stream disambiguates collisions ─────────────────────────

def test_two_streams_same_ms_do_not_collide(tmp_path):
    """Two events with IDENTICAL timestamp + prefix + suffix must
    produce DIFFERENT filenames when ``filename_stream`` differs —
    this was the data-loss bug."""
    onset = datetime.datetime(2024, 1, 2, 3, 4, 5, 678000)
    audio = np.zeros(512, dtype=np.float32)
    p1 = write_wav_sync([audio], str(tmp_path),
                        prefix='session', suffix='',
                        sample_rate=44100, onset_time=onset,
                        filename_stream='mic-A')
    p2 = write_wav_sync([audio], str(tmp_path),
                        prefix='session', suffix='',
                        sample_rate=44100, onset_time=onset,
                        filename_stream='mic-B')
    assert os.path.basename(p1) != os.path.basename(p2)
    assert 'mic-A' in os.path.basename(p1)
    assert 'mic-B' in os.path.basename(p2)


def test_reserved_windows_prefix_is_rewritten(tmp_path):
    """A user setting ``filename_prefix = 'CON'`` must not open the
    Windows console device — the sanitizer rewrites it."""
    onset = datetime.datetime(2024, 1, 2, 3, 4, 5, 678000)
    audio = np.zeros(512, dtype=np.float32)
    path = write_wav_sync([audio], str(tmp_path),
                          prefix='CON',
                          sample_rate=44100,
                          onset_time=onset)
    # Windows resolves reserved device names by comparing the FULL
    # stem (filename sans extension) against the reserved list —
    # ``CON.wav`` opens console, ``CON_r_....wav`` does not. The
    # sanitizer just needs to ensure the original reserved token is
    # no longer the stem. Verify the stem is neither ``CON`` nor
    # starts with ``CON.`` (case-insensitive).
    base = os.path.basename(path)
    stem = base.rsplit('.', 1)[0]
    assert stem.upper() != 'CON'
    # And the prefix token itself (first underscore-delimited piece)
    # has been rewritten from ``CON`` to something else (``CON_r``)
    # so the filename as a whole is distinguishable.
    assert 'CON_r' in base, (
        f'filename {base!r} did not get the _r suffix applied to CON')


# ── dph_folder_prefix sanitization (#51 case 3) ──────────────────────

def test_dph_folder_prefix_is_sanitized():
    """A ``dph_folder_prefix`` containing path separators must NOT
    escape ``output_dir`` when ``_effective_output_dir`` is called."""
    from chirp.recording.entity import RecordingEntity
    e = RecordingEntity(name='t', device_id=None)
    e.output_dir = '/safe/output'
    e.ref_date = datetime.date.today()
    e.dph_folder_prefix = '../../escape_'

    out = e._effective_output_dir()
    # The prefix must have been sanitized — no path traversal.
    # commonpath should still be /safe/output.
    real_out = os.path.realpath('/safe/output')
    real_eff = os.path.realpath(out)
    assert os.path.commonpath([real_out, real_eff]) == real_out, (
        f'_effective_output_dir escaped: {out!r}')


# ── Output-folder validation (#50) ───────────────────────────────────

def _make_window_for_folder_tests():
    from chirp.ui.window import ChirpWindow
    win = ChirpWindow.__new__(ChirpWindow)
    win._folder_edit = MagicMock()
    win._folder_edit.setStyleSheet = MagicMock()
    win._folder_edit.setToolTip = MagicMock()
    return win


def test_probe_output_dir_ok(tmp_path):
    win = _make_window_for_folder_tests()
    ok, reason = win._probe_output_dir(str(tmp_path))
    assert ok is True
    assert reason == ''


def test_probe_output_dir_missing_path(tmp_path):
    win = _make_window_for_folder_tests()
    bad = str(tmp_path / 'does-not-exist')
    ok, reason = win._probe_output_dir(bad)
    assert ok is False
    assert 'directory' in reason.lower()


def test_probe_output_dir_blank():
    win = _make_window_for_folder_tests()
    ok, reason = win._probe_output_dir('')
    assert ok is False
    assert 'empty' in reason.lower()


def test_apply_folder_validation_stamps_entity(tmp_path):
    """A valid folder sets ``output_dir_valid=True`` and clears any
    prior error message. An invalid folder flips both."""
    from chirp.recording.entity import RecordingEntity
    win = _make_window_for_folder_tests()
    e = RecordingEntity(name='t', device_id=None)

    # Valid path
    win._apply_folder_validation(e, str(tmp_path))
    assert e.output_dir_valid is True
    assert e.output_dir_error is None

    # Invalid path
    win._apply_folder_validation(e, str(tmp_path / 'nope'))
    assert e.output_dir_valid is False
    assert e.output_dir_error is not None
    # And the textbox was styled red.
    win._folder_edit.setStyleSheet.assert_called()
    # Last call was the invalid style
    call_args = win._folder_edit.setStyleSheet.call_args_list[-1]
    style_arg = call_args[0][0]
    assert 'f38ba8' in style_arg  # Catppuccin red


def test_apply_folder_validation_clears_style_on_ok(tmp_path):
    """When a previously-invalid folder becomes valid, the red style
    must be cleared (setStyleSheet with empty string)."""
    from chirp.recording.entity import RecordingEntity
    win = _make_window_for_folder_tests()
    e = RecordingEntity(name='t', device_id=None)

    win._apply_folder_validation(e, str(tmp_path / 'nope'))  # invalid
    win._apply_folder_validation(e, str(tmp_path))           # valid
    # Last style-sheet call cleared the style.
    last_call = win._folder_edit.setStyleSheet.call_args_list[-1]
    assert last_call[0][0] == ''
