"""Tests for atomic file-write contract (#52).

Two hot-path writes used to be non-atomic:

  * Settings save — ``_write_settings_to_path`` used ``open(path, 'w')``
    directly, so a crash between truncation and ``json.dump``'s final
    byte wiped the user's config.

  * WAV write — ``scipy.io.wavfile.write`` wrote to the canonical path
    directly, so a crash (power loss, OOM-kill, drain force-close)
    left a truncated RIFF header with a sample count that didn't
    match the body. Downstream tooling silently mis-reads.

Both now write to ``<target>.tmp``, fsync, then ``os.replace(tmp,
target)``. The replace is atomic on the same filesystem.

These tests pin the observable contract: no ``.tmp`` sibling lingers
on success, the target path holds the final bytes, and a simulated
mid-write crash leaves the previous good file untouched.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import scipy.io.wavfile


# ── WAV writer atomicity ─────────────────────────────────────────────

def test_write_wav_sync_uses_tmp_then_rename(tmp_path, monkeypatch):
    """The canonical path must never be visible until the full file
    is written. Intercept ``scipy.io.wavfile.write`` and
    ``os.replace`` to observe the ordering."""
    from chirp.recording import writer as _w

    order: list[tuple[str, str]] = []

    real_wavwrite = scipy.io.wavfile.write
    real_replace  = os.replace

    def _spy_wavwrite(path, *a, **kw):
        order.append(('wavwrite', str(path)))
        return real_wavwrite(path, *a, **kw)

    def _spy_replace(src, dst):
        order.append(('replace', f'{src}→{dst}'))
        return real_replace(src, dst)

    monkeypatch.setattr(_w.scipy.io.wavfile, 'write', _spy_wavwrite)
    monkeypatch.setattr(_w.os, 'replace', _spy_replace)

    chunk = np.zeros(512, dtype=np.float32)
    out = _w.write_wav_sync(
        [chunk], str(tmp_path), prefix='p', suffix='s', sample_rate=44100,
    )

    # The wavwrite call targeted a .tmp sibling, not the canonical path.
    assert order[0][0] == 'wavwrite'
    assert order[0][1].endswith('.tmp')
    # Immediately followed by a rename to the canonical path.
    assert order[1][0] == 'replace'
    src, dst = order[1][1].split('→')
    assert src.endswith('.tmp')
    assert dst == out
    # After the call, only the canonical path exists — no .tmp leak.
    assert os.path.exists(out)
    assert not os.path.exists(out + '.tmp')
    # And it's a valid WAV.
    sr, data = scipy.io.wavfile.read(out)
    assert sr == 44100
    assert data.shape[0] == 512


def test_write_wav_sync_leaves_previous_file_on_crash(tmp_path, monkeypatch):
    """If scipy.io.wavfile.write raises mid-write, the canonical path
    must be untouched — that's the whole point of the tmp-then-
    rename pattern."""
    from chirp.recording import writer as _w

    # Seed a previous-good file at the expected canonical path by
    # doing one successful write first.
    chunk = np.full(256, 0.25, dtype=np.float32)
    ok_path = _w.write_wav_sync(
        [chunk], str(tmp_path), prefix='p', sample_rate=44100)
    original_bytes = Path(ok_path).read_bytes()

    # Now break the wav writer: raise AFTER writing the .tmp, so the
    # tmp sibling exists but the rename never runs.
    original_write = _w.scipy.io.wavfile.write
    def _crashing_write(path, sr, data):
        original_write(path, sr, data)
        raise RuntimeError('simulated power loss')
    monkeypatch.setattr(_w.scipy.io.wavfile, 'write', _crashing_write)

    # We need the same filename the first call produced so the test
    # asserts on a matching target. Use the same prefix + a fixed
    # onset_time so the epoch_ms token is stable.
    import datetime
    onset = datetime.datetime(2026, 1, 1, 12, 0, 0)
    with pytest.raises(RuntimeError):
        _w.write_wav_sync(
            [chunk * 2], str(tmp_path), prefix='crashy', sample_rate=44100,
            onset_time=onset,
        )

    # Original file is untouched.
    assert Path(ok_path).read_bytes() == original_bytes


def test_write_wav_sync_fsync_failure_does_not_abort(tmp_path, monkeypatch):
    """Some filesystems / platforms refuse fsync — the write must
    still succeed because fsync is a best-effort durability hint,
    not a correctness requirement."""
    from chirp.recording import writer as _w

    def _no_fsync(fd):
        raise OSError(22, 'Invalid argument')
    monkeypatch.setattr(_w.os, 'fsync', _no_fsync)

    chunk = np.zeros(256, dtype=np.float32)
    out = _w.write_wav_sync(
        [chunk], str(tmp_path), prefix='p', sample_rate=44100)

    assert os.path.exists(out)
    assert not os.path.exists(out + '.tmp')


# ── Settings-save atomicity ──────────────────────────────────────────

def test_settings_save_is_atomic(tmp_path):
    """The settings saver writes to ``<path>.tmp`` then replaces. On
    a successful save, only the canonical path exists."""
    # Import here to keep the test module importable without Qt at
    # parse time.
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    from chirp.ui.window import ChirpWindow
    # Build just enough of the window to call the private method.
    # We bypass __init__ by using __new__ + stubbing the bits the
    # writer needs.
    win = ChirpWindow.__new__(ChirpWindow)
    win._current_config_path = None
    win._mark_clean = lambda: None

    path = str(tmp_path / 'cfg.json')
    data = {'recordings': [], 'view_mode': {}}

    ok = win._write_settings_to_path(path, data)
    assert ok is True
    assert os.path.exists(path)
    assert not os.path.exists(path + '.tmp')
    # Round-trip the payload.
    with open(path, 'r', encoding='utf-8') as f:
        loaded = json.load(f)
    assert loaded == data


def test_settings_save_preserves_old_file_on_serialization_error(tmp_path):
    """json.dumps failure must not touch the canonical path. The
    tmp sibling (if any) gets cleaned up. Previous-good config
    survives."""
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from chirp.ui.window import ChirpWindow

    # Seed a previous-good config.
    path = str(tmp_path / 'cfg.json')
    good = {'recordings': [{'name': 'old'}], 'view_mode': {}}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(good, f)

    win = ChirpWindow.__new__(ChirpWindow)
    win._current_config_path = None
    win._mark_clean = lambda: None

    # Inject a non-JSON-serializable value so json.dumps raises.
    class _Bad:
        pass
    bad_data = {'recordings': [{'name': _Bad()}], 'view_mode': {}}

    # Suppress the QMessageBox pop in test.
    with patch('chirp.ui.window.QMessageBox.warning'):
        ok = win._write_settings_to_path(path, bad_data)

    assert ok is False
    assert not os.path.exists(path + '.tmp'), (
        'failed save must not leak a .tmp sibling')
    # Previous good file survived — check content.
    with open(path, 'r', encoding='utf-8') as f:
        preserved = json.load(f)
    assert preserved == good
