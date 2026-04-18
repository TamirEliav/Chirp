"""Tests that ``ChirpWindow._load_settings`` actually routes through
``chirp.config.schema.load_settings_dict`` (#55).

The schema's versioning, migration chain, and unknown-key warnings
existed since #22 / c17 but the real loader bypassed them — it called
``json.load`` and instantiated ``RecordingEntity.from_dict`` directly,
leaving the schema validation as dead code only exercised by
``test_config_schema.py``.

Consequences before the fix:

  - A settings JSON with ``"version": 9999`` loaded silently — the
    "newer Chirp" guard at the top of ``load_settings_dict`` never
    fired on the real path.
  - Unknown top-level keys never produced user-visible warnings.
  - A malformed ``view_mode`` (string instead of dict) raised
    ``AttributeError`` mid-load and left ``self._entities = []``,
    breaking the UI.

Tested here:

  - The real loader calls ``load_settings_dict`` exactly once.
  - A future-version JSON triggers the rejection modal and the
    existing entities are preserved (no half-load).
  - Unknown-key warnings reach a ``QMessageBox.information`` modal.
  - A malformed ``view_mode`` value (string) does NOT raise — the
    schema's defensive defaults kick in.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest


# ── Helpers ──────────────────────────────────────────────────────────

def _make_window():
    """Construct a ChirpWindow without QMainWindow.__init__ so we can
    exercise ``_load_settings`` in headless tests. Stubs the bare
    minimum widgets the loader touches."""
    from chirp.ui.window import ChirpWindow
    win = ChirpWindow.__new__(ChirpWindow)

    # Timer + entities + sidebar
    win._timer = MagicMock()
    win._entities = []
    win._selected_idx = -1
    win._monitor = MagicMock()
    win._monitor.set_source = MagicMock()
    win._sidebar = MagicMock()
    win._sidebar.clear_all = MagicMock()
    win._sidebar.add_item = MagicMock()
    win._sidebar.select = MagicMock()
    win._next_num = 1

    # View-mode spinboxes
    win._vm_n_cols = 1
    win._vm_panel_height = 300
    win._vm_cols_spin = MagicMock()
    win._vm_height_spin = MagicMock()

    # Methods invoked at end of load
    win._refresh_monitor_source_combo = MagicMock()
    win._load_params_from_entity = MagicMock()
    win._setup_axes = MagicMock()
    win._update_spec_yticks = MagicMock()
    win._refresh_transport_ui = MagicMock()
    win._setup_view_mode_axes = MagicMock()
    win._mark_clean = MagicMock()
    win._view_mode = False
    win._current_config_path = None
    return win


@contextmanager
def _patched_load(file_contents: dict, message_box_capture: list):
    """Patch QFileDialog to return a path, open() to return file_contents
    (encoded JSON), and capture any QMessageBox calls into the list.

    Yields the (msg_box_warn, msg_box_info) MagicMocks for assertion."""
    payload = json.dumps(file_contents)

    def _open_text(*args, **kwargs):
        from io import StringIO
        return StringIO(payload)

    msg_warn = MagicMock(side_effect=lambda *a, **k:
                         message_box_capture.append(('warn', a, k)))
    msg_info = MagicMock(side_effect=lambda *a, **k:
                         message_box_capture.append(('info', a, k)))

    with patch('chirp.ui.window.QFileDialog.getOpenFileName',
               return_value=('/fake/settings.json', '')), \
         patch('chirp.ui.window.open', _open_text, create=True), \
         patch('chirp.ui.window.QMessageBox.warning', msg_warn), \
         patch('chirp.ui.window.QMessageBox.information', msg_info):
        yield (msg_warn, msg_info)


# ── #55: load_settings_dict is actually called ───────────────────────

def test_real_loader_routes_through_load_settings_dict():
    """Smoke test: a happy-path load invokes ``load_settings_dict``
    exactly once with the parsed JSON dict."""
    win = _make_window()
    payload = {'version': 1, 'recordings': [],
               'view_mode': {'columns': 2, 'panel_height': 250}}
    captured = []

    real_load = None
    spy = MagicMock()
    def _spy_loader(data):
        spy(data)
        from chirp.config.schema import load_settings_dict as _real
        return _real(data)

    with _patched_load(payload, captured), \
         patch('chirp.config.load_settings_dict', _spy_loader):
        win._load_settings()

    spy.assert_called_once()
    # The argument is the parsed JSON dict.
    arg = spy.call_args[0][0]
    assert arg.get('version') == 1
    # And the view_mode actually took effect (proves the result was
    # used, not discarded).
    assert win._vm_n_cols == 2
    assert win._vm_panel_height == 250


# ── #55: future-version file is rejected ─────────────────────────────

def test_future_version_settings_file_is_rejected():
    """A settings file with a version newer than this build
    understands must trigger the schema's "newer Chirp" guard — the
    pre-fix loader silently accepted such files, dropping any new
    keys on the floor."""
    win = _make_window()
    # Pre-existing entity that must NOT be wiped out by a failed load.
    pre_existing = MagicMock()
    pre_existing.name = 'pre'
    win._entities.append(pre_existing)

    payload = {'version': 9999, 'recordings': []}
    captured = []
    with _patched_load(payload, captured):
        win._load_settings()

    # A warning modal fired with the schema's error message.
    warn_calls = [c for c in captured if c[0] == 'warn']
    assert len(warn_calls) == 1
    args = warn_calls[0][1]
    # QMessageBox.warning(self, title, message) — message contains 'newer'
    assert 'newer' in args[2].lower()

    # And the existing entity was NOT closed / wiped — the loader
    # must bail BEFORE tearing down state when the schema rejects.
    assert win._entities == [pre_existing]
    pre_existing.stop_acq.assert_not_called()
    pre_existing.close.assert_not_called()


# ── #55: unknown-key warnings reach the user ─────────────────────────

def test_unknown_top_level_key_warns_user():
    """An unknown top-level key in the JSON must surface in a
    QMessageBox.information modal — not just print to stdout."""
    win = _make_window()
    payload = {
        'version': 1,
        'recordings': [],
        'view_mode': {},
        'a_made_up_field_from_some_fork': 'hello',
    }
    captured = []
    with _patched_load(payload, captured):
        win._load_settings()

    info_calls = [c for c in captured if c[0] == 'info']
    assert len(info_calls) == 1
    body = info_calls[0][1][2]
    assert 'a_made_up_field_from_some_fork' in body


# ── #55: malformed view_mode is recovered, not crashed ───────────────

def test_view_mode_string_does_not_crash():
    """Pre-fix: a JSON with ``view_mode: "hello"`` raised
    AttributeError mid-load (called ``.get`` on a string), leaving
    self._entities = [] and the UI in a broken state. The schema
    defends against this — an info modal warns and defaults are
    used."""
    win = _make_window()
    payload = {
        'version': 1,
        'recordings': [],
        'view_mode': 'hello-this-is-the-wrong-type',
    }
    captured = []
    with _patched_load(payload, captured):
        win._load_settings()  # MUST NOT raise

    # Defaults applied
    assert win._vm_n_cols == 1
    assert win._vm_panel_height == 300
    # And the user got a warning about the bad view_mode
    info_calls = [c for c in captured if c[0] == 'info']
    assert len(info_calls) == 1
    body = info_calls[0][1][2]
    assert 'view_mode' in body.lower()


# ── #55: malformed JSON shape is rejected cleanly ───────────────────

def test_missing_recordings_key_is_rejected():
    """A JSON missing ``'recordings'`` must be rejected by the schema
    (not silently accepted). Existing UI state is preserved."""
    win = _make_window()
    payload = {'version': 1}  # no 'recordings'
    captured = []
    with _patched_load(payload, captured):
        win._load_settings()

    warn_calls = [c for c in captured if c[0] == 'warn']
    assert len(warn_calls) == 1
    # Timer was never stopped (the loader bailed before teardown).
    win._timer.stop.assert_not_called()
