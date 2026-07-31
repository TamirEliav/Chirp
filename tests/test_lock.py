"""Per-stream parameter-lock UI behaviour.

Covers the pieces that don't need the full ChirpWindow (which segfaults
at pytest teardown with many Qt widgets — see test_pg_window_integration):
the sidebar item's lock toggle + rename guard, and the config-table
lock-exemption set. Full end-to-end wiring is exercised by launching the
app.
"""

import pytest

pytest.importorskip('pyqtgraph')

from PyQt5.QtWidgets import QApplication  # noqa: E402

from chirp.ui.sidebar import RecordingSidebarItem  # noqa: E402


def _app():
    return QApplication.instance() or QApplication([])


def test_sidebar_item_lock_toggle_and_signal():
    _app()
    item = RecordingSidebarItem(0, 'Stream A')
    fired = []
    item.toggle_lock.connect(fired.append)
    # Defaults unlocked, 🔓 glyph.
    assert item.params_locked is False
    assert item._btn_lock.text() == '\U0001f513'
    # Programmatic lock flips the glyph to 🔒.
    item.set_params_locked(True)
    assert item.params_locked is True
    assert item._btn_lock.text() == '\U0001f512'
    # Clicking the button requests a toggle (window owns the confirm).
    item._btn_lock.click()
    assert fired == [0]


def test_sidebar_item_rename_blocked_while_locked():
    _app()
    item = RecordingSidebarItem(1, 'Stream B')
    item.set_params_locked(True)
    # A locked item ignores the double-click rename entry.
    item._start_edit()
    assert item._editing is False
    # Unlocking restores renaming.
    item.set_params_locked(False)
    item._start_edit()
    assert item._editing is True


def test_config_table_lock_exempts_display_keys():
    from chirp.ui.config_table import _LOCK_EXEMPT_KEYS, _ROWS
    # Display-group params + the enable switch stay editable when locked.
    for key in ('display_mode', 'gain_db', 'db_floor', 'db_ceil',
                'freq_scale', 'spec_nperseg', 'spec_window',
                'display_freq_lo', 'display_freq_hi', 'amp_scale',
                'display_seconds', 'stream_enabled'):
        assert key in _LOCK_EXEMPT_KEYS
    # Trigger / output params are NOT exempt — they lock.
    for key in ('threshold', 'min_cross_sec', 'hold_sec', 'max_rec_sec',
                'output_dir', 'filename_prefix', 'freq_lo',
                'spectral_threshold'):
        assert key not in _LOCK_EXEMPT_KEYS
    # Every exempt key is a real row key (guards against typos/drift).
    row_keys = {k for k, *_ in _ROWS if k is not None}
    assert _LOCK_EXEMPT_KEYS <= row_keys
