"""⚙ Advanced dialog — the AMPLITUDE ENVELOPE and LOST-DEVICE
AUTO-RECONNECT groups.

The dialog body is inline in ``ChirpWindow._open_advanced_settings``, so
the only way to catch a typo in a widget name or a missed apply is to
actually build it. These tests run it against a real QApplication with
``exec_`` patched, then assert on what reached ``_audio_cfg`` and the
module-level ``chirp.dsp.envelope`` state.

Covered: OK applies the amplitude-envelope choice and the lost-device
auto-reconnect switch; Cancel applies neither; and the dialog opens with
whatever is currently in force (not with the defaults).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow

from chirp.config.schema import DEFAULT_AUDIO
from chirp.dsp import envelope as env_mod
from chirp.ui.window import ChirpWindow


@pytest.fixture(scope='module')
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _restore_globals():
    import chirp.audio.shared_stream as shared
    env_before = env_mod.current_params()
    cap_before = shared.current_params()
    yield
    env_mod.configure(*env_before)
    shared.configure(*cap_before)


def _win():
    win = ChirpWindow.__new__(ChirpWindow)
    # The dialog is parented to the window, so the Qt side has to exist —
    # but ChirpWindow.__init__ would build the whole app (and open audio
    # devices). Initialise only the QMainWindow base.
    QMainWindow.__init__(win)
    win._audio_cfg = dict(DEFAULT_AUDIO)
    win._zero_high_since = {}
    win._config_dirty = False
    win._update_title = MagicMock()
    win._update_save_tooltip = MagicMock()
    win._update_save_button = MagicMock()
    return win


def _open(win, accept: bool, tweak=None):
    """Run the dialog, optionally mutating its widgets first."""
    captured = {}

    def _fake_exec(self):
        captured['dlg'] = self
        if tweak is not None:
            tweak(self)
        return QDialog.Accepted if accept else QDialog.Rejected

    with patch.object(QDialog, 'exec_', _fake_exec):
        win._open_advanced_settings()
    return captured['dlg']


def _widgets(dlg):
    """Map the dialog's controls by the label of the group they sit in.
    Keyed on visible text so a renamed internal variable can't silently
    make a test pass against the wrong widget."""
    from PyQt5.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QGroupBox
    out = {}
    for box in dlg.findChildren(QGroupBox):
        out[box.title()] = {
            'checks': box.findChildren(QCheckBox),
            'combos': box.findChildren(QComboBox),
            'spins': box.findChildren(QDoubleSpinBox),
        }
    return out


def test_dialog_has_the_new_groups(qapp):
    # Keep the window alive: the dialog is its child, and letting the
    # parent be collected deletes the C++ dialog out from under us.
    win = _win()
    dlg = _open(win, accept=False)
    titles = set(_widgets(dlg))
    assert 'AMPLITUDE ENVELOPE (TRIGGER)' in titles
    assert 'LOST-DEVICE AUTO-RECONNECT' in titles


def test_ok_applies_envelope_choice(qapp):
    win = _win()

    def tweak(dlg):
        g = _widgets(dlg)['AMPLITUDE ENVELOPE (TRIGGER)']
        combo = g['combos'][0]
        combo.setCurrentIndex(combo.findData('rectify'))
        g['spins'][0].setValue(120.0)

    _open(win, accept=True, tweak=tweak)

    assert env_mod.current_params() == ('rectify', 120.0)
    assert win._audio_cfg['envelope_method'] == 'rectify'
    assert win._audio_cfg['envelope_cutoff_hz'] == 120.0
    assert win._config_dirty is True


def test_ok_applies_reconnect_switch(qapp):
    win = _win()

    def tweak(dlg):
        chk = _widgets(dlg)['LOST-DEVICE AUTO-RECONNECT']['checks'][0]
        chk.setChecked(False)

    _open(win, accept=True, tweak=tweak)

    assert win._audio_cfg['auto_recover_capture_stall'] is False


def test_cancel_applies_nothing(qapp):
    win = _win()
    env_mod.configure('hilbert', 50.0)

    def tweak(dlg):
        g = _widgets(dlg)['AMPLITUDE ENVELOPE (TRIGGER)']
        g['combos'][0].setCurrentIndex(g['combos'][0].findData('rectify'))
        _widgets(dlg)['LOST-DEVICE AUTO-RECONNECT']['checks'][0].setChecked(False)

    _open(win, accept=False, tweak=tweak)

    assert env_mod.current_params() == ('hilbert', 50.0)
    assert win._audio_cfg['auto_recover_capture_stall'] is True
    assert win._config_dirty is False


def test_dialog_opens_showing_current_state(qapp):
    """Reopening must show what is in force, not the defaults — the
    classic way a settings dialog silently reverts a user's choice."""
    win = _win()
    env_mod.configure('rectify', 90.0)
    win._audio_cfg['auto_recover_capture_stall'] = False

    dlg = _open(win, accept=False)
    g = _widgets(dlg)
    assert g['AMPLITUDE ENVELOPE (TRIGGER)']['combos'][0].currentData() == 'rectify'
    assert g['AMPLITUDE ENVELOPE (TRIGGER)']['spins'][0].value() == 90.0
    assert g['LOST-DEVICE AUTO-RECONNECT']['checks'][0].isChecked() is False


def test_cutoff_spinbox_disabled_under_hilbert(qapp):
    """The cutoff only means something for the rectify follower."""
    win = _win()
    env_mod.configure('hilbert', 50.0)
    dlg = _open(win, accept=False)
    g = _widgets(dlg)['AMPLITUDE ENVELOPE (TRIGGER)']
    assert g['spins'][0].isEnabled() is False
    g['combos'][0].setCurrentIndex(g['combos'][0].findData('rectify'))
    assert g['spins'][0].isEnabled() is True
