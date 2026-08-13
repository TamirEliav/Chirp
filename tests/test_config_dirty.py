"""Unsaved-changes tracking for structural config edits.

The bug this pins: changing a stream's INPUT DEVICE (e.g. moving the
same endpoint from WASAPI to WDM-KS) left ``_config_dirty`` False. Two
user-visible consequences, reported together:

  1. no "•" in the window title, so the change looked already-saved; and
  2. the Save button stays disabled while clean (``_update_save_button``),
     so pressing Save did nothing and the config file kept the old
     device.

A whole family of persisted edits had the same gap — device, channel
mode, trigger mode, sample rate, display buffer, display frequency
range, reference date, day-post-hatch prefix, WAV-simulation source and
loop flag, auto-calibrate, the view-mode grid geometry, and Reset. Each
one is checked here against the SAME rule: if ``to_dict()`` (or the
``view_mode`` section) would serialize differently afterwards, the
window must be dirty.

The tests drive the real handlers on a ``__new__``-constructed window
with stub widgets — the same technique as test_sr_change.py — because
the bug lives in the handlers, not in ``_mark_dirty`` itself.
"""

from __future__ import annotations

import datetime
from unittest.mock import MagicMock

import pytest

from chirp.ui.window import ChirpWindow


def _win(entity=None):
    """A ChirpWindow with real dirty tracking and stubbed Qt surfaces."""
    win = ChirpWindow.__new__(ChirpWindow)
    win._config_dirty = False
    # _mark_dirty fans out to three refreshers that touch Qt; the window
    # never ran QMainWindow.__init__, so setWindowTitle would raise.
    win._update_title = MagicMock()
    win._update_save_tooltip = MagicMock()
    win._update_save_button = MagicMock()

    ent = entity if entity is not None else MagicMock()
    win._entities = [ent]
    win._selected_idx = 0
    win._monitor = MagicMock()
    win._monitor.source_id = None
    win._busy_cursor = MagicMock()
    win._busy_cursor.return_value.__enter__ = MagicMock()
    win._busy_cursor.return_value.__exit__ = MagicMock(return_value=False)
    win._refresh_transport_ui = MagicMock()
    win._refresh_wav_controls = MagicMock()
    win._apply_monitor_source = MagicMock()
    win._apply_folder_validation = MagicMock()
    return win, ent


# ── Input device (the reported bug) ──────────────────────────────────

@pytest.mark.parametrize('open_ok', [True, False])
def test_device_change_marks_dirty(monkeypatch, open_ok):
    """Switching host API (WASAPI → WDM-KS) is a persisted change. It
    counts even when the open FAILED: ``change_device`` has already
    written ``device_id``, and that is what gets serialized."""
    win, ent = _win()
    ent.channel_mode = 'Mono'
    ent.change_device = MagicMock(return_value=open_ok)
    win._device_combo = MagicMock()
    win._device_combo.currentData = MagicMock(return_value=7)
    win._device_combo.currentText = MagicMock(return_value='X [WDM-KS]')
    win._chan_combo = MagicMock()
    win._trig_combo = MagicMock()
    monkeypatch.setattr('chirp.ui.window.sd.query_devices',
                        lambda idx: {'max_input_channels': 2})
    monkeypatch.setattr('chirp.ui.window.QMessageBox.warning',
                        lambda *a, **k: None)

    win._on_device_changed(0)

    ent.change_device.assert_called_once_with(7, 1)
    assert win._config_dirty is True


def test_wav_sim_selection_does_not_go_through_the_device_path():
    """The WAV-sim sentinel has its own handler; the device branch must
    not run (and must not dirty on its own — the sim handler does)."""
    win, ent = _win()
    win._device_combo = MagicMock()
    win._device_combo.currentData = MagicMock(
        return_value=ChirpWindow.WAV_SIM_SENTINEL)
    win._handle_wav_sim_selection = MagicMock()

    win._on_device_changed(0)

    win._handle_wav_sim_selection.assert_called_once_with(ent)


# ── The rest of the structural edits ─────────────────────────────────

def test_channel_mode_change_marks_dirty(monkeypatch):
    win, ent = _win()
    ent.channel_mode = 'Mono'
    ent.change_device = MagicMock(return_value=True)
    win._device_combo = MagicMock()
    win._device_combo.currentData = MagicMock(return_value=3)
    win._chan_combo = MagicMock()
    win._trig_combo = MagicMock()
    monkeypatch.setattr('chirp.ui.window.sd.query_devices',
                        lambda idx: {'max_input_channels': 2})

    win._on_channel_mode_changed('Stereo')

    assert win._config_dirty is True


def test_channel_mode_downgrade_to_mono_marks_dirty(monkeypatch):
    """The early-return branch (asked for stereo on a mono device) also
    writes ``channel_mode`` — it used to slip past the dirty mark."""
    win, ent = _win()
    ent.channel_mode = 'Mono'
    win._device_combo = MagicMock()
    win._device_combo.currentData = MagicMock(return_value=3)
    win._chan_combo = MagicMock()
    win._trig_combo = MagicMock()
    monkeypatch.setattr('chirp.ui.window.sd.query_devices',
                        lambda idx: {'max_input_channels': 1})

    win._on_channel_mode_changed('Stereo')

    assert ent.channel_mode == 'Mono'
    assert win._config_dirty is True


def test_trigger_mode_change_marks_dirty():
    win, ent = _win()
    win._on_trigger_mode_changed('Both Channels')
    assert ent.trigger_mode == 'Both Channels'
    assert win._config_dirty is True


def test_display_buffer_change_marks_dirty():
    win, ent = _win()
    ent.display_seconds = 10.0
    win._buf_combo = MagicMock()
    win._buf_combo.currentData = MagicMock(return_value=30.0)

    win._on_display_buffer_changed(0)

    ent.change_display_seconds.assert_called_once_with(30.0)
    assert win._config_dirty is True


def test_display_freq_range_change_marks_dirty():
    win, ent = _win()
    win._sb_disp_freq_lo = MagicMock()
    win._sb_disp_freq_lo.value = MagicMock(return_value=100.0)
    win._sb_disp_freq_hi = MagicMock()
    win._sb_disp_freq_hi.value = MagicMock(return_value=9000.0)

    win._on_disp_freq_changed(0)

    assert (ent.display_freq_lo, ent.display_freq_hi) == (100.0, 9000.0)
    assert win._config_dirty is True


def test_ref_date_toggle_and_edit_mark_dirty():
    win, ent = _win()
    win._date_line = MagicMock()
    win._date_line.text = MagicMock(return_value='2026-01-05')
    win._btn_pick_date = MagicMock()
    win._dph_prefix_edit = MagicMock()
    win._dph_prefix_edit.text = MagicMock(return_value='day_')
    win._lbl_day_count = MagicMock()
    win._chk_ref_date = MagicMock()
    win._chk_ref_date.isChecked = MagicMock(return_value=True)

    win._on_ref_date_toggled(True)
    assert ent.ref_date == datetime.date(2026, 1, 5)
    assert win._config_dirty is True

    win._config_dirty = False
    win._on_dph_prefix_changed()
    assert ent.dph_folder_prefix == 'day_'
    assert win._config_dirty is True

    win._config_dirty = False
    win._date_line.text = MagicMock(return_value='2026-02-09')
    win._on_ref_date_text_changed()
    assert ent.ref_date == datetime.date(2026, 2, 9)
    assert win._config_dirty is True


def test_wav_loop_toggle_marks_dirty():
    win, ent = _win()
    ent.capture = MagicMock()
    win._on_wav_loop_toggled(False)
    assert ent.wav_loop is False
    assert win._config_dirty is True


def test_view_mode_geometry_marks_dirty():
    """columns / tile height live in the persisted ``view_mode`` section
    even though they are not per-stream."""
    win, _ent = _win()
    win._view_mode = False
    win._pg_grid = None

    win._on_vm_cols_changed(3)
    assert win._vm_n_cols == 3
    assert win._config_dirty is True

    win._config_dirty = False
    win._on_vm_height_changed(420)
    assert win._vm_panel_height == 420
    assert win._config_dirty is True


def test_autocalibrate_result_marks_dirty():
    """Auto-calibrate writes ``threshold`` through ``_set_thr_silent``,
    which deliberately blocks the spinbox signal — so the dirty mark the
    spinbox handler would have made has to happen here."""
    import numpy as np

    win, ent = _win()
    win._btn_calibrate = MagicMock()
    win._lbl_calib_status = MagicMock()
    win._sb_calib_margin = MagicMock()
    win._sb_calib_margin.value = MagicMock(return_value=3.0)
    win._set_thr_silent = MagicMock()
    win._sync_thr_line = MagicMock()
    win._calib_samples = [0.01, 0.02, 0.015]

    win._finish_calibrate()

    assert ent.threshold == pytest.approx(
        min(1.0, float(np.percentile(win._calib_samples, 95)) * 3.0))
    assert win._config_dirty is True


def test_reset_params_writes_text_fields_and_marks_dirty():
    """Reset uses setText(), which does NOT emit ``editingFinished`` —
    the folder / prefix / suffix / DPH values have to be pushed into the
    entity by hand or the fields and the entity disagree."""
    from chirp.constants import RECORDINGS_DIR

    win, ent = _win()
    ent.output_dir = r'D:\old'
    ent.filename_prefix = 'old'
    ent.filename_suffix = 'old'
    ent.dph_folder_prefix = 'old'
    ent.sample_rate = 44100

    for name in ('_sb_thr', '_sb_mc', '_sb_min_total', '_sb_hold',
                 '_sb_post_trig', '_sb_maxr', '_sb_pre', '_chk_freq',
                 '_sb_freq_lo', '_sb_freq_hi', '_combo_detect_mode',
                 '_sb_entropy_thr', '_sb_entropy_mc', '_combo_rec_mode',
                 '_sb_gain', '_sb_floor', '_sb_ceil', '_combo_fft',
                 '_combo_win', '_combo_fscale', '_sb_disp_freq_lo',
                 '_sb_disp_freq_hi', '_chk_ref_date', '_date_line',
                 '_combo_display_mode'):
        setattr(win, name, MagicMock())
    win._folder_edit = MagicMock()
    win._folder_edit.text = MagicMock(return_value='')
    win._prefix_edit = MagicMock()
    win._prefix_edit.text = MagicMock(return_value='')
    win._suffix_edit = MagicMock()
    win._suffix_edit.text = MagicMock(return_value='')
    win._dph_prefix_edit = MagicMock()
    win._dph_prefix_edit.text = MagicMock(return_value='')

    win._on_reset_params()

    assert ent.output_dir == RECORDINGS_DIR
    assert ent.filename_prefix == ''
    assert ent.filename_suffix == ''
    assert ent.dph_folder_prefix == ''
    assert win._config_dirty is True
