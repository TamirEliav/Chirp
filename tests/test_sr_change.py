"""Tests for sample-rate change hardening (#46).

Three failure modes used to stack when the session SR changed mid-run:

  1. Mixed-SR WAV corruption. ``change_sample_rate`` drained the
     queue and rebuilt buffers but left ``recorder._active_events``
     intact — pre-trigger samples in the lookback deque were
     captured at the old SR; samples appended after the rebuild at
     the new SR. The resulting WAV had a single header tagged at
     SR-at-flush-time, playing the pre-trigger portion at the wrong
     speed.

  2. Reentrancy. ``change_sample_rate`` is a multi-second operation
     (close stream, drain, rebuild, reopen). The combo signal was
     not blocked — a rapid wheel-scroll could re-enter while the
     prior call was inside ``sd.InputStream.close()``, causing a
     double-close / PortAudio crash.

  3. No sync. Other sync-eligible panels (freq filter, FFT) propagate
     across all entities when "Sync settings" is on. SR change
     applied only to ``self._sel``, silently leaving other streams
     at mismatched rates.

The fix adds three belts:

  - Each event dict now carries ``'sample_rate'`` set at open time,
    and ``_flush_event`` uses it instead of the caller-supplied
    session SR — even if a flush slips past the PR 2 teardown hook,
    the WAV header is still labelled with the true capture rate.
  - ``_on_sample_rate_changed`` is gated by ``_sr_change_busy`` and
    disables the SR combo for the duration of the rebuild.
  - When ``_chk_shared_spec`` is checked, every other entity is also
    re-rated to the new SR (each via its own per-entity flush path).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from chirp.constants import CHUNK_FRAMES, SAMPLE_RATE
from chirp.recording.entity import RecordingEntity
from chirp.recording.trigger import ThresholdRecorder


# ── Event-level SR pinning ───────────────────────────────────────────

@pytest.fixture
def captured_flushes(monkeypatch):
    """Replace ``ThresholdRecorder._start_flush`` with an in-memory
    capture."""
    flushes: list[dict] = []
    def _capture(buf_snapshot, output_dir, prefix='', suffix='',
                 sample_rate=SAMPLE_RATE, onset_time=None,
                 filename_stream=''):
        flushes.append({
            'sample_rate':     sample_rate,
            'filename_stream': filename_stream,
        })
    monkeypatch.setattr(
        ThresholdRecorder, '_start_flush', staticmethod(_capture))
    return flushes


def _loud_chunk() -> np.ndarray:
    return np.full(CHUNK_FRAMES, 0.5, dtype=np.float32)


def _drive_trigger_to_open(e: RecordingEntity, n_loud: int = 3) -> None:
    e.rec_enabled = True
    for i in range(n_loud):
        e.recorder.process_chunk(
            _loud_chunk(), trigger_peak=0.5, threshold=e.threshold,
            min_cross_sec=e.min_cross_sec, hold_sec=e.hold_sec,
            post_trig_sec=e.post_trig_sec, max_rec_sec=e.max_rec_sec,
            pre_trig_sec=e.pre_trig_sec, output_dir=e.output_dir,
            enabled=True, filename_prefix=e.filename_prefix,
            filename_suffix=e.filename_suffix,
            sample_rate=e.sample_rate, filename_stream=e.name,
            global_chunk_end=(i + 1) * CHUNK_FRAMES,
        )


def test_event_carries_sample_rate_at_open_time(captured_flushes):
    """Each event dict stores the SR captured at its open — so a
    later flush with a *different* ``sample_rate`` argument still
    writes the WAV at the original rate."""
    e = RecordingEntity(name='t', device_id=None, sample_rate=44100)
    _drive_trigger_to_open(e, n_loud=3)
    assert e.recorder.is_recording
    # The event should have sample_rate=44100 pinned.
    ev = e.recorder._active_events[0]
    assert ev['sample_rate'] == 44100

    # Now flush with a DIFFERENT sample_rate — the pinned one should
    # win.  (This is the belt that protects against any code path
    # that misses the PR 2 pre-SR-change flush.)
    e.recorder.flush_all(
        output_dir=e.output_dir, filename_prefix='p',
        sample_rate=48000,  # <<< different from event's pinned rate
        filename_stream=e.name, reason='unit-test',
    )
    assert len(captured_flushes) == 1
    assert captured_flushes[0]['sample_rate'] == 44100, (
        'flush used the session SR instead of the event-pinned SR — '
        'a mid-event SR change would mislabel the WAV')


def test_event_sample_rate_differs_per_event(captured_flushes):
    """Two events opened at different SRs must each retain their own
    pinned rate even if a single ``flush_all`` sweeps them together."""
    e = RecordingEntity(name='t', device_id=None, sample_rate=44100)
    _drive_trigger_to_open(e, n_loud=3)
    # Manually change pinned SR on the event (simulating what would
    # happen if a future refactor opened a second event at a
    # different rate).
    e.recorder._active_events[0]['sample_rate'] = 96000

    # Open a second event at 22050 by directly forging one — easier
    # than driving two full trigger sequences and enough to pin the
    # per-event behaviour.
    second_ev = dict(e.recorder._active_events[0])
    second_ev['sample_rate'] = 22050
    e.recorder._active_events.append(second_ev)

    e.recorder.flush_all(output_dir=e.output_dir, filename_prefix='p',
                         sample_rate=44100, filename_stream=e.name)

    rates = [f['sample_rate'] for f in captured_flushes]
    assert 96000 in rates
    assert 22050 in rates


# ── _on_sample_rate_changed reentrancy + sync ───────────────────────

def _make_window_for_sr_tests(n_entities: int, initial_sr: int = 44100):
    from chirp.ui.window import ChirpWindow
    win = ChirpWindow.__new__(ChirpWindow)

    # Build fake entities with the subset of attrs/methods used by
    # _on_sample_rate_changed.
    entities = []
    for i in range(n_entities):
        ent = MagicMock()
        ent.name = f'ent{i}'
        ent.sample_rate = initial_sr
        ent.channel_mode = 'Mono'
        ent.change_sample_rate = MagicMock(
            side_effect=lambda sr, _ent=ent: setattr(_ent, 'sample_rate', sr))
        entities.append(ent)
    win._entities = entities
    # _sel is a read-only property derived from _selected_idx.
    win._selected_idx = 0
    # getattr falls through to QMainWindow.__getattr__ which raises
    # when __init__ wasn't called — initialise the flag explicitly.
    win._sr_change_busy = False

    # Stub Qt widgets that closeEvent / SR change touch.
    win._sr_combo           = MagicMock()
    win._sb_freq_lo         = MagicMock()
    win._sb_freq_hi         = MagicMock()
    win._sb_disp_freq_lo    = MagicMock()
    win._sb_disp_freq_hi    = MagicMock()
    win._sb_disp_freq_hi.value = MagicMock(return_value=0)
    win._chk_shared_spec    = MagicMock()
    win._chk_shared_spec.isChecked = MagicMock(return_value=False)
    # Monitor stub
    win._monitor = MagicMock()
    win._monitor.source_id = None
    # Methods called near the end of the handler
    win._apply_monitor_source = MagicMock()
    win._setup_axes           = MagicMock()
    win._update_spec_yticks   = MagicMock()
    win._refresh_transport_ui = MagicMock()
    return win


def test_on_sr_change_disables_combo_and_calls_change(qapp=None):
    """The SR combo is disabled for the duration of the rebuild so a
    rapid re-entry can't fire while InputStream.close() is in
    progress."""
    win = _make_window_for_sr_tests(n_entities=1, initial_sr=44100)
    win._sr_combo.currentData = MagicMock(return_value=48000)

    # Observe whether setEnabled(False) was called.
    win._on_sample_rate_changed(0)

    win._sr_combo.setEnabled.assert_any_call(False)
    win._sr_combo.setEnabled.assert_any_call(True)
    # change_sample_rate did run on the selected entity.
    win._entities[0].change_sample_rate.assert_called_once_with(48000)


def test_on_sr_change_is_reentrancy_safe():
    """A reentrant call while ``_sr_change_busy`` is set must be a
    no-op — it must NOT re-enter change_sample_rate."""
    win = _make_window_for_sr_tests(n_entities=1, initial_sr=44100)
    win._sr_change_busy = True  # simulate a prior call still in progress
    win._sr_combo.currentData = MagicMock(return_value=48000)

    win._on_sample_rate_changed(0)

    win._entities[0].change_sample_rate.assert_not_called()


def test_on_sr_change_syncs_to_all_entities_when_shared():
    """When sync-settings is on, every other entity must also get
    change_sample_rate called with the new rate."""
    win = _make_window_for_sr_tests(n_entities=3, initial_sr=44100)
    win._chk_shared_spec.isChecked = MagicMock(return_value=True)
    win._sr_combo.currentData = MagicMock(return_value=22050)

    win._on_sample_rate_changed(0)

    # Every entity had change_sample_rate called.
    for ent in win._entities:
        ent.change_sample_rate.assert_called_once_with(22050)


def test_on_sr_change_does_not_sync_when_not_shared():
    """When sync-settings is off, only the selected entity changes."""
    win = _make_window_for_sr_tests(n_entities=3, initial_sr=44100)
    win._chk_shared_spec.isChecked = MagicMock(return_value=False)
    win._sr_combo.currentData = MagicMock(return_value=22050)

    win._on_sample_rate_changed(0)

    win._entities[0].change_sample_rate.assert_called_once_with(22050)
    win._entities[1].change_sample_rate.assert_not_called()
    win._entities[2].change_sample_rate.assert_not_called()


def test_on_sr_change_sync_failure_on_one_entity_does_not_abort_others():
    """If entity #1 throws inside change_sample_rate, entity #2 must
    still be re-rated — one device failure shouldn't leave the rest
    of the session at mismatched SRs."""
    win = _make_window_for_sr_tests(n_entities=3, initial_sr=44100)
    win._chk_shared_spec.isChecked = MagicMock(return_value=True)
    win._sr_combo.currentData = MagicMock(return_value=22050)
    # Poison entity #1.
    def _bad(sr): raise RuntimeError('simulated SR failure')
    win._entities[1].change_sample_rate = MagicMock(side_effect=_bad)

    # Must not propagate the exception.
    win._on_sample_rate_changed(0)

    # Entities #0 and #2 still changed.
    win._entities[0].change_sample_rate.assert_called_once_with(22050)
    win._entities[2].change_sample_rate.assert_called_once_with(22050)


def test_on_sr_change_does_not_touch_entities_already_at_new_rate():
    """If an entity is already at the target SR, sync must skip it
    (avoid an unnecessary rebuild / flush)."""
    win = _make_window_for_sr_tests(n_entities=3, initial_sr=44100)
    win._entities[2].sample_rate = 48000  # already there
    win._chk_shared_spec.isChecked = MagicMock(return_value=True)
    win._sr_combo.currentData = MagicMock(return_value=48000)

    win._on_sample_rate_changed(0)

    win._entities[0].change_sample_rate.assert_called_once_with(48000)
    win._entities[1].change_sample_rate.assert_called_once_with(48000)
    win._entities[2].change_sample_rate.assert_not_called()
