"""Tests for ``_update_plot`` top-level exception guard (#58).

Pre-fix: the matplotlib blit body of ``_update_plot`` was unguarded.
A single matplotlib / numpy / shape exception (NaN sample, axes
referenced before a View↔Config rebuild completes, shape mismatch
between a newly-reallocated buffer and a cached artist) propagated
out of the slot. Qt logged the traceback to stderr (invisible in
packaged builds) and kept firing the timer, but the half-finished
slot left blit-cache invariants broken (``_axes_changed`` half-set,
background bbox captured mid-reallocation). Subsequent ticks
compounded the problem; the display froze. The user assumed the app
was dead and force-killed it — orphaning the writer pool (per #56)
and silently losing in-flight events.

Post-fix:

  - Body is wrapped in a top-level try/except.
  - On exception: ``_canvas.draw()`` is attempted to re-baseline the
    blit cache, the consecutive-error counter is bumped, and after
    ``_update_plot_freeze_threshold`` straight failures a sticky note
    appears in ``_lbl_trig_status`` ("DISPLAY HALTED — acquisition
    still running") so the user knows the mic and writer are still
    working and they should NOT force-kill.
  - Counter resets on the next successful tick — a transient blip
    (one bad chunk) does not leave the sticky note up after recovery.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _make_window():
    from chirp.ui.window import ChirpWindow
    win = ChirpWindow.__new__(ChirpWindow)

    # Sidebar polling section needs these attrs.
    win._entities = []
    win._sidebar = MagicMock()
    win._update_error_sticky = MagicMock()

    # Body branch needs these.
    win._view_mode = False
    win._update_wav_time_label = MagicMock()
    win._selected_idx = -1
    win._bg = None  # short-circuits the blit branch
    win._chk_ref_date = MagicMock()
    win._chk_ref_date.isChecked = MagicMock(return_value=False)
    win._lbl_trig_status = MagicMock()
    win._lbl_trig_status.style = MagicMock(return_value=MagicMock())
    win._lbl_entropy = MagicMock()

    # Counters initialised normally by _build_status_panel.
    win._blink_counter = 0
    win._update_plot_err_count = 0
    win._update_plot_err_total = 0
    win._update_plot_last_err = None
    win._update_plot_freeze_threshold = 5

    # Canvas / fig — the error-recovery path tries to draw().
    win._canvas = MagicMock()
    win._fig = MagicMock()
    return win


# ── Top-level guard ─────────────────────────────────────────────────

def test_update_plot_does_not_propagate_body_exception():
    """The raw exception from the body must NOT propagate out of the
    slot. Qt would log it to stderr (invisible in packaged builds)
    and the user's only signal would be a frozen display."""
    win = _make_window()
    boom = RuntimeError('shape mismatch in spec_buffer')
    with patch.object(type(win), '_update_plot_body',
                      side_effect=boom):
        # MUST NOT raise.
        win._update_plot()

    assert win._update_plot_err_count == 1
    assert win._update_plot_err_total == 1
    assert 'shape mismatch' in win._update_plot_last_err


def test_update_plot_re_baselines_blit_cache_on_error():
    """On error, ``_canvas.draw()`` runs to re-baseline the blit
    state — otherwise the next tick restores from a half-built bg
    bbox and corrupts the display further."""
    win = _make_window()
    with patch.object(type(win), '_update_plot_body',
                      side_effect=RuntimeError('x')):
        win._update_plot()

    win._canvas.draw.assert_called_once()
    win._canvas.copy_from_bbox.assert_called_once()


def test_update_plot_consecutive_errors_show_sticky_note():
    """After ``freeze_threshold`` straight failures, the trigger
    status label must show "DISPLAY HALTED" so the user doesn't
    force-kill thinking the app is dead."""
    win = _make_window()
    win._update_plot_freeze_threshold = 3
    with patch.object(type(win), '_update_plot_body',
                      side_effect=RuntimeError('x')):
        for _ in range(3):
            win._update_plot()

    assert win._update_plot_err_count == 3
    # The sticky note text was set on the trigger-status label.
    set_text_calls = [c.args[0]
                      for c in win._lbl_trig_status.setText.call_args_list]
    assert any('DISPLAY HALTED' in t for t in set_text_calls)


def test_update_plot_does_not_show_sticky_below_threshold():
    """A single transient blip must NOT show the sticky note —
    otherwise the badge thrashes on every tick that hits a one-off
    NaN sample."""
    win = _make_window()
    win._update_plot_freeze_threshold = 5
    with patch.object(type(win), '_update_plot_body',
                      side_effect=RuntimeError('x')):
        win._update_plot()  # 1 error, well under threshold

    set_text_calls = [c.args[0]
                      for c in win._lbl_trig_status.setText.call_args_list]
    assert not any('DISPLAY HALTED' in t for t in set_text_calls)


def test_update_plot_recovery_resets_counter():
    """When the next tick succeeds, the consecutive-error counter
    must reset so a brief incident doesn't leave the sticky note up
    forever."""
    win = _make_window()
    win._update_plot_err_count = 4

    # Now run a successful tick.
    win._update_plot()

    assert win._update_plot_err_count == 0
    # Total stays — that's a session-wide counter for diagnostics.
    assert win._update_plot_err_total == 0  # was never bumped


def test_update_plot_recovery_after_failures():
    """Two failed ticks then a successful tick — counter must reset
    on the success."""
    win = _make_window()
    win._update_plot_freeze_threshold = 10  # high so we never trip the note

    real_body = type(win)._update_plot_body
    fail = [True, True, False]  # third call succeeds
    def _maybe_fail(self):
        f = fail.pop(0)
        if f:
            raise RuntimeError('flaky')
        # Simulate the real body's success path by calling the
        # pieces that the counter-reset depends on. The reset
        # itself runs at the end of the body, so call the real
        # body here (it'll exit early because _bg is None).
        return real_body(self)

    with patch.object(type(win), '_update_plot_body', _maybe_fail):
        win._update_plot()
        win._update_plot()
        assert win._update_plot_err_count == 2  # mid-failure
        win._update_plot()  # third tick succeeds

    assert win._update_plot_err_count == 0
    assert win._update_plot_err_total == 2


def test_update_plot_canvas_draw_failure_is_swallowed():
    """If even ``_canvas.draw()`` raises during recovery, the slot
    must STILL not propagate — the next tick will simply skip the
    blit branch (``self._bg is None``) until the user toggles a
    redraw."""
    win = _make_window()
    win._canvas.draw = MagicMock(side_effect=RuntimeError('draw failed too'))
    with patch.object(type(win), '_update_plot_body',
                      side_effect=RuntimeError('first')):
        # MUST NOT raise.
        win._update_plot()

    assert win._update_plot_err_count == 1
