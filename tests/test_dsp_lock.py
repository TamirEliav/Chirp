"""Tests for DSP-lock + ingest-thread hardening (#53).

Two concurrency bugs used to corrupt DSP state and silently double-
spawn the ingest thread:

  1. UI-thread rebuild paths (``change_fft_params``,
     ``change_analysis_fft_params``, ``change_display_seconds``)
     reassigned ``spec_acc`` / ``spec_buffer`` / ring buffers WITHOUT
     coordinating with the live ingest thread. A chunk processed
     concurrently could read a half-freed buffer, crash with a shape
     mismatch (swallowed by #44), or worst — compute an FFT column
     from a freshly-zeroed overlap, silently suppressing the
     spectral trigger for the next ``nperseg`` samples.

  2. ``stop_acq`` joined the ingest thread with a 2-second timeout.
     If the in-flight ``ingest_chunk`` took longer, ``_ingest_thread``
     was set to ``None`` while the thread was still alive. A
     subsequent ``start_acq`` spawned a SECOND thread — both drained
     the same queue, chunks split arbitrarily between pipelines,
     ring-buffer writes raced.

Both paths are now protected by ``_dsp_lock`` (per-chunk) + a
join-failure latch (``_ingest_join_failed``) that blocks
``start_acq`` from double-spawning.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from chirp.recording.entity import RecordingEntity


# ── DSP-lock serialisation ───────────────────────────────────────────

def test_ingest_chunk_holds_dsp_lock():
    """``ingest_chunk`` is the ingest-thread's hot path. It must hold
    ``_dsp_lock`` for the duration of one chunk so rebuild paths
    on the UI thread cannot interleave mid-chunk."""
    e = RecordingEntity(name='t', device_id=None)

    # Instrument the locked implementation so we observe lock state
    # from inside the body.
    seen_held = {'ok': False}
    real_impl = e._ingest_chunk_locked
    def _spy(raw):
        # If the lock is correctly held by the caller, trying to
        # acquire it non-blocking from the same thread will fail
        # (Lock, not RLock — that's the point).
        assert not e._dsp_lock.acquire(blocking=False), (
            'ingest_chunk entered _ingest_chunk_locked without '
            'holding _dsp_lock — DSP state is unprotected')
        seen_held['ok'] = True
        return real_impl(raw)
    e._ingest_chunk_locked = _spy

    chunk = np.zeros(1024, dtype=np.float32)
    e.ingest_chunk(chunk)
    assert seen_held['ok']
    # And the lock is released after return.
    assert e._dsp_lock.acquire(blocking=False)
    e._dsp_lock.release()


def test_change_fft_params_acquires_dsp_lock_and_rebuilds():
    """``change_fft_params`` must serialise buffer / accumulator
    swaps against the ingest thread. The simple observable: the lock
    is held during the rebuild, so a concurrent ``acquire(blocking=
    False)`` from another thread returns False."""
    e = RecordingEntity(name='t', device_id=None)

    observed_locked_during_rebuild = {'ok': False}
    # Patch the accumulator constructor so we can observe lock state
    # from inside the critical section.
    from chirp.dsp import SpectrogramAccumulator as _RealAcc
    class _Probe(_RealAcc):
        def __init__(self, *a, **kw):
            # Cross-thread probe: this thread IS the caller of
            # change_fft_params, so acquire(blocking=False) would
            # return True on a released Lock and False on a held Lock.
            # To probe cross-thread, spawn a background thread.
            result = {}
            def _probe():
                result['got'] = e._dsp_lock.acquire(blocking=False)
                if result['got']:
                    e._dsp_lock.release()
            t = threading.Thread(target=_probe)
            t.start()
            t.join(timeout=1.0)
            observed_locked_during_rebuild['ok'] = not result.get('got', True)
            super().__init__(*a, **kw)

    import chirp.recording.entity as _ent
    orig = _ent.SpectrogramAccumulator
    _ent.SpectrogramAccumulator = _Probe
    try:
        e.change_fft_params(nperseg=512, window='hann')
    finally:
        _ent.SpectrogramAccumulator = orig

    assert observed_locked_during_rebuild['ok'], (
        'change_fft_params did not hold _dsp_lock during the rebuild — '
        'the ingest thread can read half-reallocated state')
    # Post-condition: lock was released.
    assert e._dsp_lock.acquire(blocking=False)
    e._dsp_lock.release()
    # And the rebuild actually took effect.
    assert e.spec_nperseg == 512
    assert e.spec_buffer.shape == (257, e._n_cols)


def test_change_analysis_fft_params_acquires_dsp_lock():
    """Same contract for the analysis-FFT rebuild path. ``_rebuild_
    analysis_split`` mutates ``_analysis_acc`` which the ingest
    thread reads via the ``analysis_acc`` property."""
    e = RecordingEntity(name='t', device_id=None)

    # Set display params first so analysis_nperseg != spec_nperseg
    # forces a dedicated accumulator.
    e.change_fft_params(nperseg=256, window='hann')

    acquired = {'n': 0}
    from chirp.dsp import SpectrogramAccumulator as _RealAcc
    class _Probe(_RealAcc):
        def __init__(self, *a, **kw):
            result = {}
            def _probe():
                result['got'] = e._dsp_lock.acquire(blocking=False)
                if result['got']:
                    e._dsp_lock.release()
            t = threading.Thread(target=_probe)
            t.start()
            t.join(timeout=1.0)
            if not result.get('got', True):
                acquired['n'] += 1
            super().__init__(*a, **kw)

    import chirp.recording.entity as _ent
    orig = _ent.SpectrogramAccumulator
    _ent.SpectrogramAccumulator = _Probe
    try:
        e.change_analysis_fft_params(nperseg=1024, window='hann')
    finally:
        _ent.SpectrogramAccumulator = orig

    assert acquired['n'] >= 1, (
        'change_analysis_fft_params did not hold _dsp_lock during '
        'accumulator reallocation')


def test_change_display_seconds_acquires_dsp_lock():
    """Ring-buffer reallocation on display-seconds change must
    serialise — the ingest thread indexes these buffers by
    ``write_head % _total_samples`` and a mid-chunk shrink would
    produce an out-of-bounds write."""
    e = RecordingEntity(name='t', device_id=None)

    observed = {'locked': False}
    real_zeros = np.zeros
    def _spy_zeros(*a, **kw):
        # First call inside change_display_seconds. Probe the lock
        # from another thread.
        result = {}
        def _probe():
            result['got'] = e._dsp_lock.acquire(blocking=False)
            if result['got']:
                e._dsp_lock.release()
        t = threading.Thread(target=_probe)
        t.start()
        t.join(timeout=1.0)
        if not result.get('got', True):
            observed['locked'] = True
        return real_zeros(*a, **kw)

    import chirp.recording.entity as _ent
    orig = _ent.np.zeros
    _ent.np.zeros = _spy_zeros
    try:
        e.change_display_seconds(20.0)
    finally:
        _ent.np.zeros = orig

    assert observed['locked']


# ── Ingest-thread double-spawn guard ─────────────────────────────────

def test_start_acq_refuses_to_double_spawn_if_previous_thread_stuck(monkeypatch):
    """If a prior ``_stop_ingest_and_flush`` timed out and left a
    stuck thread alive, ``start_acq`` must refuse to spawn a second
    one — silently double-spawning was the original bug."""
    e = RecordingEntity(name='t', device_id=None)

    # Fake capture so start_acq's ``capture.valid`` check passes.
    class _FakeCap:
        valid = True
        def resume(self): pass
        def pause(self): pass
    e.capture = _FakeCap()

    # Fake a stuck prior ingest thread. Use a real Thread target that
    # blocks on a never-set event so is_alive() is True.
    never = threading.Event()
    stuck = threading.Thread(target=lambda: never.wait(), daemon=True)
    stuck.start()
    e._ingest_thread = stuck

    e.acq_running = False
    e.start_acq()

    # Acquisition NOT started, the error flag is latched, and no
    # second thread got spawned.
    assert e.acq_running is False
    assert e._ingest_join_failed is True
    assert e.has_ever_ingest_errored is True
    assert e.last_ingest_error is not None
    assert 'still alive' in e.last_ingest_error
    # The stuck thread is still the one on the entity — nothing
    # else got attached.
    assert e._ingest_thread is stuck

    # Cleanup: unblock the stuck thread so pytest doesn't warn about
    # leaked threads.
    never.set()
    stuck.join(timeout=1.0)


def test_stop_ingest_and_flush_latches_error_on_join_timeout(monkeypatch):
    """When the ingest thread refuses to exit within 10s, the helper
    must keep the stuck reference + latch the error flag rather than
    silently clearing to ``None``."""
    e = RecordingEntity(name='t', device_id=None)

    # Build a stuck ingest thread that never checks _ingest_stop.
    never = threading.Event()
    stuck = threading.Thread(target=lambda: never.wait(), daemon=True)
    stuck.start()
    e._ingest_thread = stuck

    # Shorten the join window for the test by patching Thread.join
    # on THIS instance only — we can't monkeypatch the class without
    # affecting the rest of the process.
    orig_join = stuck.join
    def _fast_join(timeout=None):
        return orig_join(timeout=0.1)  # always "time out"
    stuck.join = _fast_join

    e._stop_ingest_and_flush(reason='unit-test')

    assert e._ingest_join_failed is True
    assert e.has_ever_ingest_errored is True
    # Critically: the stuck reference is KEPT so start_acq's guard
    # can see is_alive() and refuse to double-spawn.
    assert e._ingest_thread is stuck
    # last_ingest_error names the failure.
    assert e.last_ingest_error is not None
    assert '10s' in e.last_ingest_error or 'stop' in e.last_ingest_error

    never.set()
    stuck.join(timeout=1.0)


def test_clear_error_flag_unlatches_ingest_join_failed():
    """After the user clicks the error badge, the join-failed latch
    must clear too — otherwise the UI shows green but start_acq
    still refuses to spawn."""
    e = RecordingEntity(name='t', device_id=None)
    e._ingest_join_failed = True
    e.has_ever_ingest_errored = True

    e.clear_error_flag()

    assert e._ingest_join_failed is False
    assert e.has_ever_ingest_errored is False
