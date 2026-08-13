"""Amplitude-envelope estimators for the threshold trigger.

Two methods are available; which one runs is a global (app-wide) choice
exposed in ⚙ Advanced → AMPLITUDE ENVELOPE and persisted in the config's
``audio`` section. Both take the already-bandpassed signal and return a
smooth, non-negative amplitude estimate; the trigger compares it against
``threshold`` per sample.

``hilbert`` (default, historical behaviour)
    ``|analytic_signal(x)|`` — see :func:`analytic_envelope`. Exact for a
    pure tone, zero lag, but computed per chunk with an FFT, so it has a
    small circular-convolution transient at each chunk boundary and its
    cost grows with chunk size.

``rectify``
    ``lowpass(|x|)`` — a 2nd-order Butterworth applied to the rectified
    signal, carried statefully across chunks (:class:`RectifiedEnvelope`).
    The classic analog envelope follower, with no chunk-boundary artifact
    at all because the filter state is continuous. In exchange it has a
    real group delay (~1/cutoff) and it under-reads a tone by the
    rectified-sine DC factor, which is why the implementation
    compensates by 2/π — with that correction a steady sine of amplitude
    A reads A, matching ``hilbert`` so switching methods does not
    silently move the effective threshold.

    Cost is NOT a reason to choose between them: measured on a
    1024-sample chunk, hilbert is 18.7 µs and rectify 21.5 µs — the
    per-call ``sosfilt`` overhead cancels the FFT it avoids at this
    size. Choose on the artifact-vs-delay trade-off, not on speed.

    The cutoff is the one extra knob: it must sit below the signal band
    (so the carrier is removed) and above the modulation rate of the
    calls being detected. The default (:data:`DEFAULT_ENVELOPE_CUTOFF_HZ`)
    suits bioacoustic call envelopes; lower it for smoother/slower
    detection, raise it to track fast onsets.

Analytic-signal amplitude envelope.

Used by the threshold trigger to measure "signal present" in a way that
is insensitive to waveform zero crossings. The pre-fix code compared
``|filtered[i]| >= threshold`` per sample, which dips to zero at every
zero crossing of the raw waveform. For narrowband signals — pure tones,
bandpassed bioacoustic calls, whistles — the per-sample compare
oscillates at the signal's frequency and never accumulates enough
*consecutive* above-threshold samples to satisfy ``min_cross`` (see
test_envelope_trigger.py for the reproducer). That was a v2.1.0
regression introduced by the sample-accurate state-machine rewrite in
commit d758fbf (c18 / #15); prior versions compared a chunk-level peak
so the issue didn't manifest.

Fix: compute the *instantaneous amplitude* of the analytic signal
(Hilbert transform), which is the smooth envelope. For a pure sine
A·sin(2πft) the analytic envelope is exactly A (no zero-crossing
dips). For broadband / impulsive signals it tracks the peak closely.

Implementation notes:

  * ``scipy.signal.hilbert`` computes the analytic signal via FFT.
    Per-chunk FFT implies circular-convolution edge artifacts at the
    first/last samples of the chunk — typically 10–30 samples of
    transient dip at each boundary. For our use (threshold detection
    at low-kHz audio rates with 1024-sample chunks) this is acceptable
    and is preferable to the alternative of running a stateful IIR
    rectify-lowpass whose cutoff becomes another UI knob to tune.
  * We operate on the already-bandpassed signal (``filt``), so
    narrowband inputs stay narrowband and the envelope is smooth.
  * Float32 is preserved on output so downstream threshold compares
    stay in the native pipeline dtype.
"""

from __future__ import annotations

import numpy as np
import scipy.signal
from scipy.signal import hilbert


#: Selectable envelope methods, in the order the Advanced dialog lists
#: them. ``hilbert`` is the historical default.
ENVELOPE_METHODS = ('hilbert', 'rectify')

#: Default low-pass cutoff (Hz) for the ``rectify`` method.
DEFAULT_ENVELOPE_CUTOFF_HZ = 50.0

#: Bounds for the cutoff, enforced by the schema and the UI spinbox. The
#: upper bound is additionally clamped to just under Nyquist at design
#: time so a low sample rate can't produce an invalid filter.
ENVELOPE_CUTOFF_MIN_HZ = 1.0
ENVELOPE_CUTOFF_MAX_HZ = 5000.0

#: Order of the rectifier's low-pass. Two poles give ~12 dB/octave of
#: carrier rejection with a group delay short enough that ``min_cross``
#: still means roughly what the user set.
_LOWPASS_ORDER = 2

#: Rectified-sine DC gain: mean(|A·sin|) = 2A/π. Dividing by it makes a
#: steady tone read its true peak amplitude, so ``hilbert`` and
#: ``rectify`` agree on what "amplitude 0.1" means and a user switching
#: methods does not have to re-tune every stream's threshold.
_RECTIFY_GAIN = np.pi / 2.0


def analytic_envelope(x: np.ndarray) -> np.ndarray:
    """Amplitude envelope of ``x`` via the analytic signal.

    Returns ``|hilbert(x)|`` with the same shape and float32 dtype as
    the input (regardless of input dtype). For a real signal of length
    N the analytic signal has the same length; no trimming is needed.

    Parameters
    ----------
    x : 1-D real array

    Returns
    -------
    envelope : 1-D float32 array, same length as ``x``
    """
    # hilbert returns complex128; take magnitude and downcast.
    if x.size == 0:
        return np.empty(0, dtype=np.float32)
    # scipy.signal.hilbert accepts float inputs and internally uses
    # FFT over the chunk. For a pure sine the returned magnitude is
    # very close to the peak amplitude except for a short transient at
    # each chunk boundary (FFT edge effect).
    analytic = hilbert(x)
    return np.abs(analytic).astype(np.float32, copy=False)


class RectifiedEnvelope:
    """Stateful rectify + low-pass envelope follower.

    One instance per signal being followed (a stereo stream needs two —
    the left and right filter histories must not be shared). The IIR
    state carries across :meth:`process` calls so the envelope is
    continuous across chunk boundaries, which is the whole reason this
    is an object rather than a function.

    ``cutoff_hz`` is clamped into
    ``[ENVELOPE_CUTOFF_MIN_HZ, min(ENVELOPE_CUTOFF_MAX_HZ, 0.45·fs)]``
    at construction, so no combination of configured cutoff and sample
    rate can produce an invalid design.
    """

    def __init__(self, sample_rate: int, cutoff_hz: float = DEFAULT_ENVELOPE_CUTOFF_HZ):
        self.sample_rate = int(sample_rate)
        nyq = self.sample_rate * 0.5
        hi = min(ENVELOPE_CUTOFF_MAX_HZ, nyq * 0.9)
        #: What the caller asked for, kept verbatim so a cache can tell
        #: "still the configured cutoff" from "needs a rebuild" even when
        #: the request was clamped.
        self.requested_cutoff_hz = float(cutoff_hz)
        #: What was actually designed, after clamping.
        self.cutoff_hz = float(
            min(max(float(cutoff_hz), ENVELOPE_CUTOFF_MIN_HZ), hi))
        self._sos = scipy.signal.butter(
            _LOWPASS_ORDER, self.cutoff_hz / nyq, btype='low', output='sos')
        # sosfilt_zi is the steady-state for unit DC input; scaling it by
        # the first sample of each fresh run would be the textbook warm
        # start, but the envelope is only ever started from silence here
        # (a new stream / a rate change), so a zero-seeded state is both
        # correct and free of a start-up overshoot.
        self._zi = np.zeros((self._sos.shape[0], 2), dtype=np.float64)

    @property
    def key(self) -> tuple:
        """Identity of the request — cheap equality check for the caller
        deciding whether a cached follower still matches the config."""
        return (self.sample_rate, self.requested_cutoff_hz)

    def process(self, x: np.ndarray) -> np.ndarray:
        """Return the envelope of ``x`` as float32, same length."""
        if x.size == 0:
            return np.empty(0, dtype=np.float32)
        rect = np.abs(np.asarray(x, dtype=np.float64))
        out, self._zi = scipy.signal.sosfilt(self._sos, rect, zi=self._zi)
        # The low-pass can ring slightly negative on a sharp offset;
        # an envelope must not be negative (the trigger compares it to a
        # non-negative threshold, and the amplitude plot would show a
        # spurious dip below zero).
        np.multiply(out, _RECTIFY_GAIN, out=out)
        np.maximum(out, 0.0, out=out)
        return out.astype(np.float32, copy=False)

    def reset(self) -> None:
        """Clear the filter history (stream restart / device change)."""
        self._zi = np.zeros((self._sos.shape[0], 2), dtype=np.float64)


# ── Global method selection ──────────────────────────────────────────────────
#
# App-wide rather than per-stream, matching how the user asked for it
# (⚙ Advanced) and how the other capture-engine knobs work. Entities read
# it per chunk, so a change applies to every running stream on the next
# chunk without restarting acquisition.

_method: str = 'hilbert'
_cutoff_hz: float = DEFAULT_ENVELOPE_CUTOFF_HZ


def configure(method: str = 'hilbert',
              cutoff_hz: float = DEFAULT_ENVELOPE_CUTOFF_HZ) -> tuple[str, float]:
    """Set the app-wide envelope method + cutoff. Returns the values
    actually stored (an unknown method falls back to ``hilbert``)."""
    global _method, _cutoff_hz
    _method = method if method in ENVELOPE_METHODS else 'hilbert'
    try:
        cut = float(cutoff_hz)
    except (TypeError, ValueError):
        cut = DEFAULT_ENVELOPE_CUTOFF_HZ
    _cutoff_hz = min(max(cut, ENVELOPE_CUTOFF_MIN_HZ), ENVELOPE_CUTOFF_MAX_HZ)
    return _method, _cutoff_hz


def current_params() -> tuple[str, float]:
    """Return the app-wide ``(method, cutoff_hz)``."""
    return _method, _cutoff_hz
