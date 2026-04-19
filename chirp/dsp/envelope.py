"""Analytic-signal amplitude envelope.

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
from scipy.signal import hilbert


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
