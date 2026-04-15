"""SpectrogramAccumulator — overlapped FFT column computation.

Extracted from the monolith in the Phase 1 refactor (plan: c03). The
class computes one spectrogram column per input chunk by carrying a
rolling `_overlap` buffer of the last `nperseg` samples. Every call
emits the dB column (for display) *and* the linear magnitude (for
downstream spectral entropy).

Warm-up / `primed` flag (c10, issue #14):

  The overlap buffer is initialized to zeros, and remains zero-padded
  after a `reset()` (stream restart, sample-rate change, FFT param
  change). Until enough *live* samples have been fed to fully replace
  that zero tail, the FFT column is contaminated by stale silence and
  the derived spectral entropy is meaningless. Callers MUST skip
  spectral-trigger evaluation while `primed` is False — otherwise a
  Spectral-Only trigger can fire spuriously the instant acquisition
  restarts.

Known quirks still pinned for later phases:
  - Exactly one column is emitted per chunk regardless of chunk size,
    so display temporal resolution is implicitly chunk-quantized
    (issue #12 — hop-size control will be added in c19).
"""

import numpy as np
import scipy.signal

from chirp.constants import SPECTROGRAM_NPERSEG


class SpectrogramAccumulator:
    WINDOW_TYPES = ('hann', 'hamming', 'blackman', 'bartlett', 'flattop')
    FFT_SIZES    = (256, 512, 1024, 2048, 4096)

    def __init__(self, nperseg=SPECTROGRAM_NPERSEG, window='hann'):
        self._n       = nperseg
        self._window  = scipy.signal.windows.get_window(window, self._n).astype(np.float32)
        self._overlap = np.zeros(self._n, dtype=np.float32)
        self._live_samples = 0  # count of live samples fed since last reset

    @property
    def primed(self) -> bool:
        """True once enough live samples have flowed through to fully
        displace the zero-initialized overlap buffer."""
        return self._live_samples >= self._n

    def reset(self) -> None:
        """Clear the overlap buffer and mark the accumulator un-primed.

        Call this on stream start/stop, sample-rate change, and FFT
        parameter change. The very next `compute_column` will still
        emit a column (so the display stays consistent), but `primed`
        will be False until `_n` live samples have been consumed.
        """
        self._overlap[:] = 0.0
        self._live_samples = 0

    def compute_column(self, chunk: np.ndarray):
        """Return (dB_column, linear_magnitude).

        *dB_column* is the log-magnitude spectrogram column (float32).
        *linear_magnitude* is the raw |FFT| before dB conversion (float32),
        useful for computing spectral entropy. While `self.primed` is
        False the returned magnitudes reflect a zero-padded warm-up
        window and should NOT drive spectral-trigger decisions.
        """
        combined    = np.concatenate([self._overlap, chunk])
        window_data = combined[-self._n:]
        self._overlap = window_data.copy()
        self._live_samples += int(len(chunk))
        fft_mag = np.abs(np.fft.rfft(window_data * self._window))
        db_col  = (20.0 * np.log10(fft_mag + 1e-10)).astype(np.float32)
        return db_col, fft_mag
