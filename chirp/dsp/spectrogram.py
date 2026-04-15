"""SpectrogramAccumulator — overlapped FFT column computation.

Extracted from the monolith in the Phase 1 refactor (plan: c03). The
class computes one spectrogram column per input chunk by carrying a
rolling `_overlap` buffer of the last `nperseg` samples. Every call
emits the dB column (for display) *and* the linear magnitude (for
downstream spectral entropy).

Known quirks pinned for later phases:
  - The overlap buffer is not reset when FFT size / window changes
    (issue #14 — a "primed" flag will be added in c10).
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
        self._overlap = np.zeros(self._n, dtype=np.float32)
        self._window  = scipy.signal.windows.get_window(window, self._n).astype(np.float32)

    def compute_column(self, chunk: np.ndarray):
        """Return (dB_column, linear_magnitude).

        *dB_column* is the log-magnitude spectrogram column (float32).
        *linear_magnitude* is the raw |FFT| before dB conversion (float32),
        useful for computing spectral entropy.
        """
        combined    = np.concatenate([self._overlap, chunk])
        window_data = combined[-self._n:]
        self._overlap = window_data.copy()
        fft_mag = np.abs(np.fft.rfft(window_data * self._window))
        db_col  = (20.0 * np.log10(fft_mag + 1e-10)).astype(np.float32)
        return db_col, fft_mag
