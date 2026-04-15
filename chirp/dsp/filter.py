"""BandpassFilter — 4th-order Butterworth IIR bandpass with lazy redesign.

Extracted from the monolith in the Phase 1 refactor (plan: c03).

Semantics are identical to the pre-refactor implementation:
  - Invalid band (lo >= hi, or both outside the Nyquist window) disables
    the filter and returns the input chunk unmodified with the raw peak.
  - Filter state (`_zi`) carries across `filter_chunk` calls to keep
    the IIR response continuous across chunk boundaries.
  - `reset()` re-seeds `_zi` from the current SOS — callers invoke this
    after a stream restart to clear history.
"""

import numpy as np
import scipy.signal

from chirp.constants import SAMPLE_RATE


class BandpassFilter:
    def __init__(self, sample_rate=SAMPLE_RATE):
        self._sos    = None
        self._zi     = None
        self._params = (None, None)
        self._sample_rate = sample_rate

    def _redesign(self, low_hz: float, high_hz: float) -> bool:
        nyq = self._sample_rate * 0.5
        lo  = max(1.0, low_hz)
        hi  = min(nyq - 1.0, high_hz)
        if lo >= hi:
            self._sos = self._zi = None
            self._params = (low_hz, high_hz)
            return False
        self._sos    = scipy.signal.butter(4, [lo / nyq, hi / nyq],
                                           btype='band', output='sos')
        self._zi     = scipy.signal.sosfilt_zi(self._sos)
        self._params = (low_hz, high_hz)
        return True

    def get_peak(self, chunk: np.ndarray, low_hz: float, high_hz: float) -> float:
        _, peak = self.filter_chunk(chunk, low_hz, high_hz)
        return peak

    def filter_chunk(self, chunk: np.ndarray, low_hz: float, high_hz: float):
        """Return (filtered_signal, peak). If filter invalid, returns (chunk, peak)."""
        if (low_hz, high_hz) != self._params:
            if not self._redesign(low_hz, high_hz):
                return chunk, float(np.max(np.abs(chunk)))
        if self._sos is None:
            return chunk, float(np.max(np.abs(chunk)))
        filtered, self._zi = scipy.signal.sosfilt(self._sos, chunk, zi=self._zi)
        return filtered, float(np.max(np.abs(filtered)))

    def reset(self):
        if self._sos is not None:
            self._zi = scipy.signal.sosfilt_zi(self._sos)
