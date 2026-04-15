"""Normalized Shannon spectral entropy.

Hoisted out of `RecordingEntity.ingest_chunk` in the Phase 1 refactor
(plan: c03). The function takes a linear FFT magnitude vector and
returns a value in [0, 1] where:

  - 0   → energy concentrated in a single bin (pure tone)
  - 1   → energy uniformly distributed across bins (white noise)
  - 1.0 is also returned as a sentinel for a fully silent input
          (total magnitude below ~1e-30). This matches the legacy
          monolith semantics — callers that want to distinguish
          "silent" from "noisy" should inspect the raw magnitudes,
          not the entropy value alone.

The formula is H / log2(N) where H = -Σ p_i log2(p_i) and p_i are the
bin probabilities derived from the magnitude vector. Zero-probability
bins are dropped before the log to avoid NaNs.
"""

import numpy as np

# Threshold below which the total FFT magnitude is considered numerically
# silent. Matches the pre-refactor monolith (1e-30). Exposed as a module
# constant for tests.
SILENT_MAGNITUDE_THRESHOLD = 1e-30


def normalized_spectral_entropy(mag: np.ndarray) -> float:
    """Normalized Shannon entropy of a linear FFT magnitude vector.

    Returns a float in [0.0, 1.0].
    """
    s = mag.sum()
    if s < SILENT_MAGNITUDE_THRESHOLD:
        return 1.0
    p = mag / s
    p = p[p > 0]
    n = len(mag)
    h = -float(np.sum(p * np.log2(p)))
    return h / np.log2(n) if n > 1 else 0.0
