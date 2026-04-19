"""Chirp — DSP subpackage.

Pure-numpy / scipy signal-processing primitives: spectrogram
accumulator, Butterworth bandpass filter, and spectral entropy. These
modules must not import PyQt, matplotlib, or anything from
`chirp.recording` / `chirp.ui` — they are the leaves of the dependency
graph and should stay cheap to import and easy to unit-test.
"""

from chirp.dsp.entropy import normalized_spectral_entropy
from chirp.dsp.envelope import analytic_envelope
from chirp.dsp.filter import BandpassFilter
from chirp.dsp.spectrogram import SpectrogramAccumulator

__all__ = [
    "BandpassFilter",
    "SpectrogramAccumulator",
    "analytic_envelope",
    "normalized_spectral_entropy",
]
