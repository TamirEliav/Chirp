"""Chirp — recording subpackage.

Holds the trigger state machine, the per-stream entity, and (later, in
c06) the WAV writer pool. Kept separate from `chirp.audio` because the
recording layer owns both the audio pipeline and the triggered capture
logic — the split lets tests pin trigger semantics without touching
sounddevice.
"""

from chirp.recording.trigger import ThresholdRecorder

__all__ = ["ThresholdRecorder"]
