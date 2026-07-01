"""Tests for the WAV-file-simulation input source.

Pins the behavior of `WavFileCapture` and the `RecordingEntity.use_wav_file`
wiring: loading a file, pacing chunks through the audio queue in the
expected shape, looping, one-shot stopping at EOF, and schema round-trip
of the new persisted fields.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pytest
import scipy.io.wavfile

from chirp.audio.ringbuffer import AudioRing
from chirp.audio.wav_capture import WavFileCapture
from chirp.config.schema import build_settings_dict, load_settings_dict
from chirp.constants import CHUNK_FRAMES
from chirp.recording.entity import RecordingEntity


def _make_wav(path: str, sample_rate: int = 44100,
              duration: float = 0.1, freq: float = 1000.0,
              channels: int = 1) -> None:
    n = int(sample_rate * duration)
    t = np.arange(n, dtype=np.float32) / sample_rate
    tone = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    pcm16 = (tone * 32767.0).astype(np.int16)
    if channels == 2:
        pcm16 = np.stack([pcm16, pcm16], axis=1)
    scipy.io.wavfile.write(path, sample_rate, pcm16)


# WavFileCapture now writes into an AudioRing rather than a queue.Queue.
# These helpers mirror the queue inspection the tests used to do: the
# ring is the producer→consumer channel and the test plays the consumer.
def _ring(channels: int = 1, seconds: float = 3.0, sr: int = 44100) -> AudioRing:
    # Sized generously so the un-consumed producer never overruns during
    # the short test windows.
    return AudioRing(int(sr * seconds), channels=channels)


def _navail(ring: AudioRing) -> int:
    """Number of whole CHUNK_FRAMES blocks currently available."""
    return ring.available // CHUNK_FRAMES


def _get_chunk(ring: AudioRing) -> np.ndarray:
    _, block = ring.read(CHUNK_FRAMES)
    return block


# ── WavFileCapture -------------------------------------------------------

def test_wav_capture_opens_and_reports_metadata(tmp_path):
    wav = tmp_path / 'tone.wav'
    _make_wav(str(wav), sample_rate=22050, duration=0.05)
    cap = WavFileCapture(_ring(sr=22050), str(wav))
    try:
        assert cap.valid
        assert cap.file_sample_rate == 22050
        assert cap.file_channels == 1
    finally:
        cap.close()


def test_wav_capture_invalid_path_not_valid(tmp_path):
    cap = WavFileCapture(_ring(), str(tmp_path / 'does-not-exist.wav'))
    try:
        assert not cap.valid
    finally:
        cap.close()


def test_wav_capture_emits_mono_chunks_of_fixed_size(tmp_path):
    wav = tmp_path / 'tone.wav'
    # ~2 s of audio so the ring has room to fill at real-time pace.
    _make_wav(str(wav), sample_rate=44100, duration=2.0)
    ring = _ring()
    cap = WavFileCapture(ring, str(wav))
    try:
        cap.resume()
        # Wait for a handful of chunks to arrive.
        deadline = time.monotonic() + 1.0
        while _navail(ring) < 5 and time.monotonic() < deadline:
            time.sleep(0.01)
        cap.pause()
        assert _navail(ring) >= 5
        first = _get_chunk(ring)
        assert first.dtype == np.float32
        assert first.shape == (CHUNK_FRAMES,)
    finally:
        cap.close()


def test_wav_capture_stereo_duplicates_mono(tmp_path):
    wav = tmp_path / 'tone.wav'
    _make_wav(str(wav), sample_rate=44100, duration=0.5, channels=1)
    ring = _ring(channels=2)
    cap = WavFileCapture(ring, str(wav), channels=2)
    try:
        cap.resume()
        deadline = time.monotonic() + 1.0
        while _navail(ring) < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        cap.pause()
        chunk = _get_chunk(ring)
        assert chunk.shape == (CHUNK_FRAMES, 2)
        # Left and right should match (mono duplicated).
        np.testing.assert_array_equal(chunk[:, 0], chunk[:, 1])
    finally:
        cap.close()


def test_wav_capture_one_shot_stops_at_eof(tmp_path):
    wav = tmp_path / 'tone.wav'
    _make_wav(str(wav), sample_rate=44100, duration=0.05)  # ~2 chunks worth
    ring = _ring()
    cap = WavFileCapture(ring, str(wav), loop=False)
    try:
        cap.resume()
        # The file is ~2200 samples = 3 chunks. Give the producer a
        # generous window, then confirm no more chunks arrive. Nothing
        # consumes the ring, so write_total stops growing after EOF.
        time.sleep(0.4)
        before = ring.write_total
        time.sleep(0.2)
        after = ring.write_total
        assert after == before, (
            f'ring grew after EOF: {before} -> {after}')
        assert before >= CHUNK_FRAMES
    finally:
        cap.close()


def test_wav_capture_reports_duration_and_position(tmp_path):
    wav = tmp_path / 'tone.wav'
    _make_wav(str(wav), sample_rate=44100, duration=0.5)
    ring = _ring()
    cap = WavFileCapture(ring, str(wav))
    try:
        assert cap.duration_sec == pytest.approx(0.5, abs=1e-3)
        assert cap.position_sec == 0.0
        cap.resume()
        # Let a few chunks produce, then check position advanced.
        deadline = time.monotonic() + 1.0
        while _navail(ring) < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        cap.pause()
        assert cap.position_sec > 0.0
    finally:
        cap.close()


def test_wav_capture_reset_rewinds_to_start(tmp_path):
    wav = tmp_path / 'tone.wav'
    _make_wav(str(wav), sample_rate=44100, duration=1.0)
    ring = _ring()
    cap = WavFileCapture(ring, str(wav))
    try:
        cap.resume()
        deadline = time.monotonic() + 1.0
        while _navail(ring) < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert cap.position_sec > 0.0
        cap.reset_position()
        # Wait until the producer observes the reset flag.
        deadline = time.monotonic() + 0.5
        while cap.position_sec > 0.05 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert cap.position_sec < 0.05
    finally:
        cap.close()


def test_wav_capture_set_loop_toggles(tmp_path):
    wav = tmp_path / 'tone.wav'
    _make_wav(str(wav), sample_rate=44100, duration=0.05)
    cap = WavFileCapture(_ring(), str(wav), loop=True)
    try:
        cap.set_loop(False)
        assert cap._loop is False
        cap.set_loop(True)
        assert cap._loop is True
    finally:
        cap.close()


# ── RecordingEntity integration ------------------------------------------

def test_entity_use_wav_file_switches_source(tmp_path):
    wav = tmp_path / 'tone.wav'
    _make_wav(str(wav), sample_rate=44100, duration=0.1)
    e = RecordingEntity(name='test', device_id=None)
    try:
        ok, warning = e.use_wav_file(str(wav))
        assert ok
        assert warning is None  # file SR matches default session SR
        assert e.input_source == 'wav_file'
        assert e.wav_file_path == str(wav)
        assert isinstance(e.capture, WavFileCapture)
        assert e.capture.valid
    finally:
        e.close()


def test_entity_use_wav_file_syncs_sample_rate(tmp_path):
    wav = tmp_path / 'tone.wav'
    _make_wav(str(wav), sample_rate=22050, duration=0.1)
    e = RecordingEntity(name='test', device_id=None, sample_rate=44100)
    try:
        ok, warning = e.use_wav_file(str(wav))
        assert ok
        assert e.sample_rate == 22050
        assert warning is not None
        assert '22050' in warning
    finally:
        e.close()


def test_entity_use_wav_file_bad_path_returns_false(tmp_path):
    """#49: a missing WAV file must NOT fall back to the live device.
    Pre-fix this test pinned the buggy fall-back; post-fix the entity
    keeps the invalid WavFileCapture in place so the user can see the
    error and fix the path without silently recording from the
    default microphone."""
    e = RecordingEntity(name='test', device_id=None)
    try:
        bad = str(tmp_path / 'missing.wav')
        ok, warning = e.use_wav_file(bad)
        assert not ok
        assert warning and 'Could not open' in warning
        # #49: input_source stays 'wav_file' and wav_file_path is
        # preserved — the user retries by editing the path; the entity
        # is in a clearly-broken state (capture.valid == False) until
        # they do.
        assert e.input_source == 'wav_file'
        assert e.wav_file_path == bad
        assert e.capture.valid is False
    finally:
        e.close()


# ── Config persistence ---------------------------------------------------

def test_config_persists_wav_sim_fields(tmp_path):
    wav = tmp_path / 'tone.wav'
    _make_wav(str(wav), sample_rate=44100, duration=0.05)
    e = RecordingEntity(name='sim', device_id=None)
    try:
        e.use_wav_file(str(wav))
        data = build_settings_dict([e])
        encoded = json.dumps(data)
        decoded = json.loads(encoded)
        entities, _, warnings = load_settings_dict(decoded)
        assert len(entities) == 1
        r = entities[0]
        try:
            assert r.input_source == 'wav_file'
            assert r.wav_file_path == str(wav)
            assert r.wav_loop is True
            # Capture should actually be pointing at the file after
            # from_dict (`use_wav_file` re-opens it).
            assert isinstance(r.capture, WavFileCapture)
            assert r.capture.valid
        finally:
            r.close()
    finally:
        e.close()


def test_legacy_config_defaults_to_device_source():
    """Configs without the new keys should load as live-device sources."""
    raw = {"version": 1, "recordings": [{"name": "legacy"}]}
    entities, _, _ = load_settings_dict(raw)
    try:
        assert len(entities) == 1
        assert entities[0].input_source == 'device'
        assert entities[0].wav_file_path is None
    finally:
        for e in entities:
            e.close()
