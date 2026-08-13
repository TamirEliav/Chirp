"""Sticky "S" badge names the last clipped recording.

Before this, the badge told the user that clipping had happened at some
point but not WHERE — finding the offending WAV meant opening
``chirp_errors.log`` and matching timestamps by hand. Both publish paths
(``write_wav_sync`` and ``StreamingWavWriter.close``) already know the
final path and the peak at the moment they emit the ``saturation`` log
line, so they now also record it in a small registry keyed by stream
name; the entity picks it up on the UI tick and the tooltip names it.

Pinned here:

* both publish paths register the file;
* the registry is consume-once, so a poll doesn't keep re-copying the
  same tuple;
* a clipped file recorded while acquisition ran but nothing was written
  leaves the badge lit with an honest "no saved recording yet" tooltip
  rather than pointing at an unrelated older file;
* clearing the badge clears the remembered file (a stale name under a
  freshly-lit badge is worse than no name).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from chirp.recording import writer as W
from chirp.ui.status_util import compose_saturation_state


@pytest.fixture(autouse=True)
def _clean_registry():
    W.clear_saturated_file('s1')
    W.clear_saturated_file('s2')
    yield
    W.clear_saturated_file('s1')
    W.clear_saturated_file('s2')


class _Ent:
    """Only what compose_saturation_state / the poll touch."""

    def __init__(self, name='s1'):
        self.name = name
        self.saturated_ever = False
        self.last_saturated_path = ''
        self.last_saturated_peak = 0.0


# ── Registry ─────────────────────────────────────────────────────────

def test_registry_is_consume_once():
    W.note_saturated_file('s1', r'C:\rec\a.wav', 0.995)
    assert W.consume_saturated_file('s1') == (r'C:\rec\a.wav', 0.995)
    assert W.consume_saturated_file('s1') is None


def test_registry_keeps_streams_apart():
    W.note_saturated_file('s1', 'a.wav', 0.99)
    W.note_saturated_file('s2', 'b.wav', 1.0)
    assert W.consume_saturated_file('s2') == ('b.wav', 1.0)
    assert W.consume_saturated_file('s1') == ('a.wav', 0.99)


def test_registry_keeps_only_the_latest_per_stream():
    W.note_saturated_file('s1', 'old.wav', 0.99)
    W.note_saturated_file('s1', 'new.wav', 1.0)
    assert W.consume_saturated_file('s1') == ('new.wav', 1.0)


def test_clear_drops_a_pending_record():
    W.note_saturated_file('s1', 'a.wav', 0.99)
    W.clear_saturated_file('s1')
    assert W.consume_saturated_file('s1') is None


# ── Both publish paths register ──────────────────────────────────────

def test_write_wav_sync_registers_a_clipped_file(tmp_path):
    clipped = np.full(256, 0.999, dtype=np.float32)
    path = W.write_wav_sync([clipped], str(tmp_path), sample_rate=8000,
                            filename_stream='s1')
    assert W.consume_saturated_file('s1') == (path, pytest.approx(0.999))


def test_write_wav_sync_registers_nothing_for_clean_audio(tmp_path):
    quiet = np.full(256, 0.2, dtype=np.float32)
    W.write_wav_sync([quiet], str(tmp_path), sample_rate=8000,
                     filename_stream='s1')
    assert W.consume_saturated_file('s1') is None


def test_streaming_writer_registers_a_clipped_file(tmp_path):
    w = W.StreamingWavWriter(str(tmp_path), sample_rate=8000, channels=1,
                             filename_stream='s1')
    w.append(np.full(256, 0.999, dtype=np.float32))
    path = w.close()
    assert W.consume_saturated_file('s1') == (path, pytest.approx(0.999))


# ── Tooltip composition ──────────────────────────────────────────────

def test_tooltip_is_quiet_when_nothing_clipped():
    lit, tip = compose_saturation_state(_Ent())
    assert lit is False
    assert 'not been detected' in tip


def test_tooltip_admits_when_no_file_was_written():
    """Clipping seen live but never recorded: the badge is lit, and the
    tooltip must NOT name an older unrelated file."""
    e = _Ent()
    e.saturated_ever = True
    lit, tip = compose_saturation_state(e)
    assert lit is True
    assert 'no saved recording' in tip
    assert '.wav' not in tip


def test_tooltip_names_the_file_and_folder():
    e = _Ent()
    e.saturated_ever = True
    e.last_saturated_path = os.path.join('D:', 'recs', 'day1', 'x_123.wav')
    e.last_saturated_peak = 0.9991
    lit, tip = compose_saturation_state(e)
    assert lit is True
    assert 'x_123.wav' in tip
    assert os.path.join('D:', 'recs', 'day1') in tip
    assert '0.9991' in tip


# ── Entity integration ───────────────────────────────────────────────

def test_entity_polls_and_clears(tmp_path):
    from chirp.recording.entity import RecordingEntity
    e = RecordingEntity(name='s1', device_id=None, sample_rate=8000)
    try:
        W.note_saturated_file('s1', str(tmp_path / 'clip.wav'), 0.999)
        e.saturated_ever = True
        e.poll_saturated_file()
        assert e.last_saturated_path.endswith('clip.wav')
        lit, tip = compose_saturation_state(e)
        assert lit and 'clip.wav' in tip

        # Clearing the badge must forget the file, and must also drop a
        # record that arrived but hasn't been polled yet.
        W.note_saturated_file('s1', str(tmp_path / 'later.wav'), 1.0)
        e.clear_saturation_flag()
        assert e.last_saturated_path == ''
        assert W.consume_saturated_file('s1') is None
        e.poll_saturated_file()
        assert e.last_saturated_path == ''
        assert compose_saturation_state(e)[0] is False
    finally:
        e.close()
