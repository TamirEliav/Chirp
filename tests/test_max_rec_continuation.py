"""Tests for ``max_rec`` force-split continuation (#57).

Pre-fix: when an event exceeded ``max_rec_sec``, ``ThresholdRecorder``
flushed the first half and removed the event from ``_active_events``.
A fresh event then had to re-qualify through ``min_cross`` from
scratch before the second half could open. For a steady
above-threshold signal this meant the entire ``min_cross_sec`` window
between the two halves was silently discarded — the WAV files did NOT
butt-join. Researchers analysing long continuous vocalisations saw
gaps in the captured audio with no warning.

Post-fix:

  - On force-split, the first half is pinned to exactly
    ``max_samps`` samples (clean sample-accurate boundary).
  - A continuation event opens immediately at that boundary, with no
    ``min_cross`` requirement and no pre-trigger lookback.
  - Both halves are tagged with a ``split_index`` (1, 2, 3, ...)
    internally; filenames carry NO part token — each part's own
    onset timestamp distinguishes the series
    so the WAVs are unambiguously a contiguous series.
  - A standalone (non-split) event has ``split_index = None`` and no
    ``partNN`` token in its filename — false positives would be
    confusing.
"""

from __future__ import annotations

import datetime

import numpy as np
import pytest

from chirp.constants import CHUNK_FRAMES
from chirp.recording.trigger import ThresholdRecorder


# ── Fixture + helpers (mirrors test_trigger.py) ──────────────────────

@pytest.fixture
def captured_flushes(monkeypatch):
    flushes: list[dict] = []

    def _capture(buf_snapshot, output_dir, prefix='', suffix='',
                 sample_rate=44100, onset_time=None, filename_stream=''):
        flushes.append({
            'audio':           np.concatenate(list(buf_snapshot)),
            'n_chunks':        len(buf_snapshot),
            'output_dir':      output_dir,
            'prefix':          prefix,
            'suffix':          suffix,
            'sample_rate':     sample_rate,
            'onset_time':      onset_time,
            'filename_stream': filename_stream,
        })

    monkeypatch.setattr(
        ThresholdRecorder, '_start_flush', staticmethod(_capture),
    )
    return flushes


def _loud(level: float = 0.5) -> np.ndarray:
    return np.full(CHUNK_FRAMES, level, dtype=np.float32)


def _silent() -> np.ndarray:
    return np.zeros(CHUNK_FRAMES, dtype=np.float32)


def _params(**overrides):
    p = dict(
        threshold        = 0.1,
        min_cross_sec    = 0.0,
        hold_sec         = 0.0,
        post_trig_sec    = 0.0,
        max_rec_sec      = 10.0,
        pre_trig_sec     = 0.0,
        output_dir       = '/tmp/chirp_test',
        enabled          = True,
        filename_prefix  = '',
        filename_suffix  = '',
        sample_rate      = 44100,
    )
    p.update(overrides)
    return p


# ── Continuation: no samples lost between halves ─────────────────────

def test_continuation_butt_joins_at_boundary(captured_flushes):
    """5 chunks above threshold with a 3-chunk cap. Post-fix: a
    3-chunk part01 flushes when chunk 3 hits max_samps and a butt-
    joined continuation starts at the boundary; chunks 4-5 fill
    the continuation's buffer while it's still recording."""
    rec = ThresholdRecorder()
    max_rec = 3.0 * CHUNK_FRAMES / 44100  # 3 chunks
    p = _params(max_rec_sec=max_rec)

    for _ in range(5):
        rec.process_chunk(_loud(0.5), trigger_peak=0.5, **p)

    # Exactly one flush so far — part01 (the continuation is still open).
    assert len(captured_flushes) == 1
    part1 = captured_flushes[0]
    # Sample-accurate boundary at exactly max_samps.
    assert part1['audio'].size == 3 * CHUNK_FRAMES
    # And a continuation is still recording — pre-fix the new event
    # would have opened too on the very next chunk (because min_cross=0
    # here), but it would NOT have carried a split_index marker; the
    # researcher would see two unrelated WAVs.
    assert rec.is_recording is True

    # The still-open continuation has chunks 4 and 5 queued.
    cont = rec._active_events[0]
    assert cont['samples_kept'] == 2 * CHUNK_FRAMES
    assert cont['split_index'] == 2


def test_split_parts_keep_plain_user_suffix(captured_flushes):
    """Split parts share the SAME name format as unsplit files — no
    ``partNN`` token; each part is distinguished by its own onset
    timestamp instead."""
    rec = ThresholdRecorder()
    max_rec = 3.0 * CHUNK_FRAMES / 44100
    p = _params(max_rec_sec=max_rec, hold_sec=0.0,
                filename_suffix='song')

    # 3 loud (force flush at chunk 3), then go silent so continuation closes.
    for _ in range(3):
        rec.process_chunk(_loud(0.5), trigger_peak=0.5, **p)
    rec.process_chunk(_silent(), trigger_peak=0.0, **p)

    assert len(captured_flushes) == 2
    assert captured_flushes[0]['suffix'] == 'song'
    assert captured_flushes[1]['suffix'] == 'song'
    # The parts stay distinguishable: onsets differ by part1's duration.
    assert (captured_flushes[1]['onset_time']
            > captured_flushes[0]['onset_time'])


def test_split_parts_with_empty_user_suffix(captured_flushes):
    """With no user suffix the split parts have no suffix at all —
    identical format to a standalone file."""
    rec = ThresholdRecorder()
    max_rec = 3.0 * CHUNK_FRAMES / 44100
    p = _params(max_rec_sec=max_rec, filename_suffix='')

    for _ in range(3):
        rec.process_chunk(_loud(0.5), trigger_peak=0.5, **p)
    rec.process_chunk(_silent(), trigger_peak=0.0, **p)

    assert len(captured_flushes) == 2
    assert captured_flushes[0]['suffix'] == ''
    assert captured_flushes[1]['suffix'] == ''


def test_three_way_split_advances_onsets(captured_flushes):
    """A signal that runs long enough to hit max_rec twice should
    produce three parts whose onset timestamps advance by exactly the
    duration of the preceding part."""
    rec = ThresholdRecorder()
    max_rec = 2.0 * CHUNK_FRAMES / 44100  # cap at 2 chunks
    p = _params(max_rec_sec=max_rec)

    # 7 chunks of solid above-threshold → 3 force-splits, plus a final
    # silent chunk to close the trailing continuation.
    for _ in range(7):
        rec.process_chunk(_loud(0.5), trigger_peak=0.5, **p)
    rec.process_chunk(_silent(), trigger_peak=0.0, **p)

    assert len(captured_flushes) >= 3
    for prev, cur in zip(captured_flushes, captured_flushes[1:]):
        delta = (cur['onset_time'] - prev['onset_time']).total_seconds()
        expected = prev['audio'].size / 44100
        assert abs(delta - expected) < 1e-6, (
            f'part onset must advance by the previous part duration '
            f'(got {delta}, expected {expected})')
    # And every flush is exactly 2 chunks except possibly the final
    # part which carries whatever was left.
    for i, f in enumerate(captured_flushes[:-1]):
        assert f['audio'].size == 2 * CHUNK_FRAMES, (
            f'part{i+1:02d} should be exactly max_samps')


def test_no_samples_lost_across_splits(captured_flushes):
    """The headline data-loss assertion: the concatenation of every
    split-half must equal the input audio. Use a small (but non-zero)
    min_cross so the first event qualifies on the very first chunk
    (1024 > 100 samples) but every continuation still bypasses
    min_cross — pre-fix the continuation needed 100 samples each time
    to re-qualify and they were silently lost."""
    sr = 44100
    min_cross_samps = 100  # tiny, but pre-fix a continuation would still lose this much per split
    rec = ThresholdRecorder()
    max_rec = 2.0 * CHUNK_FRAMES / sr
    p = _params(max_rec_sec=max_rec,
                min_cross_sec=min_cross_samps / sr,
                sample_rate=sr)

    n_chunks = 8
    for _ in range(n_chunks):
        rec.process_chunk(_loud(0.5), trigger_peak=0.5, **p)
    # Close the last continuation.
    rec.process_chunk(_silent(), trigger_peak=0.0, **p)

    total_samples = sum(f['audio'].size for f in captured_flushes)
    # The first event opens 100 samples into chunk 1 (min_cross gate),
    # so the maximum recoverable count is n_chunks*CHUNK_FRAMES -
    # min_cross_samps. Pre-fix every additional split would lose
    # another min_cross_samps; post-fix only the very first event pays
    # that cost.
    expected_min = n_chunks * CHUNK_FRAMES - min_cross_samps
    assert total_samples >= expected_min, (
        f'lost samples across splits: got {total_samples}, '
        f'expected at least {expected_min}')


def test_continuation_onset_time_advances_by_first_half_duration(captured_flushes):
    """Part2's onset_time must equal part1's onset + part1 duration —
    otherwise the two WAVs would be timestamped overlapping."""
    rec = ThresholdRecorder()
    max_rec = 3.0 * CHUNK_FRAMES / 44100
    p = _params(max_rec_sec=max_rec, sample_rate=44100)

    for _ in range(3):
        rec.process_chunk(_loud(0.5), trigger_peak=0.5, **p)
    rec.process_chunk(_silent(), trigger_peak=0.0, **p)

    assert len(captured_flushes) == 2
    o1 = captured_flushes[0]['onset_time']
    o2 = captured_flushes[1]['onset_time']
    expected_delta = datetime.timedelta(
        seconds=captured_flushes[0]['audio'].size / 44100)
    actual_delta = o2 - o1
    # Sub-microsecond precision is fine.
    assert abs((actual_delta - expected_delta).total_seconds()) < 1e-6


def test_no_part_token_when_event_did_not_split(captured_flushes):
    """A standalone event that closes naturally (never trips
    max_rec) keeps the user suffix verbatim — same contract as split
    parts now that the partNN token is gone."""
    rec = ThresholdRecorder()
    p = _params(max_rec_sec=10.0, filename_suffix='song')

    # Single short burst that closes well under max_rec.
    rec.process_chunk(_loud(0.5), trigger_peak=0.5, **p)
    rec.process_chunk(_silent(), trigger_peak=0.0, **p)

    assert len(captured_flushes) == 1
    # Suffix is the user suffix verbatim — no part token.
    assert captured_flushes[0]['suffix'] == 'song'


def test_continuation_global_spans_butt_join(captured_flushes):
    """The flushed_spans report drives the visual ``record`` strip in
    the UI (#32). Part1's flushed span must end exactly where part2's
    active span starts — otherwise the strip shows a gap in the
    rendered timeline that doesn't match the actual saved files. Use
    a non-chunk-aligned max_rec so the continuation actually has
    samples in the boundary chunk and reports a non-empty active span."""
    rec = ThresholdRecorder()
    # 2.5 chunks worth — boundary lands mid-chunk (chunk 3, sample 512).
    max_rec = 2.5 * CHUNK_FRAMES / 44100
    p = _params(max_rec_sec=max_rec)

    flushed_part1_end = None
    cont_active_start = None
    flushed_at_some_chunk = False
    for chunk_idx in range(1, 6):
        report = rec.process_chunk(
            _loud(0.5), trigger_peak=0.5,
            global_chunk_end=chunk_idx * CHUNK_FRAMES,
            **p,
        )
        if report['flushed_spans']:
            flushed_at_some_chunk = True
            flushed_part1_end = report['flushed_spans'][0][1]
            # Continuation should be active in the same chunk's report
            # (boundary lands at sample 512 in this chunk so the
            # continuation's samples_kept is 512 > 0).
            assert report['active_spans'], (
                'continuation must report an active span in the same '
                'chunk as the part1 flush')
            cont_active_start = report['active_spans'][0][0]
            break

    assert flushed_at_some_chunk, 'force-split never fired'
    # Span boundaries butt-join exactly.
    assert flushed_part1_end == cont_active_start


def test_continuation_inherits_parent_sample_rate(captured_flushes):
    """If the user changes the session SR mid-event, part2's WAV
    header must STILL carry the original capture rate (#46) — the
    samples were captured at the parent's rate."""
    rec = ThresholdRecorder()
    max_rec = 3.0 * CHUNK_FRAMES / 44100
    p = _params(max_rec_sec=max_rec, sample_rate=44100)

    # Three loud chunks at 44100, then continue but with a different
    # caller-supplied sample_rate (simulating the SR-change race).
    for _ in range(3):
        rec.process_chunk(_loud(0.5), trigger_peak=0.5, **p)
    p2 = dict(p)
    p2['sample_rate'] = 48000  # caller "changed" the SR
    rec.process_chunk(_silent(), trigger_peak=0.0, **p2)

    assert len(captured_flushes) == 2
    # Part1 was captured at 44100.
    assert captured_flushes[0]['sample_rate'] == 44100
    # Part2 inherits the parent's pinned rate, NOT the caller's
    # current 48000.
    assert captured_flushes[1]['sample_rate'] == 44100
