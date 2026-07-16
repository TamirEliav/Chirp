"""Tests for the sidecar clock log (chirp_clock_log.csv).

One CSV row per stream per ``clock_log_interval_sec`` pairing the
capture-ring sample index with the derived capture time and the raw
wall clock — the offline audit trail for filename timestamps over
multi-week runs. Rows are appended via the writer pool, never on the
ingest thread, and only while ``acq_running`` (direct test feeds in
the rest of the suite never log).
"""

import csv
import datetime

import numpy as np

from chirp.constants import CHUNK_FRAMES
from chirp.recording import writer
from chirp.recording.entity import CLOCK_LOG_FILENAME, RecordingEntity


ANCHOR = datetime.datetime(2026, 7, 15, 13, 0, 0)


def _entity(tmp_path, interval=0.0):
    e = RecordingEntity(name='LogTest', device_id=None)
    e.output_dir = str(tmp_path)
    e.clock_log_interval_sec = interval
    e.acq_running = True           # gate for the log; capture stays idle
    e._wall_anchor_time = ANCHOR
    e._wall_anchor_samples = 0
    return e


def _rows(tmp_path):
    assert writer.drain(10)
    path = tmp_path / CLOCK_LOG_FILENAME
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def test_rows_written_with_anchor_source(tmp_path):
    e = _entity(tmp_path, interval=0.0)
    chunk = np.zeros(CHUNK_FRAMES, dtype=np.float32)
    try:
        for _ in range(3):
            e.ingest_chunk(chunk)
    finally:
        e.acq_running = False
    rows = _rows(tmp_path)
    assert len(rows) == 3
    for i, r in enumerate(rows):
        assert r['stream'] == 'LogTest'
        assert r['derived_source'] == 'anchor'
        assert int(r['ring_sample_index']) == (i + 1) * CHUNK_FRAMES
        # derived_epoch matches the anchor + sample-clock derivation.
        expected = (ANCHOR + datetime.timedelta(
            seconds=(i + 1) * CHUNK_FRAMES / e.sample_rate)).timestamp()
        assert abs(float(r['derived_epoch']) - expected) < 1e-3
        float(r['wall_epoch'])  # parses
        assert r['utc_iso']     # present


def test_clock_source_tagged_and_epoch_matches(tmp_path):
    e = _entity(tmp_path, interval=0.0)
    t0 = 1_700_000_000.0
    e.clock.observe(CHUNK_FRAMES, t0)
    try:
        e.ingest_chunk(np.zeros(CHUNK_FRAMES, dtype=np.float32),
                       abs_end=CHUNK_FRAMES)
    finally:
        e.acq_running = False
    rows = _rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]['derived_source'] == 'clock'
    assert abs(float(rows[0]['derived_epoch']) - t0) < 1e-3
    assert int(rows[0]['ring_sample_index']) == CHUNK_FRAMES


def test_interval_throttles_rows(tmp_path):
    e = _entity(tmp_path, interval=3600.0)
    chunk = np.zeros(CHUNK_FRAMES, dtype=np.float32)
    try:
        for _ in range(5):
            e.ingest_chunk(chunk)
    finally:
        e.acq_running = False
    rows = _rows(tmp_path)
    assert len(rows) == 1   # first chunk logs, the rest are inside the interval


def test_no_log_without_acquisition(tmp_path):
    e = _entity(tmp_path, interval=0.0)
    e.acq_running = False
    e.ingest_chunk(np.zeros(CHUNK_FRAMES, dtype=np.float32))
    assert writer.drain(10)
    assert not (tmp_path / CLOCK_LOG_FILENAME).exists()


def test_header_written_once(tmp_path):
    e = _entity(tmp_path, interval=0.0)
    chunk = np.zeros(CHUNK_FRAMES, dtype=np.float32)
    try:
        e.ingest_chunk(chunk)
        e.ingest_chunk(chunk)
    finally:
        e.acq_running = False
    assert writer.drain(10)
    text = (tmp_path / CLOCK_LOG_FILENAME).read_text(encoding='utf-8')
    assert text.count('utc_iso') == 1
    assert text.splitlines()[0].startswith('utc_iso,stream,')
