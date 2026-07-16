"""Tests for the disciplined timestamp clock (chirp/audio/clock.py).

The clock maps capture-ring sample indices to UTC epoch seconds:
a free-running sample clock steered onto wall time by per-callback
observations, filtered with a windowed MINIMUM (observation noise is
one-sided — callbacks fire at or after capture, never before), with a
slew-rate-limited correction and forward-only steps gated on "no
recording event open".
"""

import datetime

import numpy as np

from chirp.audio.clock import DisciplinedClock
from chirp.constants import CHUNK_FRAMES
from chirp.recording.entity import RecordingEntity


SR = 44100
T0 = 1_700_000_000.0  # arbitrary epoch anchor
CHUNK_SEC = CHUNK_FRAMES / SR


def _feed_ideal(clock, k_from, k_to, extra_wall=0.0, drift=0.0,
                delays=None):
    """Feed observations for chunks k_from..k_to-1: sample index
    k*CHUNK_FRAMES at wall T0 + k*CHUNK_SEC*(1+drift) + extra_wall
    (+ optional per-obs delivery delay)."""
    for i, k in enumerate(range(k_from, k_to)):
        d = delays[i % len(delays)] if delays else 0.0
        clock.observe(k * CHUNK_FRAMES,
                      T0 + k * CHUNK_SEC * (1.0 + drift) + extra_wall + d)


# ── Basics ──────────────────────────────────────────────────────────────────

def test_no_observations_returns_none():
    c = DisciplinedClock(SR)
    assert c.wall_at(CHUNK_FRAMES) is None
    assert c.wall_at(None) is None


def test_ideal_observations_map_linearly():
    c = DisciplinedClock(SR)
    _feed_ideal(c, 1, 200)
    w = c.wall_at(200 * CHUNK_FRAMES)
    assert abs(w - (T0 + 200 * CHUNK_SEC)) < 1e-6


def test_min_filter_rejects_one_sided_jitter():
    """Delivery delays are one-sided (late only): the offset target must
    track the SMALLEST observed delay, not the mean."""
    c = DisciplinedClock(SR)
    delays = [0.050, 0.020, 0.005, 0.030]
    _feed_ideal(c, 1, 100, delays=delays)
    c.wall_at(100 * CHUNK_FRAMES)
    # Base was anchored on the first obs (delay 0.050); the best later
    # obs had delay 0.005 → target = 0.005 - 0.050 = -0.045.
    assert abs(c.last_offset_target - (-0.045)) < 1e-6


# ── Discipline dynamics ─────────────────────────────────────────────────────

def test_slew_is_rate_limited_and_output_monotonic():
    """A sudden (sub-threshold) target change must be absorbed at
    <= SLEW_RATE seconds per second of audio, and the output must stay
    strictly monotonic throughout."""
    c = DisciplinedClock(SR)
    _feed_ideal(c, 1, 50)
    w_prev = c.wall_at(50 * CHUNK_FRAMES)
    # Offset jumps +0.5 s (below STEP_THRESHOLD) from chunk 50 on.
    _feed_ideal(c, 51, 400, extra_wall=0.5)
    outputs = []
    for k in range(51, 400, 10):
        w = c.wall_at(k * CHUNK_FRAMES, allow_step=True)
        outputs.append(w)
        # Correction advanced no faster than SLEW_RATE * audio time.
        audio_dt = (k - 50) * CHUNK_SEC
        nominal = T0 + k * CHUNK_SEC
        assert w - nominal <= c.SLEW_RATE * audio_dt + 1e-9
        assert w > w_prev
        w_prev = w
    # No step was taken for a sub-threshold error.
    assert c.step_count == 0
    # And the correction did make progress toward the target.
    assert outputs[-1] - (T0 + 390 * CHUNK_SEC) > 0.005


def test_drift_is_tracked_within_tolerance():
    """A 100 ppm-fast source clock (offset growing ~8.6 ms/day... here
    compressed: wall advances 1.0001x per sample) must be tracked to
    within ~the window-lag bound, not accumulate."""
    c = DisciplinedClock(SR)
    drift = 1e-4  # 100 ppm
    k = 1
    step = 10  # discipline every 10 chunks
    while k < 5000:
        _feed_ideal(c, k, k + step, drift=drift)
        k += step
        c.wall_at(k * CHUNK_FRAMES)
    true_wall = T0 + k * CHUNK_SEC * (1.0 + drift)
    got = c.wall_at(k * CHUNK_FRAMES)
    # ~116 s of audio → raw drift would be ~12 ms; disciplined error
    # must stay under the window-lag bound (~drift * window + slack).
    assert abs(got - true_wall) < 0.02
    assert c.step_count == 0


def test_step_requires_permission_and_threshold():
    """A capture hole (samples frozen while wall time advanced) beyond
    STEP_THRESHOLD: no step without permission; one clean forward step
    with it."""
    c = DisciplinedClock(SR)
    _feed_ideal(c, 1, 100)
    c.wall_at(100 * CHUNK_FRAMES)
    # Hole: 5 s of real time lost upstream — all later observations are
    # +5 s relative to the sample count. Feed >70 s of wall time so the
    # pre-hole buckets roll out of the min window.
    n_post = int(75.0 / CHUNK_SEC)
    _feed_ideal(c, 101, 101 + n_post, extra_wall=5.0)
    s_now = (101 + n_post) * CHUNK_FRAMES
    w_denied = c.wall_at(s_now, allow_step=False)
    assert c.step_count == 0            # no permission → slew only
    w_stepped = c.wall_at(s_now + CHUNK_FRAMES, allow_step=True)
    assert c.step_count == 1
    assert 4.0 < (w_stepped - w_denied) < 5.5   # jumped the hole
    assert c.last_step_sec > 4.0


def test_never_steps_backward():
    """A large NEGATIVE offset change (system clock set back) must never
    step — even with permission — only slew, keeping output monotonic."""
    c = DisciplinedClock(SR)
    _feed_ideal(c, 1, 100)
    w1 = c.wall_at(100 * CHUNK_FRAMES, allow_step=True)
    n_post = int(75.0 / CHUNK_SEC)
    _feed_ideal(c, 101, 101 + n_post, extra_wall=-5.0)
    w2 = c.wall_at((101 + n_post) * CHUNK_FRAMES, allow_step=True)
    w3 = c.wall_at((102 + n_post) * CHUNK_FRAMES, allow_step=True)
    assert c.step_count == 0
    assert w2 > w1
    assert w3 > w2


def test_same_sample_index_is_idempotent():
    c = DisciplinedClock(SR)
    _feed_ideal(c, 1, 10)
    a = c.wall_at(10 * CHUNK_FRAMES)
    b = c.wall_at(10 * CHUNK_FRAMES)
    assert a == b


# ── Entity integration ──────────────────────────────────────────────────────

def _spy_chunk_end_wall(e):
    captured = []
    orig = e.recorder.process_chunk

    def spy(chunk, **kw):
        captured.append(kw.get('chunk_end_wall'))
        return orig(chunk, **kw)

    e.recorder.process_chunk = spy
    return captured


def test_entity_uses_disciplined_clock_when_abs_end_given():
    e = RecordingEntity(name="ClkTest", device_id=None)
    captured = _spy_chunk_end_wall(e)
    chunk = np.zeros(CHUNK_FRAMES, dtype=np.float32)

    e.clock.observe(CHUNK_FRAMES, T0)
    e.ingest_chunk(chunk, abs_end=CHUNK_FRAMES)
    cw = captured[-1]
    assert cw is not None and cw.tzinfo is not None      # aware UTC
    assert abs(cw.timestamp() - T0) < 1e-6

    step = CHUNK_FRAMES / e.sample_rate
    e.clock.observe(2 * CHUNK_FRAMES, T0 + step)
    e.ingest_chunk(chunk, abs_end=2 * CHUNK_FRAMES)
    assert abs(captured[-1].timestamp() - (T0 + step)) < 1e-6
    assert captured[-1] > captured[-2]


def test_entity_falls_back_to_anchor_without_observations():
    """No clock observations (WAV playback, tests): the M5 start_acq
    anchor still drives chunk_end_wall — even when abs_end is passed."""
    e = RecordingEntity(name="ClkFallback", device_id=None)
    captured = _spy_chunk_end_wall(e)
    anchor = datetime.datetime(2026, 7, 15, 13, 0, 0)
    e._wall_anchor_time = anchor
    e._wall_anchor_samples = 0
    e.ingest_chunk(np.zeros(CHUNK_FRAMES, dtype=np.float32),
                   abs_end=CHUNK_FRAMES)
    expected = anchor + datetime.timedelta(
        seconds=CHUNK_FRAMES / e.sample_rate)
    assert abs((captured[-1] - expected).total_seconds()) < 1e-6


def test_entity_clock_is_recreated_with_capture():
    """A capture rebuild (device / SR / source switch) must start a
    fresh clock whose coordinates match the fresh ring."""
    e = RecordingEntity(name="ClkRebuild", device_id=None)
    old_clock = e.clock
    e.clock.observe(CHUNK_FRAMES, T0)
    e.change_sample_rate(22050 if e.sample_rate != 22050 else 44100)
    assert e.clock is not old_clock
    assert e.clock.wall_at(CHUNK_FRAMES) is None   # no stale observations
    e.close()


# ── Filename composition with aware onsets ──────────────────────────────────

def test_compose_filename_aware_utc_onset_renders_local_token():
    from chirp.recording.writer import _compose_filename
    onset = datetime.datetime(2026, 7, 16, 10, 30, 15, 123000,
                              tzinfo=datetime.timezone.utc)
    fname = _compose_filename('pre', 'suf', onset, 'S1')
    epoch_ms = int(onset.timestamp() * 1000)
    local_ts = onset.astimezone().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    assert fname == f'pre_{epoch_ms}_{local_ts}_S1_suf.wav'


def test_compose_filename_naive_onset_unchanged():
    from chirp.recording.writer import _compose_filename
    onset = datetime.datetime(2026, 7, 16, 10, 30, 15, 123000)
    fname = _compose_filename('pre', 'suf', onset, 'S1')
    epoch_ms = int(onset.timestamp() * 1000)
    assert fname == f'pre_{epoch_ms}_20260716_103015_123_S1_suf.wav'
