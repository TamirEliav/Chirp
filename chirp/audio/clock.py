"""DisciplinedClock — maps capture-ring sample indices to UTC wall time.

Why this exists
---------------
WAV filename onsets need two properties that no single clock provides:

* **Local smoothness** — adjacent / overlapping / split-part files must
  tile at sample precision, so timestamps must derive from the sample
  counter (querying ``time.time()`` per event adds scheduler + ingest
  backlog jitter and breaks tiling).
* **Long-run absolute accuracy** — the source's ADC crystal is off by
  tens of ppm (seconds per day, minutes over a 90-day recording run),
  and device stalls / OS-level drops silently freeze the sample counter
  while real time keeps moving. A pure "anchor + samples/rate" clock
  therefore drifts without bound.

This class is the NTP-style compromise: a free-running sample clock
steered ("disciplined") by noisy absolute observations of wall time.

Model
-----
The PortAudio callback calls :meth:`observe` with the capture ring's
cumulative frame count and ``time.time()`` right after each buffer
lands — the only point in the pipeline where a sample index and the
wall clock meet without queue backlog in between. Each observation
measures the offset between wall time and the nominal mapping::

    off_i = wall_i - (base_wall + (samples_i - base_samples) / rate)

Observation noise is ONE-SIDED: a callback fires at or after the
hardware finished capturing the buffer, never before. So the MINIMUM
offset over a sliding window (bucketed minima over ~60 s) is the best
estimate of the true offset — scheduling spikes and ingest stalls are
rejected outright instead of averaged in.

The applied correction slews toward that target at ``SLEW_RATE``
seconds per second of audio (1 ms/s = 1000 ppm — an order of magnitude
faster than any real crystal error, which is why no explicit rate term
is needed). Because the slew is rate-limited, the output is strictly
monotonic in the sample index. A hard forward STEP is taken only when
the target exceeds ``STEP_THRESHOLD`` (a real data hole: device stall,
OS overflow burst) AND the caller says no recording event is open, so
no file ever spans a discontinuity. Backward steps never happen — a
shrinking target (e.g. the system clock was set back) is always slewed.

Threading
---------
Strict SPSC, no locks: :meth:`observe` (producer, PortAudio callback)
only appends to a bounded deque — O(1), allocation-trivial, realtime
safe. :meth:`wall_at` (consumer, the entity's ingest thread) drains the
deque and owns every other piece of state. ``collections.deque``
append/popleft are atomic under the GIL.

Lifetime: one instance per capture ring (created alongside it in
``RecordingEntity._make_capture``), so its sample coordinates are the
ring's absolute frame counts and a device / sample-rate / source
rebuild naturally starts a fresh clock.
"""

import collections


class DisciplinedClock:
    #: Sliding-window geometry for the offset minimum: per-bucket minima
    #: over ``BUCKET_SEC``-wide wall-time buckets, ``N_BUCKETS`` deep.
    #: 6 x 10 s = a 50-60 s effective window — long enough to catch a
    #: low-latency callback, short enough that crystal drift inside the
    #: window (~6 ms at 100 ppm) stays negligible.
    BUCKET_SEC = 10.0
    N_BUCKETS = 6
    #: Max correction slew, in seconds of correction per second of
    #: audio. 1 ms/s = 1000 ppm of steering authority.
    SLEW_RATE = 1e-3
    #: Offset error (seconds) beyond which slewing is hopeless and a
    #: forward step is taken at the next quiet (no-open-event) chunk.
    STEP_THRESHOLD = 2.0
    #: Bound on the raw observation queue. At CHUNK_FRAMES=1024 even a
    #: 384 kHz stream produces ~375 obs/s, so this rides out ~10 s of
    #: ingest stall before old observations fall off (harmless — the
    #: window would have expired them anyway).
    OBS_MAXLEN = 4096

    def __init__(self, sample_rate: int):
        self.sample_rate = float(sample_rate)
        # SPSC observation queue: (cumulative_frames, wall_seconds).
        self._obs: collections.deque = collections.deque(maxlen=self.OBS_MAXLEN)
        # Nominal-mapping anchor, set from the first observation.
        self._base_samples: int | None = None
        self._base_wall = 0.0
        # Bucketed offset minima: deque of [bucket_id, min_offset].
        self._buckets: collections.deque = collections.deque()
        # Applied correction (seconds) and the sample index at the last
        # discipline update (meters the slew).
        self._corr = 0.0
        self._corr_init = False
        self._last_s: int | None = None
        # Telemetry for the UI / error log.
        self.step_count = 0
        self.last_step_sec = 0.0
        self.last_offset_target = 0.0

    # ── Producer side (PortAudio callback) ─────────────────────────────

    def observe(self, samples_total: int, wall: float) -> None:
        """Record one (cumulative capture frames, ``time.time()``) pair.
        Realtime-safe: a single bounded-deque append."""
        self._obs.append((samples_total, wall))

    # ── Consumer side (ingest thread) ──────────────────────────────────

    def _drain_observations(self) -> None:
        while True:
            try:
                s, w = self._obs.popleft()
            except IndexError:
                return
            if self._base_samples is None:
                self._base_samples = int(s)
                self._base_wall = float(w)
            off = w - (self._base_wall
                       + (s - self._base_samples) / self.sample_rate)
            b = int(w // self.BUCKET_SEC)
            if self._buckets and self._buckets[-1][0] == b:
                if off < self._buckets[-1][1]:
                    self._buckets[-1][1] = off
            else:
                self._buckets.append([b, off])
                while len(self._buckets) > self.N_BUCKETS:
                    self._buckets.popleft()

    def wall_at(self, samples_total: int | None, *,
                allow_step: bool = False) -> float | None:
        """Return the UTC epoch seconds of capture-frame ``samples_total``,
        or None while no observation has arrived yet (caller falls back
        to the coarse start_acq anchor).

        Strictly monotonic in ``samples_total``: the correction applied
        between two calls is bounded by ``SLEW_RATE`` times the audio
        that elapsed between them, except for a forward step, taken
        only when ``allow_step`` (no recording event open) and the
        target error exceeds ``STEP_THRESHOLD``.
        """
        if samples_total is None:
            return None
        self._drain_observations()
        if self._base_samples is None:
            return None

        target = min(m for _, m in self._buckets)
        self.last_offset_target = target

        s = int(samples_total)
        if self._last_s is None:
            self._last_s = s
        delta_audio = max(0.0, (s - self._last_s) / self.sample_rate)
        if s > self._last_s:
            self._last_s = s

        if not self._corr_init:
            # Adopt the filtered offset outright on the first call.
            #
            # The slew limiter exists to keep already-issued timestamps
            # consistent with each other; at this point none have been
            # issued, so there is nothing to stay consistent with — and
            # starting from zero is not neutral. The anchor observation
            # is the FIRST callback of the first delivery, and under
            # burst delivery (WASAPI exclusive, WDM-KS: a whole device
            # buffer handed over at once, split into several blocksize
            # callbacks with the same wall time) that is the *most late*
            # observation of its group. The filtered target is therefore
            # a whole device buffer behind it, and slewing 1 ms per
            # second of audio takes 10-30 minutes to walk out — during
            # which every filename is stamped late by up to that buffer.
            # Measured before this: +900 ms at start and still +329 ms
            # ten minutes in, on a 1 s WDM-KS buffer.
            self._corr_init = True
            self._corr = target
            return (self._base_wall
                    + (s - self._base_samples) / self.sample_rate
                    + self._corr)

        err = target - self._corr
        if err > self.STEP_THRESHOLD and allow_step:
            # A real hole in the capture (stall / drop burst): jump
            # forward in one piece, between events only.
            self.step_count += 1
            self.last_step_sec = err
            self._corr = target
        else:
            # Slew: bounded by the audio-time metered rate, so the
            # output can never run backwards (1 - SLEW_RATE > 0) and
            # adjacent files stay aligned to within ~1 ms per second.
            max_slew = self.SLEW_RATE * delta_audio
            if err > max_slew:
                err = max_slew
            elif err < -max_slew:
                err = -max_slew
            self._corr += err

        return (self._base_wall
                + (s - self._base_samples) / self.sample_rate
                + self._corr)
