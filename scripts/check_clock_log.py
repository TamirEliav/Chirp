"""Audit a run for LOST capture samples, from chirp_clock_log.csv.

Chirp already detects samples the driver *inserts* (the zero-run
detector: digital silence spliced into live audio). This script answers
the opposite question — did the driver ever fail to deliver samples at
all? — which no in-app counter can see, because omitted samples leave no
trace in the audio itself.

Method
------
Each clock-log row pairs a ring-absolute sample index with the wall
clock at the moment that sample was ingested. Over any interval, a
healthy capture advances

    delta_samples / sample_rate  ==  delta_wall_seconds

to within the crystal difference between the audio device and the PC
clock (tens of ppm; 100 ppm = 0.36 s/hour). Samples that were never
delivered make the sample count fall behind wall time, so the shortfall
accumulates. A steady shortfall of thousands of ppm, or a step in the
per-interval rate, is capture loss rather than clock drift.

The reverse sign (samples running AHEAD of wall time) is not loss — it
usually means the PC clock was stepped, e.g. by NTP.

Usage
-----
    python scripts/check_clock_log.py <path-to-chirp_clock_log.csv> [rate]

``rate`` defaults to 44100. Streams are reported separately; a stream's
rows are only comparable within one acquisition run, so gaps longer than
``--gap`` seconds (default 300) split the analysis into segments.
"""

from __future__ import annotations

import argparse
import collections
import csv
import sys

# Crystal difference between two free-running oscillators. Anything
# beyond this is not the clocks disagreeing.
DRIFT_OK_PPM = 200.0


def _load(path):
    rows = collections.defaultdict(list)
    with open(path, newline='', encoding='utf-8', errors='replace') as f:
        for row in csv.DictReader(f):
            try:
                rows[row['stream']].append(
                    (float(row['wall_epoch']), int(row['ring_sample_index'])))
            except (TypeError, ValueError, KeyError):
                continue
    for v in rows.values():
        v.sort()
    return rows


def _segments(points, gap_sec, rate):
    """Split on long gaps AND on sample-index resets (a restart rebuilds
    the ring, so the index is not comparable across one)."""
    seg = [points[0]]
    for prev, cur in zip(points, points[1:]):
        if cur[0] - prev[0] > gap_sec or cur[1] < prev[1]:
            if len(seg) > 1:
                yield seg
            seg = []
        seg.append(cur)
    if len(seg) > 1:
        yield seg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('csv_path')
    ap.add_argument('rate', nargs='?', type=int, default=44100)
    ap.add_argument('--gap', type=float, default=300.0,
                    help='seconds without a row that starts a new segment')
    args = ap.parse_args(argv)

    streams = _load(args.csv_path)
    if not streams:
        print('no usable rows found')
        return 1

    worst = 0.0
    for name in sorted(streams):
        pts = streams[name]
        print(f'\n=== {name} ({len(pts)} rows) ===')
        for i, seg in enumerate(_segments(pts, args.gap, args.rate), 1):
            wall = seg[-1][0] - seg[0][0]
            samples = seg[-1][1] - seg[0][1]
            if wall <= 0:
                continue
            audio = samples / args.rate
            short = wall - audio
            ppm = short / wall * 1e6
            worst = max(worst, ppm)
            verdict = ('OK' if abs(ppm) <= DRIFT_OK_PPM
                       else ('LOST SAMPLES' if ppm > 0 else 'clock stepped?'))
            print(f'  segment {i}: {wall / 3600:6.2f} h wall, '
                  f'{audio / 3600:6.2f} h audio, shortfall {short:+8.2f} s '
                  f'({ppm:+9.0f} ppm)  {verdict}')

            # Where did it go? Report the worst individual intervals so a
            # steady drift can be told from discrete dropouts.
            bad = []
            for (w0, s0), (w1, s1) in zip(seg, seg[1:]):
                dw = w1 - w0
                if dw <= 0:
                    continue
                d = dw - (s1 - s0) / args.rate
                if d / dw * 1e6 > DRIFT_OK_PPM * 5:
                    bad.append((d, w0, dw))
            if bad:
                bad.sort(reverse=True)
                print(f'    {len(bad)} interval(s) lost audio; worst:')
                for d, w0, dw in bad[:5]:
                    print(f'      epoch {w0:.0f}: {d:6.2f} s missing '
                          f'over a {dw:5.1f} s interval')

    print(f'\nworst shortfall: {worst:+.0f} ppm '
          f'(crystal difference alone stays under {DRIFT_OK_PPM:.0f})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
