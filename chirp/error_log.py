"""Append-only error log written to ``chirp_errors.log`` in the
current working directory.

Surfaces the same error categories tracked for the sidebar `!` badge
(queue.Full drops, OS-level audio overflows, ingest-thread
exceptions, capture open failures, WAV-writer failures) so the user
can trace any indicator back to a precise timestamp + stream + (for
writer / WAV-playback failures) the file path involved.

Categories
----------
``queue_full``  — Python-side audio queue overflow (ingestion thread fell
                  behind; chunk dropped). Throttled.
``os_drop``     — PortAudio ``input_overflow`` flag (driver / OS lost
                  samples upstream of our queue). Throttled.
``ingest``      — Exception raised inside the per-entity ingest loop
                  (DSP / FFT / trigger). Logged every event.
``open``        — Capture failed to open the device or the WAV input
                  file. Logged every event.
``wav_writer``  — Worker pool failed while writing a triggered WAV.
                  Logged every event; ``wav_path`` carries the target
                  folder so the user can see which output was affected.
``saturation``  — A successfully-written WAV contained clipped samples
                  (peak >= 0.99 of full scale). Logged once per file —
                  not per sample — and ``wav_path`` carries the full
                  path of the recording so the user can identify
                  affected files for review or exclusion.

Throttling
----------
``queue_full`` and ``os_drop`` can fire on every audio chunk
(50+/second at the default sample rate / chunk size). To keep the log
useful and bounded, those two categories are limited to one entry per
(stream, category) per ``_THROTTLE_SECONDS``. The first occurrence in
any burst always logs immediately; subsequent events within the
window are suppressed. The cumulative count stamped on each line lets
the reader see how many events were actually suppressed between two
lines.

The logger never raises — any I/O failure (path locked, disk full,
permission error) is swallowed silently. Losing log lines is strictly
preferable to crashing the audio pipeline.
"""

import datetime
import os
import threading
import time

_LOG_FILENAME = 'chirp_errors.log'
_THROTTLE_SECONDS = 1.0
_THROTTLED_CATEGORIES = frozenset({'queue_full', 'os_drop'})

_lock = threading.Lock()
_last_log_at: dict[tuple[str, str], float] = {}


def _path() -> str:
    return os.path.join(os.getcwd(), _LOG_FILENAME)


def log(category: str, stream: str, message: str,
        wav_path: str | None = None) -> None:
    """Append one error entry to ``chirp_errors.log``. Never raises."""
    if category in _THROTTLED_CATEGORIES:
        now = time.monotonic()
        key = (stream or '', category)
        with _lock:
            last = _last_log_at.get(key, 0.0)
            if now - last < _THROTTLE_SECONDS:
                return
            _last_log_at[key] = now

    ts = datetime.datetime.now().isoformat(timespec='milliseconds')
    fields = [ts, category, f'stream={stream or "?"}']
    if wav_path:
        fields.append(f'file={wav_path}')
    fields.append((message or '').replace('\n', ' | ').strip())
    line = '\t'.join(fields) + '\n'
    try:
        with _lock:
            with open(_path(), 'a', encoding='utf-8') as f:
                f.write(line)
    except Exception:
        pass
