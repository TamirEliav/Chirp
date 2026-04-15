"""WAV writer — synchronous helper + legacy daemon-thread launcher.

Extracted from `ThresholdRecorder._write_wav` in the Phase 1 refactor
(plan: c06). This file exists primarily as a target for issue #17
(safe shutdown + writer pool): c16 will replace the daemon-thread
launcher with a bounded worker pool that can be drained on close.

For now, the API is:

  - `write_wav_sync(buf_snapshot, output_dir, ...)` — synchronous,
    matches the old `_write_wav` body exactly.
  - `start_flush_thread(buf_snapshot, ...)` — the legacy daemon-thread
    launcher. `ThresholdRecorder._start_flush` delegates to this so
    the test monkeypatch on `_start_flush` still takes precedence.

Behavior is intentionally unchanged from the monolith.
"""

import datetime
import os
import threading

import numpy as np
import scipy.io.wavfile

from chirp.constants import SAMPLE_RATE


def write_wav_sync(buf_snapshot: list, output_dir: str,
                   prefix: str = '', suffix: str = '',
                   sample_rate: int = SAMPLE_RATE,
                   onset_time=None) -> str:
    """Concatenate chunks and write a 16-bit PCM WAV synchronously.

    Returns the output path. Raises on I/O failure — the legacy
    daemon-thread wrapper swallows exceptions by killing the thread.
    """
    audio = np.concatenate(buf_snapshot)
    if audio.ndim == 1:
        audio = audio.flatten()
    pcm16 = (audio * 32767.0).clip(-32768, 32767).astype(np.int16)
    os.makedirs(output_dir, exist_ok=True)
    n_samples = audio.shape[0]
    audio_dur = n_samples / sample_rate
    if onset_time is not None:
        onset = onset_time
    else:
        onset = datetime.datetime.now() - datetime.timedelta(seconds=audio_dur)
    epoch_ms = int(onset.timestamp() * 1000)
    local_ts = onset.strftime('%Y%m%d_%H%M%S_%f')[:-3]
    parts = [p for p in [prefix.rstrip('_'), str(epoch_ms), local_ts,
                         suffix.lstrip('_')] if p]
    fname = '_'.join(parts) + '.wav'
    path  = os.path.join(output_dir, fname)
    scipy.io.wavfile.write(path, sample_rate, pcm16)
    ch_str = 'stereo' if audio.ndim == 2 else 'mono'
    print(f'[REC] saved {path}  ({n_samples/sample_rate:.2f} s, {ch_str})')
    return path


def start_flush_thread(buf_snapshot: list, output_dir: str,
                       prefix: str = '', suffix: str = '',
                       sample_rate: int = SAMPLE_RATE,
                       onset_time=None) -> None:
    """Legacy fire-and-forget daemon-thread writer.

    c16 (#17) will replace this with a bounded worker pool that can be
    drained at application shutdown.
    """
    threading.Thread(
        target=write_wav_sync,
        args=(list(buf_snapshot), output_dir, prefix, suffix, sample_rate, onset_time),
        daemon=True,
    ).start()
