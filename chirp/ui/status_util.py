"""Shared per-stream status composition (Phase 3 / v3.3.0).

The sticky-error badge logic (ingest / OS-drop / open / channel
truncation / writer failures composed into one flag + tooltip) was
originally embedded in ``ChirpWindow._update_error_sticky``. It now
lives here so the sidebar badges (Config mode) and the per-tile headers
(View mode) render identical state from one implementation.
"""

from __future__ import annotations


def compose_error_state(e) -> tuple[bool, str]:
    """Return ``(any_error, tooltip)`` for one RecordingEntity.

    Composes ingest errors, OS-level input overflows, capture open
    failures, WAV-replay channel truncation, and the process-global
    writer-pool error stats (surfaced on every stream — a write failure
    can't be attributed to its stream from here).
    """
    cap = getattr(e, 'capture', None)
    ingest_ever = bool(getattr(e, 'has_ever_ingest_errored', False))
    os_drop_ever = bool(getattr(cap, 'has_ever_os_dropped', False))
    underflow_ever = bool(getattr(cap, 'has_ever_underflowed', False))
    zero_run_ever = bool(getattr(e, 'has_ever_zero_run', False))
    open_err = getattr(cap, 'open_error', None)
    ch_trunc = bool(getattr(cap, 'channels_truncated', False))
    ch_trunc_msg = getattr(cap, 'channels_truncated_msg', '') or ''
    try:
        from chirp.recording import writer as _writer
        wr_has, wr_total, wr_last = _writer.error_stats()
    except Exception:
        wr_has, wr_total, wr_last = False, 0, None

    any_err = (ingest_ever or os_drop_ever or underflow_ever
               or zero_run_ever or bool(open_err) or ch_trunc or wr_has)
    if not any_err:
        return False, 'No pipeline errors recorded for this stream.'
    parts = []
    if ingest_ever:
        n = int(getattr(e, 'ingest_error_count_total', 0))
        last = getattr(e, 'last_ingest_error', None) or '?'
        parts.append(
            f'{n} DSP ingestion error{"s" if n != 1 else ""} '
            f'(last: {last})')
    if os_drop_ever:
        n = int(getattr(cap, 'os_drop_count_total', 0))
        parts.append(
            f'{n} OS-level input overflow{"s" if n != 1 else ""} — '
            f'samples lost before our queue')
    if underflow_ever:
        n = int(getattr(cap, 'underflow_count_total', 0))
        parts.append(
            f'{n} input underflow{"s" if n != 1 else ""} — PortAudio '
            f'inserted ZERO samples into the captured audio (zero runs '
            f'are being recorded)')
    if zero_run_ever:
        n = int(getattr(e, 'zero_run_count_total', 0))
        longest = int(getattr(e, 'zero_run_longest', 0))
        sr = int(getattr(e, 'sample_rate', 0)) or 1
        parts.append(
            f'{n} inserted-silence run{"s" if n != 1 else ""} detected '
            f'in captured audio (exact-zero gaps, longest '
            f'{longest / sr * 1000:.1f} ms) — the device/engine is '
            f'zero-filling; restart acquisition on ALL streams of this '
            f'input device to clear')
    if open_err:
        parts.append(f'Capture open failed: {open_err}')
    if ch_trunc:
        parts.append(f'WAV channel truncation: {ch_trunc_msg}')
    if wr_has:
        last = wr_last or '?'
        parts.append(
            f'{int(wr_total)} WAV write failure'
            f'{"s" if wr_total != 1 else ""} (last: {last})')
    return True, ' · '.join(parts) + ' — click to clear.'
