"""Device enumeration and robust name matching (#21 / c20).

Consolidates the device-lookup logic that was previously spread across
`RecordingEntity.from_dict` and `ChirpWindow._populate_devices` into a
single, testable module.

Match strategy (in priority order):
  1. **Exact** — device name matches the saved name verbatim AND host
     API matches (if a host API hint was saved).
  2. **Exact name, any host API** — same name but the host API differs
     (e.g. device moved from WASAPI to MME after a driver update).
  3. **Prefix** — saved name starts with the candidate name (or vice
     versa). Handles Windows truncation (`name[:32]`).
  4. **Substring** — the pre-bracket portion of the saved name appears
     inside the candidate name (legacy `.chirp` files often stored only
     the portion before the `[host api]` suffix).

Within each tier, ties are broken by `max_input_channels` (prefer the
device with more channels, since a stereo config saved against it is
more likely to work).
"""

from __future__ import annotations

from typing import List, Tuple

import sounddevice as sd


def list_input_devices() -> List[Tuple[int, dict]]:
    """Return `[(device_id, info_dict), ...]` for every input-capable device."""
    return [
        (i, info)
        for i, info in enumerate(sd.query_devices())
        if info.get('max_input_channels', 0) > 0
    ]


def host_api_name(device_info: dict) -> str:
    """Return the human-readable host API name for a device info dict."""
    try:
        idx = device_info.get('hostapi', -1)
        apis = sd.query_hostapis()
        if 0 <= idx < len(apis):
            return apis[idx].get('name', '')
    except Exception:
        pass
    return ''


def _strip_bracket_suffix(name: str) -> str:
    """Return the portion of *name* before the first `[`, stripped."""
    return name.split('[')[0].strip()


def find_device_by_name(
    name: str,
    hostapi_hint: str = '',
    candidates: list | None = None,
) -> Tuple[int | None, str | None]:
    """Resolve a saved device name to a live device ID.

    Parameters
    ----------
    name : str
        The device name as saved in the config file.
    hostapi_hint : str
        Optional host API name (e.g. ``"Windows WASAPI"``). When
        present, exact-name matches on the same host API are preferred.
    candidates : list or None
        Override for ``list_input_devices()`` — used by tests to inject
        a fake device list.

    Returns
    -------
    (device_id, warning)
        *device_id* is an int or None. *warning* is a human-readable
        string when the match was non-exact (or missing).
    """
    if not name:
        return None, None

    if candidates is None:
        candidates = list_input_devices()

    if not candidates:
        return None, f"Device '{name}' not found — no input devices available"

    name_stripped = _strip_bracket_suffix(name)

    # ── Tier 1: exact name + matching host API ───────────────────────
    if hostapi_hint:
        for dev_id, info in candidates:
            if (info['name'] == name
                    and host_api_name(info) == hostapi_hint
                    and info['max_input_channels'] > 0):
                return dev_id, None

    # ── Tier 2: exact name, any host API ─────────────────────────────
    exact = [(dev_id, info) for dev_id, info in candidates
             if info['name'] == name and info['max_input_channels'] > 0]
    if exact:
        best = max(exact, key=lambda t: t[1]['max_input_channels'])
        if hostapi_hint:
            return best[0], (
                f"Device '{name}' found on different host API "
                f"(expected {hostapi_hint})")
        return best[0], None

    # ── Tier 3: prefix match (handles Windows name truncation) ───────
    prefix = []
    for dev_id, info in candidates:
        cand = info['name']
        if cand.startswith(name) or name.startswith(cand):
            prefix.append((dev_id, info))
    if prefix:
        best = max(prefix, key=lambda t: t[1]['max_input_channels'])
        return best[0], (
            f"Device '{name}' not found exactly — using prefix match "
            f"'{best[1]['name']}'")

    # ── Tier 4: substring on pre-bracket portion ─────────────────────
    if name_stripped:
        substr = []
        for dev_id, info in candidates:
            if name_stripped in info['name']:
                substr.append((dev_id, info))
        if substr:
            best = max(substr, key=lambda t: t[1]['max_input_channels'])
            return best[0], (
                f"Device '{name}' not found exactly — using substring match "
                f"'{best[1]['name']}'")

    return None, f"Device '{name}' not found — using default"
