"""Device enumeration and name matching.

Thin wrappers around `sounddevice.query_devices` + `query_hostapis`.
Landed as a stub in c05 to give #21 a clear target — the monolith's
device-lookup logic (currently spread across `RecordingEntity.from_dict`
and `ChirpWindow._populate_devices`) will be consolidated here in c20.

For now, this module only exposes two small helpers that the rest of
the codebase can migrate to incrementally.
"""

import sounddevice as sd


def list_input_devices():
    """Return a list of `(device_id, info_dict)` for every input-capable
    device reported by sounddevice.

    `info_dict` is the raw dict from `sd.query_devices(i)` — left
    untouched so callers can inspect `max_input_channels`, `name`,
    `hostapi`, etc. directly.
    """
    return [
        (i, info)
        for i, info in enumerate(sd.query_devices())
        if info.get('max_input_channels', 0) > 0
    ]


def find_device_by_name(name: str):
    """Return the device_id whose name matches `name` exactly, or None.

    This is the minimal primitive that #21 will replace with a more
    robust matcher (hostapi-aware, substring fallback, tiebreaking on
    max_input_channels).
    """
    if not name:
        return None
    for i, info in enumerate(sd.query_devices()):
        if info.get('name') == name:
            return i
    return None
