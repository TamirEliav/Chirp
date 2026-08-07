"""Chirp settings schema (serialize / deserialize / migrate).

Extracted from the monolith in the Phase 1 refactor (plan: c07) and
upgraded in c17 (#22) with proper versioning, a migration dispatch,
and unknown-key warnings.

Schema shape (current — version 1):

    {
        "version": 1,
        "view_mode": {
            "columns": int,
            "panel_height": int,
            "use_opengl": bool,
            "active_only": bool,
            "fill_order": "row" | "column",
        },
        "monitor": {
            "output_device_name": str,
            "output_device_hostapi": str,
            "gain_percent": int,
            "muted": bool,
            "follow": bool,
            "source_index": int,
        },
        "recordings": [ RecordingEntity.to_dict(), ... ],
    }

When the on-disk format eventually changes, bump
`CONFIG_SCHEMA_VERSION` and add a `_migrate_vN_to_vN+1(data)` step
to `_MIGRATIONS`. `load_settings_dict` walks the chain from the file's
declared version up to the current version, so older files keep
loading cleanly.
"""

from __future__ import annotations

from typing import Iterable

from chirp.constants import (CAPTURE_BLOCKSIZE, CAPTURE_EXCLUSIVE,
                             CAPTURE_LATENCY)
from chirp.recording.entity import RecordingEntity


CONFIG_SCHEMA_VERSION = 1
# ``use_opengl`` (Phase 4) toggles GPU-accelerated view-mode rendering.
# Added as an optional view_mode key; older files simply omit it and get
# the default, so no schema version bump is needed.
DEFAULT_VIEW_MODE = {"columns": 1, "panel_height": 300, "use_opengl": True,
                     "active_only": True, "fill_order": "column"}

# Audio-monitor loopback settings (#7). Previously session-scoped and
# not persisted; now saved so a config restores the monitor routing.
# ``output_device_name`` / ``output_device_hostapi`` are resolved by name
# on load (indices shift), matching how input devices are handled.
# ``source_index`` is the position of the routed stream in ``recordings``
# (-1 = Off); ``gain_percent`` is the 0–200 slider value.
DEFAULT_MONITOR = {
    "output_device_name": "",
    "output_device_hostapi": "",
    "gain_percent": 100,
    "muted": False,
    "follow": False,
    "source_index": -1,
}


# Capture-engine tuning (v3.8.1). Machine/hardware-specific knobs for
# keeping the realtime callback on time: with sounddevice the callback is
# Python and must take the GIL, and a callback that misses its deadline
# is what lets a driver zero-fill or drop samples. ``capture_blocksize``
# sets how often it fires (bigger = less often, at the cost of monitor /
# display / trigger latency, none of which affects recorded audio);
# ``capture_latency`` is passed to PortAudio as the suggested input
# latency — 'low', 'high', or an explicit float in seconds — and is what
# actually buys the driver slack. ``capture_exclusive`` goes a layer
# lower: it asks WASAPI for exclusive use of the endpoint, taking the
# Windows audio engine (the layer field logs implicate in the
# inserted-silence fault) out of the capture path entirely.
DEFAULT_AUDIO = {
    "capture_blocksize": CAPTURE_BLOCKSIZE,
    "capture_latency": CAPTURE_LATENCY,
    "capture_exclusive": CAPTURE_EXCLUSIVE,
    # Inserted-silence auto-recovery. When the zero-sample duty cycle
    # stays above ``zero_recover_percent`` for ``zero_recover_seconds``,
    # acquisition is restarted on every stream sharing the affected
    # device — the only reset known to clear a capture session that has
    # latched into zero-filling, and the one the user otherwise has to
    # perform by hand. ``zero_recover_cooldown_sec`` keeps a persistent
    # fault from turning into a restart loop.
    "auto_recover_zero_runs": True,
    "zero_recover_percent": 5.0,
    "zero_recover_seconds": 15.0,
    "zero_recover_cooldown_sec": 120.0,
}

# Set of top-level keys recognized by the loader. Anything else triggers
# a warning so users notice typos and forks notice schema drift.
_KNOWN_TOP_KEYS: frozenset[str] = frozenset({
    "version", "view_mode", "recordings", "monitor", "audio",
})

_KNOWN_AUDIO_KEYS: frozenset[str] = frozenset(DEFAULT_AUDIO.keys())

# Set of keys recognized inside each recording's dict. Mirrors
# `RecordingEntity.to_dict` exactly.
_KNOWN_RECORDING_KEYS: frozenset[str] = frozenset({
    "name", "device_name", "device_hostapi", "sample_rate", "display_seconds",
    "channel_mode", "trigger_mode",
    "threshold", "min_cross_sec", "hold_sec", "post_trig_sec",
    "max_rec_sec", "pre_trig_sec",
    "freq_filter_enabled", "freq_lo", "freq_hi",
    "spec_nperseg", "spec_window",
    "freq_scale", "gain_db", "db_floor", "db_ceil",
    "display_freq_lo", "display_freq_hi",
    "output_dir", "filename_prefix", "filename_suffix",
    "ref_date", "dph_folder_prefix",
    "amp_ylim", "amp_scale",
    "spectral_trigger_mode", "spectral_threshold",
    # v3.3.0 additions — optional per-recording keys; older files simply
    # omit them and get the defaults, so no schema-version bump.
    "entropy_min_cross_sec", "rec_mode",
    "display_mode",
    # v3.4.0 additions — min-total-crossing file filter + per-stream
    # enable switch; optional, older files omit them and get defaults.
    "min_total_cross_sec",
    "stream_enabled",
    "analysis_nperseg", "analysis_window",
    "input_source", "wav_file_path", "wav_loop",
    # v3.6.1 additions — per-stream recognition color.
    "color",
    # v3.8.0 additions — per-stream parameter lock.
    "params_locked",
})

_KNOWN_VIEW_MODE_KEYS: frozenset[str] = frozenset({
    "columns", "panel_height", "use_opengl", "active_only", "fill_order",
})

_KNOWN_MONITOR_KEYS: frozenset[str] = frozenset(DEFAULT_MONITOR.keys())


# ── Migration chain ──────────────────────────────────────────────────────────

# Each entry maps `from_version -> callable(data) -> data` and is expected
# to bump `data["version"]` to `from_version + 1`. The chain is currently
# empty because v1 is the current format; this is the seam where future
# bumps land.
_MIGRATIONS: dict = {}


def _migrate(data: dict, warnings: list) -> dict:
    """Walk migrations from `data['version']` up to current."""
    v = data.get("version")
    if v is None:
        # Pre-versioned files (legacy `.chirp`). Treat as v1.
        warnings.append("Settings file has no version — assuming v1")
        v = 1
        data["version"] = 1
    if not isinstance(v, int):
        raise ValueError(f"Invalid settings version: {v!r}")
    if v > CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"Settings file is from a newer Chirp ({v}); this build "
            f"understands up to version {CONFIG_SCHEMA_VERSION}.")
    while v < CONFIG_SCHEMA_VERSION:
        step = _MIGRATIONS.get(v)
        if step is None:
            raise ValueError(f"No migration registered for version {v}")
        data = step(data)
        v = data.get("version", v + 1)
    return data


# ── Public API ───────────────────────────────────────────────────────────────

def _normalize_fill_order(value) -> str:
    """Coerce a fill-order value to 'row' or 'column' (default)."""
    return "row" if str(value).lower().startswith("row") else "column"


def parse_audio_settings(data: dict) -> tuple[dict, list[str]]:
    """Extract the ``audio`` section from a settings dict.

    Standalone (rather than part of :func:`load_settings_dict`'s return)
    because it must be applied BEFORE entities are constructed —
    building a RecordingEntity opens its capture, which is exactly when
    the blocksize takes effect.
    """
    warnings: list[str] = []
    raw = data.get("audio") or {}
    if not isinstance(raw, dict):
        warnings.append("audio is not a dict — using defaults")
        raw = {}
    unknown = sorted(set(raw.keys()) - _KNOWN_AUDIO_KEYS)
    if unknown:
        warnings.append(
            f"Ignoring unknown audio key(s): {', '.join(unknown)}")
    out = dict(DEFAULT_AUDIO)
    for k in _KNOWN_AUDIO_KEYS:
        if k in raw:
            out[k] = raw[k]
    # Coerce the numeric / boolean knobs so a hand-edited config can't
    # feed a string into the watchdog arithmetic.
    out["capture_exclusive"] = bool(out["capture_exclusive"])
    try:
        out["auto_recover_zero_runs"] = bool(out["auto_recover_zero_runs"])
        out["zero_recover_percent"] = max(0.1, float(out["zero_recover_percent"]))
        out["zero_recover_seconds"] = max(1.0, float(out["zero_recover_seconds"]))
        out["zero_recover_cooldown_sec"] = max(
            5.0, float(out["zero_recover_cooldown_sec"]))
    except (TypeError, ValueError):
        warnings.append("audio: invalid auto-recovery value(s) — using defaults")
        for k in ("auto_recover_zero_runs", "zero_recover_percent",
                  "zero_recover_seconds", "zero_recover_cooldown_sec"):
            out[k] = DEFAULT_AUDIO[k]
    return out, warnings


def build_settings_dict(entities: Iterable[RecordingEntity],
                        view_mode: dict | None = None,
                        monitor: dict | None = None,
                        audio: dict | None = None) -> dict:
    """Serialize a collection of entities + view-mode + monitor + audio
    settings to a plain dict."""
    vm = dict(DEFAULT_VIEW_MODE)
    if view_mode:
        vm.update(view_mode)
    mon = dict(DEFAULT_MONITOR)
    if monitor:
        mon.update(monitor)
    aud = dict(DEFAULT_AUDIO)
    if audio:
        aud.update(audio)
    return {
        "version": CONFIG_SCHEMA_VERSION,
        "view_mode": {
            "columns":      vm.get("columns", DEFAULT_VIEW_MODE["columns"]),
            "panel_height": vm.get("panel_height", DEFAULT_VIEW_MODE["panel_height"]),
            "use_opengl":   bool(vm.get("use_opengl", DEFAULT_VIEW_MODE["use_opengl"])),
            "active_only":  bool(vm.get("active_only", DEFAULT_VIEW_MODE["active_only"])),
            "fill_order":   _normalize_fill_order(vm.get("fill_order", DEFAULT_VIEW_MODE["fill_order"])),
        },
        "monitor": {
            "output_device_name":    str(mon.get("output_device_name", "") or ""),
            "output_device_hostapi": str(mon.get("output_device_hostapi", "") or ""),
            "gain_percent":          int(mon.get("gain_percent", DEFAULT_MONITOR["gain_percent"])),
            "muted":                 bool(mon.get("muted", DEFAULT_MONITOR["muted"])),
            "follow":                bool(mon.get("follow", DEFAULT_MONITOR["follow"])),
            "source_index":          int(mon.get("source_index", DEFAULT_MONITOR["source_index"])),
        },
        "audio": {
            "capture_blocksize": int(aud.get("capture_blocksize",
                                             DEFAULT_AUDIO["capture_blocksize"])),
            "capture_latency":   aud.get("capture_latency",
                                         DEFAULT_AUDIO["capture_latency"]),
            "capture_exclusive": bool(
                aud.get("capture_exclusive",
                        DEFAULT_AUDIO["capture_exclusive"])),
            "auto_recover_zero_runs": bool(
                aud.get("auto_recover_zero_runs",
                        DEFAULT_AUDIO["auto_recover_zero_runs"])),
            "zero_recover_percent": float(
                aud.get("zero_recover_percent",
                        DEFAULT_AUDIO["zero_recover_percent"])),
            "zero_recover_seconds": float(
                aud.get("zero_recover_seconds",
                        DEFAULT_AUDIO["zero_recover_seconds"])),
            "zero_recover_cooldown_sec": float(
                aud.get("zero_recover_cooldown_sec",
                        DEFAULT_AUDIO["zero_recover_cooldown_sec"])),
        },
        "recordings": [e.to_dict() for e in entities],
    }


def load_settings_dict(data: dict) -> tuple[list[RecordingEntity], dict, dict, list[str]]:
    """Parse a settings dict into `(entities, view_mode, monitor, warnings)`.

    Raises ValueError if `data` is malformed. Unknown keys at any
    level produce warnings instead of failures so a config file from
    a slightly newer (compatible) build still loads.
    """
    if not isinstance(data, dict) or "recordings" not in data:
        raise ValueError("Invalid settings file format: missing 'recordings' array")

    warnings: list[str] = []
    data = _migrate(dict(data), warnings)  # shallow copy — never mutate caller

    # Top-level unknown keys
    unknown_top = sorted(set(data.keys()) - _KNOWN_TOP_KEYS)
    if unknown_top:
        warnings.append(
            f"Ignoring unknown top-level setting(s): {', '.join(unknown_top)}")

    vm_raw = data.get("view_mode") or {}
    if not isinstance(vm_raw, dict):
        warnings.append("view_mode is not a dict — using defaults")
        vm_raw = {}
    unknown_vm = sorted(set(vm_raw.keys()) - _KNOWN_VIEW_MODE_KEYS)
    if unknown_vm:
        warnings.append(
            f"Ignoring unknown view_mode key(s): {', '.join(unknown_vm)}")
    view_mode = {
        "columns":      vm_raw.get("columns", DEFAULT_VIEW_MODE["columns"]),
        "panel_height": vm_raw.get("panel_height", DEFAULT_VIEW_MODE["panel_height"]),
        "use_opengl":   bool(vm_raw.get("use_opengl", DEFAULT_VIEW_MODE["use_opengl"])),
        "active_only":  bool(vm_raw.get("active_only", DEFAULT_VIEW_MODE["active_only"])),
        "fill_order":   _normalize_fill_order(vm_raw.get("fill_order", DEFAULT_VIEW_MODE["fill_order"])),
    }

    mon_raw = data.get("monitor") or {}
    if not isinstance(mon_raw, dict):
        warnings.append("monitor is not a dict — using defaults")
        mon_raw = {}
    unknown_mon = sorted(set(mon_raw.keys()) - _KNOWN_MONITOR_KEYS)
    if unknown_mon:
        warnings.append(
            f"Ignoring unknown monitor key(s): {', '.join(unknown_mon)}")
    monitor = {
        "output_device_name":    str(mon_raw.get("output_device_name", "") or ""),
        "output_device_hostapi": str(mon_raw.get("output_device_hostapi", "") or ""),
        "gain_percent":          int(mon_raw.get("gain_percent", DEFAULT_MONITOR["gain_percent"])),
        "muted":                 bool(mon_raw.get("muted", DEFAULT_MONITOR["muted"])),
        "follow":                bool(mon_raw.get("follow", DEFAULT_MONITOR["follow"])),
        "source_index":          int(mon_raw.get("source_index", DEFAULT_MONITOR["source_index"])),
    }

    entities: list[RecordingEntity] = []
    for i, rec_d in enumerate(data["recordings"]):
        if not isinstance(rec_d, dict):
            warnings.append(f"Recording #{i} is not a dict — skipping")
            continue
        unknown_rec = sorted(set(rec_d.keys()) - _KNOWN_RECORDING_KEYS)
        if unknown_rec:
            warnings.append(
                f"Recording '{rec_d.get('name', f'#{i}')}': "
                f"ignoring unknown key(s): {', '.join(unknown_rec)}")
        ent, warn = RecordingEntity.from_dict(rec_d)
        entities.append(ent)
        if warn:
            warnings.append(warn)

    return entities, view_mode, monitor, warnings
