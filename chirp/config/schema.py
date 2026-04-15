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
        },
        "recordings": [ RecordingEntity.to_dict(), ... ],
    }

When the on-disk format eventually changes, bump
`CONFIG_SCHEMA_VERSION` and add a `_migrate_vN_to_vN+1(data)` step
to `_MIGRATIONS`. `load_settings_dict` walks the chain from the file's
declared version up to the current version, so older files keep
loading cleanly.
"""

from typing import Iterable, Tuple, List

from chirp.recording.entity import RecordingEntity


CONFIG_SCHEMA_VERSION = 1
DEFAULT_VIEW_MODE = {"columns": 1, "panel_height": 300}


# Set of top-level keys recognized by the loader. Anything else triggers
# a warning so users notice typos and forks notice schema drift.
_KNOWN_TOP_KEYS: frozenset[str] = frozenset({
    "version", "view_mode", "recordings",
})

# Set of keys recognized inside each recording's dict. Mirrors
# `RecordingEntity.to_dict` exactly.
_KNOWN_RECORDING_KEYS: frozenset[str] = frozenset({
    "name", "device_name", "sample_rate", "display_seconds",
    "channel_mode", "trigger_mode",
    "threshold", "min_cross_sec", "hold_sec", "post_trig_sec",
    "max_rec_sec", "pre_trig_sec",
    "freq_filter_enabled", "freq_lo", "freq_hi",
    "spec_nperseg", "spec_window",
    "freq_scale", "gain_db", "db_floor", "db_ceil",
    "display_freq_lo", "display_freq_hi",
    "output_dir", "filename_prefix", "filename_suffix",
    "ref_date", "dph_folder_prefix",
    "amp_ylim",
    "spectral_trigger_mode", "spectral_threshold",
    "display_mode",
})

_KNOWN_VIEW_MODE_KEYS: frozenset[str] = frozenset({
    "columns", "panel_height",
})


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

def build_settings_dict(entities: Iterable[RecordingEntity],
                        view_mode: dict | None = None) -> dict:
    """Serialize a collection of entities + view-mode to a plain dict."""
    vm = dict(DEFAULT_VIEW_MODE)
    if view_mode:
        vm.update(view_mode)
    return {
        "version": CONFIG_SCHEMA_VERSION,
        "view_mode": {
            "columns":      vm.get("columns", DEFAULT_VIEW_MODE["columns"]),
            "panel_height": vm.get("panel_height", DEFAULT_VIEW_MODE["panel_height"]),
        },
        "recordings": [e.to_dict() for e in entities],
    }


def load_settings_dict(data: dict) -> Tuple[List[RecordingEntity], dict, List[str]]:
    """Parse a settings dict into `(entities, view_mode, warnings)`.

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

    return entities, view_mode, warnings
