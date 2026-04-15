"""Chirp settings schema (serialize / deserialize).

Extracted from the monolith in the Phase 1 refactor (plan: c07).
Wraps `RecordingEntity.to_dict` / `from_dict` plus the view-mode
metadata block so the whole config round-trip is callable from pure
code (no Qt dialogs, no `QMessageBox`).

Schema shape (version 1):

    {
        "version": 1,
        "view_mode": {
            "columns": int,
            "panel_height": int,
        },
        "recordings": [ RecordingEntity.to_dict(), ... ],
    }

c17 (#22) will add real versioning + migration dispatch and warn on
unknown keys. For now this file intentionally mirrors the monolith's
behavior verbatim so the round-trip test pins the current shape.
"""

from typing import Iterable, Tuple, List

from chirp.recording.entity import RecordingEntity


CONFIG_SCHEMA_VERSION = 1
DEFAULT_VIEW_MODE = {"columns": 1, "panel_height": 300}


def build_settings_dict(entities: Iterable[RecordingEntity],
                        view_mode: dict | None = None) -> dict:
    """Serialize a collection of entities + view-mode to a plain dict.

    Matches the shape that `ChirpWindow._build_settings_data` was
    producing in the monolith (minus the `_flush_params_to_entity`
    call, which is a Qt widget concern and stays in the UI layer).
    """
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

    Raises ValueError if `data` is not a dict or lacks a `recordings`
    array — the monolith handled this with a QMessageBox in the UI
    layer; here we surface it as an exception so callers can decide
    how to present it.
    """
    if not isinstance(data, dict) or "recordings" not in data:
        raise ValueError("Invalid settings file format: missing 'recordings' array")

    vm_raw = data.get("view_mode") or {}
    view_mode = {
        "columns":      vm_raw.get("columns", DEFAULT_VIEW_MODE["columns"]),
        "panel_height": vm_raw.get("panel_height", DEFAULT_VIEW_MODE["panel_height"]),
    }

    entities: list[RecordingEntity] = []
    warnings: list[str] = []
    for rec_d in data["recordings"]:
        ent, warn = RecordingEntity.from_dict(rec_d)
        entities.append(ent)
        if warn:
            warnings.append(warn)

    return entities, view_mode, warnings
