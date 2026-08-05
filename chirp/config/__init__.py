"""Chirp — config subpackage.

Owns the serialization schema for Chirp settings files (`.json` and the
legacy `.chirp` format). Kept separate from `chirp.ui` so the schema
can be tested without booting Qt.

c17 (#22) will add versioning + migration dispatch here.
"""

from chirp.config.schema import (
    build_settings_dict,
    load_settings_dict,
    parse_audio_settings,
    CONFIG_SCHEMA_VERSION,
    DEFAULT_VIEW_MODE,
    DEFAULT_MONITOR,
    DEFAULT_AUDIO,
)

__all__ = [
    "build_settings_dict",
    "load_settings_dict",
    "parse_audio_settings",
    "CONFIG_SCHEMA_VERSION",
    "DEFAULT_VIEW_MODE",
    "DEFAULT_MONITOR",
    "DEFAULT_AUDIO",
]
