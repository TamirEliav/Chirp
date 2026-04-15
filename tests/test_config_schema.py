"""Round-trip tests for `chirp.config.schema`.

Pins the shape and semantics of the settings file before c17 (#22)
adds versioning + migration. If this test breaks, the on-disk config
format has probably drifted and users' saved `.json`/`.chirp` files
will fail to load cleanly on upgrade.
"""

import json

import pytest

from chirp.config.schema import (
    CONFIG_SCHEMA_VERSION,
    DEFAULT_VIEW_MODE,
    build_settings_dict,
    load_settings_dict,
)
from chirp.recording.entity import RecordingEntity


def _fresh_entity(name="RoundTrip"):
    """Construct a RecordingEntity with device_id=None.

    AudioCapture opens a default-device InputStream in __init__ but
    swallows failures and leaves `_stream = None`, so the entity is
    safe to construct in headless test environments.
    """
    return RecordingEntity(name=name, device_id=None)


def test_empty_config_roundtrip():
    data = build_settings_dict([], view_mode={"columns": 3, "panel_height": 250})
    assert data["version"] == CONFIG_SCHEMA_VERSION
    assert data["view_mode"] == {"columns": 3, "panel_height": 250}
    assert data["recordings"] == []

    # JSON round-trip should be lossless
    encoded = json.dumps(data)
    decoded = json.loads(encoded)

    entities, vm, warnings = load_settings_dict(decoded)
    assert entities == []
    assert vm == {"columns": 3, "panel_height": 250}
    assert warnings == []


def test_view_mode_defaults_applied_on_missing_key():
    data = build_settings_dict([], view_mode=None)
    assert data["view_mode"] == DEFAULT_VIEW_MODE


def test_view_mode_missing_block_on_load():
    """A file missing the view_mode block should get defaults."""
    raw = {"version": 1, "recordings": []}
    entities, vm, warnings = load_settings_dict(raw)
    assert entities == []
    assert vm == DEFAULT_VIEW_MODE


def test_invalid_shape_raises():
    with pytest.raises(ValueError):
        load_settings_dict("not a dict")
    with pytest.raises(ValueError):
        load_settings_dict({"version": 1})  # missing 'recordings'


def test_single_entity_roundtrip_preserves_scalar_params():
    """All scalar config fields should survive a build → json → load trip."""
    e = _fresh_entity(name="Chickadee")
    # Poke non-default values to verify they persist
    e.threshold = 0.234
    e.min_cross_sec = 0.05
    e.hold_sec = 0.75
    e.post_trig_sec = 0.25
    e.max_rec_sec = 15.0
    e.pre_trig_sec = 0.6
    e.freq_filter_enabled = True
    e.freq_lo = 2000.0
    e.freq_hi = 9000.0
    e.gain_db = 3.5
    e.spectral_threshold = 0.4
    e.spectral_trigger_mode = "Amp OR Spectral"
    e.filename_prefix = "test"
    e.filename_suffix = "_v1"
    e.dph_folder_prefix = "day_"
    e.display_mode = "Both"

    data = build_settings_dict([e])
    encoded = json.dumps(data)
    decoded = json.loads(encoded)

    entities, vm, warnings = load_settings_dict(decoded)
    assert len(entities) == 1
    r = entities[0]

    assert r.name == "Chickadee"
    assert r.threshold == pytest.approx(0.234)
    assert r.min_cross_sec == pytest.approx(0.05)
    assert r.hold_sec == pytest.approx(0.75)
    assert r.post_trig_sec == pytest.approx(0.25)
    assert r.max_rec_sec == pytest.approx(15.0)
    assert r.pre_trig_sec == pytest.approx(0.6)
    assert r.freq_filter_enabled is True
    assert r.freq_lo == pytest.approx(2000.0)
    assert r.freq_hi == pytest.approx(9000.0)
    assert r.gain_db == pytest.approx(3.5)
    assert r.spectral_threshold == pytest.approx(0.4)
    assert r.spectral_trigger_mode == "Amp OR Spectral"
    assert r.filename_prefix == "test"
    assert r.filename_suffix == "_v1"
    assert r.dph_folder_prefix == "day_"
    assert r.display_mode == "Both"

    # View mode should also survive
    assert vm == DEFAULT_VIEW_MODE
    # No warnings when device_name is empty
    assert warnings == []


def test_multiple_entities_preserve_order():
    e1 = _fresh_entity(name="First")
    e2 = _fresh_entity(name="Second")
    e3 = _fresh_entity(name="Third")

    data = build_settings_dict([e1, e2, e3])
    decoded = json.loads(json.dumps(data))
    entities, _, _ = load_settings_dict(decoded)

    assert [r.name for r in entities] == ["First", "Second", "Third"]
