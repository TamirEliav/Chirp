"""Round-trip tests for `chirp.config.schema`.

Pins the shape and semantics of the settings file before c17 (#22)
adds versioning + migration. If this test breaks, the on-disk config
format has probably drifted and users' saved `.json`/`.chirp` files
will fail to load cleanly on upgrade.
"""

import json

import numpy as np

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


def test_use_opengl_roundtrip_and_default():
    from chirp.config.schema import load_settings_dict
    # Explicit False survives a round-trip.
    data = build_settings_dict([], view_mode={"columns": 1, "panel_height": 300,
                                              "use_opengl": False})
    assert data["view_mode"]["use_opengl"] is False
    _, vm, _, _ = load_settings_dict(json.loads(json.dumps(data)))
    assert vm["use_opengl"] is False
    # Legacy file without the key loads with the default (True).
    _, vm2, _, _ = load_settings_dict({"version": 1, "recordings": [],
                                    "view_mode": {"columns": 2, "panel_height": 200}})
    assert vm2["use_opengl"] is True


def test_empty_config_roundtrip():
    data = build_settings_dict([], view_mode={"columns": 3, "panel_height": 250})
    assert data["version"] == CONFIG_SCHEMA_VERSION
    # use_opengl (Phase 4) / active_only (v3.3.0) default True when the
    # caller doesn't set them.
    assert data["view_mode"] == {"columns": 3, "panel_height": 250,
                                 "use_opengl": True, "active_only": True,
                                 "fill_order": "column"}
    assert data["recordings"] == []

    # JSON round-trip should be lossless
    encoded = json.dumps(data)
    decoded = json.loads(encoded)

    entities, vm, _, warnings = load_settings_dict(decoded)
    assert entities == []
    assert vm == {"columns": 3, "panel_height": 250, "use_opengl": True,
                  "active_only": True, "fill_order": "column"}
    assert warnings == []


def test_view_mode_defaults_applied_on_missing_key():
    data = build_settings_dict([], view_mode=None)
    assert data["view_mode"] == DEFAULT_VIEW_MODE


def test_view_mode_missing_block_on_load():
    """A file missing the view_mode block should get defaults."""
    raw = {"version": 1, "recordings": []}
    entities, vm, _, warnings = load_settings_dict(raw)
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
    e.amp_scale = "linear"

    data = build_settings_dict([e])
    encoded = json.dumps(data)
    decoded = json.loads(encoded)

    entities, vm, _, warnings = load_settings_dict(decoded)
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
    assert r.amp_scale == "linear"

    # View mode should also survive
    assert vm == DEFAULT_VIEW_MODE
    # No warnings when device_name is empty
    assert warnings == []


# ── #22 / c17: versioning + unknown-key warnings ───────────────────────────

def test_unknown_top_level_key_warns():
    raw = {"version": 1, "recordings": [], "view_mode": {},
           "totally_made_up_key": 42}
    entities, vm, _, warnings = load_settings_dict(raw)
    assert entities == []
    assert any("totally_made_up_key" in w for w in warnings)


def test_unknown_view_mode_key_warns():
    raw = {"version": 1, "recordings": [],
           "view_mode": {"columns": 2, "panel_height": 100, "weird": "?"}}
    _, vm, _, warnings = load_settings_dict(raw)
    assert vm["columns"] == 2
    assert any("weird" in w for w in warnings)


def test_unknown_recording_key_warns():
    raw = {
        "version": 1,
        "recordings": [{"name": "X", "future_field": 9001}],
    }
    entities, _, _, warnings = load_settings_dict(raw)
    assert len(entities) == 1
    assert any("future_field" in w for w in warnings)


def test_missing_version_treated_as_v1_with_warning():
    raw = {"recordings": []}
    _, _, _, warnings = load_settings_dict(raw)
    assert any("version" in w for w in warnings)


def test_future_version_raises():
    import pytest
    with pytest.raises(ValueError, match="newer Chirp"):
        load_settings_dict({"version": 9999, "recordings": []})


def test_multiple_entities_preserve_order():
    e1 = _fresh_entity(name="First")
    e2 = _fresh_entity(name="Second")
    e3 = _fresh_entity(name="Third")

    data = build_settings_dict([e1, e2, e3])
    decoded = json.loads(json.dumps(data))
    entities, _, _, _ = load_settings_dict(decoded)

    assert [r.name for r in entities] == ["First", "Second", "Third"]


def test_stream_enabled_roundtrip_and_default():
    """Per-stream enable switch: defaults True (older files omit the
    key), serializes, and survives a to_dict/from_dict round-trip."""
    e = RecordingEntity(name='se', device_id=None)
    try:
        assert e.stream_enabled is True
        e.stream_enabled = False
        d = e.to_dict()
        assert d['stream_enabled'] is False
        e2, _warn = RecordingEntity.from_dict(d)
        try:
            assert e2.stream_enabled is False
        finally:
            e2.close()
        # Legacy dict without the key → default True.
        d.pop('stream_enabled')
        e3, _warn = RecordingEntity.from_dict(d)
        try:
            assert e3.stream_enabled is True
        finally:
            e3.close()
    finally:
        e.close()


def test_params_locked_roundtrip_and_default():
    """Per-stream parameter lock: defaults False (older files omit the
    key), serializes, survives a to_dict/from_dict round-trip, and is a
    recognized schema key (no unknown-key warning)."""
    e = RecordingEntity(name='pl', device_id=None)
    try:
        assert e.params_locked is False
        e.params_locked = True
        d = e.to_dict()
        assert d['params_locked'] is True
        e2, _warn = RecordingEntity.from_dict(d)
        try:
            assert e2.params_locked is True
        finally:
            e2.close()
        # Legacy dict without the key → default False.
        d.pop('params_locked')
        e3, _warn = RecordingEntity.from_dict(d)
        try:
            assert e3.params_locked is False
        finally:
            e3.close()
    finally:
        e.close()

    # A locked stream survives a full build → json → load trip with no
    # unknown-key warning (the key is registered in the schema).
    e = RecordingEntity(name='pl2', device_id=None)
    try:
        e.params_locked = True
        data = build_settings_dict([e])
        entities, _vm, _mon, warnings = load_settings_dict(
            json.loads(json.dumps(data)))
        assert entities[0].params_locked is True
        assert not any('params_locked' in w for w in warnings)
        for r in entities:
            r.close()
    finally:
        e.close()


def test_color_roundtrip_and_default():
    """Per-stream recognition color: defaults empty, serializes, and
    survives a to_dict/from_dict round-trip."""
    e = RecordingEntity(name='c', device_id=None)
    try:
        assert e.color == ''
        e.color = '#ff8800'
        d = e.to_dict()
        assert d['color'] == '#ff8800'
        e2, _warn = RecordingEntity.from_dict(d)
        try:
            assert e2.color == '#ff8800'
        finally:
            e2.close()
        # Legacy dict without the key → default empty (window assigns).
        d.pop('color')
        e3, _warn = RecordingEntity.from_dict(d)
        try:
            assert e3.color == ''
        finally:
            e3.close()
    finally:
        e.close()


def test_fill_order_roundtrip_and_default():
    """view_mode.fill_order defaults to 'column', normalizes junk, and
    round-trips a 'row' choice."""
    # Default when the caller omits it.
    data = build_settings_dict([])
    assert data["view_mode"]["fill_order"] == "column"
    # Explicit 'row' survives.
    data = build_settings_dict([], view_mode={"fill_order": "row"})
    _, vm, _, _ = load_settings_dict(json.loads(json.dumps(data)))
    assert vm["fill_order"] == "row"
    # Junk / legacy-missing → 'column'.
    _, vm2, _, _ = load_settings_dict(
        {"version": 1, "recordings": [], "view_mode": {"fill_order": "bogus"}})
    assert vm2["fill_order"] == "column"
    _, vm3, _, _ = load_settings_dict({"version": 1, "recordings": []})
    assert vm3["fill_order"] == "column"


def test_monitor_roundtrip_and_default():
    """The monitor section serializes and round-trips; older files
    without it get the defaults."""
    mon = {
        "output_device_name": "Speakers (Realtek)",
        "output_device_hostapi": "Windows WASAPI",
        "gain_percent": 150,
        "muted": True,
        "follow": True,
        "source_index": 2,
    }
    data = build_settings_dict([], monitor=mon)
    assert data["monitor"] == mon
    _, _, got, _ = load_settings_dict(json.loads(json.dumps(data)))
    assert got == mon
    # Legacy file with no monitor block → defaults.
    _, _, default_mon, warnings = load_settings_dict(
        {"version": 1, "recordings": []})
    assert default_mon["gain_percent"] == 100
    assert default_mon["muted"] is False
    assert default_mon["source_index"] == -1
    assert not any("monitor" in w for w in warnings)


def test_unknown_monitor_key_warns():
    raw = {"version": 1, "recordings": [],
           "monitor": {"gain_percent": 80, "bogus_key": 1}}
    _, _, mon, warnings = load_settings_dict(raw)
    assert mon["gain_percent"] == 80
    assert any("bogus_key" in w for w in warnings)


# ── audio section: capture-engine tuning (v3.8.1) ────────────────────────
#
# Tunable in the config file because the fault it exists for (a late
# Python callback letting the driver zero-fill) is machine- and
# hardware-specific — a field user must be able to raise the buffer
# without editing code.

def test_audio_section_round_trip():
    from chirp.config.schema import build_settings_dict, load_settings_dict
    e = _fresh_entity('aud')
    data = build_settings_dict([e], audio={'capture_blocksize': 16384,
                                           'capture_latency': 0.25})
    decoded = json.loads(json.dumps(data))
    assert decoded['audio']['capture_blocksize'] == 16384
    assert decoded['audio']['capture_latency'] == 0.25
    # The audio section must not upset the rest of the loader.
    _ents, _vm, _mon, warnings = load_settings_dict(decoded)
    assert not [w for w in warnings if 'audio' in w.lower()]


def test_capture_exclusive_round_trips_and_defaults_off():
    """Exclusive mode must survive a save/load — it is set once on a
    field machine and has to still be in force after the next launch.
    It defaults OFF because it locks other applications out of the
    input endpoint."""
    from chirp.config.schema import (DEFAULT_AUDIO, build_settings_dict,
                                     parse_audio_settings)
    assert DEFAULT_AUDIO['capture_exclusive'] is False
    e = _fresh_entity('excl')
    data = build_settings_dict([e], audio={'capture_exclusive': True})
    decoded = json.loads(json.dumps(data))
    assert decoded['audio']['capture_exclusive'] is True
    cfg, warnings = parse_audio_settings(decoded)
    assert cfg['capture_exclusive'] is True
    assert not [w for w in warnings if 'audio' in w.lower()]
    # A hand-edited config must not feed a string into the open path.
    cfg, _ = parse_audio_settings({'audio': {'capture_exclusive': 'yes'}})
    assert cfg['capture_exclusive'] is True
    cfg, _ = parse_audio_settings({'audio': {'capture_exclusive': 0}})
    assert cfg['capture_exclusive'] is False


def test_capture_stall_recovery_round_trips_and_defaults_on():
    """The RDP auto-reconnect switch. It defaults ON (the historical
    behaviour) but has to survive a save/load — a rig where the
    reconnect does more harm than good is configured once and left."""
    from chirp.config.schema import (DEFAULT_AUDIO, build_settings_dict,
                                     parse_audio_settings)
    assert DEFAULT_AUDIO['auto_recover_capture_stall'] is True
    e = _fresh_entity('stall')
    data = build_settings_dict(
        [e], audio={'auto_recover_capture_stall': False})
    decoded = json.loads(json.dumps(data))
    assert decoded['audio']['auto_recover_capture_stall'] is False
    cfg, warnings = parse_audio_settings(decoded)
    assert cfg['auto_recover_capture_stall'] is False
    assert not [w for w in warnings if 'audio' in w.lower()]
    # Absent from an older config → the historical behaviour.
    cfg, _ = parse_audio_settings({'audio': {'capture_blocksize': 4096}})
    assert cfg['auto_recover_capture_stall'] is True


def test_envelope_settings_round_trip():
    from chirp.config.schema import (DEFAULT_AUDIO, build_settings_dict,
                                     parse_audio_settings)
    # Default must stay the historical estimator — switching it silently
    # would change every existing user's detection behaviour on upgrade.
    assert DEFAULT_AUDIO['envelope_method'] == 'hilbert'
    e = _fresh_entity('env')
    data = build_settings_dict([e], audio={'envelope_method': 'rectify',
                                           'envelope_cutoff_hz': 80.0})
    decoded = json.loads(json.dumps(data))
    cfg, warnings = parse_audio_settings(decoded)
    assert cfg['envelope_method'] == 'rectify'
    assert cfg['envelope_cutoff_hz'] == 80.0
    assert not [w for w in warnings if 'audio' in w.lower()]


def test_envelope_method_typo_warns_and_falls_back():
    from chirp.config.schema import parse_audio_settings
    cfg, warnings = parse_audio_settings(
        {'audio': {'envelope_method': 'rectifie'}})
    assert cfg['envelope_method'] == 'hilbert'
    assert any('envelope_method' in w for w in warnings)


def test_envelope_cutoff_is_clamped_and_junk_tolerant():
    from chirp.config.schema import parse_audio_settings
    from chirp.dsp.envelope import (DEFAULT_ENVELOPE_CUTOFF_HZ,
                                    ENVELOPE_CUTOFF_MAX_HZ,
                                    ENVELOPE_CUTOFF_MIN_HZ)
    cfg, _ = parse_audio_settings({'audio': {'envelope_cutoff_hz': 1e9}})
    assert cfg['envelope_cutoff_hz'] == ENVELOPE_CUTOFF_MAX_HZ
    cfg, _ = parse_audio_settings({'audio': {'envelope_cutoff_hz': -5}})
    assert cfg['envelope_cutoff_hz'] == ENVELOPE_CUTOFF_MIN_HZ
    cfg, warnings = parse_audio_settings(
        {'audio': {'envelope_cutoff_hz': 'fifty'}})
    assert cfg['envelope_cutoff_hz'] == DEFAULT_ENVELOPE_CUTOFF_HZ
    assert warnings


def test_audio_defaults_when_section_absent():
    from chirp.config.schema import DEFAULT_AUDIO, parse_audio_settings
    cfg, warnings = parse_audio_settings({'recordings': []})
    assert cfg == DEFAULT_AUDIO
    assert warnings == []


def test_audio_unknown_key_warns():
    from chirp.config.schema import parse_audio_settings
    cfg, warnings = parse_audio_settings(
        {'audio': {'capture_blocksize': 8192, 'bogus': 1}})
    assert cfg['capture_blocksize'] == 8192
    assert any('bogus' in w for w in warnings)


def test_audio_section_not_a_dict_warns():
    from chirp.config.schema import DEFAULT_AUDIO, parse_audio_settings
    cfg, warnings = parse_audio_settings({'audio': 'nope'})
    assert cfg == DEFAULT_AUDIO
    assert warnings


def test_blocksize_is_clamped_to_supported_range():
    """Absurd values must not reach PortAudio: too small defeats the
    purpose, too large starves the timestamp clock's observations and
    trips the capture-stall watchdog."""
    import chirp.audio.shared_stream as shared
    from chirp.constants import (CAPTURE_BLOCKSIZE_MAX, CAPTURE_BLOCKSIZE_MIN,
                                 CAPTURE_BLOCKSIZE)
    before = shared.current_params()
    try:
        bs, _, _ = shared.configure(blocksize=10 ** 9)
        assert bs == CAPTURE_BLOCKSIZE_MAX
        bs, _, _ = shared.configure(blocksize=1)
        assert bs == CAPTURE_BLOCKSIZE_MIN
        bs, _, _ = shared.configure(blocksize='not-a-number')
        assert bs == CAPTURE_BLOCKSIZE
        bs, lat, _ = shared.configure(blocksize=8192, latency='high')
        assert (bs, lat) == (8192, 'high')
        _bs, lat, _ = shared.configure(latency=0.3)
        assert lat == 0.3
        _bs, lat, _ = shared.configure(latency='garbage')
        assert lat in ('low', 'high') or isinstance(lat, float)
    finally:
        shared.configure(*before)


def test_configured_blocksize_reaches_the_stream(monkeypatch):
    import chirp.audio.shared_stream as shared
    from chirp.audio.capture import AudioCapture
    from chirp.audio.ringbuffer import AudioRing

    opened = {}

    class _FakeStream:
        def __init__(self, **kw):
            opened.update(kw)

        def start(self): pass
        def stop(self): pass
        def close(self): pass

    before = shared.current_params()
    shared.reset_registry()
    monkeypatch.setattr(shared, '_stream_factory', _FakeStream)
    monkeypatch.setattr(shared, '_device_input_channels', lambda d: 2)
    monkeypatch.setattr(shared, '_warn_samplerate_mismatch', lambda *a, **k: None)
    try:
        shared.configure(blocksize=16384, latency=0.2)
        cap = AudioCapture(AudioRing(44100 * 10, channels=1), device=3,
                           channels=1, samplerate=44100)
        assert cap.valid
        assert opened['blocksize'] == 16384
        assert opened['latency'] == 0.2
    finally:
        shared.reset_registry()
        shared.configure(*before)


# ── Loading a config must open each device exactly once ──────────────────
#
# RecordingEntity.__init__ used to open the capture with channels=1 and
# from_dict then called change_device(..., 2) for every non-Mono stream,
# so each endpoint was opened, closed and reopened milliseconds apart on
# every config load. Field logs showed the pair of opens per stream. The
# fault under investigation is a latched per-endpoint capture session, so
# needless create/destroy churn of exactly that session is worth removing.

def _count_captures(monkeypatch):
    """Count RecordingEntity._make_capture calls."""
    calls = []
    orig = RecordingEntity._make_capture

    def counting(self, channels):
        calls.append(channels)
        return orig(self, channels)

    monkeypatch.setattr(RecordingEntity, '_make_capture', counting)
    return calls


def test_from_dict_opens_capture_once_for_mono(monkeypatch):
    calls = _count_captures(monkeypatch)
    e, _ = RecordingEntity.from_dict({'name': 'm', 'channel_mode': 'Mono'})
    assert calls == [1]
    e.close()


def test_from_dict_opens_capture_once_for_stereo_modes(monkeypatch):
    for mode in ('Left', 'Right', 'Stereo'):
        calls = _count_captures(monkeypatch)
        e, _ = RecordingEntity.from_dict({'name': mode, 'channel_mode': mode})
        assert calls == [2], f'{mode}: expected one 2-channel open, got {calls}'
        assert e.channel_mode == mode
        e.close()


def test_stereo_mode_downgraded_when_device_is_mono(monkeypatch):
    """The downgrade has to happen before the capture opens, or we are
    back to opening the device twice."""
    import chirp.recording.entity as ent

    monkeypatch.setattr(ent, 'find_device_by_name',
                        lambda name, hostapi: (7, None))
    monkeypatch.setattr(ent.sd, 'query_devices',
                        lambda dev: {'max_input_channels': 1})
    calls = _count_captures(monkeypatch)
    e, _ = RecordingEntity.from_dict(
        {'name': 'mono-dev', 'device_name': 'X', 'channel_mode': 'Stereo'})
    assert e.channel_mode == 'Mono'
    assert calls == [1]
    e.close()


def test_wav_file_source_survives_a_stereo_channel_mode(tmp_path, monkeypatch):
    """Regression: from_dict's change_device() call reset input_source to
    'device', so a non-Mono config whose input was a WAV file silently
    lost its file and came back as a live-device stream."""
    import scipy.io.wavfile

    wav = tmp_path / 'src.wav'
    scipy.io.wavfile.write(str(wav), 44100,
                           np.zeros((2048, 2), dtype=np.int16))
    e, _warn = RecordingEntity.from_dict({
        'name': 'wavstereo',
        'channel_mode': 'Stereo',
        'input_source': 'wav_file',
        'wav_file_path': str(wav),
    })
    try:
        assert e.input_source == 'wav_file'
        assert e.wav_file_path == str(wav)
    finally:
        e.close()
