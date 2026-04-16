"""Tests for `chirp.audio.devices` — robust device name matching (#21 / c20).

Uses fake candidate lists injected via the `candidates` parameter so
no real audio hardware is required.
"""

from chirp.audio.devices import find_device_by_name, _strip_bracket_suffix


# ── Helpers ──────────────────────────────────────────────────────────────────

def _dev(name, dev_id=0, channels=2, hostapi=0):
    """Build a minimal (dev_id, info) pair for the test candidate list."""
    return (dev_id, {
        'name': name,
        'max_input_channels': channels,
        'hostapi': hostapi,
    })


# ── strip_bracket_suffix ─────────────────────────────────────────────────────

def test_strip_bracket_suffix():
    assert _strip_bracket_suffix("Mic Array [WASAPI]") == "Mic Array"
    assert _strip_bracket_suffix("Plain Name") == "Plain Name"
    assert _strip_bracket_suffix("[All Bracket]") == ""


# ── Empty / missing ──────────────────────────────────────────────────────────

def test_empty_name_returns_none():
    dev_id, warn = find_device_by_name("", candidates=[_dev("Mic")])
    assert dev_id is None
    assert warn is None


def test_no_candidates_returns_warning():
    dev_id, warn = find_device_by_name("Mic", candidates=[])
    assert dev_id is None
    assert "no input devices" in warn


# ── Tier 1: exact name + host API ────────────────────────────────────────────

def test_exact_match_with_hostapi():
    cands = [_dev("Mic A", dev_id=0, hostapi=0),
             _dev("Mic A", dev_id=1, hostapi=1)]
    # Without hint → picks first exact (tier 2, breaks tie by channels)
    dev_id, warn = find_device_by_name("Mic A", candidates=cands)
    assert dev_id in (0, 1)
    assert warn is None


# ── Tier 2: exact name, any host API ─────────────────────────────────────────

def test_exact_name_different_hostapi_warns():
    cands = [_dev("Mic A", dev_id=0, hostapi=0)]
    # Hint is "WASAPI" but device is on hostapi 0 (not WASAPI)
    dev_id, warn = find_device_by_name("Mic A", hostapi_hint="WASAPI",
                                       candidates=cands)
    assert dev_id == 0
    assert "different host API" in warn


def test_exact_match_tiebreaks_on_channels():
    cands = [_dev("Mic A", dev_id=0, channels=1),
             _dev("Mic A", dev_id=1, channels=4)]
    dev_id, _ = find_device_by_name("Mic A", candidates=cands)
    assert dev_id == 1  # more channels wins


# ── Tier 3: prefix match ─────────────────────────────────────────────────────

def test_prefix_match_saved_longer():
    """Saved name is longer than the truncated live name."""
    cands = [_dev("Realtek High Definition Au", dev_id=5)]
    dev_id, warn = find_device_by_name(
        "Realtek High Definition Audio", candidates=cands)
    assert dev_id == 5
    assert "prefix match" in warn


def test_prefix_match_saved_shorter():
    """Saved name is a prefix of the live name."""
    cands = [_dev("Realtek High Definition Audio (2- High Def)", dev_id=3)]
    dev_id, warn = find_device_by_name(
        "Realtek High Definition Audio", candidates=cands)
    assert dev_id == 3
    assert "prefix match" in warn


# ── Tier 4: substring match ──────────────────────────────────────────────────

def test_substring_match_bracket_stripped():
    """Legacy configs saved 'Mic Array [WASAPI]' but live name is
    'Mic Array (Realtek)'."""
    cands = [_dev("Mic Array (Realtek)", dev_id=7)]
    dev_id, warn = find_device_by_name(
        "Mic Array [WASAPI]", candidates=cands)
    assert dev_id == 7
    assert "substring match" in warn


# ── No match ─────────────────────────────────────────────────────────────────

def test_no_match_returns_none_with_warning():
    cands = [_dev("Completely Different", dev_id=0)]
    dev_id, warn = find_device_by_name("My Mic", candidates=cands)
    assert dev_id is None
    assert "not found" in warn
