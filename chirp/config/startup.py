"""Startup configuration preference (persisted via QSettings).

Controls what Chirp loads when it launches:

  - ``empty`` : start with a single fresh recording (historical default)
  - ``last``  : reload the config file that was last saved/loaded
  - ``file``  : always load a specific, user-chosen config file

The choice + associated paths live in ``QSettings`` (the registry on
Windows, a plist/ini elsewhere), independent of any one ``.json`` config
file — so the preference survives even when no config has ever been
saved, and does not travel with a shared config file.

Every accessor is best-effort: QSettings access is wrapped so a headless
context without a ``QApplication`` (the test suite) can call these
without crashing. On any failure the getters return their safe default
and the setters no-op.
"""

from __future__ import annotations

MODE_EMPTY = 'empty'
MODE_LAST = 'last'
MODE_FILE = 'file'
_VALID_MODES = (MODE_EMPTY, MODE_LAST, MODE_FILE)

_ORG = 'Chirp'
_APP = 'Chirp'

_KEY_MODE = 'startup/mode'
_KEY_FILE = 'startup/file'
_KEY_LAST = 'startup/last'


def _settings():
    """Construct a QSettings scoped to Chirp. Explicit org/app so it
    resolves the same store regardless of whether the QApplication set
    them."""
    from PyQt5.QtCore import QSettings
    return QSettings(_ORG, _APP)


def _get(key: str, default: str = '') -> str:
    try:
        val = _settings().value(key, default)
        return val if isinstance(val, str) else default
    except Exception:
        return default


def _set(key: str, value: str) -> None:
    try:
        _settings().setValue(key, value or '')
    except Exception:
        pass


def get_startup_mode() -> str:
    mode = _get(_KEY_MODE, MODE_EMPTY)
    return mode if mode in _VALID_MODES else MODE_EMPTY


def set_startup_mode(mode: str) -> None:
    _set(_KEY_MODE, mode if mode in _VALID_MODES else MODE_EMPTY)


def get_startup_file() -> str:
    """The explicit config path used in ``file`` mode."""
    return _get(_KEY_FILE, '')


def set_startup_file(path: str) -> None:
    _set(_KEY_FILE, path)


def get_last_config() -> str:
    """The config path last saved or loaded (used in ``last`` mode)."""
    return _get(_KEY_LAST, '')


def set_last_config(path: str) -> None:
    _set(_KEY_LAST, path)


def resolve_startup_path() -> str | None:
    """Return the config path Chirp should load at launch, or ``None``
    to start with an empty config. Never raises; returns ``None`` when
    the chosen source is unset."""
    mode = get_startup_mode()
    if mode == MODE_FILE:
        return get_startup_file() or None
    if mode == MODE_LAST:
        return get_last_config() or None
    return None
