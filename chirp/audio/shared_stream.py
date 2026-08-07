"""SharedInputStream — one ``sd.InputStream`` per (device, samplerate),
fanned out to every Chirp stream that uses that device.

Why
---
Windows opens a per-endpoint capture session when the first client
connects and tears it down when the last one disconnects. Chirp used to
open one ``InputStream`` per RecordingEntity, so two streams splitting
one stereo input (Left on one, Right on the other) were **two engine
clients on one endpoint**.

Field evidence tied exactly that configuration to a latched
driver/engine fault: the capture was periodically zero-filled for 2–8 ms
at a time, *identically and time-locked* on both streams (95.8% of zero
runs coincident within 1 ms at a fixed lag), while other endpoints of
the same interface stayed clean. No PortAudio status flag was raised.
Stopping one of the two streams did not clear it — only stopping both,
i.e. releasing the endpoint entirely, did.

Sharing one stream per endpoint makes Chirp exactly one client per
capture session, and makes the session lifecycle atomic across the
streams that use it: the session is created when the first stream starts
acquiring and destroyed when the last one stops.

Model
-----
* A :class:`SharedInputStream` owns the PortAudio stream and a
  copy-on-write tuple of *sinks*. A sink is an
  :class:`~chirp.audio.capture.AudioCapture` — it owns its entity's ring
  buffer, disciplined clock, monitor wiring and drop/error stats.
* The callback iterates the sink tuple and hands each **active** sink
  the column slice it asked for. Realtime-safe: an attribute read of an
  immutable tuple, no locks, no allocation beyond numpy views.
* Live-device streams are opened with 2 channels whenever the device has
  at least 2 inputs, regardless of what the first requester asked for.
  A mono sink simply takes column 0. This keeps the physical channel
  count a property of the *device* rather than of whichever stream
  happened to open it first, so a later stereo joiner never forces a
  disruptive reopen.
* Lifecycle is refcounted by attachment, not by acquisition:
  ``start()`` starts the PortAudio stream on the first active sink,
  ``stop_if_idle()`` stops it when no sink is active any more, and the
  stream is closed and evicted from the registry when its **last** sink
  detaches. That is what preserves both the L6 "release the device while
  idle" behavior and the zero-fill reset semantics.
"""

from __future__ import annotations

import threading

import sounddevice as sd

from chirp.constants import (CAPTURE_BLOCKSIZE, CAPTURE_BLOCKSIZE_MAX,
                             CAPTURE_BLOCKSIZE_MIN, CAPTURE_EXCLUSIVE,
                             CAPTURE_LATENCY, DTYPE)
from chirp.error_log import log as _err_log

# Test seam: monkeypatch to substitute a fake stream class so the shared
# layer can be exercised without an audio device. ``None`` → the real
# ``sd.InputStream``.
_stream_factory = None

# Capture parameters applied to streams opened from now on. Settable
# from the config file (``audio`` section) via :func:`configure` so the
# buffer can be tuned in the field without editing code — the field
# fault this exists for (a late Python callback letting the driver
# zero-fill) is hardware- and machine-specific.
_capture_blocksize: int = CAPTURE_BLOCKSIZE
_capture_latency = CAPTURE_LATENCY
_capture_exclusive: bool = CAPTURE_EXCLUSIVE


def configure(blocksize=None, latency=None,
              exclusive=None) -> tuple[int, object, bool]:
    """Set the capture blocksize / latency / exclusive-mode flag for
    streams opened after this call, clamping the blocksize into the
    supported range. Already-open streams keep their parameters until
    they are reopened (Stop/Start Acq, or a config load, which rebuilds
    every capture).

    Returns the effective ``(blocksize, latency, exclusive)``.
    """
    global _capture_blocksize, _capture_latency, _capture_exclusive
    if blocksize is not None:
        try:
            bs = int(blocksize)
        except (TypeError, ValueError):
            bs = CAPTURE_BLOCKSIZE
        _capture_blocksize = max(CAPTURE_BLOCKSIZE_MIN,
                                 min(CAPTURE_BLOCKSIZE_MAX, bs))
    if latency is not None:
        # 'low' / 'high' or an explicit float in seconds.
        if isinstance(latency, str) and latency.strip().lower() in ('low', 'high'):
            _capture_latency = latency.strip().lower()
        else:
            try:
                _capture_latency = max(0.0, float(latency))
            except (TypeError, ValueError):
                _capture_latency = CAPTURE_LATENCY
    if exclusive is not None:
        _capture_exclusive = bool(exclusive)
    return _capture_blocksize, _capture_latency, _capture_exclusive


def current_params() -> tuple[int, object, bool]:
    """The capture parameters the next stream will open with."""
    return _capture_blocksize, _capture_latency, _capture_exclusive

# (device, samplerate) → SharedInputStream. Guarded by ``_registry_lock``
# for mutation; the audio callback never touches it.
_registry: dict[tuple, "SharedInputStream"] = {}
_registry_lock = threading.RLock()


def _device_input_channels(device) -> int:
    """Max input channels of ``device`` (0 when unknown). ``None`` means
    PortAudio's default input device."""
    try:
        dev = device
        if dev is None:
            dev = sd.default.device[0]
        info = sd.query_devices(dev)
        return int(info.get('max_input_channels', 0) or 0)
    except Exception:
        return 0


def _device_label(device) -> str:
    """``17 "Analogue 1 + 2 (Focusrite USB)" [Windows WASAPI]`` — what
    was actually opened, for the log.

    A bare device index is useless after the fact: indices shift when
    endpoints appear or disappear, one physical interface publishes
    several endpoints with near-identical names, and the same endpoint
    exists once per host API. Which API a stream ended up on is exactly
    what an experiment comparing shared / exclusive / WDM-KS needs to
    confirm from the log rather than from memory.
    """
    try:
        dev = device
        if dev is None:
            dev = sd.default.device[0]
        info = sd.query_devices(dev)
        label = f'{dev} "{info.get("name", "?")}"'
        try:
            api = sd.query_hostapis(int(info['hostapi']))
            label += f' [{api.get("name", "?")}]'
        except Exception:
            pass
        return label
    except Exception:
        return 'default input' if device is None else str(device)


def _exclusive_settings(device, name: str = ''):
    """PortAudio ``extra_settings`` requesting WASAPI exclusive mode, or
    ``None`` when this device cannot honour it.

    Exclusive mode is a WASAPI concept: the endpoint is handed to a
    single client and the Windows audio engine's mixing / buffering path
    — the layer observed inserting whole zero-filled periods into the
    capture — is out of the picture. Asking for it on an MME,
    DirectSound or WDM-KS entry would make the open fail, so the request
    is logged and dropped instead (WDM-KS already bypasses the engine on
    its own).
    """
    try:
        dev = device
        if dev is None:
            dev = sd.default.device[0]
        info = sd.query_devices(dev)
        api = sd.query_hostapis(int(info['hostapi']))
        api_name = str(api.get('name', ''))
    except Exception as exc:
        _err_log('open', name,
                 f'exclusive mode requested but the device could not be '
                 f'probed ({type(exc).__name__}: {exc}) — opening shared')
        return None
    if 'wasapi' not in api_name.lower():
        _err_log('open', name,
                 f'exclusive mode requested but device is on host API '
                 f'"{api_name}", which has no exclusive mode — opening '
                 f'normally (only WASAPI entries support it)')
        return None
    try:
        return sd.WasapiSettings(exclusive=True)
    except Exception as exc:
        _err_log('open', name,
                 f'exclusive mode unavailable in this sounddevice build '
                 f'({type(exc).__name__}: {exc}) — opening shared')
        return None


def _warn_samplerate_mismatch(device, samplerate: int, name: str) -> None:
    """Log once per physical open when the endpoint's default format
    differs from the stream rate — in WASAPI shared mode that inserts an
    OS resampler between the driver and our callback, which smears
    driver-level glitch packets and can swallow the discontinuity flags
    that would otherwise light the `!` badge."""
    try:
        dev = device
        if dev is None:
            dev = sd.default.device[0]
        info = sd.query_devices(dev)
        dev_sr = float(info.get('default_samplerate') or 0.0)
        if dev_sr > 0 and abs(dev_sr - float(samplerate)) > 0.5:
            msg = (f'stream rate {samplerate} Hz != device default '
                   f'{dev_sr:.0f} Hz — the OS is resampling this capture '
                   f'(WASAPI shared). Driver glitches may surface as '
                   f'smeared zero-sample runs. Match the endpoint default '
                   f'format to the stream rate (Sound Control Panel → '
                   f'Recording → Advanced).')
            print(f'[SharedInputStream] {name or dev}: {msg}')
            _err_log('open', name, msg)
    except Exception:
        pass


class SharedInputStream:
    """One PortAudio input stream shared by N sinks. Construct via
    :func:`acquire` — direct construction skips the registry."""

    def __init__(self, device, samplerate: int, channels: int,
                 name: str = ''):
        self.device      = device
        self.samplerate  = int(samplerate)
        self.channels    = int(channels)
        self.open_error: str | None = None
        self._stream     = None
        self._sinks: tuple = ()
        self._lock       = threading.RLock()
        self._started    = False
        self._dead       = False
        self._key        = (device, int(samplerate))
        factory = _stream_factory or sd.InputStream
        self.blocksize = int(_capture_blocksize)
        self.latency = _capture_latency
        # Resolved below: what the stream actually opened with, not what
        # was asked for. A device that cannot do exclusive mode still
        # opens — shared — so an overnight rig never dies on a setting.
        self.exclusive = False
        extra = (_exclusive_settings(device, name)
                 if _capture_exclusive else None)

        def _open(extra_settings):
            return factory(
                samplerate=self.samplerate, channels=self.channels,
                dtype=DTYPE, blocksize=self.blocksize, device=device,
                latency=self.latency, callback=self._callback,
                extra_settings=extra_settings,
            )

        try:
            self._stream = _open(extra)
            self.exclusive = extra is not None
        except Exception as exc:
            if extra is not None:
                # Exclusive open refused — the format isn't supported
                # natively or another application holds the endpoint.
                # Fall back rather than leave the stream down, but say so
                # loudly: an experiment that silently ran in shared mode
                # would look like exclusive mode failing to help.
                msg = (f'EXCLUSIVE mode open failed on device '
                       f'{_device_label(device)} '
                       f'({type(exc).__name__}: {exc}) — falling back to '
                       f'shared mode. Another app may hold this endpoint, '
                       f'or the hardware may not accept '
                       f'{self.channels} ch @ {self.samplerate} Hz natively.')
                print(f'[SharedInputStream] {msg}')
                _err_log('open', name, msg)
                try:
                    self._stream = _open(None)
                except Exception as exc2:
                    exc = exc2
                    self._stream = None
            if self._stream is None:
                label = _device_label(device)
                self.open_error = f'{type(exc).__name__}: {exc}'[:200]
                print(f'[SharedInputStream] Failed to open device {label}: '
                      f'{exc}')
                _err_log('open', name,
                         f'failed to open device {label}: '
                         f'{type(exc).__name__}: {exc}')
                return
        # Report what the driver actually granted: ``latency`` is only a
        # *suggestion*, and on WASAPI the device's own 'high' default can
        # be as little as 10 ms — far too thin a margin for a shared
        # desktop. Logging the achieved value makes that visible in the
        # field instead of having to be inferred.
        try:
            actual = float(getattr(self._stream, 'latency', 0.0) or 0.0)
        except (TypeError, ValueError):
            actual = 0.0
        if actual:
            _err_log('open', name,
                     f'capture stream open: device {_device_label(device)}, '
                     f'blocksize={self.blocksize} '
                     f'({self.blocksize / max(1, self.samplerate) * 1000:.0f} ms), '
                     f'requested latency={self.latency}, '
                     f'granted latency={actual * 1000:.1f} ms, '
                     f'{self.channels} ch @ {self.samplerate} Hz, '
                     f'mode={"EXCLUSIVE" if self.exclusive else "shared"}')
        if not self.exclusive:
            # Exclusive mode owns the endpoint outright, so the shared
            # engine's resampler isn't in the path and the mismatch
            # warning would be misleading.
            _warn_samplerate_mismatch(device, self.samplerate, name)

    # ── Introspection ────────────────────────────────────────────────

    @property
    def stream(self):
        return self._stream

    @property
    def sink_count(self) -> int:
        return len(self._sinks)

    @property
    def active_sink_count(self) -> int:
        return sum(1 for s in self._sinks if s._active)

    @property
    def started(self) -> bool:
        return self._started

    @property
    def dead(self) -> bool:
        return self._dead

    # ── Sink management ──────────────────────────────────────────────

    def attach(self, sink) -> bool:
        """Register a sink. Returns False when this stream is closed or
        has been marked dead (the caller must acquire a fresh one)."""
        with self._lock:
            if self._dead or self._stream is None:
                return False
            if sink not in self._sinks:
                # Copy-on-write: the callback reads the tuple without a
                # lock, so it must never observe a partially-built list.
                self._sinks = self._sinks + (sink,)
            return True

    def detach(self, sink) -> None:
        """Unregister a sink; close the PortAudio stream and drop this
        entry from the registry once the last sink is gone (releasing
        the endpoint, which is what resets a latched capture session)."""
        with self._lock:
            if sink in self._sinks:
                self._sinks = tuple(s for s in self._sinks if s is not sink)
            if self._sinks:
                self.stop_if_idle()
                return
        self._close_and_evict(force=False)

    # ── Transport ────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the PortAudio stream if it isn't running. Idempotent —
        every sink calls this on resume; only the first one acts."""
        with self._lock:
            if self._stream is None or self._started:
                return
            self._stream.start()
            self._started = True

    def stop_if_idle(self) -> None:
        """Stop the PortAudio stream once no sink is active. The stream
        stays *open* (the device is released by ``detach``)."""
        with self._lock:
            if self._stream is None or not self._started:
                return
            if self.active_sink_count:
                return
            try:
                self._stream.stop()
            except Exception:
                pass
            self._started = False

    def mark_dead(self) -> None:
        """Declare this stream unusable (capture-recovery path).

        Evicts the registry entry so the next acquisition opens a fresh
        stream, and closes the PortAudio stream immediately — when a
        device stalls, every sink on it is stalled, so there is no
        healthy sibling to protect. Still-attached sinks simply stop
        receiving data; their own ``close()`` becomes a no-op.
        """
        with self._lock:
            if self._dead:
                return
            self._dead = True
        self._close_and_evict(force=True)

    # ── Internals ────────────────────────────────────────────────────

    def _close_and_evict(self, *, force: bool) -> None:
        """Close the PortAudio stream and drop the registry entry.

        Lock order is registry → stream, matching :func:`acquire`, so an
        acquisition racing a teardown either completes before eviction
        (and is then seen by the ``self._sinks`` re-check below) or waits
        and finds no entry, opening a fresh stream. ``force`` is the
        ``mark_dead`` path: the stream goes even if sinks are attached.
        """
        with _registry_lock:
            with self._lock:
                if not force and self._sinks:
                    # A sink attached between the detach and this call —
                    # the endpoint is in use again, so leave it open.
                    return
                if _registry.get(self._key) is self:
                    del _registry[self._key]
                self._dead = True
                stream, self._stream = self._stream, None
                self._started = False
        if stream is None:
            return
        try:
            stream.stop()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass

    def _callback(self, indata, frames, time_info, status):
        # PortAudio thread. Reads an immutable tuple snapshot — no lock,
        # no allocation beyond the per-sink column view. A sink raising
        # must never take down capture for its siblings.
        nch = self.channels
        for sink in self._sinks:
            if not sink._active:
                continue
            try:
                sch = sink._channels
                sink._callback(indata if sch >= nch else indata[:, :sch],
                               frames, time_info, status)
            except Exception:
                pass


# ── Registry ─────────────────────────────────────────────────────────────


def acquire(sink, device, samplerate: int, channels: int,
            name: str = '') -> tuple["SharedInputStream | None", str | None]:
    """Attach ``sink`` to the shared stream for ``(device, samplerate)``,
    opening one if needed.

    Returns ``(shared, error)``. On failure ``shared`` is None and
    ``error`` is a short message for ``AudioCapture.open_error``.
    """
    key = (device, int(samplerate))
    with _registry_lock:
        shared = _registry.get(key)
        if shared is not None and (shared.dead or shared.stream is None):
            shared = None
        want = int(channels)
        if shared is None:
            dev_ch = _device_input_channels(device)
            if dev_ch and want > dev_ch:
                # Reject before opening: the device physically cannot
                # feed this stream. Same outcome as the per-stream open
                # failing on PaError, but with a message that says why,
                # and without leaving an unusable stream open.
                msg = (f'device provides {dev_ch} input channel(s); '
                       f'this stream needs {want}')
                _err_log('open', name, msg)
                return None, msg
            # Pin the physical channel count to the device, not to this
            # requester, so a later stereo joiner never needs a reopen.
            open_ch = max(want, 2) if dev_ch >= 2 else want
            shared = SharedInputStream(device, samplerate, open_ch, name=name)
            if shared.stream is None:
                return None, shared.open_error or 'could not open device'
            _registry[key] = shared
        elif want > shared.channels:
            # Existing stream on this endpoint is narrower than this
            # stream needs (only reachable when the device's channel
            # count was unknown at open time).
            msg = (f'device provides {shared.channels} input channel(s); '
                   f'this stream needs {want}')
            _err_log('open', name, msg)
            return None, msg
        if not shared.attach(sink):
            return None, 'shared capture stream was closed during open'
        return shared, None


def registry_size() -> int:
    """Number of live shared streams (diagnostics / tests)."""
    with _registry_lock:
        return len(_registry)


def reset_registry() -> None:
    """Close every shared stream and clear the registry (tests)."""
    with _registry_lock:
        streams = list(_registry.values())
    for s in streams:
        s.mark_dead()
    with _registry_lock:
        _registry.clear()
