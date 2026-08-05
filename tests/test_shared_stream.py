"""One PortAudio stream per (device, samplerate), fanned out to sinks.

Chirp used to open one ``sd.InputStream`` per RecordingEntity, so two
streams splitting one stereo input were two clients of the same Windows
capture session. Field evidence tied that configuration to a latched
driver/engine fault that zero-filled 2–8 ms of the capture at a time,
identically on both streams, clearing only when every client released
the endpoint (see chirp/audio/shared_stream.py). These tests pin the
sharing contract that makes Chirp a single client per endpoint:

* one underlying stream per (device, samplerate); distinct keys don't share;
* every active sink receives each buffer, sliced to the channels it asked for;
* a paused sink receives nothing while its siblings keep running;
* the PortAudio stream starts on the first active sink and stops only when
  the last one pauses;
* the stream is closed — releasing the OS capture session — when the last
  sink detaches, and a later acquisition opens a genuinely fresh stream;
* ``mark_stream_dead`` (capture recovery) retires the shared stream even
  while siblings are still attached;
* a sink that needs more channels than the device has fails to open, with
  the same ``valid`` / ``open_error`` contract as a failed device open.
"""

import numpy as np
import pytest

import chirp.audio.shared_stream as shared
from chirp.audio.capture import AudioCapture
from chirp.audio.ringbuffer import AudioRing
from chirp.constants import CHUNK_FRAMES


class FakeStream:
    """Stand-in for sd.InputStream that records transport calls and can
    push buffers into the registered callback on demand."""

    instances: list = []

    def __init__(self, **kw):
        self.kw = kw
        self.callback = kw.get('callback')
        self.channels = kw.get('channels')
        self.started = False
        self.closed = False
        self.start_calls = 0
        self.stop_calls = 0
        FakeStream.instances.append(self)

    def start(self):
        self.started = True
        self.start_calls += 1

    def stop(self):
        self.started = False
        self.stop_calls += 1

    def close(self):
        self.closed = True

    def push(self, indata, status=None):
        self.callback(indata, indata.shape[0], None, status)


@pytest.fixture(autouse=True)
def _fake_audio(monkeypatch):
    FakeStream.instances = []
    shared.reset_registry()
    monkeypatch.setattr(shared, '_stream_factory', FakeStream)
    # Deterministic device probing — no real audio hardware involved.
    monkeypatch.setattr(shared, '_device_input_channels', lambda dev: 2)
    monkeypatch.setattr(shared, '_warn_samplerate_mismatch',
                        lambda *a, **kw: None)
    yield
    shared.reset_registry()


def _cap(device=7, channels=1, samplerate=44100, ring_channels=None):
    ring = AudioRing(CHUNK_FRAMES * 8,
                     channels=ring_channels or channels)
    return AudioCapture(ring, device=device, channels=channels,
                        samplerate=samplerate, name='ut'), ring


def _buf(n=CHUNK_FRAMES, channels=2, value=1.0):
    return np.full((n, channels), value, dtype=np.float32)


# ── Sharing ──────────────────────────────────────────────────────────────

def test_same_device_and_rate_share_one_stream():
    a, _ = _cap(device=7)
    b, _ = _cap(device=7)
    assert a.valid and b.valid
    assert len(FakeStream.instances) == 1
    assert shared.registry_size() == 1
    assert a._shared is b._shared
    assert a._shared.sink_count == 2


def test_different_device_or_rate_do_not_share():
    a, _ = _cap(device=7, samplerate=44100)
    b, _ = _cap(device=8, samplerate=44100)
    c, _ = _cap(device=7, samplerate=48000)
    assert len({id(a._shared), id(b._shared), id(c._shared)}) == 3
    assert len(FakeStream.instances) == 3


def test_stream_opens_two_channels_for_mono_sink_on_stereo_device():
    """The physical channel count follows the DEVICE, so a later stereo
    joiner never forces a reopen of a stream a mono stream opened."""
    a, _ = _cap(device=7, channels=1)
    assert FakeStream.instances[0].channels == 2
    b, _ = _cap(device=7, channels=2)
    assert len(FakeStream.instances) == 1     # no reopen
    assert b.valid


# ── Fan-out ──────────────────────────────────────────────────────────────

def test_every_active_sink_receives_each_buffer():
    a, ring_a = _cap(device=7, channels=2, ring_channels=2)
    b, ring_b = _cap(device=7, channels=2, ring_channels=2)
    a.resume()
    b.resume()
    FakeStream.instances[0].push(_buf())
    assert ring_a.write_total == CHUNK_FRAMES
    assert ring_b.write_total == CHUNK_FRAMES


def test_mono_sink_gets_column_zero_only():
    a, ring_a = _cap(device=7, channels=1, ring_channels=1)
    a.resume()
    buf = np.zeros((CHUNK_FRAMES, 2), dtype=np.float32)
    buf[:, 0] = 0.25          # left
    buf[:, 1] = 0.75          # right
    FakeStream.instances[0].push(buf)
    _, frames = ring_a.read()
    assert np.allclose(frames, 0.25)


def test_paused_sink_receives_nothing_while_sibling_runs():
    a, ring_a = _cap(device=7, channels=2, ring_channels=2)
    b, ring_b = _cap(device=7, channels=2, ring_channels=2)
    a.resume()
    b.resume()
    stream = FakeStream.instances[0]
    stream.push(_buf())
    b.pause()
    stream.push(_buf())
    assert ring_a.write_total == 2 * CHUNK_FRAMES
    assert ring_b.write_total == CHUNK_FRAMES      # frozen while paused
    assert stream.started is True                  # sibling keeps it alive


def test_sink_exception_does_not_break_siblings():
    a, ring_a = _cap(device=7, channels=2, ring_channels=2)
    b, ring_b = _cap(device=7, channels=2, ring_channels=2)
    a.resume()
    b.resume()

    def _boom(*args, **kw):
        raise RuntimeError('sink blew up')

    a._callback = _boom
    FakeStream.instances[0].push(_buf())
    assert ring_b.write_total == CHUNK_FRAMES


# ── Transport lifecycle ──────────────────────────────────────────────────

def test_stream_starts_once_and_stops_only_when_all_paused():
    a, _ = _cap(device=7)
    b, _ = _cap(device=7)
    stream = FakeStream.instances[0]
    a.resume()
    assert stream.started and stream.start_calls == 1
    b.resume()
    assert stream.start_calls == 1        # already running
    a.pause()
    assert stream.started is True         # b still active
    b.pause()
    assert stream.started is False


def test_last_detach_closes_stream_and_next_acquire_opens_fresh():
    a, _ = _cap(device=7)
    b, _ = _cap(device=7)
    stream = FakeStream.instances[0]
    a.close()
    assert stream.closed is False         # b still holds the endpoint
    assert shared.registry_size() == 1
    b.close()
    assert stream.closed is True          # OS capture session released
    assert shared.registry_size() == 0
    assert a.valid is False and b.valid is False
    c, _ = _cap(device=7)
    assert len(FakeStream.instances) == 2
    assert c.valid is True


def test_close_is_idempotent():
    a, _ = _cap(device=7)
    a.close()
    a.close()
    assert a.valid is False


# ── Recovery ─────────────────────────────────────────────────────────────

def test_mark_stream_dead_retires_stream_even_with_siblings():
    a, _ = _cap(device=7)
    b, _ = _cap(device=7)
    stream = FakeStream.instances[0]
    a.mark_stream_dead()
    assert stream.closed is True
    assert shared.registry_size() == 0
    # The replacement capture must NOT inherit the dead session.
    c, _ = _cap(device=7)
    assert len(FakeStream.instances) == 2
    assert c._shared is not b._shared


def test_dead_stream_is_not_reused_by_new_sinks():
    a, _ = _cap(device=7)
    a._shared.mark_dead()
    b, _ = _cap(device=7)
    assert b.valid is True
    assert b._shared is not a._shared


# ── Failure contract ─────────────────────────────────────────────────────

def test_sink_needing_more_channels_than_device_fails_to_open(monkeypatch):
    monkeypatch.setattr(shared, '_device_input_channels', lambda dev: 1)
    cap, _ = _cap(device=7, channels=2, ring_channels=2)
    assert cap.valid is False
    assert cap.open_error is not None
    assert 'input channel' in cap.open_error


# ── End-to-end: the split-stereo configuration that caused the bug ───────

def _entity(name, device_id, channel_mode):
    from chirp.recording.entity import RecordingEntity
    e = RecordingEntity(name=name, device_id=device_id)
    e.channel_mode = channel_mode
    # Rebuild the capture for the channel mode, exactly as the UI does.
    e.capture.close()
    e.capture = e._make_capture(channels=2)
    return e


def test_split_stereo_entities_share_one_endpoint():
    """Streams 1 and 2 taking Left / Right of one stereo input must
    present ONE client to the OS capture session, not two."""
    left = _entity('L', 7, 'Left')
    right = _entity('R', 7, 'Right')
    try:
        assert left.capture.valid and right.capture.valid
        assert left.capture._shared is right.capture._shared
        # One physical stream open for the endpoint, whatever the
        # per-stream channel selection is.
        assert shared.registry_size() == 1
    finally:
        left.capture.close()
        right.capture.close()


def test_endpoint_released_only_when_all_its_streams_stop():
    """The reset the user has to perform by hand: the OS capture session
    survives one stream stopping and is torn down only when the last
    stream on the endpoint stops."""
    left = _entity('L', 7, 'Left')
    right = _entity('R', 7, 'Right')
    stream = left.capture._shared.stream
    left.capture.close()
    assert stream.closed is False        # right still holds the endpoint
    right.capture.close()
    assert stream.closed is True         # session destroyed → state cleared


def test_open_failure_leaves_registry_clean(monkeypatch):
    class _Bad:
        def __init__(self, **kw):
            raise OSError('no such device [PaError -9996]')

    monkeypatch.setattr(shared, '_stream_factory', _Bad)
    cap, _ = _cap(device=7)
    assert cap.valid is False
    assert 'OSError' in cap.open_error
    assert shared.registry_size() == 0     # failed opens are not cached


# ── Capture parameters actually reach PortAudio ──────────────────────────

def test_default_latency_is_an_explicit_float_not_high():
    """Regression guard. On WASAPI the device's own 'high' default can be
    10 ms (measured on a Focusrite Scarlett 18i20), which is far too thin
    a margin for the driver + USB stack + GIL + unrelated DPCs — and it
    reads as generous, so it invites being 'restored'. The default must
    stay an explicit number of seconds."""
    from chirp.constants import CAPTURE_LATENCY
    assert isinstance(CAPTURE_LATENCY, (int, float)), \
        'CAPTURE_LATENCY must be explicit seconds, not a device default'
    assert CAPTURE_LATENCY >= 0.1


def test_stream_opens_with_configured_params():
    a, _ = _cap(device=7)
    kw = FakeStream.instances[0].kw
    from chirp.audio import shared_stream as sh
    blocksize, latency = sh.current_params()
    assert kw['blocksize'] == blocksize
    assert kw['latency'] == latency
