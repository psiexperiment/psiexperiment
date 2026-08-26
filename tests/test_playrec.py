"""Unit tests for psi.controller.engines.soundcard.playrec._on_main_thread."""
import threading
import time
from types import SimpleNamespace

import pytest

import psi.controller.engines.soundcard.playrec as playrec_module
from psi.controller.engines.soundcard.playrec import _on_main_thread


def test_on_main_thread_direct_call_when_no_application(monkeypatch):
    # No GUI application running (e.g. the standalone script at the
    # bottom of playrec.py, or a test) -- just call fn on the current
    # thread, no marshaling.
    monkeypatch.setattr(playrec_module, 'Application',
                        SimpleNamespace(instance=lambda: None))
    calling_thread = threading.current_thread()
    seen = {}

    def fn():
        seen['thread'] = threading.current_thread()
        return 'value'

    assert _on_main_thread(fn) == 'value'
    assert seen['thread'] is calling_thread


def test_on_main_thread_direct_call_when_already_on_main_thread(monkeypatch):
    app = SimpleNamespace(is_main_thread=lambda: True)
    monkeypatch.setattr(playrec_module, 'Application',
                        SimpleNamespace(instance=lambda: app))
    assert _on_main_thread(lambda: 'value') == 'value'


def test_on_main_thread_marshals_to_main_thread_via_deferred_call(monkeypatch):
    # When the calling thread isn't the GUI thread, fn must run via
    # deferred_call (mirrors psi.controller.dispatcher's control-plane
    # thread calling into PlayRec, which needs the actual stream
    # open/start/stop calls to happen on the GUI thread -- see
    # _on_main_thread's docstring) and _on_main_thread must block until
    # that completes.
    app = SimpleNamespace(is_main_thread=lambda: False)
    monkeypatch.setattr(playrec_module, 'Application',
                        SimpleNamespace(instance=lambda: app))

    seen = {}

    def fake_deferred_call(fn, *args, **kwargs):
        def run():
            time.sleep(0.05)
            seen['thread'] = threading.current_thread()
            fn(*args, **kwargs)
        threading.Thread(target=run, name='fake-main-thread', daemon=True).start()

    monkeypatch.setattr(playrec_module, 'deferred_call', fake_deferred_call)

    calling_thread = threading.current_thread()
    assert _on_main_thread(lambda: 'value') == 'value'
    assert seen['thread'] is not calling_thread
    assert seen['thread'].name == 'fake-main-thread'


def test_on_main_thread_propagates_exceptions(monkeypatch):
    app = SimpleNamespace(is_main_thread=lambda: False)
    monkeypatch.setattr(playrec_module, 'Application',
                        SimpleNamespace(instance=lambda: app))

    def fake_deferred_call(fn, *args, **kwargs):
        threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=True).start()

    monkeypatch.setattr(playrec_module, 'deferred_call', fake_deferred_call)

    def boom():
        raise ValueError('boom')

    with pytest.raises(ValueError, match='boom'):
        _on_main_thread(boom)


class _FakeStream:
    def __init__(self, **kw):
        self.kw = kw
        self.samplerate = kw['samplerate']
        self.blocksize = kw['blocksize']


class _FakeSD:
    '''
    Minimal ``sounddevice`` stand-in for exercising ``PlayRec.configure``
    without real hardware. Records the kwargs the stream is opened with.
    '''

    def __init__(self):
        self.opened = {}
        self._device = {
            'name': 'RME Babyface', 'hostapi': 0,
            'max_input_channels': 8, 'max_output_channels': 8,
        }

    def query_devices(self, device=None):
        if device is None:
            return [dict(self._device)]
        return dict(self._device, index=0)

    def query_hostapis(self, index):
        return {'name': 'ASIO'}

    def AsioSettings(self, channels):
        return ('asio', tuple(channels))

    def InputStream(self, **kw):
        self.opened = dict(kw)
        return _FakeStream(**kw)


def test_input_stream_opened_with_fully_qualified_selector(monkeypatch):
    # Regression: the input (record-only) path must open the stream with the
    # *original* device selector it was handed -- a fully-qualified
    # "<name>, <host API>" string -- not the bare device name resolved from
    # it. The bare name re-opens sounddevice's ambiguous substring matching
    # that the fully-qualified selector exists to avoid (multiple drivers
    # exposing the same name). See PlayRec.configure.
    fake = _FakeSD()
    monkeypatch.setattr(playrec_module, 'sd', fake)
    monkeypatch.setattr(playrec_module, 'Application',
                        SimpleNamespace(instance=lambda: None))

    selector = 'RME Babyface, ASIO'
    playrec_module.PlayRec(
        fs=96000, device=selector, ai_channels=[0, 1],
        ai_cb=lambda *a: None, blocksize=4096,
    )
    assert fake.opened['device'] == selector
