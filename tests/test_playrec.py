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
