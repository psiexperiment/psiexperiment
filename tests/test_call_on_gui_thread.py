"""Unit tests for psi.util.call_on_gui_thread."""
import threading
import time
from types import SimpleNamespace

import pytest

import psi.util as util_module
from psi.util import call_on_gui_thread


def test_call_on_gui_thread_direct_call_when_no_application(monkeypatch):
    # No GUI application running (e.g. the standalone script at the
    # a script or a test) -- just call fn on the current
    # thread, no marshaling.
    monkeypatch.setattr(util_module, 'Application',
                        SimpleNamespace(instance=lambda: None))
    calling_thread = threading.current_thread()
    seen = {}

    def fn():
        seen['thread'] = threading.current_thread()
        return 'value'

    assert call_on_gui_thread(fn) == 'value'
    assert seen['thread'] is calling_thread


def test_call_on_gui_thread_direct_call_when_already_on_gui_thread(monkeypatch):
    app = SimpleNamespace(is_main_thread=lambda: True)
    monkeypatch.setattr(util_module, 'Application',
                        SimpleNamespace(instance=lambda: app))
    assert call_on_gui_thread(lambda: 'value') == 'value'


def test_call_on_gui_thread_marshals_to_gui_thread_via_deferred_call(monkeypatch):
    # When the calling thread isn't the GUI thread, fn must run via
    # deferred_call (mirrors psi.controller.dispatcher's control-plane
    # thread needing the GUI thread -- see
    # call_on_gui_thread's docstring) and call_on_gui_thread must block until
    # that completes.
    app = SimpleNamespace(is_main_thread=lambda: False)
    monkeypatch.setattr(util_module, 'Application',
                        SimpleNamespace(instance=lambda: app))

    seen = {}

    def fake_deferred_call(fn, *args, **kwargs):
        def run():
            time.sleep(0.05)
            seen['thread'] = threading.current_thread()
            fn(*args, **kwargs)
        threading.Thread(target=run, name='fake-main-thread', daemon=True).start()

    monkeypatch.setattr(util_module, 'deferred_call', fake_deferred_call)

    calling_thread = threading.current_thread()
    assert call_on_gui_thread(lambda: 'value') == 'value'
    assert seen['thread'] is not calling_thread
    assert seen['thread'].name == 'fake-main-thread'


def test_call_on_gui_thread_propagates_exceptions(monkeypatch):
    app = SimpleNamespace(is_main_thread=lambda: False)
    monkeypatch.setattr(util_module, 'Application',
                        SimpleNamespace(instance=lambda: app))

    def fake_deferred_call(fn, *args, **kwargs):
        threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=True).start()

    monkeypatch.setattr(util_module, 'deferred_call', fake_deferred_call)

    def boom():
        raise ValueError('boom')

    with pytest.raises(ValueError, match='boom'):
        call_on_gui_thread(boom)
