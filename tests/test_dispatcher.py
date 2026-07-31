"""Unit tests for psi.controller.dispatcher.ControlDispatcher."""
import sys
import threading
import time

import pytest

import psi.controller.dispatcher as dispatcher_module
from psi.controller.dispatcher import ControlDispatcher


@pytest.fixture
def dispatcher():
    d = ControlDispatcher(name='test-dispatcher')
    yield d
    d.stop()


def test_submit_sync_returns_value(dispatcher):
    assert dispatcher.submit_sync(lambda: 42) == 42


def test_submit_sync_runs_on_dispatcher_thread(dispatcher):
    thread = dispatcher.submit_sync(threading.current_thread)
    assert thread.name == 'test-dispatcher'
    assert thread is not threading.current_thread()


def test_submit_sync_propagates_exceptions(dispatcher):
    def boom():
        raise ValueError('boom')
    with pytest.raises(ValueError, match='boom'):
        dispatcher.submit_sync(boom)


def test_submit_sync_reentrant(dispatcher):
    # A dispatched call may synchronously dispatch more work (an action
    # triggering another action) without deadlocking.
    def outer():
        return dispatcher.submit_sync(lambda: 'inner')
    assert dispatcher.submit_sync(outer) == 'inner'


def test_submit_fire_and_forget(dispatcher):
    done = threading.Event()
    dispatcher.submit(done.set)
    assert done.wait(2)


def test_submit_logs_exceptions_without_raising(dispatcher):
    def boom():
        raise ValueError('boom')
    dispatcher.submit(boom)
    # The dispatcher must survive and keep processing work.
    assert dispatcher.submit_sync(lambda: 'alive') == 'alive'


def test_serialization(dispatcher):
    # Concurrent submissions from many threads execute one at a time.
    active = []
    overlaps = []
    results = []

    def work(i):
        active.append(i)
        if len(active) > 1:
            overlaps.append(list(active))
        time.sleep(0.001)
        active.remove(i)
        results.append(i)

    threads = [threading.Thread(target=dispatcher.submit_sync, args=(work, i))
               for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)

    assert len(results) == 8
    assert overlaps == []


def test_call_later_fires_on_dispatcher_thread(dispatcher):
    done = threading.Event()
    seen = {}

    def cb():
        seen['thread'] = threading.current_thread().name
        done.set()

    dispatcher.call_later('t', 0.01, cb)
    assert done.wait(2)
    assert seen['thread'] == 'test-dispatcher'


def test_call_later_cancel(dispatcher):
    fired = threading.Event()
    dispatcher.call_later('t', 0.05, fired.set)
    dispatcher.cancel('t')
    assert not fired.wait(0.2)


def test_call_later_same_name_cancels_pending(dispatcher):
    calls = []
    dispatcher.call_later('t', 0.05, lambda: calls.append('first'))
    dispatcher.call_later('t', 0.01, lambda: calls.append('second'))
    time.sleep(0.2)
    assert calls == ['second']


def test_cancel_all(dispatcher):
    fired = threading.Event()
    dispatcher.call_later('a', 0.05, fired.set)
    dispatcher.call_later('b', 0.05, fired.set)
    dispatcher.cancel_all()
    assert not fired.wait(0.2)


def test_submit_sync_pump_called_while_waiting(dispatcher):
    # A caller that passes `pump` needs it invoked repeatedly until the
    # dispatched work completes -- this is how a GUI thread keeps its
    # event loop alive (servicing a deferred_call issued from the
    # dispatcher thread mid-flight) instead of blocking outright.
    release = threading.Event()
    pump_calls = []

    def slow():
        assert release.wait(2)
        return 'done'

    def pump():
        pump_calls.append(threading.current_thread().name)
        if len(pump_calls) == 3:
            release.set()

    assert dispatcher.submit_sync(slow, pump=pump) == 'done'
    assert len(pump_calls) >= 3
    assert all(name == threading.current_thread().name for name in pump_calls)


def test_submit_sync_pump_propagates_exceptions(dispatcher):
    def boom():
        raise ValueError('boom')

    with pytest.raises(ValueError, match='boom'):
        dispatcher.submit_sync(boom, pump=lambda: None)


def test_submit_sync_pump_not_called_when_inline(dispatcher):
    # Reentrant calls (already on the dispatcher thread) run inline and
    # never touch the pump, since there's nothing to wait for.
    calls = []

    def outer():
        return dispatcher.submit_sync(lambda: 'inner', pump=lambda: calls.append(1))

    assert dispatcher.submit_sync(outer) == 'inner'
    assert calls == []


@pytest.mark.skipif(sys.platform != 'win32', reason='COM init is Windows-only')
def test_com_initialized_on_dispatcher_thread(monkeypatch):
    # PortAudio's ASIO host API requires the thread that calls
    # Pa_OpenStream to have initialized COM itself, or opening an ASIO
    # stream hangs (https://github.com/PortAudio/portaudio/issues/250).
    # Engine configuration always runs on the dispatcher thread, so it
    # must CoInitialize on entry and CoUninitialize on exit.
    calls = []

    class FakePythoncom:
        def CoInitialize(self):
            calls.append(('init', threading.current_thread().name))

        def CoUninitialize(self):
            calls.append(('uninit', threading.current_thread().name))

    monkeypatch.setattr(dispatcher_module, 'pythoncom', FakePythoncom())
    d = ControlDispatcher(name='com-test-dispatcher')
    d.submit_sync(lambda: None)
    thread = d._thread
    d.stop()
    thread.join(2)

    assert calls == [
        ('init', 'com-test-dispatcher'),
        ('uninit', 'com-test-dispatcher'),
    ]


def test_stop_rejects_sync_work():
    d = ControlDispatcher()
    d.submit_sync(lambda: None)
    d.stop()
    with pytest.raises(RuntimeError, match='stopped'):
        d.submit_sync(lambda: None)
