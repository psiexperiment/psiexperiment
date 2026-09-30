'''
Tests for :mod:`psi.launcher.process_manager`.

The launcher and the experiments it starts talk over a websocket: psi's
`WebsocketClientPlugin` on one end, the `ProcessManager`'s server on the
other. These tests run both ends for real, so that a change to either side
of that exchange (event names, command names, message layout) shows up here
rather than as a launcher that silently stops advancing its queue.
'''
import os
import queue
import re
from pathlib import Path

import enaml
import pytest

with enaml.imports():
    from psi.paradigms.core.websocket_mixins import WebsocketClientPlugin

from psi.launcher import process_manager
from psi.launcher.process_manager import ProcessManager, RELAYED_EVENTS


PSI_ROOT = Path(__file__).parents[2] / 'src' / 'psi'


class FakeProcess:

    def poll(self):
        return None


@pytest.fixture
def manager(app):
    return ProcessManager()


@pytest.fixture
def client(manager):
    '''
    A psi experiment's end of the connection, registered with the manager
    the way `open_next_subprocess` registers a real one.
    '''
    received = queue.Queue()
    # The server identifies a client by its parent's PID, which for a real
    # experiment is the process the launcher started.
    process = {'client_id': os.getppid(), 'process': FakeProcess(),
               'state': None, 'uid': 'row-1'}
    manager.current_subprocess = process
    manager.subprocesses.append((process, 'row-1'))

    plugin = WebsocketClientPlugin(uri=manager.ws_server.connected_uri,
                                   recv_cb=received.put)
    plugin.start_thread()
    yield plugin, received, process
    plugin._disconnect()


def expect(received, command):
    while True:
        mesg = received.get(timeout=5)
        if mesg['command'] == command:
            return mesg


def send_event(plugin, event, info=None):
    # Same layout as ControllerPlugin._log_event.
    plugin.send_message({'event': event, 'timestamp': 0, 'info': info})


def wait_for(condition, timeout=5):
    import time
    end = time.time() + timeout
    while time.time() < end:
        if condition():
            return
        time.sleep(0.01)
    raise AssertionError('Timed out')


def test_client_is_told_which_events_to_relay(client):
    plugin, received, _ = client
    mesg = expect(received, 'websocket.set_event_filter')
    event_filter = re.compile(mesg['parameters']['event_filter'])
    for event in RELAYED_EVENTS:
        assert event_filter.match(event)


def test_autostart_starts_the_experiment(manager, client):
    plugin, received, process = client
    events = []
    manager.subscribe(lambda event, uid: events.append((event, uid)))
    manager.autostart = True
    send_event(plugin, 'plugins_started')
    # Sent to this client only, which exercises the server's lookup of a
    # client by ID.
    expect(received, 'psi.show_window')
    expect(received, 'psi.controller.start')
    assert process['state'] == 'connected'
    assert events == [('plugins_started', 'row-1')]


def test_clean_end_opens_next(manager, client, monkeypatch):
    plugin, received, process = client
    started = []

    class Popen(FakeProcess):
        pid = -1

        def __init__(self, cmd, env):
            started.append(cmd)

    monkeypatch.setattr(process_manager.subprocess, 'Popen', Popen)
    manager.autostart = True
    manager.add_command(['psi', 'next'], {}, uid='row-2')
    send_event(plugin, 'experiment_start')
    send_event(plugin, 'experiment_end', {'stop_reason': '', 'error_message': ''})
    wait_for(lambda: process['state'] == 'complete')
    assert started == [['psi', 'next']]
    assert manager.current_subprocess['uid'] == 'row-2'


def test_error_end_clears_queue(manager, client):
    plugin, received, process = client
    manager.autostart = True
    manager.add_command(['psi', 'next'], {})
    send_event(plugin, 'experiment_end', {'stop_reason': 'error', 'error_message': 'x'})
    wait_for(lambda: process['state'] == 'complete')
    assert manager.commands == []
    assert not manager.autostart


def test_window_closed_forgets_process(manager, client):
    plugin, received, process = client
    send_event(plugin, 'window_closed')
    wait_for(lambda: not manager.subprocesses)
    assert manager.current_subprocess is None


def test_clear_commands(manager):
    manager.add_command(['psi', 'a'], {})
    manager.clear_commands()
    assert manager.commands == []


def _source(pattern):
    '''
    True if `pattern` appears in psi's source, outside the launcher (which
    would otherwise always match itself).
    '''
    launcher = PSI_ROOT / 'launcher'
    for path in [*PSI_ROOT.rglob('*.enaml'), *PSI_ROOT.rglob('*.py')]:
        if launcher in path.parents:
            continue
        if re.search(pattern, path.read_text(encoding='utf-8')):
            return True
    return False


@pytest.mark.parametrize('command', ['psi.show_window', 'psi.controller.start',
                                     'websocket.set_event_filter'])
def test_commands_sent_to_experiments_exist(command):
    assert _source(rf"id\s*=\s*'{re.escape(command)}'"), command


@pytest.mark.parametrize('event', RELAYED_EVENTS)
def test_relayed_events_are_emitted_by_psi(event):
    # Each relayed event must be something psi actually emits, or the
    # launcher waits for it forever.
    assert _source(rf"['\"]{event}['\"]"), event


def test_message_for_unknown_client_is_not_delivered(manager, client):
    plugin, received, _ = client
    expect(received, 'websocket.set_event_filter')
    # Used to fall through to whichever clients the previous message went to
    # (here, the broadcast above).
    manager.ws_server.send_message({'command': 'for.someone.else'}, -12345)
    manager.ws_server.send_message({'command': 'psi.show_window'}, os.getppid())
    assert received.get(timeout=5)['command'] == 'psi.show_window'
