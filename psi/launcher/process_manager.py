import logging
log = logging.getLogger(__name__)

import pprint
import os
import subprocess
import threading
import time

from atom.api import Atom, Bool, Float, List, Typed, Value
import enaml
from enaml.application import timed_call

with enaml.imports():
    from psi.paradigms.core.websocket_mixins import WebsocketServerPlugin


#: Events that experiments invoked by `psi` relay to the launcher. See
#: `ProcessManager.recv_cb` for how each is handled.
RELAYED_EVENTS = ['plugins_started', 'experiment_start', 'experiment_end',
                  'window_closed']


def synchronized(fn):
    '''
    Decorator that ensures lock is obtained before calling method
    '''
    def wrapped(self, *args, **kw):
        with self.lock:
            return fn(self, *args, **kw)
    return wrapped


class ProcessManager(Atom):
    '''
    Manager for running one or more psiexperiments in sequence

    Each experiment is run as a `psi` subprocess that connects back to a
    websocket server owned by this manager (its URI is passed to the
    subprocess via the `PSI_WEBSOCKETS_URI` environment variable). The
    subprocess reports its progress over the websocket, which is how the
    manager knows when to start the next experiment in the queue.
    '''
    #: List of callbacks requesting notifications. Each is called as
    #: `cb(event, uid)`, where `event` is one of `RELAYED_EVENTS` or
    #: 'subprocess_exited' and `uid` is the value passed to `add_command`.
    callbacks = List()

    #: List of commands the order they should be executed.
    commands = List()

    #: List of (subprocess, uid) tuples for subprocesses that are currently
    #: open.
    subprocesses = List()

    #: Current active subprocess (we retain references to all subprocesses that
    #: are currently open since we do not auto-close the GUI).
    current_subprocess = Value()

    #: Should experiments be auto-started?
    autostart = Bool(False)

    #: Used for communicating with experiments invoked by `psi`.
    ws_server = Typed(WebsocketServerPlugin)

    #: Used to track runtime of individual experiments.
    exp_start_time = Float()

    #: Duration of last experiment from experiment_start to experiment_end event.
    duration = Float()

    lock = Value()

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        # The lock must exist before the server thread starts, since the
        # server may call `recv_cb` as soon as a client connects.
        self.lock = threading.Lock()
        # Set up a websocket server. Whenever clients (i.e., programs invoked
        # by `psi`) connect they will be instructed to relay only a handful of
        # events to keep load on the websocket low.
        self.ws_server = WebsocketServerPlugin(
            recv_cb=self.recv_cb,
            connect_cb=self.connect_cb,
        )
        self.ws_server.start_thread()
        log.info('Websocket server listening on %s', self.ws_server.connected_uri)
        timed_call(1000, self.check_status)

    def connect_cb(self):
        self.ws_server.send_message({
            'command': 'websocket.set_event_filter',
            'parameters': {'event_filter': '|'.join(RELAYED_EVENTS)},
        })

    def open_next_subprocess(self):
        try:
            cmd, env, uid = self.commands.pop(0)
        except IndexError:
            log.info('No more commands queued')
            return
        process = subprocess.Popen(cmd, env=dict(os.environ, **env))
        log.info('Command: %s', ' '.join(cmd))
        log.info('Environment Variables:\n%s', pprint.pformat(env))
        self.current_subprocess = {
            'cmd': cmd,
            'env': env,
            'process': process,
            'running': False,
            'client_id': process.pid,
            'state': None,
            'uid': uid,
        }
        self.subprocesses.append((self.current_subprocess, uid))

    def subscribe(self, cb):
        self.callbacks.append(cb)

    def notify(self, event, uid):
        for cb in self.callbacks:
            cb(event, uid)

    @synchronized
    def recv_cb(self, message):
        # Find which process the message is from.
        for entry in self.subprocesses:
            process, uid = entry
            if process['client_id'] == message['client_id']:
                break
        else:
            raise ValueError(f'No process with client ID {message["client_id"]}')

        self.notify(message['event'], uid)

        if message['event'] == 'plugins_started':
            process['state'] = 'connected'
            # Start the subprocess if it is the first one in the list to run.
            if self.autostart:
                mesg = {'command': 'psi.show_window'}
                self.ws_server.send_message(mesg, process['client_id'])
                mesg = {'command': 'psi.controller.start'}
                self.ws_server.send_message(mesg, process['client_id'])
        elif message['event'] == 'experiment_start':
            process['state'] = 'running'
            self.exp_start_time = time.time()
        elif message['event'] == 'experiment_end':
            if process == self.current_subprocess:
                self.current_subprocess = None
            process['state'] = 'complete'
            self.duration = round(time.time() - self.exp_start_time)
            # If the experiment ended in an error, then don't continue with the
            # queue. Clear the queue as well.
            if message['info'].get('stop_reason') != '':
                self.autostart = False
                self.commands = []
            # Now, start the next one if one exists. Make sure
            # current_subprocess is not None just in case we get multiple stop
            # events from a subprocess.
            if self.autostart and self.current_subprocess is None:
                self.open_next_subprocess()
        elif message['event'] == 'window_closed':
            # Window closed. Remove from the list of subprocesses.
            if process == self.current_subprocess:
                self.current_subprocess = None
            self.subprocesses = [s for s in self.subprocesses if s is not entry]

    @synchronized
    def add_command(self, cmd, env, uid=None):
        log.info('Queueing command: %s', ' '.join(cmd))
        self.commands.append((cmd, env, uid))

    @synchronized
    def clear_commands(self):
        '''
        Discard all queued commands (e.g., left over from an aborted sequence).
        '''
        self.commands = []

    @synchronized
    def pause_sequence(self):
        # TODO: What to do if we need to re-sequence an experiment?
        self.autostart = False

    @synchronized
    def check_status(self):
        self.subprocesses = [p for p in self.subprocesses if p[0]['process'].poll() is None]
        # This means the current subprocess has been closed.
        if self.current_subprocess is not None:
            if self.current_subprocess['process'].poll() is not None:
                self.notify('subprocess_exited', self.current_subprocess['uid'])
                self.current_subprocess = None
        timed_call(1000, self.check_status)
