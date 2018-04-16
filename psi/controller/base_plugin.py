import logging
log = logging.getLogger(__name__)

from functools import partial
import operator as op
import threading

import numpy as np

from atom.api import Enum, Bool, Typed
from enaml.application import timed_call
from enaml.qt.QtCore import QTimer
from enaml.workbench.api import Extension
from enaml.workbench.plugin import Plugin

from .channel import Channel, OutputChannel, InputChannel
from .engine import Engine
from .output import Output, Synchronized
from .input import Input
from .device import Device

from .experiment_action import (ExperimentAction, ExperimentEvent,
                                ExperimentState)
from .output import ContinuousOutput, EpochOutput


IO_POINT = 'psi.controller.io'
ACTION_POINT = 'psi.controller.actions'


def get_inputs(input):
    inputs = []
    for child in input.children:
        inputs.extend(get_inputs(child))
    inputs.append(input)
    return inputs


def get_outputs(output):
    outputs = []
    for child in output.children:
        outputs.extend(get_outputs(child))
    outputs.append(output)
    return outputs


def log_actions(plugin):
    accumulator = {}
    for action in plugin._actions:
        a = accumulator.setdefault(action.event, [])
        info = '{} (weight={})'.format(action.command, action.weight)
        a.append(info)

    for event, actions in accumulator.items():
        log.info(event)
        for action in actions:
            log.info('\t{}'.format(action))


def find_devices(point):
    devices = {}
    for extension in point.extensions:
        for device in extension.get_children(Device):
            devices[device.name] = device
    return devices


def find_engines(point):
    master_engine = None
    engines = {}
    for extension in point.extensions:
        for e in extension.get_children(Engine):
            engines[e.name] = e
            if e.master_clock:
                if master_engine is not None:
                    m = 'Only one engine can be defined as the master'
                    raise ValueError(m)
                master_engine = e
    return engines, master_engine


def find_channels(engines):
    channels = {}
    for e in engines.values():
        for c in e.get_channels(active=False):
            channels[c.name] = c
    return channels


def find_outputs(channels, point):
    outputs = {}
    supporting = {}

    # Find all the outputs already connected to a channel
    for c in channels.values():
        if isinstance(c, OutputChannel):
            for o in c.outputs:
                outputs[o.name] = o

    # Find unconnected outputs and inputs (these are allowed so that we can
    # split processing hierarchies across multiple manifests).
    for extension in point.extensions:
        for o in extension.get_children(Output):
            outputs[o.name] = o
        for s in extension.get_children(Synchronized):
            supporting[s.name] = s
            for o in s.outputs:
                outputs[o.name] = o

    return outputs, supporting


def find_inputs(channels, point):
    inputs = {}

    # Find all the outputs already connected to a channel
    for c in channels.values():
        if isinstance(c, InputChannel):
            for i in c.children:
                for ci in get_inputs(i):
                    inputs[ci.name] = ci

    for extension in point.extensions:
        for i in extension.get_children(Input):
            # Recurse through input tree. Currently we assume that
            # inputs can be nested/hierarchial while outputs are not.
            for ci in get_inputs(i):
                inputs[ci.name] = ci

    return inputs


class BasePlugin(Plugin):

    # Tracks the state of the controller.
    experiment_state = Enum('initialized', 'running', 'paused', 'stopped')

    # Provides direct access to plugins rather than going through the core
    # command system. Right now the context plugin is so fundamentally important
    # to the controller that it would be cumbersome to use the core command
    # system.
    core = Typed(Plugin)
    context = Typed(Plugin)
    data = Typed(Plugin)

    # We should not respond to changes during the course of a trial. These
    # flags indicate changes or requests from the user are pending and should
    # be processed when the opportunity arises (e.g., at the end of the trial).
    _apply_requested = Bool(False)
    _remind_requested = Bool(False)
    _pause_requested = Bool(False)

    # Can the experiment be paused?
    _pause_ok = Bool(False)

    # Available engines
    _engines = Typed(dict, {})

    # Available channels
    _channels = Typed(dict, {})

    # Available outputs
    _outputs = Typed(dict, {})

    # Available supporting classes for outputs (right now only Synchronized)
    _supporting = Typed(dict, {})

    # Available inputs
    _inputs = Typed(dict, {})

    # Available devices
    _devices = Typed(dict, {})

    # This determines which engine is responsible for the clock
    _master_engine = Typed(Engine)

    # List of events and actions that can be associated with the event
    _events = Typed(dict, {})
    _states = Typed(dict, {})
    _actions = Typed(list, {})
    _action_context = Typed(dict, {})
    _timers = Typed(dict, {})

    def start(self):
        log.debug('Starting controller plugin')
        self._refresh_io()
        self._refresh_actions()
        self._bind_observers()
        self.core = self.workbench.get_plugin('enaml.workbench.core')
        self.context = self.workbench.get_plugin('psi.context')
        self.data = self.workbench.get_plugin('psi.data')

    def stop(self):
        self._unbind_observers()

    def _bind_observers(self):
        self.workbench.get_extension_point(IO_POINT) \
            .observe('extensions', self._refresh_io)
        self.workbench.get_extension_point(ACTION_POINT) \
            .observe('extensions', self._refresh_actions)

    def _unbind_observers(self):
        self.workbench.get_extension_point(IO_POINT) \
            .unobserve('extensions',self._refresh_io)
        self.workbench.get_extension_point(ACTION_POINT) \
            .unobserve('extensions', self._refresh_actions)

    def _refresh_io(self, event=None):
        # TODO: Allow disabling of devices.
        log.debug('Loading IO')
        point = self.workbench.get_extension_point(IO_POINT)

        self._devices = find_devices(point)
        self._engines, self._master_engine = find_engines(point)
        self._channels = find_channels(self._engines)
        self._outputs, self._supporting = find_outputs(self._channels, point)
        self._inputs = find_inputs(self._channels, point)

        for d in self._devices.values():
            d.load_manifest(self.workbench)

        for s in self._supporting.values():
            s.load_manifest(self.workbench)

        for e in self._engines.values():
            e.load_manifest(self.workbench)

        for o in self._outputs.values():
            # First, make sure that the output is connected to a target. Check
            # to see if the target is named. If not, then check to see if it
            # has a parent one can use.
            if o.target is None and o.target_name:
                self.connect_output(o.name, o.target_name)
            elif o.target is None and not isinstance(o.parent, Extension):
                o.parent.add_output(o)
            elif o.target is None:
                log.warn('Unconnected output %s', o.name)
            o.load_manifest(self.workbench)

        for i in self._inputs.values():
            # First, make sure the input is connected to a source
            if i.source is None and i.source_name:
                self.connect_input(i.name, i.source_name)
            elif i.source is None and not isinstance(i.parent, Extension):
                i.parent.add_input(i)
            elif i.source is None:
                log.warn('Unconnected input %s', i.name)

            i.load_manifest(self.workbench)

    def connect_output(self, output_name, target_name):
        # Link up outputs with channels if needed.  TODO: Can another output be
        # the target (e.g., if one wanted to combine multiple tokens into a
        # single stream)?
        if target_name in self._channels:
            target = self._channels[target_name]
        else:
            m = "Unknown target {}".format(target_name)
            raise ValueError(m)

        o = self._outputs[output_name]
        target.add_output(o)
        m = 'Connected output %s to target %s'
        log.debug(m, output_name, target_name)

    def connect_input(self, input_name, source_name):
        if source_name in self._inputs:
            source = self._inputs[source_name]
        elif source_name in self._channels:
            source = self._channels[source_name]
        else:
            m = "Unknown source {}".format(source_name)
            raise ValueError(m)

        i = self._inputs[input_name]
        source.add_input(i)
        m = 'Connected input %s to source %s'
        log.debug(m, input_name, source_name)

    def _refresh_actions(self, event=None):
        actions = []
        events = {}
        states = {}

        point = self.workbench.get_extension_point(ACTION_POINT)
        for extension in point.extensions:
            found_states = extension.get_children(ExperimentState)
            found_events = extension.get_children(ExperimentEvent)
            found_actions = extension.get_children(ExperimentAction)

            for state in found_states:
                if state.name in states:
                    m = '{} state already exists'.format(state.name)
                    raise ValueError(m)
                states[state.name] = state
                found_events.extend(state._generate_events())

            for event in found_events:
                if event.name in events:
                    m = '{} event already exists'.format(event.name)
                    raise ValueError(m)
                events[event.name] = event

            actions.extend(found_actions)

        context = {}
        for state_name in states:
            context[state_name + '_active'] = False
        for event_name in events:
            context[event_name] = False

        actions.sort(key=lambda a: a.weight)
        self._states = states
        self._events = events
        self._actions = actions
        self._action_context = context

        log_actions(self)

    def configure_engines(self):
        log.debug('Configuring engines')
        for engine in self._engines.values():
            # Check to see if engine is being used
            if engine.get_channels():
                engine.configure()
                cb = partial(self.invoke_actions, '{}_end'.format(engine.name))
                engine.register_done_callback(cb)
        self.invoke_actions('engines_configured')

    def start_engines(self):
        log.debug('Starting engines')
        for engine in self._engines.values():
            engine.start()

    def stop_engines(self):
        for name, timer in list(self._timers.items()):
            timer.timeout.disconnect()
            timer.stop()
            del self._timers[name]
        for engine in self._engines.values():
            engine.stop()

    def reset_engines(self):
        for engine in self._engines.values():
            engine.reset()

    def get_output(self, output_name):
        return self._outputs[output_name]

    def get_input(self, input_name):
        return self._inputs[input_name]

    def set_input_attr(self, input_name, attr_name, value):
        setattr(self._inputs[input_name], attr_name, value)

    def get_channel(self, channel_name):
        return self._channels[channel_name]

    def get_channels(self, mode=None, direction=None, timing=None,
                     active=True):
        '''
        Return channels matching criteria across all engines

        Parameters
        ----------
        mode : {None, 'analog', 'digital'
            Type of channel
        direction : {None, 'input, 'output'}
            Direction
        timing : {None, 'hardware', 'software'}
            Hardware or software-timed channel. Hardware-timed channels have a
            sampling frequency greater than 0.
        active : bool
            If True, return only channels that have configured inputs or
            outputs.
        '''
        channels = []
        for engine in self._engines.values():
            ec = engine.get_channels(mode, direction, timing, active)
            channels.extend(ec)
        return channels

    def invoke_actions(self, event_name, timestamp=None, delayed=False,
                       cancel_existing=True, **kw):
        if cancel_existing:
            self.stop_timer(event_name)
        if delayed:
            delay = timestamp-self.get_ts()
            if delay > 0:
                cb = lambda: self._invoke_actions(event_name, timestamp)
                self.start_timer(event_name, delay, cb)
                return
        self._invoke_actions(event_name, timestamp, **kw)

    def event_used(self, event_name):
        for action in self._actions:
            if event_name in action._dependencies:
                return True
        return False

    def _invoke_actions(self, event_name, timestamp=None, **kw):
        log.trace('Triggering event {}'.format(event_name))

        if timestamp is not None:
            # The event is logged only if the timestamp is provided
            params = {'event': event_name, 'timestamp': timestamp}
            self.core.invoke_command('psi.data.process_event', params)

        # If this is a stateful event, update the associated state.
        if event_name.endswith('_start'):
            key = event_name[:-6]
            self._action_context[key + '_active'] = True
        elif event_name.endswith('_end'):
            key = event_name[:-4]
            self._action_context[key + '_active'] = False

        # Make a copy of the context and set the event to True. We don't want
        # to set the state on the main context since it may affect recursive
        # notifications.
        context = self._action_context.copy()
        context[event_name] = True

        for action in self._actions:
            if action.match(context):
                self._invoke_action(action, event_name, timestamp, **kw)

    def _invoke_action(self, action, event_name, timestamp, **kw):
        # Add the event name and timestamp to the parameters passed to the
        # command.
        kwargs = action.kwargs.copy()
        kwargs.update(kw)
        kwargs['timestamp'] = timestamp
        kwargs['event'] = event_name
        self.core.invoke_command(action.command, parameters=kwargs)

    def request_apply(self):
        if not self.apply_changes():
            log.debug('Apply requested')
            self._apply_requested = True

    def request_remind(self):
        self._remind_requested = True

    def request_pause(self):
        if not self.pause_experiment():
            log.debug('Pause requested')
            self._pause_requested = True

    def request_resume(self):
        self._pause_requested = False
        self.experiment_state = 'running'

    def apply_changes(self):
        raise NotImplementedError

    def prepare_experiment(self):
        self.invoke_actions('experiment_prepare')

    def start_experiment(self):
        self.invoke_actions('experiment_start')
        self.experiment_state = 'running'

    def stop_experiment(self):
        self.invoke_actions('experiment_end', self.get_ts())
        self.experiment_state = 'stopped'

    def pause_experiment(self):
        raise NotImplementedError

    def get_ts(self):
        return self._master_engine.get_ts()

    def set_pause_ok(self, value):
        self._pause_ok = value

    def start_timer(self, name, duration, callback):
        log.debug('Starting timer {}'.format(name))
        timer = QTimer()
        timer.timeout.connect(callback)
        timer.setSingleShot(True)
        timer.start(duration*1e3)
        self._timers[name] = timer

    def stop_timer(self, name):
        if self._timers.get(name) is not None:
            self._timers[name].timeout.disconnect()
            self._timers[name].stop()
            del self._timers[name]
            log.debug('Disabled deferred event %s', name)
