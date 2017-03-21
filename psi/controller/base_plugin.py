import logging
log = logging.getLogger(__name__)

from functools import partial
import operator as op

import numpy as np

from atom.api import Enum, Bool, Typed
from enaml.application import timed_call
from enaml.qt.QtCore import QTimer
from enaml.workbench.plugin import Plugin

from .channel import Channel, AOChannel
from .engine import Engine
from .output import Output
from .input import Input
from .device import Device

from .experiment_action import (ExperimentAction, ExperimentEvent,
                                ExperimentState)
from .output import ContinuousOutput, EpochOutput


IO_POINT = 'psi.controller.io'
ACTION_POINT = 'psi.controller.actions'


def get_named_inputs(input):
    named_inputs = []
    for child in input.children:
        named_inputs.extend(get_named_inputs(child))
    if input.name:
        named_inputs.append(input)
    return named_inputs


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
        log.debug('Loading IO')
        channels = {}
        engines = {}
        outputs = {}
        inputs = {}
        devices = {}
        master_engine = None

        point = self.workbench.get_extension_point(IO_POINT)
        for extension in point.extensions:
            for device in extension.get_children(Device):
                log.debug('Found device {}'.format(device.name))
                devices[device.name] = device
                try:
                    manifest = device.load_manifest()
                    self.workbench.register(manifest)
                except NotImplementedError:
                    m = 'No manifest defind for device {}'.format(device.name)
                    log.warn(m)

            for engine in extension.get_children(Engine):
                engines[engine.name] = engine
                if engine.master_clock:
                    if master_engine is not None:
                        m = 'Only one engine can be defined as the master'
                        raise ValueError(m)
                    master_engine = engine

                for channel in engine.channels:
                    channels[channel.name] = channel
                    for output in getattr(channel, 'outputs', []):
                        outputs[output.name] = output
                    for all_inputs in getattr(channel, 'inputs', []):
                        for input in get_named_inputs(all_inputs):
                            inputs[input.name] = input

        # Find unconnected outputs and inputs (these are allowed so that we can
        # split processing hierarchies across multiple manifests).
        for extension in point.extensions:
            for output in extension.get_children(Output):
                outputs[output.name] = output

        for extension in point.extensions:
            for all_inputs in extension.get_children(Input):
                for input in get_named_inputs(all_inputs):
                    inputs[input.name] = input

        # Link up outputs with channels if needed.  TODO: Can another output be
        # the target (e.g., if one wanted to combine multiple tokens into a
        # single stream)?
        for output in outputs.values():
            if output.target is None:
                if output.target_name in channels:
                    target = channels[output.target_name]
                else:
                    m = "Unknown target {}".format(output.target_name)
                    raise ValueError(m)
                log.debug('Connecting output {} to target {}' \
                            .format(output.name, output.target_name))
                output.target = target

        for input in inputs.values():
            if input.source is None:
                if input.source_name in inputs:
                    source = inputs[input.source_name]
                elif input.source_name in channels:
                    source = channels[input.source_name]
                else:
                    m = "Unknown source {}".format(input.source_name)
                    raise ValueError(m)
                log.debug('Connecting input {} to source {}' \
                          .format(input.name, input.source_name))
                input.source = source

        # Remove channels that do not have an input or output defined. TODO: We
        # need to figure out how to configure which inputs/outputs are active
        # (via GUI or config file) since some IO manifests may define
        # additional (unneeded) inputs and outputs.
        for channel in channels.values():
            if not channel.children:
                channel.engine = None
                del channels[channel.name]

        # Hack for channels that don't define a continuous output. TODO?
        # Somehow find a better way to do this?
        for channel in channels.values():
            if isinstance(channel, AOChannel):
                if channel.continuous_output is not None:
                    break
                else:
                    plugin = self.workbench.get_plugin('psi.token')
                    output = ContinuousOutput(name=channel.name + '_default_',
                                              visible=False)
                    t = plugin.generate_continuous_token('Silence',
                                                         output.name,
                                                         output.label)
                    output._token = t
                    output.target = channel
                    outputs[output.name] = output

        self._master_engine = master_engine
        self._channels = channels
        self._engines = engines
        self._outputs = outputs
        self._inputs = inputs
        self._devices = devices

        log.debug('Available inputs: {}'.format(inputs.keys()))
        log.debug('Available outputs: {}'.format(outputs.keys()))

    def _refresh_actions(self, event=None):
        actions = []
        events = {}
        states = {}

        point = self.workbench.get_extension_point(ACTION_POINT)
        for extension in point.extensions:
            found_states = extension.get_children(ExperimentState)
            found_events = extension.get_children(ExperimentEvent)
            found_actions = extension.get_children(ExperimentAction)
            if extension.factory is not None:
                objs = extension.factory(self.workbench, ExperimentState)
                found_states.extend(objs)
                objs = extension.factory(self.workbench, ExperimentEvent)
                found_events.extend(objs)
                objs = extension.factory(self.workbench, ExperimentAction)
                found_actions.extend(objs)

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

        actions.sort(key=lambda a: a.weight)
        self._states = states
        self._events = events
        self._actions = actions

        log.debug('Configured experiment actions')
        log.debug(' * States: %r', self._states.keys())
        log.debug(' * Events: %r', self._events.keys())
        log.debug(' * Actions')
        for action in self._actions:
            log.info('    - {} linked to {}' \
                     .format(action.event, action.command))

    def configure_engines(self):
        log.debug('Configuring engines')
        for engine in self._engines.values():
            engine.configure(self)

    def start_engines(self):
        log.debug('Starting engines')
        for engine in self._engines.values():
            engine.start()

    def stop_engines(self):
        for engine in self._engines.values():
            engine.stop()

    def configure_output(self, output_name, token_name):
        # Link the specified token to the output
        log.debug('Setting {} to {}'.format(output_name, token_name))
        output = self._outputs[output_name]
        if output._token_name == token_name:
            return

        plugin = self.workbench.get_plugin('psi.token')
        if isinstance(output, ContinuousOutput):
            t = plugin.generate_continuous_token(token_name, output.name,
                                                 output.label)
        else:
            t = plugin.generate_epoch_token(token_name, output.name,
                                            output.label)
        output._token = t
        output._token_name = token_name
        self.context._refresh_items()

    def get_output(self, output_name):
        return self._outputs[output_name]

    def get_epoch_output(self, output_name):
        output = self.get_output(output_name)
        if not isinstance(output, EpochOutput):
            m = 'Epoch output {} does not exist'
            raise AttributeError(m.format(output_name))
        return output

    def prepare_epoch_output(self, output_name, context=None):
        output = self.get_epoch_output(output_name)
        if context is None:
            context = self.context.get_values()
        output.initialize(self, context)

    def start_epoch_output(self, output_name, start=None, delay=0):
        if start is None:
            start = self.get_ts()
        duration = self.get_epoch_output(output_name).start(self, start, delay)
        self.invoke_actions('{}_start'.format(output_name), start+delay)
        self.invoke_actions('{}_end'.format(output_name), start+delay+duration,
                            delayed=True)

    def clear_epoch_output(self, output_name, end=None, delay=0):
        if end is None:
            end = self.get_ts()
        self.get_epoch_output(output_name).clear(self, end, delay)
        self.invoke_actions('{}_end'.format(output_name), end+delay)

    def _get_action_context(self):
        context = {}
        for state in self._states.values():
            context[state.name] = state.active
        for event in self._events.values():
            context[event.name] = event.active
        return context

    def invoke_actions(self, event_name, timestamp=None, delayed=False,
                       cancel_existing=True):
        if cancel_existing:
            self.stop_timer(event_name)
        if delayed:
            delay = timestamp-self.get_ts()
            if delay > 0:
                cb = lambda: self._invoke_actions(event_name, timestamp)
                self.start_timer(event_name, delay, cb)
                return  
        self._invoke_actions(event_name, timestamp)

    def _invoke_actions(self, event_name, timestamp=None):
        if timestamp is not None:
            # The event is logged only if the timestamp is provided
            params = {'event': event_name, 'timestamp': timestamp}
            self.core.invoke_command('psi.data.process_event', params)

        # Load the state of all events so that we can determine which actions
        # should be performed.
        log.debug('Triggering event {}'.format(event_name))
        with self._events[event_name]:
            context = self._get_action_context()

        # TODO: We cannot invoke this inside the with block because it may
        # result in infinite loops if one of the commands calls invoke_actions
        # again. Should we wrap it in a deferred call?
        for action in self._actions:
            if action.match(context):
                m = 'Invoking command {} with parameters {}'
                log.debug(m.format(action.command, action.kwargs))
                
                # Add the event name and timestamp to the parameters passed to
                # the command. 
                kwargs = action.kwargs.copy()
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

    def ai_callback(self, name, data):
        log.trace('Acquired {} samples from {}'.format(data.shape, name))
        parameters = {'name': name, 'data': data}
        self.core.invoke_command('psi.data.process_ai', parameters)

    def et_callback(self, name, edge, timestamp):
        raise NotImplementedError

    def get_ts(self):
        return self._master_engine.get_ts()

    def set_pause_ok(self, value):
        self._pause_ok = value

    def start_timer(self, name, duration, callback):
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
