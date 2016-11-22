import logging
log = logging.getLogger(__name__)

from functools import partial
import operator as op

import numpy as np

from atom.api import Enum, Bool, Typed
from enaml.application import timed_call
from enaml.workbench.plugin import Plugin

from .channel import Channel, AOChannel
from .engine import Engine
from .output import Output
from .input import Input
from .experiment_action import ExperimentAction, ExperimentEvent
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

    # This determines which engine is responsible for the clock
    _master_engine = Typed(Engine)

    # List of events and actions that can be associated with the event
    _events = Typed(dict, {})
    _actions = Typed(dict, {})

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
        master_engine = None

        point = self.workbench.get_extension_point(IO_POINT)
        for extension in point.extensions:
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
            for input in extension.get_children(Input):
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
                            .format(output.name, output.target_name))
                input.source = source

        # Remove channels that do not have an input or output defined.
        for channel in channels.values():
            if not channel.children:
                channel.engine = None
                del channels[channel.name]

        # Hack for channels that don't define a continuous output
        for channel in channels.values():
            if isinstance(channel, AOChannel):
                for o in channel.outputs:
                    if isinstance(o, ContinuousOutput):
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

        log.debug('Available inputs: {}'.format(inputs.keys()))
        log.debug('Available outputs: {}'.format(outputs.keys()))

    def _refresh_actions(self, event=None):
        actions = {}
        events = {}
        point = self.workbench.get_extension_point(ACTION_POINT)
        for extension in point.extensions:
            for event in extension.get_children(ExperimentEvent):
                if event in actions:
                    raise ValueError('{} already exists'.format(event.name))
                events[event.name] = event
                actions[event.name] = []

        for extension in point.extensions:
            for action in extension.get_children(ExperimentAction):
                try:
                    actions[action.event].append(action)
                except KeyError:
                    m = 'Unknown event {} for {}'
                    raise ValueError(m.format(action.event, action.command))

        # Sort based on weight
        for action_set in actions.values():
            action_set.sort(key=lambda a: a.weight)

        self._events = events
        self._actions = actions

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

    def start_epoch_output(self, output_name, start=None, delay=0):
        output = self.get_output(output_name)
        if not isinstance(output, EpochOutput):
            raise ValueError('This only works for epoch outputs')
        if start is None:
            start = self.get_ts()
        output.start(self, start, delay)

    def invoke_actions(self, event, timestamp=None, delay=False):
        if delay:
            delay_ms = (timestamp-self.get_ts())*1e3
            if delay_ms > 0:
                timed_call(delay_ms, self.invoke_actions, event, timestamp)
                return
            else:
                log.warning('Not delaying {} at {}'.format(event, timestamp))

        if timestamp is not None:
            params = {'event': event, 'timestamp': timestamp}
            self.core.invoke_command('psi.data.process_event', params)

        log.debug('Invoking actions for {}'.format(event))
        for action in self._actions.get(event, []):
            m = 'Invoking command {} with parameters {}'
            log.debug(m.format(action.command, action.kwargs))
            self.core.invoke_command(action.command, parameters=action.kwargs)

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

    def initialize_experiment(self):
        self.invoke_actions('experiment_initialize')

    def start_experiment(self):
        self.invoke_actions('experiment_start')
        self.experiment_state = 'running'

    def stop_experiment(self):
        self.invoke_actions('experiment_stop', self.get_ts())
        self.experiment_state = 'stopped'

    def pause_experiment(self):
        raise NotImplementedError

    def start_trial(self):
        raise NotImplementedError

    def end_trial(self):
        raise NotImplementedError

    def ao_callback(self, name):
        log.debug('Updating output {}'.format(name))
        self._outputs[name].update(self)

    def ai_callback(self, name, data):
        log.trace('Acquired {} samples from {}'.format(data.shape, name))
        parameters = {'name': name, 'data': data}
        self.core.invoke_command('psi.data.process_ai', parameters)

    #def et_callback(self, name, edge, timestamp):
    #    log.debug('Detected {} on {} at {}'.format(edge, name, timestamp))
    #    parameters = {'name': name, 'edge': edge, 'timestamp': timestamp}
    #    # TODO: add callback to save et
    #    #self.core.invoke_command('psi.data.process_et', parameters)

    def get_ts(self):
        return self._master_engine.get_ts()

    def set_pause_ok(self, value):
        self._pause_ok = value
