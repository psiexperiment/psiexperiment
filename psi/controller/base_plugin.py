import logging
log = logging.getLogger(__name__)

from functools import partial

import numpy as np

from atom.api import Enum, Bool, Typed
from enaml.workbench.plugin import Plugin

from .channel import Channel
from .engine import Engine
from .experiment_action import ExperimentAction
from .output import ContinuousOutput
from ..token import get_token_manifest


IO_POINT = 'psi.controller.io'
ACTION_POINT = 'psi.controller.actions'

def get_named_inputs(input):
    named_inputs = []
    for child in input.children:
        named_inputs.extend(get_named_inputs(child))
    if input.name:
        named_inputs.append(input)
    return named_inputs


class BaseController(Plugin):

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

    # Available engines
    _engines = Typed(dict, {})

    # Available outputs
    _outputs = Typed(dict, {})

    # Available inputs
    _inputs = Typed(dict, {})

    # This determines which engine is responsible for the clock
    _master_engine = Typed(Engine)

    # TODO: Define action groups to minimize errors.
    _actions = Typed(dict, {})

    def start(self):
        self.core = self.workbench.get_plugin('enaml.workbench.core')
        self.context = self.workbench.get_plugin('psi.context')
        self.data = self.workbench.get_plugin('psi.data')
        self._refresh_io()
        self._refresh_actions()
        self._bind_observers()

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

    def _refresh_io(self):
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
                    for output in getattr(channel, 'outputs', []):
                        outputs[output.name] = output
                    for all_inputs in getattr(channel, 'inputs', []):
                        for input in get_named_inputs(all_inputs):
                            inputs[input.name] = input

        self._master_engine = master_engine
        self._engines = engines
        self._outputs = outputs
        self._inputs = inputs

    def _refresh_actions(self, event=None):
        actions = {}
        point = self.workbench.get_extension_point(ACTION_POINT)
        for extension in point.extensions:
            for action in extension.get_children(ExperimentAction):
                subgroup = actions.setdefault(action.event, [])
                subgroup.append(action)
        self._actions = actions

    def start_engines(self):
        for engine in self._engines.values():
            engine.configure(self)
        for engine in self._engines.values():
            engine.start()

    def stop_engines(self):
        for engine in self._engines.values():
            engine.stop()

    def configure_output(self, output_name, token_name):
        log.debug('Setting {} to {}'.format(output_name, token_name))
        output = self._outputs[output_name]
        if output._token_name == token_name:
            return
        if output._plugin:
            self.workbench.unregister(output._plugin_id)
        manifest_description = get_token_manifest(token_name)
        if isinstance(output, ContinuousOutput):
            scope = 'experiment'
        else:
            scope = 'trial'
        manifest = manifest_description(output.name, scope=scope,
                                        label_base=output.label)
        self.workbench.register(manifest)
        output._plugin_id = manifest.id
        output._plugin = self.workbench.get_plugin(manifest.id)
        output._token_name = token_name

    def get_output(self, output_name):
        return self._outputs[output_name]

    def invoke_actions(self, event, timestamp):
        params = {'event': event, 'timestamp': timestamp}
        self.core.invoke_command('psi.data.process_event', params)

        log.debug('Invoking actions for {}'.format(event))
        for action in self._actions.get(event, []):
            log.debug('Invoking command {}'.format(action.command))
            self.core.invoke_command(action.command)

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

    def start_experiment(self):
        raise NotImplementedError

    def stop_experiment(self):
        raise NotImplementedError

    def pause_experiment(self):
        raise NotImplementedError

    def start_trial(self):
        raise NotImplementedError

    def end_trial(self):
        raise NotImplementedError

    def ao_callback(self, name, data):
        raise NotImplementedError

    def ai_callback(self, name, data):
        raise NotImplementedError

    def et_callback(self, name, edge, timestamp):
        raise NotImplementedError

    def get_ts(self):
        return self._master_engine.get_ts()
