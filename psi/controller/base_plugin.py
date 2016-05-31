from atom.api import Enum, Bool, Typed, Property
from enaml.workbench.plugin import Plugin

from .channel import Channel
from .engine import Engine
from ..token import get_token_manifest


CHANNEL_POINT = 'psi.controller.channels'
ENGINE_POINT = 'psi.controller.engines'


class BaseController(Plugin):

    # Tracks the state of the controller.
    state = Enum('initialized', 'running', 'paused', 'stopped')
    running = Property()

    # Provides direct access to plugins rather than going through the core
    # command system. Right now the context plugin is so fundamentally important
    # to the controller that it would be cumbersome to use the core command
    # system.
    core = Typed(Plugin)
    context = Typed(Plugin)

    # We should not respond to changes during the course of a trial. These flags
    # indicate changes or requests from the user are pending and should be
    # processed when the opportunity arisese (e.g., at the end of the trial).
    _apply_requested = Bool(False)
    _remind_requested = Bool(False)
    _pause_requested = Bool(False)

    _engines = Typed(dict, {})
    _channels = Typed(dict, {})
    _outputs = Typed(dict, {})

    def start(self):
        self.core = self.workbench.get_plugin('enaml.workbench.core')
        self.context = self.workbench.get_plugin('psi.context')
        self.core.invoke_command('psi.data.prepare')
        self._refresh_engines()
        self._refresh_channels()
        self._bind_observers()

    def stop(self):
        self._unbind_observers()

    def _bind_observers(self):
        self.workbench.get_extension_point(CHANNEL_POINT) \
            .observe('extensions', self._refresh_channels)
        self.workbench.get_extension_point(ENGINE_POINT) \
            .observe('extensions', self._refresh_engines)

    def _unbind_observers(self):
        self.workbench.get_extension_point(CHANNEL_POINT) \
            .unobserve('extensions', self._refresh_channels)
        self.workbench.get_extension_point(ENGINE_POINT) \
            .unobserve('extensions', self._refresh_engines)

    def _refresh_channels(self):
        point = self.workbench.get_extension_point(CHANNEL_POINT)
        channels = {}
        outputs = {}
        for extension in point.extensions:
            for channel in extension.get_children(Channel):
                engine_config = channels.setdefault(channel.engine, {})
                engine_config.setdefault(channel.io_type, []).append(channel)
                if channel.io_type == 'hw_ao':
                    outputs[channel.name] = channel
        self._channels = channels
        self._outputs = outputs

    def _refresh_engines(self):
        engines = {}
        point = self.workbench.get_extension_point(ENGINE_POINT)
        for extension in point.extensions:
            for engine in extension.get_children(Engine):
                engines[engine.name] = engine
        self._engines = engines

    def configure_engines(self):
        for engine_name, engine_config in self._channels.items():
            engine = self._engines[engine_name]
            engine.configure(engine_config)

    def configure_output(self, output_name, token_name):
        output = self._outputs[output_name]
        if output._token_name == token_name:
            return
        if output._plugin_id:
            self.workbench.unregister(output._plugin_id)
        manifest_description = get_token_manifest(token_name)
        scope = 'experiment' if output.mode == 'continuous' else 'trial'
        manifest = manifest_description(output.name, scope=scope,
                                        label_base=output.label)
        output._plugin_id = manifest.id
        output._token_name = token_name
        self.workbench.register(manifest)

    def request_apply(self):
        self._apply_requested = True

    def request_remind(self):
        self._remind_requested = True

    def request_pause(self):
        self._pause_requested = True

    def request_resume(self):
        self.state == 'running'
        self.start_trial()

    def start_continuous_outputs(self):
        for output in self._outputs:
            if isinstance(output, Continuous):
                output.start()

    def get_epoch_waveforms(self):
        for output in self._outputs.values():
            if isinstance(output, Epoch):
                output.prepare(self.workbench)

    def start_experiment(self):
        raise NotImplementedError

    def stop_experiment(self):
        self.state = 'stopped'

    def start_trial(self):
        raise NotImplementedError

    def end_trial(self):
        raise NotImplementedError
