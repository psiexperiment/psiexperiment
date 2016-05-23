from atom.api import Enum, Bool, Typed, Property
from enaml.workbench.plugin import Plugin

from .output import Output


OUTPUT_POINT = 'psi.controller.output'

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

    _outputs = Typed(list, [])
    _engines = Typed(dict, {})

    def start(self):
        self.core = self.workbench.get_plugin('enaml.workbench.core')
        self.context = self.workbench.get_plugin('psi.context')
        self.core.invoke_command('psi.data.prepare')
        self._refresh_outputs()
        self._bind_observers()

    def stop(self):
        self._unbind_observers()

    def _bind_observers(self):
        self.workbench.get_extension_point(OUTPUT_POINT) \
            .observe('extensions', self._refresh_outputs)

    def _unbind_observers(self):
        self.workbench.get_extension_point(OUTPUT_POINT) \
            .unobserve('extensions', self._refresh_outputs)

    def _refresh_outputs(self):
        outputs = []
        point = self.workbench.get_extension_point(OUTPUT_POINT)
        for extension in point.extensions:
            for output in extension.get_children(Output):
                outputs.append(output)
        self._outputs = outputs

    def _refresh_engines(self):
        engines = {}
        point = self.workbench.get_extension_point(ENGINE_POINT)
        for extension in point.extensions:
            for engine in selector.get_children(Engine):
                engines[engine.name] = engine
        self._engines = engines

    def configure_output(self, output, manifest_description):
        if output._plugin_id:
            self.workbench.unregister(output._plugin_id)
        manifest = manifest_description(output.name, label_base=output.label,
                                        scope=output.scope)
        output._plugin_id = manifest.id
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

    def start_experiment(self):
        raise NotImplementedError

    def stop_experiment(self):
        self.state = 'stopped'

    def start_trial(self):
        raise NotImplementedError

    def end_trial(self):
        raise NotImplementedError
