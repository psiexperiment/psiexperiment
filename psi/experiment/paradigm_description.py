from psi.core.enaml.api import load_manifest


class ParadigmManager:

    def __init__(self):
        self.paradigms = {}
        self.broken_paradigms = {}

    def register(self, paradigm, exception=None):
        self.paradigms[paradigm.name] = paradigm
        if exception is not None:
            raise
            #self.paradigm_errors[paradigm.name] = exception

    def available_paradigms(self):
        return list(self.paradigms.keys())

    def iter_paradigms(self, experiment_type=None):
        for paradigm in self.paradigms.values():
            if experiment_type is None:
                yield paradigm
            elif paradigm.experiment_type == experiment_type:
                yield paradigm

    def list_paradigms(self, experiment_type=None):
        return list(self.iter_paradigms(experiment_type))

    def get_paradigm(self, name):
        return self.paradigms[name]


class PluginDescription:

    def __init__(self, manifest, selected=False, required=None, name=None, title=None):
        manifest = load_manifest(manifest)()
        self.manifest = manifest
        self.selected = selected

        # Default values are loaded directly from the PluginManifest if the
        # provided value is None.
        self.setdefault('required', required)
        self.setdefault('name', name)
        self.setdefault('title', title)

    def setdefault(self, attr, value):
        if value is not None:
            setattr(self, attr, value)
        else:
            setattr(self, attr, getattr(self.manifest, attr))


class ParadigmDescription:

    def __init__(self, name, title, experiment_type, plugin_info):
        '''
        Parameters
        ----------
        name : str
            Simple name that will be used to identify experiment. Must be
            globally unique. Will often be used at the command line to start
            the experment (e.g., `psi name`).
        title : str
            Title to show in main window of the experiment as well as in any
            user experience where the user is asked to select from a list of
            experiments.
        experiment_type : {'ear', 'animal', 'cohort', 'calibration', str}
            Type of experiment. This is mainly used to organize the list of
            available experments in different user interfaces.
        '''
        self.name = name
        self.title = title
        self.experiment_type = experiment_type

        global paradigm_manager
        try:
            self.plugins = [PluginDescription(*pi) for pi in plugin_info]
            paradigm_manager.register(self)
        except Exception as exc:
            paradigm_manager.register(self, exc)

    def enable_plugin(self, plugin_name):
        for p in self.plugins:
            if p.name == plugin_name:
                p.selected = True

    def disable_plugin(self, plugin_name):
        for p in self.plugins:
            if p.name == plugin_name:
                p.selected = False

    def disable_all_plugins(self):
        for p in self.plugins:
            p.selected = False


paradigm_manager = ParadigmManager()
