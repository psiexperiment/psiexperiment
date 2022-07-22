from psi.core.enaml.api import load_manifest


class ParadigmManager:
    '''
    Core class for managing experiment paradigms available to psiexperiment
    '''
    def __init__(self):
        self.paradigms = {}
        self.broken_paradigms = {}

    def register(self, paradigm, exception=None):
        self.paradigms[paradigm.name] = paradigm
        if exception is not None:
            raise

    def list_paradigms(self, experiment_type=None):
        '''
        Iterate through available experiment paradigms

        Parameters
        ----------
        experiment_type : {None, str}
            If None, yield all paradigms. If experiment_type is
            specified, yield only paradigms matching that experiment type.

        Returns
        -------
        iterator
            Iterator over set of matching paradigm descriptions
        '''
        matches = []
        for paradigm in self.paradigms.values():
            if experiment_type is None:
                matches.append(paradigm)
            elif paradigm.experiment_type == experiment_type:
                matches.append(paradigm)
        return matches

    def list_paradigm_names(self, experiment_type=None):
        matches = self.list_paradigms()
        return sorted(m.name for m in matches)

    def get_paradigm(self, name):
        '''
        Return definition for experiment paradigm

        Parameters
        ----------
        name : str
            Name of experiment paradigm to return

        Returns
        -------
        description
            Instance of ParadigmDescription for the experiment paradigm
        '''
        return self.paradigms[name]


class PluginDescription:

    def __init__(self, manifest, selected=False, required=None, name=None,
                 title=None, attrs=None):
        if attrs is None:
            attrs = {}
        self.manifest = load_manifest(manifest)(**attrs)
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
        plugin_info : list
            List of tuples containing information about the plugins that are
            available for this particular paradigm.
        '''
        self.name = name
        self.title = title
        self.experiment_type = experiment_type

        global paradigm_manager
        try:
            self.plugins = [PluginDescription(**d) for d in plugin_info]
            paradigm_manager.register(self)
        except Exception as exc:
            print(plugin_info)
            paradigm_manager.register(self, exc)

    def enable_plugin(self, plugin_name):
        for p in self.plugins:
            if p.name == plugin_name:
                p.selected = True
                break
        else:
            choices = ', '.join(p.name for p in self.plugins)
            raise ValueError(f'Plugin {plugin_name} not found. ' \
                             f'Valid plugins are {choices}.')

    def disable_plugin(self, plugin_name):
        for p in self.plugins:
            if p.name == plugin_name:
                p.selected = False

    def disable_all_plugins(self):
        for p in self.plugins:
            p.selected = False


paradigm_manager = ParadigmManager()
