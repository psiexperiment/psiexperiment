import logging
log = logging.getLogger(__name__)

from pathlib import Path

from atom.api import Atom, Dict, List, Str, Value

from psi.experiment.api import paradigm_manager, ParadigmNotFound


class Experiment(Atom):
    '''
    One row in a launcher: a paradigm plus the preferences file and optional
    plugins to run it with.

    Subclasses add their own per-row settings as Atom members. Anything a
    subclass needs to persist (or carry over into `freeze`) should be added to
    `__getstate__` and accepted as a keyword by `__init__`; `freeze` rebuilds
    the frozen copy from `__getstate__`, so such members carry over without
    `freeze` having to be overridden.
    '''
    #: Class to create in `freeze`. A subclass that also subclasses
    #: `FrozenExperiment` should point this at its own frozen class. When None,
    #: an `Experiment` freezes to `FrozenExperiment` and a `FrozenExperiment`
    #: to its own class.
    frozen_class = None

    paradigm = Value()

    preference = Str()

    #: List of available preferences. Primarily used for updating the GUI.
    preferences = List()

    #: Plugins selected for load
    plugins = List(Str())

    #: Supplemental note to append based on the button clicked.
    mode_notes = Dict()

    def iter_selectable_plugins(self):
        for plugin in self.paradigm.plugins:
            if plugin.required:
                continue
            if plugin.info.get('hide', False):
                continue
            yield plugin

    def update_preferences(self):
        # This will force a change notification in the Enaml ObjectCombo,
        # thereby refreshing the list of options.
        self.preferences = self.paradigm.list_preferences().copy()

    def __init__(self, paradigm, plugins=None, preference=None, **kwargs):
        if isinstance(paradigm, str):
            paradigm = paradigm_manager.get_paradigm(paradigm)
        self.paradigm = paradigm
        self.update_preferences()

        # Make sure the plugins saved to the config file are valid plugins (we
        # sometimes remove or rename plugins). If the plugin is no longer
        # valid, remove it. If the plugin is required, remove it as a plugin
        # that the user can select from in the GUI (since it automatically gets
        # loaded).
        plugins = set() if plugins is None else set(plugins)
        valid_plugins = set(p.id for p in self.iter_selectable_plugins())
        plugins = list(plugins & valid_plugins)

        # We only save preference name, not the full path to the preference. We
        # need to restore the full path to the preference by scanning through
        # the list of avaialble preferences. This allows for portability across
        # systems.
        if not preference:
            preference = ''
        else:
            preference = Path(preference)
            for valid_preference in self.preferences:
                if valid_preference.stem == preference.stem:
                    preference = str(valid_preference)
                    break
            else:
                log.warning('Invalid preference requested for %s: %s', paradigm,
                            preference)
                preference = ''

        super().__init__(paradigm=paradigm, plugins=plugins,
                         preference=preference, **kwargs)

    def __getstate__(self):
        state = super().__getstate__()
        # Convert some keys to things that can be JSON-serialized. Don't save
        # the full path to the preference because we want this to be portable
        # across environments and installs. The list of available preferences
        # is rebuilt on load.
        state['preference'] = Path(state['preference']).name
        state['paradigm'] = state['paradigm'].name
        state.pop('preferences', None)
        return state

    @property
    def modes(self):
        '''
        Modes the paradigm can be run under (e.g., 'run', 'ipsi', 'contra').
        '''
        return self.paradigm.info.get('modes', ['Run'])

    def freeze(self, mode=None, **kwargs):
        '''
        Return a copy of this experiment with the mode fixed. Used in running
        sequences.

        Parameters
        ----------
        mode : str, optional
            One of the modes the paradigm can be run under (e.g., 'run',
            'ipsi', 'contra'). Defaults to the first mode.
        **kwargs
            Additional members of `frozen_class` to set (e.g., a subclass that
            also fixes the ear being tested).
        '''
        if mode is None:
            mode = self.modes[0]
        frozen_class = self.frozen_class
        if frozen_class is None:
            # Refreezing a frozen experiment keeps its own class.
            frozen_class = type(self) if isinstance(self, FrozenExperiment) \
                else FrozenExperiment
        state = self.__getstate__()
        # The state holds only the paradigm's name (for saving); hand over the
        # paradigm itself rather than looking it up again.
        state['paradigm'] = self.paradigm
        state.update(kwargs, mode=mode)
        return frozen_class(**state)


class FrozenExperiment(Experiment):
    '''
    Subclass of Experiment in which we have frozen the mode (used for
    sequences).
    '''
    mode = Str()

    def freeze(self, mode=None, **kwargs):
        if mode is None:
            mode = self.mode
        return super().freeze(mode, **kwargs)


def load_experiments(seq, experiment_class=Experiment):
    '''
    Helper function for loading experiments from JSON file
    '''
    experiments = []
    for s in seq:
        s = dict(s)
        # Remove obsolete label argument from saved paradigms and
        # legacy modes since we changed how this is handled (it was
        # always a hack to put in the file).
        s.pop('label', None)
        s.pop('modes', None)
        s.pop('preferences', None)
        mode_notes = s.get('mode_notes', {})
        s['mode_notes'] = {k.lower(): v for k, v in mode_notes.items()}
        try:
            experiments.append(experiment_class(**s))
        except ParadigmNotFound:
            log.warning('Skipping unknown paradigm %s', s.get('paradigm'))
    return experiments
