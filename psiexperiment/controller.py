import types

from atom.api import Typed, Bool, Atom, ContainerList, Dict, Enum, observe

from .parameter import Parameter
from .selector import BaseSelector
from .expression import Expr, ExpressionNamespace


class Controller(Atom):

    # Tracks the state of the controller.
    state = Enum('initialized', 'running', 'paused', 'stopped')

    # Provided on class initialization.
    symbols = Dict()
    parameters = ContainerList(Typed(Parameter))
    selectors = Dict()

    # Internal state.
    _expressions = Dict()
    _sequences = Typed(type({}))
    _namespace = Typed(ExpressionNamespace, ())
    _prior_values = ContainerList()

    # We should not respond to changes during the course of a trial. These flags
    # indicate changes or requests from the user are pending and should be
    # processed accordingly.
    _apply_requested = Bool(True)
    _remind_requested = Bool(False)
    _changes_pending = Bool(False)

    def get_parameter(self, name):
        '''
        Get parameter by name
        '''
        for p in self.parameters:
            if p.name == name:
                return p
        raise KeyError('No parameter %s', name)

    def _observe_parameters(self, event):
        # When the list of parameters changes, ensure that we subscribe to
        # changes in the 'rove' attribute (so we can add/remove the parameter
        # from the selector).
        # TODO - this seems like a great candidate for a Workbench plugin
        # system.
        for p in event['value']:
            p.observe('rove', self._observe_parameters_rove)
            self._update_selectors(p)
            p.observe('expression', self._observe_parameters_expression)

    def _observe_selectors(self, event):
        if event['type'] == 'update':
            map(self._update_selectors, self.parameters)
            for selector in self.selectors.values():
                selector.observe('updated', self._observe_selectors_updated)

    def _observe_parameters_rove(self, event):
        self._update_selectors(event['object'])
        if self.state != 'initialized':
            self._changes_pending = True

    def _observe_parameters_expression(self, event):
        if self.state != 'initialized':
            self._changes_pending = True

    def _observe_selectors_updated(self, event):
        if self.state != 'initialized':
            self._changes_pending = True

    def _update_selectors(self, parameter):
        if parameter.rove:
            for selector in self.selectors.values():
                if parameter not in selector.parameters:
                    selector.append_parameter(parameter)
        else:
            for selector in self.selectors.values():
                if parameter in selector.parameters:
                    selector.remove_parameter(parameter)

    def request_apply(self):
        self._apply_requested = True

    def initialize_values(self):
        if self._apply_requested:
            self._expressions = dict((p.name, Expr(p.expression)) \
                                     for p in self.parameters if not p.rove)
            self._sequences = dict((k, v.get_iterator()) \
                                    for k, v in self.selectors.items())
            self._apply_requested = False
            self._changes_pending = False
        self._namespace = ExpressionNamespace(self._expressions, self.symbols)

    def value_changed(self, parameter):
        '''
        True if the value has changed from the previous trial
        '''
        current = self.get_value(parameter)
        prior = self.get_value(parameter, -1)
        return current != prior

    def get_value(self, parameter, trial=None):
        if trial is not None:
            try:
                return self._prior_values[trial][parameter]
            except IndexError:
                return None
        return self._namespace.get_value(parameter)

    def get_values(self, trial=None):
        if trial is not None:
            return self._prior_values[trial]
        return self._namespace.get_values()

    def set_value(self, parameter, value):
        self._namespace.set_value(parameter.name, value)

    def set_values(self, values):
        self._namespace.set_values(values)

    def apply_requested(self):
        self._apply_requested = True

    def start_experiment(self, initial_ttype='default'):
        self.initialize_values()
        self.state = 'running'
        self.next_trial(initial_ttype)

    def pause_experiment(self):
        self.state = 'paused'

    def resume_experiment(self):
        self.state = 'running'

    def stop_experiment(self):
        self.state = 'stopped'

    def next_trial(self, ttype='default'):
        self.initialize_values()
        self.set_values(self._sequences[ttype].next())
        self.get_values()

    def save_trial(self, **data):
        values = self.get_values()
        self._prior_values.append(values)
        data.update(values)
        # save to file


def main():
    import enaml
    from enaml.qt.qt_application import QtApplication
    from parameter import Parameter
    from selector import SequenceSelector, SingleSetting
    import choice
    import numpy as np

    trials = Parameter(name='trials', expression='80', dtype= np.int,
                       label='Trials', rove=True)
    level = Parameter( name='level', expression='np.random.randint(80)',
                      dtype=np.float, label='Level (dB SPL)')
    center_frequency = Parameter(name='center_frequency',
                                 expression='32e3/trials', dtype=np.float,
                                 label='Center freq. (Hz)')

    selectors = {'nogo': SingleSetting(), 'go': SequenceSelector(),
                 'remind': SingleSetting()}
    controller = Controller(parameters=[trials, level, center_frequency],
                            selectors=selectors, symbols={'np': np})
    controller.selectors['go'].add_setting({'trials': 60})
    controller.selectors['go'].add_setting({'trials': 40})
    controller.selectors['go'].add_setting({'trials': 20})
    controller.selectors['go'].add_setting({'trials': 10})
    controller.selectors['go'].order = choice.descending
    controller.start_experiment('go')

    with enaml.imports():
        from experiment_view import ExperimentView
    app = QtApplication()
    view = ExperimentView(controller=controller)
    view.show()
    app.start()

if __name__ == '__main__':
    main()
