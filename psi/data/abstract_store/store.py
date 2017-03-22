import logging
log = logging.getLogger(__name__)

from atom.api import Bool, Typed
from ..plugin import Sink

from .data_source import DataTable, EventDataTable, DataChannel


class AbstractStore(Sink):

    _channels = Typed(dict)

    def prepare(self, plugin):
        self._prepare_event_log()
        self._prepare_trial_log(plugin.context_info)
        self._prepare_inputs(plugin.inputs.values())
        # TODO: This seems a bit hackish. Do we really need this?
        self._channels['trial_log'] = self.trial_log
        self._channels['event_log'] = self.event_log

    def get_source(self, source_name):
        try:
            return self._channels[source_name]
        except KeyError:
            # TODO: Once we port to Python 3, add exception chaining.
            raise AttributeError

    def set_current_time(self, name, timestamp):
        self._channels[name].set_current_time(timestamp)

    def _prepare_trial_log(self, context_info):
        data = self._create_trial_log(context_info)
        self.trial_log = DataTable(data=data)

    def _prepare_event_log(self):
        data = self._create_event_log()
        self.event_log = EventDataTable(data=data)

    def _prepare_inputs(self, inputs):
        channels = {}
        for input in inputs:
            log.debug('Preparing file for input {}'.format(input.name))
            create_function_name = '_create_{}_input'.format(input.mode)
            if not hasattr(self, create_function_name):
                m = 'No method for createaring datasource {} of type {}'
                log.debug(m.format(input.name, input.mode))
            else:
                create_function = getattr(self, create_function_name)
                data = create_function(input)
                channels[input.name] = DataChannel(data=data, fs=input.fs)

        self._channels = channels
