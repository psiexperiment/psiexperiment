import logging
log = logging.getLogger(__name__)

import os.path

from atom.api import Unicode, Typed

import numpy as np
import bcolz

from psi.util import get_tagged_values
from .abstract_store.store import AbstractStore


class BColzStore(AbstractStore):
    '''
    Simple class for storing acquired trial data in a HDF5 file. No analysis or
    further processing is done.
    '''
    base_path = Unicode()

    trial_log = Typed(object)
    event_log = Typed(object)

    def process_trials(self, results):
        names = self.trial_log.data.dtype.names
        rows = [r[n] for r in results for n in names]
        self.trial_log.append(rows)

    def process_event(self, event, timestamp):
        self.event_log.append([timestamp, event])

    def process_ai_continuous(self, name, data):
        if self._channels[name] is not None:
            self._channels[name].append(data)

    def process_ai_epochs(self, name, data):
        epochs = [d['epoch'] for d in data]
        if self._channels[name] is not None:
            self._channels[name].append(epochs)

    def _get_filename(self, name):
        if self.base_path != '<memory>':
            return os.path.join(self.base_path, name)
        else:
            return None

    def _create_trial_log(self, context_info):
        '''
        Create a table to hold the event log.
        '''
        filename = self._get_filename('trial_log')
        dtype = [(str(n), i.dtype) for n, i in context_info.items()]
        return bcolz.zeros(0, rootdir=filename, mode='w', dtype=dtype)

    def _create_event_log(self):
        '''
        Create a table to hold the event log.
        '''
        filename = self._get_filename('event_log')
        dtype = [('timestamp', 'float32'), ('event', 'S512')]
        return bcolz.zeros(0, rootdir=filename, mode='w', dtype=dtype)

    def _create_continuous_input(self, input):
        n = int(input.fs*60*60)
        filename = self._get_filename(input.name)
        carray = bcolz.carray([], rootdir=filename, mode='w',
                              dtype=input.channel.dtype, expectedlen=n)

        # Copy some attribute metadata over
        values = get_tagged_values(input, 'metadata')
        for name, value in values.items():
            carray.attrs[name] = value

        values = get_tagged_values(input.channel, 'metadata')
        for name, value in values.items():
            carray.attrs['channel_' + name] = value

        values = get_tagged_values(input.engine, 'metadata')
        for name, value in values.items():
            carray.attrs['engine_' + name] = value

        return carray

    def _create_epochs_input(self, input):
        filename = self._get_filename(input.name)
        epoch_samples = int(input.fs*input.epoch_size)
        base = np.empty((0, epoch_samples))
        carray = bcolz.carray(base, rootdir=filename, mode='w',
                              dtype=input.channel.dtype)
        return carray

    def finalize(self, workbench):
        log.debug('Flushing all data to disk')
        for channel in self._channels.values():
            channel.data.flush()
        if self.base_path != '<memory>':
            cmd = 'psi.save_preferences'
            filename = os.path.join(self.base_path, 'final')
            params = {'filename': filename}
            core = workbench.get_plugin('enaml.workbench.core')
            core.invoke_command(cmd, params)

    def set_base_path(self, base_path):
        self.base_path = base_path
