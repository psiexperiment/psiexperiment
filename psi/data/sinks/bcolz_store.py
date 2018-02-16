import logging
log = logging.getLogger(__name__)

import os.path
import atexit
import tempfile
import shutil

from atom.api import Unicode, Typed, List

import numpy as np
import bcolz

from .abstract_store import (AbstractStore, ContinuousDataChannel,
                             EpochDataChannel)


class BColzStore(AbstractStore):
    '''
    Simple class for storing acquired trial data in a HDF5 file. No analysis or
    further processing is done.
    '''
    base_path = Unicode()
    temp_base_path = Unicode()
    trial_log = Typed(object)
    event_log = Typed(object)

    def _default_temp_base_path(self):
        # Create a temporary folder. Be sure to delete when the program exits.
        base_path = tempfile.mkdtemp()
        atexit.register(shutil.rmtree, base_path)
        return base_path

    def process_trials(self, results):
        names = self.trial_log.data.dtype.names
        columns = []
        for name in names:
            rows = [result[name] for result in results]
            columns.append(rows)
        self.trial_log.append(columns)

    def process_event(self, event, timestamp):
        self.event_log.append([timestamp, event])

    def process_ai_continuous(self, name, data):
        self._stores[name].append(data)

    def process_ai_epochs(self, name, data):
        self._stores[name].append(data)

    def _get_filename(self, name, save=True):
        # Even if we use a memory store, we need to ensure that it has access
        # to the disk (to avoid running out of memory)
        if save and (self.base_path != '<memory>'):
            filename = os.path.join(self.base_path, name)
        else:
            filename = os.path.join(self.temp_base_path, name)
        log.info('Saving %s to %s', name, filename)
        return filename

    def _create_trial_log(self, context_info):
        '''
        Create a table to hold the event log.
        '''
        filename = self._get_filename('trial_log')
        dtype = [(str(n), i.dtype) for n, i in context_info.items()]
        carray = bcolz.zeros(0, rootdir=filename, mode='w', dtype=dtype)
        atexit.register(carray.flush)
        return carray

    def _create_event_log(self):
        '''
        Create a table to hold the event log.
        '''
        filename = self._get_filename('event_log')
        dtype = [('timestamp', 'float32'), ('event', 'S512')]
        carray = bcolz.zeros(0, rootdir=filename, mode='w', dtype=dtype)
        atexit.register(carray.flush)
        return carray

    def create_ai_continuous(self, name, fs, dtype, save, **metadata):
        n = int(fs*60*60)
        filename = self._get_filename(name, save)
        carray = bcolz.carray([], rootdir=filename, mode='w', dtype=dtype,
                              expectedlen=n)
        carray.attrs['fs'] = fs
        for key, value in metadata.items():
            carray.attrs[key] = value
        self._stores[name] = ContinuousDataChannel(data=carray, fs=fs)
        atexit.register(carray.flush)

    def create_ai_epochs(self, name, fs, epoch_size, dtype, save, **metadata):
        filename = self._get_filename(name, save)
        epoch_samples = int(fs*epoch_size)
        base = np.empty((0, epoch_samples))
        carray = bcolz.carray(base, rootdir=filename, mode='w', dtype=dtype)
        carray.attrs['fs'] = fs
        for key, value in metadata.items():
            carray.attrs[key] = value
        self._stores[name] = EpochDataChannel(data=carray, fs=fs,
                                              epoch_size=epoch_size)
        atexit.register(carray.flush)

    def finalize(self, workbench):
        if self.base_path != '<memory>':
            # Save the settings file
            cmd = 'psi.save_preferences'
            filename = os.path.join(self.base_path, 'final')
            params = {'filename': filename}
            core = workbench.get_plugin('enaml.workbench.core')
            core.invoke_command(cmd, params)

    def create_table(self, name, dataframe):
        filename = self._get_filename(name + '.csv')
        if os.path.exists(filename):
            raise IOError('{} already exists'.format(filename))
        dataframe.to_csv(filename)

    def set_base_path(self, base_path):
        self.base_path = base_path
