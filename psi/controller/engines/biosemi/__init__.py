import logging
log = logging.getLogger(__name__)

import configparser

from atom.api import Bool, Dict, Float, Int, List, Property, set_default, Str, Typed, Value
from enaml.core.api import d_

from threading import Event, Thread

from pyactivetwo import ActiveTwoClient
from psi.controller.api import (Engine, HardwareAIChannel)

from .channel_maps import map_64


import time

################################################################################
# Utility functions
################################################################################
def load_cfg(filename):
    '''
    Load ActiView config file
    '''
    config = configparser.ConfigParser()
    config.read(filename)

    channel_map = {}
    for key, value in config['Labels'].items():
        if key.startswith('chan'):
            # This will map to the index
            channel = int(key[4:]) - 1
            channel_map[channel] = value
    return {
        'channel_map': channel_map
    }

################################################################################
# Core classes
################################################################################
class BiosemiThread(Thread):

    def __init__(self, client, poll_interval, callback):
        log.debug('Initializing acquisition thread')
        super().__init__()
        self.client = client
        self.poll_interval = poll_interval
        self.callback = callback
        self.stop_requested = Event()

    def read(self):
        return self.client.read(duration=self.poll_interval)

    def run(self):
        log.debug('Starting acquisition thread')
        self.client.connect()
        while not self.stop_requested.wait(0):
            data = self.read()
            self.callback(data)
        self.client.disconnect()
        log.debug('Exiting acquistion thread')


class BaseBiosemiChannel(HardwareAIChannel):
    pass


class BiosemiEEGChannels(BaseBiosemiChannel):
    n_channels = d_(Int(8))
    name = 'biosemi_eeg'


class BiosemiSensorChannels(BaseBiosemiChannel):
    n_channels = d_(Int(7))
    name = 'biosemi_sensors'

    def _default_channel_labels(self):
        return ['GSR1', 'GSR2', 'Erg1', 'Erg2', 'Resp', 'Plet', 'Temp']


class BiosemiTriggerChannel(BaseBiosemiChannel):
    name = set_default('biosemi_trigger')


class BiosemiEngine(Engine):
    name = set_default('biosemi')

    host_ip = Str('127.0.0.1')
    host_port = Int(8888)

    monitor_period = Float(0.25)
    fs = Int(1024)

    _client = Typed(ActiveTwoClient)
    _thread = Value()
    _callbacks = Dict()
    _channels = List()

    eeg_channels = d_(Int(8))
    trigger_included = d_(Bool(True))
    sensors_included = d_(Bool(True))
    ex_included = d_(Bool(True))

    total_eeg_channels = Property()

    _samples_acquired = Int(0)

    def _get_total_eeg_channels(self):
        eeg_channels = self.eeg_channels
        if self.ex_included:
            eeg_channels += 8
        return eeg_channels

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._callbacks = {}
        self._channels = []

        if self.total_eeg_channels:
            channel = BiosemiEEGChannels(n_channels=self.total_eeg_channels,
                                         fs=self.fs, parent=self)
            self._channels.append(channel)

        if self.trigger_included:
            channel = BiosemiTriggerChannel(fs=self.fs, parent=self)
            self._channels.append(channel)

        if self.sensors_included:
            channel = BiosemiSensorChannels(fs=self.fs, parent=self)
            self._channels.append(channel)

    def get_channels(self, mode=None, direction=None, timing=None,
                     active=True):
        # The only channels we can return from the Biosemi are hardware-timed
        # analog input channels.
        if mode not in ('analog', None):
            return []
        if direction not in ('input', None):
            return []
        if timing not in ('hardware', None):
            return []
        return self._channels

    def configure(self):
        log.debug('Configuring {} engine'.format(self.name))
        self._client = ActiveTwoClient(host=self.host_ip,
                                       port=self.host_port,
                                       eeg_channels=self.eeg_channels,
                                       ex_included=self.ex_included,
                                       sensors_included=self.sensors_included,
                                       trigger_included=self.trigger_included,
                                       fs=self.fs)
        self._thread = BiosemiThread(self._client, self.monitor_period,
                                     self._hw_callback)
        super().configure()

    def _hw_callback(self, data):
        '''
        Data is a dictionary. Keys will indicate the datatype (e.g., 'eeg',
        'sensors', 'trigger', etc.). Values will be a 2D array (channel x time)
        containing the data for that datatype.
        '''
        for channel_name, cb in self._callbacks.get('ai'):
            channel_name = channel_name.split('_', 1)[1]
            d = data[channel_name]
            if d.shape[-1] != 0:
                cb(d)

    def get_ts(self):
        return self._samples_acquired / self.fs

    def start(self):
        self._thread.start()

    def stop(self):
        if not self.configured:
            return
        self._thread.stop_requested.set()
        log.debug('Waiting for acquistion thread to exit')
        self._thread.join()
        self.complete()

    def complete(self):
        log.debug('Triggering "done" callbacks')
        for cb in self._callbacks.get('done', []):
            cb()

    def register_done_callback(self, callback):
        self._callbacks.setdefault('done', []).append(callback)

    def register_ai_callback(self, callback, channel_name):
        self._callbacks.setdefault('ai', []).append((channel_name, callback))

    def unregister_done_callback(self, callback):
        try:
            self._callbacks['done'].remove(callback)
        except KeyError:
            log.warning('Callback no longer exists.')

    def unregister_ai_callback(self, callback, channel_name=None):
        try:
            self._callbacks['ai'].remove((channel_name, callback))
        except (KeyError, AttributeError):
            log.warning('Callback no longer exists.')


if __name__ == '__main__':
    from pathlib import Path
    path = Path(r'C:\Users\biosemi\Desktop\ActiView900-Win\Configuring')
    filename = path / '10-20system256+8.cfg'
    load_cfg(filename)
