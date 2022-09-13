import logging
log = logging.getLogger(__name__)

import datetime as dt
from pathlib import Path
import threading
import time

from atom.api import Dict, Float, Str
from ..engine import Engine


class NullEngine(Engine):

    buffer_size = Float()
    handles = Dict()
    base_path = Str()

    t0 = Float()
    t_ao = Float()
    t_ai = Float()

    def configure(self, active=True):
        counter_channels = self.get_channels('counter', active=active)
        sw_do_channels = self.get_channels('digital', 'output', 'software',
                                           active=active)
        hw_ai_channels = self.get_channels('analog', 'input', 'hardware',
                                           active=active)
        hw_di_channels = self.get_channels('digital', 'input', 'hardware',
                                           active=active)
        hw_ao_channels = self.get_channels('analog', 'output', 'hardware',
                                           active=active)

        base_path = Path(self.base_path) / \
            dt.datetime.now().strftime('%Y%m%d-%H%M%S')

        for channel in hw_ao_channels:
            filename = (base_path / channel.name).with_suffix('.bin')
            log.debug('***** %r', filename)

    def _hw_ao_callback(self):
        elapsed_time = time.time() - self.t0
        for channel_name, cb in self._callbacks.get('ai', []):
            fs = self.get_channel(channel_name).fs
            samples = int(round(elapsed_time * fs))
            cb(samples)
        self.t0 = elapsed_time

    def register_done_callback(self, callback):
        pass

    def register_ao_callback(self, callback, channel_name):
        self._callbacks.setdefault('ao', []).append((channel_name, callback))

    def register_ai_callback(self, callback, channel_name):
        pass

    def register_et_callback(self, callback, channel_name):
        pass

    def unregister_ao_callback(self, callback, channel_name):
        pass

    def unregister_ai_callback(self, callback, channel_name):
        pass

    def unregister_et_callback(self, callback, channel_name):
        pass

    def start(self):
        self.t0 = time.time()
        self.t_ao = 0
        self.t_ai = 0
        #self.timer = threading.Timer(self.hw_ao_monitor_period)

    def stop(self):
        pass

    def get_ts(self):
        return time.time() - self.t0

    def get_buffer_size(self, channel):
        return self.buffer_size

    def update_hw_ao_multiple(self, offsets, channel_names, method):
        for (o, c_name) in zip((offsets, channel_names)):
            self.update_hw_ao(o, c, method)

    def update_hw_ao(self, offset, channel_name=None, method='space_available'):
        return
        #if method == 'space_available':
        #    samples
        #data = self._get_hw_ao_samples()

    def get_space_available(self, offset=None, channel_name=None):
        samples_remaining = self.t_ai - time.time()
        if samples_remaining < 0:
            raise IOError('Ran out of samples')
        samples_needed = self.buffer_size - samples_remaining
        fs = self.get_channel(channel_name).fs
        return int(round(samples_needed * fs))
