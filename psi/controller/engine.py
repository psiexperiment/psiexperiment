import logging
log = logging.getLogger(__name__)

import threading

import numpy as np

from atom.api import (Unicode, Float, Bool, observe, Property, Int, Typed,
                      Long, Value)
from enaml.core.api import Declarative, d_

from .channel import Channel, AIChannel, AOChannel, DIChannel, DOChannel


class Engine(Declarative):

    name = d_(Unicode()).tag(metadata=True)
    master_clock = d_(Bool(False)).tag(metadata=True)
    lock = Value()

    channels = Property()
    hw_ao_channels = Property()
    hw_ai_channels = Property()
    hw_do_channels = Property()
    hw_di_channels = Property()
    sw_do_channels = Property()

    def _default_lock(self):
        return threading.RLock()

    def _get_channels(self):
        return [c for c in self.children if isinstance(c, Channel)]

    def _get_hw_ao_channels(self):
        return [c for c in self.children if \
                isinstance(c, AOChannel) and c.fs != 0]

    def _get_hw_ai_channels(self):
        return [c for c in self.children if \
                isinstance(c, AIChannel) and c.fs != 0]

    def _get_hw_do_channels(self):
        return [c for c in self.children if \
                isinstance(c, DOChannel) and c.fs != 0]

    def _get_hw_di_channels(self):
        return [c for c in self.children if \
                isinstance(c, DIChannel) and c.fs != 0]

    def _get_sw_do_channels(self):
        return [c for c in self.children if \
                isinstance(c, DOChannel) and c.fs == 0]

    def configure(self, plugin=None):
        for channel in self.channels:
            log.debug('Configuring channel {}'.format(channel.name))
            channel.configure(plugin)

    def register_ai_callback(self, callback, channel_name=None):
        raise NotImplementedError

    def register_et_callback(self, callback, channel_name=None):
        raise NotImplementedError

    def unregister_ai_callback(self, callback, channel_name=None):
        raise NotImplementedError

    def unregister_et_callback(self, callback, channel_name=None):
        raise NotImplementedError

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def get_ts(self):
        raise NotImplementedError

    def get_buffer_size(self, channel_name):
        raise NotImplementedError

    def get_offset(self, channel_name):
        raise NotImplementedError
