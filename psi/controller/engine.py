import logging
log = logging.getLogger(__name__)

import threading

import numpy as np

from atom.api import (Unicode, Float, Bool, observe, Property, Int, Typed,
                      Long, Value)
from enaml.core.api import Declarative, d_

from psi.core.enaml.api import PSIContribution
from ..util import copy_declarative
from .channel import Channel, AIChannel, AOChannel, DIChannel, DOChannel


def log_configuration(engine):
    info = ['Engine configuration']
    info.append('Engine {}'.format(engine.name))
    for channel in engine.get_channels(direction='input', active=True):
        info.append('\t channel {}'.format(channel.name))
        for i in channel.inputs:
            info.append('\t\t input {}'.format(i.name))
    for channel in engine.get_channels(direction='output', active=True):
        info.append('\t channel {}'.format(channel.name))
        for o in channel.outputs:
            info.append('\t\t output {}'.format(o.name))
    log.info('\n'.join(info))


class Engine(PSIContribution):

    name = d_(Unicode()).tag(metadata=True)
    master_clock = d_(Bool(False)).tag(metadata=True)
    lock = Value()

    def _default_lock(self):
        return threading.Lock()

    def get_channels(self, mode=None, direction=None, timing=None,
                     active=True):
        '''
        Return channels matching criteria

        Parameters
        ----------
        mode : {None, 'analog', 'digital'
            Type of channel
        direction : {None, 'input, 'output'}
            Direction
        timing : {None, 'hardware', 'software'}
            Hardware or software-timed channel. Hardware-timed channels have a
            sampling frequency greater than 0.
        active : bool
            If True, return only channels that have configured inputs or
            outputs.
        '''
        channels = [c for c in self.children if isinstance(c, Channel)]

        if active:
            channels = [c for c in channels if c.active]

        if timing is not None:
            if timing in ('hardware', 'hw'):
                channels = [c for c in channels if c.fs != 0]
            elif timing in ('software', 'sw'):
                channels = [c for c in channels if c.fs == 0]
            else:
                raise ValueError('Unsupported timing')

        if direction is not None:
            if direction in ('input', 'in'):
                matches = (AIChannel, DIChannel)
            elif direction in ('output', 'out'):
                matches = (AOChannel, DOChannel)
            else:
                raise ValueError('Unsupported direction')
            channels = [c for c in channels if isinstance(c, matches)]

        if mode is not None:
            if mode == 'analog':
                matches = (AIChannel, AOChannel)
            elif mode == 'digital':
                matches = (DIChannel, DOChannel)
            else:
                raise ValueError('Unsupported mode')
            channels = [c for c in channels if isinstance(c, matches)]

        return tuple(channels)

    def get_channel(self, channel_name):
        channels = self.get_channels(active=False)
        for channel in channels:
            if channel.name == channel_name:
                return channel
        m = '{} channel does not exist'.format(channel_name)
        raise AttributeError(m)

    def remove_channel(self, channel):
        channel.set_parent(None)

    def configure(self):
        log_configuration(self)
        for channel in self.get_channels():
            log.debug('Configuring channel {}'.format(channel.name))
            channel.configure()

    def register_ai_callback(self, callback, channel_name=None):
        raise NotImplementedError

    def register_et_callback(self, callback, channel_name=None):
        raise NotImplementedError

    def unregister_ai_callback(self, callback, channel_name=None):
        raise NotImplementedError

    def unregister_et_callback(self, callback, channel_name=None):
        raise NotImplementedError

    def register_done_callback(self, callback):
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

    def clone(self, channel_names=None):
        '''
        Return a copy of this engine with specified channels incldued

        This is intended as a utility function to assist various routines that
        may need to do a quick operation before starting the experiment. For
        example, calibration may only need to run a subset of the channels.
        '''
        new = copy_declarative(self)
        if channel_names is not None:
            for channel_name in channel_names:
                channel = self.get_channel(channel_name)
                copy_declarative(channel, parent=new)
        return new
