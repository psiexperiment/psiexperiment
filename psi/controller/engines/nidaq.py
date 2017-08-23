import logging
log = logging.getLogger(__name__)

import ctypes

import numpy as np

from atom.api import Float, Typed, Unicode, Int
from enaml.core.api import Declarative, d_

from ..engine import Engine
from daqengine import ni
import PyDAQmx as mx


def get_channel_property(channels, property, allow_unique=False):
    values = [getattr(c, property) for c in channels]
    if allow_unique:
        return values
    elif len(set(values)) != 1:
        m = 'NIDAQEngine does not support per-channel {} as specified: {}' \
            .format(property, values)
        raise ValueError(m)
    else:
        return values[0]


################################################################################
# engine
################################################################################
class NIDAQEngine(ni.Engine, Engine):
    # Even though data is written to the analog outputs, it is buffered in
    # computer memory until it's time to be transferred to the onboard buffer of
    # the NI acquisition card. NI-DAQmx handles this behind the scenes (i.e.,
    # when the acquisition card needs additional samples, NI-DAQmx will transfer
    # the next chunk of data from the computer memory). We can overwrite data
    # that's been buffered in computer memory (e.g., so we can insert a target
    # in response to a nose-poke). However, we cannot overwrite data that's
    # already been transfered to the onboard buffer. So, the onboard buffer size
    # determines how quickly we can change the analog output in response to an
    # event.

    # TODO: this is not configurable on some systems. How do we figure out if
    # it's configurable?
    hw_ao_onboard_buffer = d_(Int(4095)).tag(metadata=True)

    # Since any function call takes a small fraction of time (e.g., nanoseconds
    # to milliseconds), we can't simply overwrite data starting at
    # hw_ao_onboard_buffer+1. By the time the function calls are complete, the
    # DAQ probably has already transferred a couple hundred samples to the
    # buffer. This parameter will likely need some tweaking (i.e., only you can
    # determine an appropriate value for this based on the needs of your
    # program).
    hw_ao_min_writeahead = d_(Int(8191 + 1000)).tag(metadata=True)

    hw_ai_monitor_period = d_(Float(0.1)).tag(metadata=True)

    hw_ao_monitor_period = d_(Float(1)).tag(metadata=True)
    hw_ao_buffer_size = d_(Float(10)).tag(metadata=True)

    # These need to be redefined here even though we define them in the parent
    # class.
    _tasks = Typed(dict)
    _callbacks = Typed(dict)
    _timers = Typed(dict)
    _uint32 = Typed(ctypes.c_uint32)
    _uint64 = Typed(ctypes.c_uint64)
    _int32 = Typed(ctypes.c_int32)

    ao_fs = Typed(float).tag(metadata=True)

    terminal_mode_map = {
        'differential': mx.DAQmx_Val_Diff,
        'pseudodifferential': mx.DAQmx_Val_PseudoDiff,
        'RSE': mx.DAQmx_Val_RSE,
        'NRSE': mx.DAQmx_Val_NRSE,
        'default': mx.DAQmx_Val_Cfg_Default,
    }

    terminal_coupling_map = {
        None: None,
        'AC': mx.DAQmx_Val_AC,
        'DC': mx.DAQmx_Val_DC,
        'ground': mx.DAQmx_Val_GND,
    }

    def __init__(self, *args, **kwargs):
        ni.Engine.__init__(self)
        Engine.__init__(self, *args, **kwargs)

    def configure(self, plugin):
        log.debug('Configuring {} engine'.format(self.name))
        if self.sw_do_channels:
            log.debug('Configuring SW DO channels')
            channels = self.sw_do_channels
            lines = ','.join(get_channel_property(channels, 'channel', True))
            names = get_channel_property(channels, 'name', True)
            self.configure_sw_do(lines, names)

        if self.hw_ai_channels:
            log.debug('Configuring HW AI channels')
            channels = self.hw_ai_channels
            lines = ','.join(get_channel_property(channels, 'channel', True))
            names = get_channel_property(channels, 'name', True)
            fs = get_channel_property(channels, 'fs')
            start_trigger = get_channel_property(channels, 'start_trigger')
            expected_range = get_channel_property(channels, 'expected_range')

            tmode = get_channel_property(channels, 'terminal_mode')
            tcoupling = get_channel_property(channels, 'terminal_coupling')
            terminal_mode = self.terminal_mode_map[tmode]
            terminal_coupling = self.terminal_coupling_map[tcoupling]

            self.configure_hw_ai(fs, lines, expected_range, names,
                                 start_trigger, terminal_mode,
                                 terminal_coupling)

        if self.hw_di_channels:
            log.debug('Configuring HW DI channels')
            channels = self.hw_di_channels
            lines = ','.join(get_channel_property(channels, 'channel', True))
            names = get_channel_property(channels, 'name', True)
            fs = get_channel_property(channels, 'fs')
            start_trigger = get_channel_property(channels, 'start_trigger')
            # Required for M-series to enable hardware-timed digital
            # acquisition. TODO: Make this a setting that can be configured
            # since X-series doesn't need this hack.
            device = channels[0].channel.strip('/').split('/')[0]
            clock = '/{}/Ctr0'.format(device)
            self.configure_hw_di(fs, lines, names, start_trigger, clock)

        # Configure the analog output last because acquisition is synced with
        # the analog output signal (i.e., when the analog output starts, the
        # analog input begins acquiring such that sample 0 of the input
        # corresponds with sample 0 of the output).
        # TODO: eventually we should be able to inspect the  'start_trigger'
        # property on the channel configuration to decide the order in which the
        # tasks are started.
        if self.hw_ao_channels:
            log.debug('Configuring HW AO channels')
            channels = self.hw_ao_channels
            lines = ','.join(get_channel_property(channels, 'channel', True))
            names = get_channel_property(channels, 'name', True)
            fs = get_channel_property(channels, 'fs')
            start_trigger = get_channel_property(channels, 'start_trigger')
            expected_range = get_channel_property(channels, 'expected_range')
            tmode = get_channel_property(channels, 'terminal_mode')
            terminal_mode = self.terminal_mode_map[tmode]
            self.configure_hw_ao(fs, lines, expected_range, names,
                                 start_trigger, terminal_mode)
            self.ao_fs = fs

        super(NIDAQEngine, self).configure(plugin)
        log.debug('Completed engine configuration')

    def get_ts(self):
        return self.ao_sample_clock()/self.ao_fs

    def get_offset(self, channel_name=None):
        # It doesn't matter what the output channel is. Offset will be the same
        # for all.
        return self.ao_write_position()

    def get_space_available(self, channel_name=None, offset=None):
        # It doesn't matter what the output channel is. Write space will be the
        # same for all.
        return np.clip(self.ao_write_space_available(offset),
                       0, self.hw_ao_buffer_samples)
