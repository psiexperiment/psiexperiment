import logging
log = logging.getLogger(__name__)

import threading

import numpy as np

from atom.api import (Unicode, Float, Bool, observe, Property, Int, Typed,
                      Long, Value)
from enaml.core.api import Declarative, d_

from .channel import Channel, AIChannel, AOChannel, DIChannel, DOChannel
from psi import SimpleState


class Engine(SimpleState, Declarative):

    name = d_(Unicode())
    master_clock = d_(Bool(False))
    lock = Value()

    hw_ao_buffer_samples = Long().tag(transient=True)
    hw_ao_buffer_offset = Long().tag(transient=True)
    hw_ao_buffer = Typed(np.ndarray).tag(transient=True)
    hw_ao_buffer_map = Typed(dict).tag(transient=True)

    channels = Property().tag(transient=True)
    hw_ao_channels = Property().tag(transient=True)
    hw_ai_channels = Property().tag(transient=True)
    hw_do_channels = Property().tag(transient=True)
    hw_di_channels = Property().tag(transient=True)
    sw_do_channels = Property().tag(transient=True)

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

    def configure(self, plugin):
        self.lock = threading.Lock()

        if self.hw_ao_channels:
            # Generate a 1-based output map (0 is always reserved for the
            # continuous output). Max outputs is number of epoch outputs plus
            # the continuous output.
            output_map = {}
            max_outputs = 0
            for channel in self.hw_ao_channels:
                for i, output in enumerate(channel.epoch_outputs):
                    output_map[output.name] = i+1
                max_outputs = max(max_outputs, len(channel.epoch_outputs)+1)
            self.hw_ao_buffer_map = output_map

            # Setup the ring buffer (so we can meld in existing data without
            # having to regenerate samples for the other outputs in the
            # channel)
            n_channels = len(self.hw_ao_channels)
            buffer_shape = (n_channels, max_outputs, self.hw_ao_buffer_samples)
            self.hw_ao_buffer = np.zeros(buffer_shape, dtype=np.double)
            self.hw_ao_buffer_offset = -self.hw_ao_buffer_samples

        for channel in self.channels:
            log.debug('Configuring channel {}'.format(channel.name))
            channel.configure(plugin)

    def append_hw_ao(self, data):
        '''
        This can only be used for the continuous output.
        '''
        # TODO: need to build-in support for multiple output channels. This
        # needs to be linked to the callback somehow.

        # Store information regarding the data we have written to the output
        # buffer. This allows us to insert new signals from the epoch output
        # into the ongoing stream without having to recompute the data.  If the
        # length of the data is greater than our buffer, just overwrite the
        # entire buffer. If less, shift all the samples back and write the new
        # data to the end of the buffer.
        m = 'Appending {} samples to end of hw ao buffer'
        log.trace(m.format(data.shape))

        # Write this immediately to minimize delays. TODO: At some point add a
        # delay so we can modify the buffer with ongoing epoch outputs as well
        # (i.e., this should minimize function overhead)? NOTE: calling the
        # function without offset argument means that it appends to the end of
        # the existing buffer.
        self.write_hw_ao(data, timeout=1)

        # The data will be provided in 2D form (channel, sample), but we need
        # to expand this to the 3D form of (channel, output, sample) where the
        # data is the first output (the continuous output)
        padding = (0, 0), (0, self.hw_ao_buffer.shape[1]-1), (0, 0)
        data = np.pad(data[np.newaxis, np.newaxis, :], padding, 'constant')

        data_samples = data.shape[-1]
        if data_samples >= self.hw_ao_buffer_samples:
            self.hw_ao_buffer[:] = data[..., -self.hw_ao_buffer_samples:]
        else:
            self.hw_ao_buffer = np.roll(self.hw_ao_buffer, -data_samples, -1)
            self.hw_ao_buffer[..., -data_samples:] = data

        # Track the trailing edge of the buffer (i.e., what is the sample number
        # of the first sample in the buffer).
        self.hw_ao_buffer_offset += data_samples

        m = 'Current hw ao buffer offset {}'
        log.trace(m.format(self.hw_ao_buffer_offset))

    def modify_hw_ao(self, data, offset, output_name, reference='start'):
        if reference == 'current':
            offset += self.ao_sample_clock()

        buffer_lb = self.hw_ao_buffer_offset
        buffer_ub = self.hw_ao_buffer_offset + self.hw_ao_buffer_samples
        if not(buffer_lb <= offset < buffer_ub):
            m = 'Buffer from {} to {}, requested offset is {}'
            log.debug(m.format(buffer_lb, buffer_ub, offset))
            raise IndexError('Segment falls outside of buffered stream')

        # TODO: Support other types of operations? (e.g., multiply, replace)
        lb = offset - self.hw_ao_buffer_offset
        ub = lb + data.shape[-1]
        oi = self.hw_ao_buffer_map[output_name]
        self.hw_ao_buffer[:, oi, lb:ub] = data
        combined_data = self.hw_ao_buffer[..., lb:].sum(axis=1)
        self.write_hw_ao(combined_data, offset, timeout=1)

    def get_buffered_samples(self, channel_name, offset=0):
        buffer_offset = offset-self.hw_ao_buffer_offset
        return self.hw_ao_buffer_samples-buffer_offset

    def get_epoch_offset(self):
        pass

    def register_ao_callback(self, callback):
        raise NotImplementedError

    def register_ai_callback(self, callback):
        raise NotImplementedError

    def register_et_callback(self, callback):
        raise NotImplementedError

    def unregister_ao_callback(self, callback):
        raise NotImplementedError

    def unregister_ai_callback(self, callback):
        raise NotImplementedError

    def unregister_et_callback(self, callback):
        raise NotImplementedError

    def start(self):
        raise NotImplementedError

    def get_ts(self):
        raise NotImplementedError
