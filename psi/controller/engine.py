import logging
log = logging.getLogger(__name__)

import numpy as np

from atom.api import Unicode, Float, Bool, observe, Property, Int, Typed, Long
from enaml.core.api import Declarative, d_

from .channel import Channel, AIChannel, AOChannel, DIChannel, DOChannel
from psi import SimpleState


class Engine(SimpleState, Declarative):

    name = d_(Unicode())
    master_clock = d_(Bool(False))

    hw_ao_buffer_samples = Long().tag(transient=True)
    hw_ao_buffer_offset = Long().tag(transient=True)
    hw_ao_buffer = Typed(np.ndarray).tag(transient=True)

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
        if self.hw_ao_channels:
            # Setup the ring buffer (so we can meld in existing data without
            # having to regenerate samples for the other outputs in the
            # channel)
            buffer_shape = len(self.hw_ao_channels), self.hw_ao_buffer_samples
            self.hw_ao_buffer = np.empty(buffer_shape, dtype=np.double)
            self.hw_ao_buffer_offset = -self.hw_ao_buffer_samples

        for channel in self.channels:
            log.debug('Configuring channel {}'.format(channel.name))
            channel.configure(plugin)

    def append_hw_ao(self, data, offset=None):
        # Store information regarding the data we have written to the output
        # buffer. This allows us to insert new signals into the ongoing stream
        # without having to recompute the data.  If the length of the data is
        # greater than our buffer, just overwrite the entire buffer. If less,
        # shift all the samples back and write the new data to the end of the
        # buffer.
        log.trace('Appending {} samples to end of hw ao buffer' \
                  .format(data.shape))
        data_samples = data.shape[-1]
        if data_samples >= self.hw_ao_buffer_samples:
            self.hw_ao_buffer[:] = data[..., -self.hw_ao_buffer_samples:]
        else:
            self.hw_ao_buffer = np.roll(self.hw_ao_buffer, -data_samples, -1)
            self.hw_ao_buffer[..., -data_samples:] = data

        # Track the trailing edge of the buffer (i.e., what is the sample number
        # of the first sample in the buffer).
        self.hw_ao_buffer_offset += data_samples
        log.trace('Current hw ao buffer offset {}' \
                  .format(self.hw_ao_buffer_offset))

        # Now, we actually write it.
        self.write_hw_ao(data)

    def modify_hw_ao(self, data, offset, method='merge', reference='start'):
        if reference == 'current':
            offset += self.ao_sample_clock()

        buffer_lb = self.hw_ao_buffer_offset
        buffer_ub = self.hw_ao_buffer_offset + self.hw_ao_buffer_samples
        if not(buffer_lb <= offset < buffer_ub):
            m = 'Buffer from {} to {}, requested offset is {}'
            log.debug(m.format(buffer_lb, buffer_ub, offset))
            raise IndexError('Segment falls outside of buffered stream')

        lb = offset - self.hw_ao_buffer_offset
        ub = lb + data.shape[-1]
        if method == 'merge':
            self.hw_ao_buffer[..., lb:ub] += data
        elif method == 'replace':
            self.hw_ao_buffer[..., lb:ub] = data
        elif method == 'multiply':
            self.hw_ao_buffer[..., lb:ub] *= data
        else:
            raise ValueError('Unsupported method')

        self.write_hw_ao(self.hw_ao_buffer[..., lb:], offset)

    def get_epoch_offset(self):
        pass

    def register_ao_callback(self, callback):
        raise NotImplementedError

    def register_ai_callback(self, callback):
        raise NotImplementedError

    def register_et_callback(self, callback):
        raise NotImplementedError

    def start(self):
        raise NotImplementedError

    def get_ts(self):
        raise NotImplementedError
