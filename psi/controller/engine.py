import numpy as np

from atom.api import Unicode, Float, Bool, observe, Property, Int, Typed, Long
from enaml.core.api import Declarative, d_


class Engine(Declarative):

    name = d_(Unicode())
    ao_fs = d_(Float(100e3))
    ai_fs = d_(Float(25e3))
    master_clock = d_(Bool(False))

    hw_ao_buffer_samples = Long()
    hw_ao_buffer_offset = Long()
    hw_ao_buffer = Typed(np.ndarray)

    def configure(self, configuration):
        if 'hw_ao' in configuration:
            channels = configuration['hw_ao']
            # Setup the ring buffer (so we can meld in existing data)
            buffer_shape = len(channels), self.hw_ao_buffer_samples
            self.hw_ao_buffer = np.empty(buffer_shape, dtype=np.double)
            self.hw_ao_buffer_offset = -self.hw_ao_buffer_samples

    def append_hw_ao(self, data, offset=None):
        # Store information regarding the data we have written to the output
        # buffer. This allows us to insert new signals into the ongoing stream
        # without having to recompute the data.  If the length of the data is
        # greater than our buffer, just overwrite the entire buffer. If less,
        # shift all the samples back and write the new data to the end of the
        # buffer.
        data_samples = data.shape[-1]
        if data_samples >= self.hw_ao_buffer_samples:
            self.hw_ao_buffer[:] = data[..., -self.hw_ao_buffer_samples:]
        else:
            self.hw_ao_buffer = np.roll(self.hw_ao_buffer, -data_samples, -1)
            self.hw_ao_buffer[..., -data_samples:] = data

        # Track the trailing edge of the buffer (i.e., what is the sample number
        # of the first sample in the buffer).
        self.hw_ao_buffer_offset += data_samples

        # Now, we actually write it.
        self.write_hw_ao(data)

    def modify_hw_ao(self, data, offset, method='merge'):
        buffer_lb = self.hw_ao_buffer_offset
        buffer_ub = self.hw_ao_buffer_offset + self.hw_ao_buffer_samples
        if not(buffer_lb <= offset < buffer_ub):
            raise IndexError('Segment falls outside of buffered stream')

        lb = offset - self.hw_ao_buffer_offset
        ub = lb + data.shape[-1]
        if method == 'merge':
            self.hw_ao_buffer[..., lb:ub] += data
        elif method == 'replace':
            self.hw_ao_buffer[..., lb:ub] = data
        else:
            raise ValueError('Unsupported method')

        # Now, write the modified buffer to the hardware.
        self.write_hw_ao(self.hw_ao_buffer[..., lb:], offset)

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
