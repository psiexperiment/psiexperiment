import os
os.environ['SD_ENABLE_ASIO'] = '1'

import logging
log = logging.getLogger(__name__)

import threading

import numpy as np
import sounddevice as sd


class BaseCallbackContext:

    def __init__(self):
        self.s0 = 0

    def __call__(self, *args):
        raise NotImplementedError


class RecordCallbackContext(BaseCallbackContext):

    def __init__(self, ai_cb):
        super().__init__()
        self.ai_cb = ai_cb

    def __call__(self, indata, samples, time, status):
        # Read the next segment to the input buffer
        self.ai_cb(self.s0, samples, indata.T.copy())
        self.s0 += samples


class PlayCallbackContext(BaseCallbackContext):

    def __init__(self, ao_cb):
        super().__init__()
        self.ao_cb = ao_cb

    def __call__(self, outdata, samples, time, status):
        # Write the next segment to the output buffer
        outdata[:] = self.ao_cb(samples)
        self.s0 += samples


class PlayRecordCallbackContext(BaseCallbackContext):

    def __init__(self, ai_cb, ao_cb):
        super().__init__()
        self.ai_cb = ai_cb
        self.ao_cb = ao_cb

    def __call__(self, indata, outdata, samples, time, status):
        # Read the next segment to the input buffer
        buffer = np.zeros_like(outdata.T)
        self.ao_cb(self.s0, samples, buffer)
        outdata[:] = buffer.T
        self.ai_cb(self.s0, samples, indata.T.copy())
        self.s0 += samples


class PlayRec:
    '''
    Basic wrapper around an audio device that provides play and record
    functionality.

    Parameters
    ----------
    fs : float
        Sampling rate to set device to.
    device : string
        Name of device as seen by portaudio.
    ai_channels : {None, list of int}
        List of channels to record from. If None, do not record.
    ao_channels : {None, list of int}
        List of channels to play. If None, do not play.
    ai_cb : {None, callable}
        Callback that takes recorded samples.
    ao_cb : {None, callable}
        Callback that generates samples to play. First argument to the callback
        is the number of samples needed. Callback must return a 2D array with
        one row for each output channel.
    blocksize : {None, int}
        Blocksize of read/write. Smaller blocksizes generally have better
        latency. Set to None to let PortAudio decide what's best.

    All channel numbers are specified using 0 based numbering. To determine the
    channel number for RME devices, you can open up TotalMixFX and count from
    left to right starting at 0.
    '''

    def __init__(self, fs, device, ai_channels=None, ao_channels=None,
                 ai_cb=None, ao_cb=None, blocksize=4096):
        self.fs = fs
        self.device = device
        self.ai_channels = ai_channels
        self.ao_channels = ao_channels
        self.ai_cb = ai_cb
        self.ao_cb = ao_cb
        self.blocksize = blocksize

        self.device_info = sd.query_devices(self.device)
        self.event = threading.Event()
        self.configure()

    def configure(self):
        '''
        Configure the acquisition stream
        '''
        stream_kw = {
            'samplerate': self.fs,
            'blocksize': self.blocksize,
        }
        log.error('Configuring PortAudio stream with AI %r and AO %r for device %s',
                  self.ai_channels, self.ao_channels, self.device)

        if self.ao_channels is not None and self.ai_channels is not None:
            stream_class = sd.Stream
            stream_kw['device'] = (self.device, self.device)
            stream_kw['channels'] = (len(self.ai_channels), len(self.ao_channels))
            stream_kw['callback'] = PlayRecordCallbackContext(self.ai_cb, self.ao_cb)
            stream_kw['extra_settings'] = (
                sd.AsioSettings(self.ai_channels),
                sd.AsioSettings(self.ao_channels),
            )
        elif self.ao_channels is not None:
            stream_class = sd.OutputStream
            stream_kw['device'] = self.device
            stream_kw['channels'] = len(self.ao_channels)
            stream_kw['callback'] = PlayCallbackContext(self.ao_cb)
            stream_kw['extra_settings'] = sd.AsioSettings(self.ao_channels)
        elif self.ai_channels is not None:
            stream_class = sd.InputStream
            stream_kw['device'] = self.device
            stream_kw['channels'] = len(self.ai_channels)
            stream_kw['callback'] = RecordCallbackContext(self.ai_cb)
            stream_kw['extra_settings'] = sd.AsioSettings(self.ai_channels)
        else:
            raise ValueError('No input or output channels specified')

        self.stream = stream_class(**stream_kw, finished_callback=self.event.set)

    def start(self):
        self.stream.start()

    def stop(self):
        self.stream.stop()

    def __del__(self):
        if hasattr(self, 'stream'):
            self.stream.close()


def test_delay():
    # This can be used to characterize the delays in the system
    import time
    import matplotlib.pyplot as plt
    import numpy as np

    fs = 96e3

    data = []
    signal = []

    def ai_cb(s0, samples, d):
        nonlocal data
        data.append(d)

    def ao_cb(s0, samples, buffer):
        nonlocal signal
        t = (np.arange(samples) + s0) / fs
        w = 0.001 * np.sin(2 * np.pi * 1e3 * t)
        w = np.zeros(samples)
        if len(signal) == 0:
            w[:10] = 1
        signal.append(w)
        buffer[0] = w
        #return w[np.newaxis]

    dev = PlayRec(fs, 'ASIO Fireface USB', [0, 12], [0], ai_cb, ao_cb)
    dev.start()
    time.sleep(0.5)
    dev.stop()

    data = np.concatenate(data, axis=-1)
    signal = np.concatenate(signal, axis=-1)
    loopback_delay = np.flatnonzero(data[0] > 0)[0]
    #output_delay = np.flatnonzero(data[1] > 0.01)[0]
    #print('Loopback delay samples: %d, mic input delay samples: %d', loopback_delay, output_delay)
    print(loopback_delay, loopback_delay / fs)

    plt.plot(data.T)
    plt.plot(signal, 'k')
    plt.show()


if __name__ == '__main__':
    test_delay()
