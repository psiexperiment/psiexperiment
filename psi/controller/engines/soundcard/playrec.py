import os
os.environ['SD_ENABLE_ASIO'] = '1'

import logging
log = logging.getLogger(__name__)

import threading

import numpy as np
import sounddevice as sd


class BaseCallbackContext:

    def __init__(self):
        self.ao_s0 = 0
        self.ai_s0 = 0

    def process_ao(self, outdata, samples):
        buffer = np.zeros_like(outdata.T)
        self.ao_cb(self.ao_s0, samples, buffer)
        outdata[:] = buffer.T
        self.ao_s0 += samples

    def _process_ai_firstcall(self, indata, samples):
        # Drop first input buffer since we are priming the output buffer on the first call.
        self.process_ai = self._process_ai

    def _process_ai(self, indata, samples):
        self.ai_cb(self.ai_s0, samples, indata.T.copy())
        self.ai_s0 += samples

    process_ai = _process_ai_firstcall

    def __call__(self, *args):
        raise NotImplementedError


class RecordCallbackContext(BaseCallbackContext):

    def __init__(self, ai_cb):
        super().__init__()
        self.ai_cb = ai_cb

    def __call__(self, indata, samples, time, status):
        # Read the next segment to the input buffer
        self.process_ai(indata, samples)


class PlayCallbackContext(BaseCallbackContext):

    def __init__(self, ao_cb):
        super().__init__()
        self.ao_cb = ao_cb

    def __call__(self, outdata, samples, time, status):
        # Write the next segment to the output buffer
        self.process_ao(outdata, samples)


class PlayRecordCallbackContext(BaseCallbackContext):

    def __init__(self, ai_cb, ao_cb):
        super().__init__()
        self.ai_cb = ai_cb
        self.ao_cb = ao_cb

    def __call__(self, indata, outdata, samples, time, status):
        self.process_ao(outdata, samples)
        self.process_ai(indata, samples)


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
        if self.stream.samplerate != self.fs:
            raise ValueError('Could not get desired sample rate')
        if self.stream.blocksize != self.blocksize:
            raise ValueError('Could not get desired blocksize')

    def start(self):
        self.stream.start()

    def stop(self):
        self.stream.stop()
        self.stream.close()
