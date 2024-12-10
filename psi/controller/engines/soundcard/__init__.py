import logging
log = logging.getLogger(__name__)

import time
from threading import Event

from atom.api import Bool, Dict, Enum, Float, List, Int, Str, Typed, Value
from enaml.core.api import d_
import numpy as np
import rtmixer
import sounddevice as sd

from psiaudio import util
from psiaudio.pipeline import PipelineData
from psi.controller.api import (Engine, HardwareAIChannel, HardwareAOChannel)
from psi.controller.engines.thread import DAQThread
from psi.controller.engines.callback import ChannelSliceCallbackMixin

# Must be a power of two (presumably since the underlying portaudio library
# manages the RingBuffer and splits it into two sections).
QSIZE = 512
STEPSIZE = 4096


def halt_on_error(f):
    def wrapper(self, *args, **kwargs):
        try:
            f(self, *args, **kwargs)
        except Exception as e:
            try:
                self._stream.stop()
                self._stream.close()
            except:
                raise
                pass
            # Be sure to raise the exception so it can be recaptured by our
            # custom excepthook handler
            raise
    return wrapper


class SoundcardHardwareAIChannel(HardwareAIChannel):

    channel = d_(Int()).tag(metadata=True)


class SoundcardHardwareAOChannel(HardwareAOChannel):

    channel = d_(Int()).tag(metadata=True)


class SoundcardEngine(ChannelSliceCallbackMixin, Engine):

    #: Must be a valid device name as visible to sounddevice. To get a list of
    #: devices type `python -m sounddevice` at the command prompt.
    device_name = d_(Str()).tag(metadata=True)

    #: Flag indicating whether engine was configured
    _configured = Bool(False)

    #: Dictionary of threads that need to be started/polled.
    _tasks = Typed(dict, {})

    _stream = Value()
    _stop_requested = Value()

    _data = Value()
    _actions = Dict()
    _hw_ai_buffer = Value()
    _hw_ao_buffer = Dict()

    #: Total samples read from the portaudio ringbuffer.
    _total_samples_read = Int()

    #: Total samples to read. Set to 0 to acquire continuously.
    _samples_to_read = Int()

    #: Total samples written to the portaudio ringbuffer.
    _total_samples_written = Int()

    #: These are standard sampling rates in most sound cards. Not all sound
    #: cards may support the highest sampling rates. Calling the result of
    #: `Enum` with a value sets the default.
    fs = Enum(44100, 48000, 96000, 192000)(96000)

    def configure(self, active=True):
        self._data = []
        log.info('Initializing sound card %s', self.device_name)
        ai_channels = self.get_channels('analog', 'input', 'hardware', active=active)
        ao_channels = self.get_channels('analog', 'output', 'hardware', active=active)
        info = sd.query_devices(self.device_name)
        self._stop_requested = Event()

        ai_chan_number = [c.channel for c in ai_channels]
        ao_chan_number = [c.channel for c in ao_channels]

        for channel in ai_channels:
            channel.fs = self.fs
        for channel in ao_channels:
            channel.fs = self.fs

        if ai_channels and ao_channels:
            self._stream = rtmixer.MixerAndRecorder(
                device=self.device_name,
                samplerate=self.fs,
                channels=(len(ai_channels), len(ao_channels)),
                extra_settings=(sd.AsioSettings(ai_chan_number),
                                sd.AsioSettings(ao_chan_number)),
            )
            self._configure_ai_cb(ai_channels)
            self._configure_ao_cb(ao_channels)

        elif ai_channels:
            self._stream = rtmixer.Recorder(
                device=self.device_name,
                samplerate=self.fs,
                channels=len(ai_channels),
                extra_settings=sd.AsioSettings(ai_chan_number),
            )
            self._configure_ai_cb(ai_channels)
        elif ao_channels:
            self._stream = rtmixer.Mixer(
                device=self.device_name,
                samplerate=self.fs,
                channels=len(ao_channels),
                extra_settings=sd.AsioSettings(ao_chan_number),
            )
            self._configure_ao_cb(ao_channels)

        self._configured = True
        super().configure()

    def _configure_ai_cb(self, ai_channels):
        log.info('Configuring AI callback for %r', ai_channels)
        if isinstance(self._stream.samplesize, tuple):
            samplesize = self._stream.samplesize[0]
        else:
            samplesize = self._stream.samplesize
        elementsize = len(ai_channels) * samplesize
        self._hw_ai_buffer = rtmixer.RingBuffer(elementsize, STEPSIZE * QSIZE)
        self._actions['hw_ai'] = self._stream.record_ringbuffer(self._hw_ai_buffer)
        task = self._tasks['hw_ai'] = DAQThread(
            1e-3,
            self._stop_requested,
            lambda: self._hw_ai_callback(len(ai_channels)),
            name='hw_ai'
        )
        task._properties = {
            'names': [c.name for c in ai_channels],
        }
        self._total_samples_read = 0
        # Find the maximum number of samples we need. If 0, then we will
        # acquire continuously.
        self._samples_to_read = max(c.samples for c in ai_channels)

    def _hw_ai_callback(self, n_channels):
        while self._hw_ai_buffer.read_available > STEPSIZE:
            samples = np.frombuffer(self._hw_ai_buffer.read(), dtype='float32')
            samples.shape = -1, n_channels

            data = PipelineData(samples.T, fs=96000, s0=self._total_samples_read)
            for channel_name, s, cb in self._callbacks.get('ai', []):
                cb(data[s])

            self._total_samples_read += len(samples)
            if self._samples_to_read > 0:
                if self._total_samples_read > self._samples_to_read:
                    cancel_action = self._stream.cancel(self._actions['hw_ai'])
                    log.info('Waiting for cancel action to complete')
                    self._stream.wait(cancel_action)
                    self._check_done()

    def _configure_ao_cb(self, ao_channels):
        log.info('Configuring AO callback for %r', ao_channels)
        if isinstance(self._stream.samplesize, tuple):
            samplesize = self._stream.samplesize[1]
        else:
            samplesize = self._stream.samplesize

        for c in ao_channels:
            # First argument to RingBuffer should be channel number x
            # samplesize. Since we are create a separate buffer for each
            # channel, this should work well.
            self._hw_ao_buffer[c] = buffer = rtmixer.RingBuffer(samplesize, STEPSIZE * QSIZE)
            self._actions[f'hw_ao_{c.name}'] = self._stream.play_ringbuffer(buffer)
            task = self._tasks[f'hw_ao_{c.name}'] = DAQThread(
                1e-3,
                self._stop_requested,
                lambda: self._hw_ao_callback(c),
                name='hw_ao'
            )
            task._properties = {'name': c.name}
            task._total_samples_written = 0

    @halt_on_error
    def update_hw_ao(self, name, offset):
        with self.lock:
            if isinstance(self._stream.samplesize, tuple):
                samplesize = self._stream.samplesize[1]
            else:
                samplesize = self._stream.samplesize
            channel = self.get_channel(name)
            task = self._tasks[f'hw_ao_{channel.name}']

            log.error('Update requested for %s at %d', name, offset)
            cancel_action = self._stream.cancel(self._actions[f'hw_ao_{name}'])
            start_time = offset / self._stream.samplerate

            self._hw_ao_buffer[channel] = buffer = rtmixer.RingBuffer(samplesize, STEPSIZE * QSIZE)

            data = channel.get_samples(offset, buffer.write_available)
            task._total_samples_written = offset + data.shape[-1]
            buffer.write(data)

            self._hw_ao_callback(channel)
            self._actions[f'hw_ao_{channel.name}'] = \
                self._stream.play_ringbuffer(buffer,
                                            start=start_time,
                                            allow_belated=True)
            log.error('Done updating. Start @ %r', start_time)

    @halt_on_error
    def _hw_ao_callback(self, channel):
        with self.lock:
            buffer = self._hw_ao_buffer[channel]
            task = self._tasks[f'hw_ao_{channel.name}']
            if (samples := buffer.write_available) <= 0:
                return
            data = channel.get_samples(task._total_samples_written, samples)
            task._total_samples_written += data.shape[-1]
            buffer.write(data)

    def _check_done(self):
        log.info('Checking to see if stream is complete')
        log.info('Current actions: %r', self._stream.actions)
        if len(self._stream.actions) == 0:
            self.stop()

    def play(self, data, channel_names, start=0, allow_belated=True):
        i = self._get_channel_slice('hw_ao', channel_names)
        if len(i) != len(data):
            raise ValueError('Number of channels does not match data shape')
        buffer = data.astype('float32').T.tobytes()
        return self._stream.play_buffer(buffer, channels=i)

    #: Size of buffer (in seconds). This defines how much data is pregenerated
    #: for the buffer before starting acquisition. This is important because
    hw_ao_buffer_size = d_(Float(10)).tag(metadata=True)

    def get_buffer_size(self, name):
        return self.hw_ao_buffer_size

    def start(self):
        # Preload with data if we have an output task
        for channel in self._hw_ao_buffer.keys():
            self._hw_ao_callback(channel)
        for thread in self._tasks.values():
            thread.start()
        log.error('Starting stream')
        self._stream.start()

    def stop(self):
        if not self._configured:
            return
        self._stop_requested.set()
        self.complete()
        self._configured = False

    def complete(self):
        log.debug('Triggering "done" callbacks')
        for cb in self._callbacks.get('done', []):
            cb()

    def get_ts(self):
        try:
            return self._stream.time
        except Exception as e:
            log.exception(e)
            return np.nan
