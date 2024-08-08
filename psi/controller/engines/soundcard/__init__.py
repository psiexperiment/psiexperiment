import logging
log = logging.getLogger(__name__)

import time
from threading import Event

from atom.api import Bool, Dict, Enum, Float, List, Int, Str, Typed, Value
from enaml.core.api import d_
import numpy as np
import rtmixer
import sounddevice as sd

from psiaudio.pipeline import PipelineData
from psi.controller.api import (Engine, HardwareAIChannel, HardwareAOChannel)
from psi.controller.engines.thread import DAQThread
from psi.controller.engines.callback import ChannelSliceCallbackMixin

# Must be a power of two (presumably since the underlying portaudio library
# manages the RingBuffer and splits it into two sections).
QSIZE = 512
STEPSIZE = 4096


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
    _threads = Typed(dict, {})

    _stream = Value()
    _stop_requested = Value()

    _data = Value()
    _actions = List()
    _hw_ai_buffer = Value()
    _hw_ao_buffer = Value()

    #: Total samples read from the portaudio ringbuffer.
    _total_samples_read = Int()
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

    def _configure_ao_cb(self, ao_channels):
        if isinstance(self._stream.samplesize, tuple):
            samplesize = self._stream.samplesize[1]
        else:
            samplesize = self._stream.samplesize
        elementsize = len(ao_channels) * samplesize
        self._hw_ao_buffer = rtmixer.RingBuffer(elementsize, STEPSIZE * QSIZE)
        self._actions.append(self._stream.play_ringbuffer(self._hw_ao_buffer))
        self._threads['hw_ao'] = DAQThread(
            1e-3,
            self._stop_requested,
            lambda: self._hw_ao_callback(len(ao_channels)),
            name='hw_ao'
        )
        self._total_samples_written = 0

    def _configure_ai_cb(self, ai_channels):
        if isinstance(self._stream.samplesize, tuple):
            samplesize = self._stream.samplesize[0]
        else:
            samplesize = self._stream.samplesize
        elementsize = len(ai_channels) * samplesize
        self._hw_ai_buffer = rtmixer.RingBuffer(elementsize, STEPSIZE * QSIZE)
        self._actions.append(self._stream.record_ringbuffer(self._hw_ai_buffer))
        self._threads['hw_ai'] = DAQThread(
            1e-3,
            self._stop_requested,
            lambda: self._hw_ai_callback(len(ai_channels)),
            name='hw_ai'
        )
        self._total_samples_read = 0

    def _hw_ai_callback(self, n_channels):
        while self._hw_ai_buffer.read_available > STEPSIZE:
            samples = np.frombuffer(self._hw_ai_buffer.read(), dtype='float32')
            samples.shape = -1, n_channels

            data = PipelineData(samples.T, fs=96000, s0=self._total_samples_read)
            for channel_name, cb in self._callbacks.get('ai', []):
                cb(data)

            self._total_samples_read += len(samples)

    def _get_hw_ao_samples(self, offset, samples):
        channels = self.get_channels('analog', 'output', 'hardware')
        data = np.empty((len(channels), samples), dtype=np.double)
        for channel, ch_data in zip(channels, data):
            channel.get_samples(offset, samples, out=ch_data)
        return data

    def _hw_ao_callback(self, n_channels=None):
        if (n := self._hw_ao_buffer.write_available) <= 0:
            return
        data = self._get_hw_ao_samples(self._total_samples_written, n)
        buffer = data.astype('float32').T.tobytes()
        result = self._hw_ao_buffer.write(buffer)

    def play(self, data, name, start=0, allow_belated=True):
        buffer = data.astype('float32').T.tobytes()
        return self._stream.play_buffer(buffer, channels=[1])

    #: Size of buffer (in seconds). This defines how much data is pregenerated
    #: for the buffer before starting acquisition. This is important because
    hw_ao_buffer_size = d_(Float(10)).tag(metadata=True)

    def get_buffer_size(self, name):
        return self.hw_ao_buffer_size

    def start(self):
        # Preload with data
        self._hw_ao_callback()
        for thread in self._threads.values():
            thread.start()
        self._stream.start()

    def stop(self):
        if not self._configured:
            return
        self._stop_requested.set()
        self.complete()
        self._configured = False

    def register_done_callback(self, callback):
        self._callbacks.setdefault('done', []).append(callback)

    def register_ao_callback(self, callback, channel_name=None):
        self._callbacks.setdefault('ao', []).append((channel_name, callback))

    def register_ai_callback(self, callback, channel_name=None):
        self._callbacks.setdefault('ai', []).append((channel_name, callback))

    def unregister_done_callback(self, callback):
        try:
            self._callbacks['done'].remove(callback)
        except KeyError:
            log.warning('Callback no longer exists.')

    def unregister_ao_callback(self, callback, channel_name=None):
        try:
            self._callbacks['ao'].remove((channel_name, callback))
        except (KeyError, AttributeError):
            log.warning('Callback no longer exists.')

    def unregister_ai_callback(self, callback, channel_name=None):
        try:
            self._callbacks['ai'].remove((channel_name, callback))
        except (KeyError, AttributeError):
            log.warning('Callback no longer exists.')

    def complete(self):
        log.debug('Triggering "done" callbacks')
        for cb in self._callbacks.get('done', []):
            cb()
