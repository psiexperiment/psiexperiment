import logging
log = logging.getLogger(__name__)

import time
from threading import Event

from atom.api import Bool, Dict, Float, List, Int, Str, Typed, Value
from enaml.core.api import d_
import numpy as np
import rtmixer
import sounddevice as sd

from psiaudio.pipeline import PipelineData
from psi.controller.api import (Engine, HardwareAIChannel, HardwareAOChannel)
from psi.controller.engines.thread import DAQThread

# Recommended to be a power of two
QSIZE = 512
STEPSIZE = 4096


class SoundcardHardwareAIChannel(HardwareAIChannel):

    channel = d_(Int()).tag(metadata=True)


class SoundcardEngine(Engine):

    #: Must be a valid device name as visible to sounddevice. To get a list of
    #: devices type `python -m sounddevice` at the command prompt.
    device_name = d_(Str()).tag(metadata=True)

    #: Flag indicating whether engine was configured
    _configured = Bool(False)
    _callbacks = Typed(dict, {})
    _threads = Typed(dict, {})

    _stream = Value()
    _t0 = Float()
    _s0 = Int(0)
    _stop_requested = Value()

    _data = Value()
    _actions = List()
    _hw_ai_buffer = Value()

    #: Total samples read from the portaudio ringbuffer.
    _total_samples_read = Int()

    fs = Int(96000)

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
                samplerate=96000,
                channels=(len(ai_channels), len(ao_channels)),
                extra_settings=(sd.AsioSettings(ai_chan_number),
                                sd.AsioSettings(ao_chan_number)),
            )
        elif ai_channels:
            self._stream = rtmixer.Recorder(
                device=self.device_name,
                samplerate=96000,
                channels=len(ai_channels),
                extra_settings=sd.AsioSettings(ai_chan_number),
            )
            elementsize = len(ai_channels) * self._stream.samplesize
            self._hw_ai_buffer = rtmixer.RingBuffer(elementsize, STEPSIZE * QSIZE)
            self._actions.append(self._stream.record_ringbuffer(self._hw_ai_buffer))
            self._threads['hw_ai'] = DAQThread(
                1e-3,
                self._stop_requested,
                lambda: self._hw_ai_callback(len(ai_channels)),
                name='hw_ai'
            )
            self._total_samples_read = 0

        elif ao_channels:
            self._stream = rtmixer.Mixer(
                device=self.device_name,
                samplerate=96000,
                channels=len(ao_channels),
                extra_settings=sd.AsioSettings(ao_chan_number),
            )
        self._configured = True

        super().configure()

    def _hw_ai_callback(self, n_channels):
        while self._hw_ai_buffer.read_available > STEPSIZE:
            read, buf1, buf2 = self._hw_ai_buffer.get_read_buffers(STEPSIZE)
            if read != STEPSIZE:
                raise ValueError
            if buf2:
                raise ValueError
            samples = np.frombuffer(buf1, dtype='float32')
            samples.shape = -1, n_channels

            data = PipelineData(samples.T, fs=96000, s0=self._total_samples_read)
            for channel_name, cb in self._callbacks.get('ai', []):
                #if channel_name == name:
                cb(data)

            self._total_samples_read += STEPSIZE
            self._hw_ai_buffer.advance_read_index(STEPSIZE)

    def start(self):
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
