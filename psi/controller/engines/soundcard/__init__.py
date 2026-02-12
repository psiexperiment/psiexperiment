import os
os.environ['SD_ENABLE_ASIO'] = '1'


import logging
log = logging.getLogger(__name__)

import sys
import time
from collections import deque
from threading import Thread

from atom.api import (Bool, Dict, Enum, Float, List, Int, Property, Str,
                      FixedTuple, Tuple, Typed, Value)
from enaml.application import deferred_call
from enaml.core.api import d_
import numpy as np

from psiaudio import util
from psiaudio.pipeline import PipelineData
from psi.controller.api import (Engine, HardwareAIChannel, HardwareAOChannel)
from psi.controller.engines.callback import ChannelSliceCallbackMixin

from .playrec import PlayRec


class SDAIThread(Thread):

    def __init__(self, fs, queue, sf, callbacks, poll_interval=0.001):
        super().__init__(daemon=True)
        self.fs = fs
        self.queue = queue
        self.callbacks = callbacks
        self.sf = sf
        self.poll_interval = poll_interval

    def run(self):
        # This is a rather complicated piece of code because we need to
        # override the threading module's built-in exception handling as well
        # as defe the exception back to the main thread (where it will properly
        # handle exceptions). If we call psi.application.exception_handler
        # directly from the thread, it will not have access to the application
        # instance (or workspace).
        try:
            while True:
                data, s0_all = [], []
                # Pull all pending data from queue and chunk it together before
                # sending to downstream pipelines (to avoid falling behind with
                # repeated calls to the callbacks).
                while self.queue:
                    d, s0 = self.queue.popleft()
                    data.append(d)
                    s0_all.append(s0)
                if len(data):
                    data = np.concatenate(data, axis=-1)
                    s0 = s0_all[0]
                    data = PipelineData(data / self.sf[..., np.newaxis], fs=self.fs, s0=s0)
                    for channel_name, s, cb in self.callbacks.get('ai', []):
                        cb(data[s])
                time.sleep(self.poll_interval)
        except:
            deferred_call(sys.excepthook, *sys.exc_info())


def halt_on_error(f):
    def wrapper(self, *args, **kwargs):
        try:
            f(self, *args, **kwargs)
        except Exception as e:
            self._stream.stop()
            # Be sure to raise the exception so it can be recaptured by our
            # custom excepthook handler
            raise
    return wrapper


class SoundcardTimingMixin:

    def sync_start(self, channel):
        # If channels are on the same engine, they will already be in sync.
        if self.engine != channel.engine:
            raise ValueError('Cannot sync channels on separate engines')


class SoundcardHardwareAIChannel(SoundcardTimingMixin, HardwareAIChannel):

    channel = d_(Int()).tag(metadata=True)

    # This is the default input sensitivity for most RME devices that are
    # controlled via TotalmixFX. Some inputs can be configured to have
    # different sensitivity. This is the value reported in dBu. Likely either
    # +13 or +19 dBu. This is in addition to any external gain (e.g., from an
    # input preamp).
    dBFS = d_(FixedTuple(Float(19), Enum('dBu', 'dBV'))).tag(metadata=True)

    # Defines total gain in units of dBV so that we can convert the unit
    # reported by the sound card API to dBV.
    total_gain = Property().tag(metadata=True)

    # This is set by the sound card containing this engine, so we cannot set it
    # manually on a per-channel basis.
    fs = Property().tag(metadata=True)

    def _get_total_gain(self):
        s, unit = self.dBFS
        if unit == 'dBu':
            #s += util.db(0.7746)
            pass
        elif unit == 'dBV':
            pass
        return -s + self.gain

    def _get_fs(self):
        return self.engine.fs


class SoundcardHardwareAOChannel(SoundcardTimingMixin, HardwareAOChannel):

    channel = d_(Int()).tag(metadata=True)

    # This is set by the sound card containing this engine, so we cannot set it
    # manually on a per-channel basis.
    fs = Property().tag(metadata=True)

    def _get_fs(self):
        return self.engine.fs


class SoundcardEngine(ChannelSliceCallbackMixin, Engine):

    #: Must be a valid device name as visible to sounddevice. To get a list of
    #: devices type `python -m sounddevice` at the command prompt.
    device_name = d_(Str()).tag(metadata=True)

    #: Flag indicating whether engine was configured
    _configured = Bool(False)

    _stream = Value()

    #: Mapping of output channel name (which is used by the generic
    #: psiexperiment API to refer to individual channels) to the channel number
    #: in the port audio stream. Used internally to determine which channel is
    #: provided to the portaudio library.
    _hw_ao_channel_map = Dict()

    #: List of hardware analog output channels that need to be combined to
    #: generate the data requested by the audio stream.
    _hw_ao_channels = Tuple()

    #: Total samples read from the portaudio ringbuffer.
    _total_samples_read = Int()

    #: Total samples to read. Set to 0 to acquire continuously.
    _samples_to_read = Int()

    #: Total samples written to the portaudio ringbuffer.
    _total_samples_written = Int()

    #: Samples to discard from input to correct for delays between output and
    #: input. This is dependent on block size and other things. See
    #: measure_delay module.
    latency_samples = d_(Int(0))

    #: Tracks remaining samples to discard to correct for latency.
    _samples_to_discard = Int(0)

    #: Factor to scale analog inputs by to correct for gain and other factors.
    _hw_ai_sf = Value()

    #: Flag indicating experiment is running and actively streaming data.
    _running = Bool(False)

    #: These are standard sampling rates in most sound cards. Not all sound
    #: cards may support the highest sampling rates. Calling the result of
    #: `Enum` with a value sets the default.
    fs = d_(Enum(32000, 44100, 48000, 64000, 88200, 96000, 128000, 176400, 192000)(96000))

    #: Blocksize for callback. Ideally this will match what is configured for
    #: the Fireface settings. TODO! Is there a way to find and set this
    #: parameter via the API? I don't think so ...
    blocksize = d_(Int(8192))

    #: Initial timestamp of the stream when it's first started (apparently the
    #: behavior of the stream start time is undefined, so we just need to
    #: capture this and then subtract from calls to `get_ts`).
    t0 = Float()

    hw_ai_queue = Typed(deque)
    hw_ai_thread = Typed(Thread)

    def configure(self, active=True):
        log.info('Initializing sound card %s', self.device_name)
        ai_channels = self.get_channels('analog', 'input', 'hardware', active=active)
        ao_channels = self.get_channels('analog', 'output', 'hardware', active=active)

        pr_kw = {
            'fs': self.fs,
            'device': self.device_name,
            'blocksize': self.blocksize,
        }
        if ai_channels:
            pr_kw['ai_channels'] = [c.channel for c in ai_channels]
            pr_kw['ai_cb'] = self._hw_ai_callback
            self._configure_ai_cb(ai_channels)
        if ao_channels:
            pr_kw['ao_channels'] = [c.channel for c in ao_channels]
            pr_kw['ao_cb'] = self._hw_ao_callback
            self._configure_ao_cb(ao_channels)

        self._stream = PlayRec(**pr_kw)

        # Pass in callbacks by reference so that the thread has updated info on
        # all the callbacks as they are created and added.
        self.hw_ai_queue = deque()
        self.hw_ai_thread = SDAIThread(self.fs, self.hw_ai_queue,
                                       self._hw_ai_sf, self._callbacks)
        self.hw_ai_thread.start()

        self._configured = True
        super().configure()

    def _configure_ai_cb(self, ai_channels):
        log.info('Configuring AI callback for %r', ai_channels)
        self._hw_ai_sf = util.dbi(np.array([c.total_gain for c in ai_channels]))
        self._total_samples_read = 0
        # Find the maximum number of samples we need. If 0, then we will
        # acquire continuously.
        self._samples_to_read = max(c.samples for c in ai_channels)
        self._channel_names['hw_ai'] = [c.name for c in ai_channels]
        self._samples_to_discard = self.latency_samples

    def _hw_ai_callback(self, offset, samples, data):
        # Process the incoming data and push through the pipeline.
        if offset <= self._samples_to_discard:
            to_discard = self._samples_to_discard - offset
            samples -= to_discard
            data = data[..., to_discard:]

        if data.shape[-1] == 0:
            return

        s0 = max(0, offset - self._samples_to_discard)
        if s0 != self._total_samples_read:
            raise ValueError

        self.hw_ai_queue.append((data, s0))
        self._total_samples_read += samples

        # Check whether we are supposed to stop after a certain number of
        # samples. If so, stop the engine.
        if (self._samples_to_read > 0) and \
                (self._total_samples_read > self._samples_to_read):
            deferred_call(self.stop)

    def _configure_ao_cb(self, ao_channels):
        log.info('Configuring AO callback for %r', ao_channels)
        # Create mapping of channel name to input channel.
        self._hw_ao_channel_map = {c.name: i+1 for i, c in enumerate(ao_channels)}
        self._total_samples_written = 0
        self._channel_names['hw_ao'] = [c.name for c in ao_channels]
        self._hw_ao_channels = ao_channels

    def _hw_ao_callback(self, offset, samples, buffer):
        if offset != self._total_samples_written:
            raise ValueError(f'Offset alignment issue. Requested data at {offset}. Expected {self._total_samples_written}.')
        for i, channel in enumerate(self._hw_ao_channels):
            channel.get_samples(offset, samples, buffer[i])
        self._total_samples_written += samples

    def start(self):
        # Preload with data if we have an output task
        self._stream.start()
        self._running = True
        self.t0 = self._stream.stream.time
        log.info('Stream time %f', self.t0)

    def stop(self):
        if self._running:
            self._running = False
        else:
            return
        log.info('Stopping sound card %s', self.name)
        self._configured = False
        self._stream.stop()
        self.complete()

    def get_ts(self):
        try:
            return self._stream.stream.time - self.t0
        except Exception as e:
            log.exception(e)
            return np.nan
