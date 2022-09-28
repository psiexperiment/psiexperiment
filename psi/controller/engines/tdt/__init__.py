'''
Defines the TDT engine interface

General notes for developers
----------------------------
This is a wraper around the tdtpy. Refer to the tdtpy documentation for
additional details.
'''

import logging
log = logging.getLogger(__name__)
log_ai = logging.getLogger(__name__ + '.ai')
log_ao = logging.getLogger(__name__ + '.ao')

from functools import partial
from pathlib import Path
import sys
from time import time
from threading import current_thread, Thread, Event

import numpy as np
from atom.api import (Float, Typed, Str, Int, Bool, Callable, Enum,
                      Property, Value)
from enaml.application import deferred_call
from enaml.core.api import Declarative, d_

from psiaudio.pipeline import PipelineData
from psiaudio.util import dbi

from psi import get_config
from psi.controller.api import (Engine, HardwareAIChannel, HardwareAOChannel)


from tdt import DSPCircuit, DSPProject

###############################################################################
# Engine-specific channels
###############################################################################
class TDTGeneralMixin(Declarative):

    #: Tag connected to WriteTagV or ReadTagV
    tag = d_(Str()).tag(metadata=True)

    #: Poll period (in seconds). This defines how often callbacks for the
    #: channel are triggered to read or write data. If the poll period is too
    #: long, then we may have buffer overflow/underflow errors.
    monitor_period = d_(Float(0.1)).tag(metadata=True)

    def __str__(self):
        return f'{self.label} ({self.tag})'

    def sync_start(self, channel):
        # All channels are synchronized. Nothing to be done.
        pass


class TDTHardwareAOChannel(TDTGeneralMixin, HardwareAOChannel):

    #: Maximum number of seconds to write on each call. Value should be greater
    #: than monitor_period. It shouldn't be too big though since TDT writes are
    #: very slow.
    max_write = d_(Float(2.5)).tag(metadata=True)

    # TODO: Add sanity check to verify that initial and max writes are
    # reasonable compared to monitor_period.


class TDTHardwareAIChannel(TDTGeneralMixin, HardwareAIChannel):

    #: Decimation factor. The analog inputs allow for the data to be
    #: downsampled to a rate that's an integer divisor of the circuit
    #: frequency.
    decimation = d_(Int(1)).tag(metadata=True)

    #: Filter delay in samples.
    filter_delay_samples = Property().tag(metadata=True)

    def _get_filter_delay_samples(self):
        return int(round(self.filter_delay * self.fs))


###############################################################################
# Helpers
###############################################################################
class DAQThread(Thread):

    def __init__(self, poll_interval, stop_requested, callback, name):
        log.debug('Initializing acquisition thread')
        super().__init__()
        self.poll_interval = poll_interval
        self.stop_requested = stop_requested
        self.callback = callback
        self.name = name

    def run(self):
        # This is a rather complicated piece of code because we need to
        # override the threading module's built-in exception handling as well
        # as defe the exception back to the main thread (where it will properly
        # handle exceptions). If we call psi.application.exception_handler
        # directly from the thread, it will not have access to the application
        # instance (or workspace).
        try:
            self._run()
        except:
            log.info('Caught exception')
            deferred_call(sys.excepthook, *sys.exc_info())

    def _run(self):
        profile = get_config('PROFILE', False)
        if profile:
            import cProfile
            pr = cProfile.Profile()
            pr.enable()

        log.debug('Starting acquisition thread')
        while not self.stop_requested.wait(self.poll_interval):
            stop = self.callback()
            if stop:
                break

        if profile:
            pr.disable()
            path = get_config('LOG_ROOT') / f'{self.name}_thread.pstat'
            pr.dump_stats(path)

        log.debug('Exiting acquistion thread')


################################################################################
# Engine
################################################################################
class TDTEngine(Engine):
    '''
    Hardware interface for TDT RP/RZ devices.

    This can only support one device. If you have multiple TDT devices,
    configure a separate engine for each device.
    '''
    #: Device name (e.g., RZ6, etc.)
    device_name = d_(Enum('RZ6')).tag(metadata=True)
    circuit = d_(Enum('RZ6-standard-RA4PAx20',
                      'RZ6-standard-Medusa4Z',
                      'RZ6-debugging')).tag(metadata=True)
    circuit_path = Property()

    #: Device ID (required only if you have more than one of the same device).
    #: Use zBUSmon utility to look up the correct device ID.
    device_id = d_(Int(1)).tag(metadata=True)

    #: Sampling rate.
    fs = d_(Float(195312.5)).tag(metadata=True)

    #: Flag indicating whether engine was configured
    _configured = Bool(False)

    _task_done = Typed(dict)
    _callbacks = Typed(dict, {})

    _project = Value()
    _circuit = Value()
    _buffers = Typed(dict, {})
    _threads = Typed(dict, {})
    _stop_requested = Value()
    _sf = Typed(dict, {})

    def _get_circuit_path(self):
        return Path(__file__).parent / f'{self.circuit}.rcx'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        log.info('Loading DSP circuit %s to %s %d', self.circuit_path,
                 self.device_name, self.device_id)
        self._project = DSPProject()
        self._circuit = self._project.load_circuit(self.circuit_path,
                                                   self.device_name,
                                                   self.device_id)
        self._circuit.start()
        log.debug('Loaded DSP circuit')

        # Now, we need to inspect the circuit to set the sampling rate of the
        # channels since that's hard-coded by the circuit.
        ai_chans = self.get_channels('analog', 'input', 'hardware', False)
        ao_chans = self.get_channels('analog', 'output', 'hardware', False)

        # First, verify that all channels are supported by the engine. If not,
        # raise an error. We don't actually load the channels until we call
        # `configure`, but we need to do some preliminary introspection to help
        # bootstrap other parts of psiexperiment
        for c in ai_chans:
            if not isinstance(c, TDTGeneralMixin):
                m = f'Channel {c} type {type(c)} not supported.'
                raise ValueError(m)
            if c.fs != 0:
                m = f'Sampling rate for {c} cannot be set in IO manifest'
                raise ValueError(m)
            c.fs = self._circuit.get_buffer(c.tag, 'r', dec_factor=c.decimation).fs
            log.debug('Updated sampling rate for %d %s %s to %f', id(c), c, c.name, c.fs)

        for c in ao_chans:
            if not isinstance(c, TDTGeneralMixin):
                m = f'Channel {c} type {type(c)} not supported.'
                raise ValueError(m)
            if c.fs != 0:
                m = f'Sampling rate for {c} cannot be set in IO manifest'
                raise ValueError(m)
            c.fs = self._circuit.get_buffer(c.tag, 'w').fs
            log.debug('Updated sampling rate for %d %s %s to %f', id(c), c, c.name, c.fs)

    def configure(self, active=True):
        log.debug('Configuring %s engine', self.name)
        hw_ai_channels = self.get_channels('analog', 'input', 'hardware',
                                           active=active)
        hw_ao_channels = self.get_channels('analog', 'output', 'hardware',
                                           active=active)

        # Flag used by threads to respond to stop requested event.
        self._stop_requested = Event()
        if hw_ai_channels:
            log.debug('Configuring HW AI channels')
            self.configure_hw_ai(hw_ai_channels)
        if hw_ao_channels:
            log.debug('Configuring HW AO channels')
            self.configure_hw_ao(hw_ao_channels)

        super().configure()

        # Required by start. This allows us to do the configuration on the fly
        # when starting the engines if the configure method hasn't been called
        # yet.
        self._configured = True
        log.debug('Completed engine configuration')

    def complete(self):
        log.debug('Triggering "done" callbacks')
        for cb in self._callbacks.get('done', []):
            cb()

    def configure_hw_ao(self, channels):
        '''
        Initialize hardware-timed analog output

        A single acquisition thread will be created for all buffers
        '''
        buffers = {}
        for c in channels:
            b = self._circuit.get_buffer(c.tag, 'w')
            b._max_samples = int(np.round(c.max_write * b.fs))
            buffers[c.name] = b

        self._buffers['hw_ao'] = buffers
        self._threads['hw_ao'] = DAQThread(c.monitor_period,
                                           self._stop_requested,
                                           self._hw_ao_callback,
                                           name='hw_ao')

    def configure_hw_ai(self, channels):
        '''
        Initialize hardware-timed analog input

        A single acquisition thread will be created for all buffers
        '''
        buffers = {}
        sf = {}
        for c in channels:
            b = self._circuit.get_buffer(c.tag, 'r', dec_factor=c.decimation)
            b._sf = dbi(c.gain)
            b._discard = c.filter_delay_samples
            b._discarded = 0
            # This will track the total samples read *after* discarding the
            # samples for filter delay. See _hw_ai_callback.
            b._total_samples_read = 0
            buffers[c.name] = b
        self._buffers['hw_ai'] = buffers
        self._threads['hw_ai'] = DAQThread(c.monitor_period,
                                           self._stop_requested,
                                           self._hw_ai_callback,
                                           name='hw_ai')

    def _hw_ai_callback(self):
        # TODO: Get lock?
        for name, b in self._buffers['hw_ai'].items():
            samples = b.read() / b._sf
            if b._discarded < b._discard:
                to_discard = min(samples.shape[-1], b._discard)
                b._discarded += to_discard

                log.info('Discarding %.0f samples to compensate for AI filter delay '
                         '(total %.0f discarded)', to_discard, b._discarded)
                samples = samples[..., to_discard:]

            if samples.shape[-1] > 0:
                data = PipelineData(samples, fs=b.fs, s0=b._total_samples_read,
                                    channel=[name])
                b._total_samples_read += samples.shape[-1]
                for channel_name, cb in self._callbacks.get('ai', []):
                    if channel_name == name:
                        cb(data)

    def get_buffer_size(self, name):
        #return self._buffers['hw_ao'][name].sample_time
        # This is currently hard-coded into the RPvdsEx circuit. Eventually we
        # should set it up so that the circuit is loaded at runtime and
        # inspected to set all variables needed (e.g., fs, etc.).
        return 38.4

    def get_space_available(self, name, offset=None):
        b = self._buffers['hw_ao'][name]
        size = b.available(offset)
        return min(size, b._max_samples)

    def _hw_ao_callback(self):
        # Get the next set of samples to upload to the buffer
        with self.lock:
            # `DSPBuffer.available` method doesn't need the offset as it
            # defaults to total_samples_written; however, we need to get the
            # offset so we can pass this to _get_hw_ao_samples.
            for name, b in self._buffers['hw_ao'].items():
                offset = b.total_samples_written
                samples = self.get_space_available(name, offset)
                if samples <= 0:
                    log_ao.trace('No update of %s required', name)
                    continue
                log_ao.debug('%d samples available for %s at offset %d',
                             samples, name, offset)

                data = self.get_channel(name).get_samples(offset, samples)
                self.write_hw_ao(name, data, offset)

    def update_hw_ao(self, name, offset):
        '''
        Update hardware-timed analog output starting at specified offset

        Data will be obtained from the channel for the buffer.
        '''
        # Get the number of samples that need to be written. If offset is
        # *after* the total number of samples, do nothing because this will be
        # handled by the next thread callback for the buffer.
        b = self._buffers['hw_ao'][name]
        samples = b.total_samples_written - offset
        log.debug('Updating HW AO: %d samples required', samples)
        samples = min(samples, b._max_samples)
        if samples <= 0:
            log_ao.trace('No update of hw ao required')
            return

        log_ao.trace('Update %s at %d with %d samples', name, offset, samples)
        data = self.get_channel(name).get_samples(offset, samples)
        self.write_hw_ao(name, data, offset)

    def update_hw_ao_multiple(self, names, offsets):
        '''
        Update multiple hardware-timed analog outputs at once
        '''
        # We just loop through the outputs. In other backends (e.g., NIDAQmx),
        # it's more efficient to use a different strategy since all channels
        # must be written to simultaneously. In TDTPy, we can write to each
        # channel individually.
        for n, o in zip(names, offsets):
            self.update_hw_ao(n, o)

    def write_hw_ao(self, name, data, offset, timeout=1):
        '''
        Write data to the named buffer starting at specified offset
        '''
        # Note, the TDTPy backend does not support timeout, so we ignore the
        # value.
        self._buffers['hw_ao'][name].write(data, offset)

    def get_ts(self):
        return self._circuit.cget_tag('zTime', 'n', 's')

    def start(self):
        if not self._configured:
            log.debug('Tasks were not configured yet')
            self.configure()

        # Preload waveforms into output buffers
        if 'hw_ao' in self._threads:
            self._hw_ao_callback()

        # Start the threads
        for thread in self._threads.values():
            log.debug('Started thread %r', thread)
            thread.start()

        self._stop_requested.clear()
        self._project.trigger('A', 'high')

    def stop(self):
        if not self._configured:
            return

        # Note that we don't actually verify that the threads have stopped
        # running. This seemed to be causing some sort of weird race condition
        # so I decided to eliminate that code.
        log.debug('Stopping engine')
        self._project.trigger('A', 'low')
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
