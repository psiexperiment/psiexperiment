import logging
log = logging.getLogger(__name__)

from functools import partial
import numpy as np

from atom.api import (Str, Dict, Event, Typed, Property, Float, Int,
                      Bool, List, set_default, Callable)

import enaml
from enaml.core.api import Declarative, d_

from psiaudio.stim import cos2envelope, FixedWaveform
from psiaudio.queue import AbstractSignalQueue

from psi.core.enaml.api import PSIContribution


class Synchronized(PSIContribution):

    outputs = Property()
    engines = Property()

    def _get_outputs(self):
        return self.children

    def _get_engines(self):
        return set(o.engine for o in self.outputs)


class BaseOutput(PSIContribution):

    #: Name of target. Currently only channels are supported, but we may
    #: eventually update this to support specifying another output as the
    #: target.
    target_name = d_(Str()).tag(metadata=True)

    #: The actual target.
    target = d_(Typed(Declarative).tag(metadata=True), writable=False)

    #: The channel this target eventually feeds into.
    channel = Property().tag(metadata=True)

    #: The engine controlling the channel this target eventually feeds into.
    engine = Property().tag(metadata=True)

    def _get_engine(self):
        if self.channel is None:
            return None
        return self.channel.engine

    def _get_channel(self):
        from .channel import Channel
        target = self.target
        while True:
            if target is None:
                return None
            elif isinstance(target, Channel):
                return target
            else:
                target = target.target

    def is_ready(self):
        raise NotImplementedError

    def add_output(self, output):
        raise ValueError(f'Output {name} does not accept downstream outputs')


class HardwareOutput(BaseOutput):

    # These two are defined as properties because it's theoretically possible
    # for the output to transform these (e.g., an output could upsample
    # children or "equalize" something before passing it along).
    fs = Property().tag(metadata=True)
    calibration = Property().tag(metadata=True)
    filter_delay = Property().tag(metadata=True)

    # Can the user configure properties (such as the token) via the GUI?
    configurable = d_(Bool(True))

    #: Datatype of output (double is usually good for analog outputs and bool
    #: is good for digital outputs)
    dtype = Str('double').tag(metadata=True)

    callbacks = List()

    #: Starting time of output
    start_sample = Int()

    def connect(self, cb):
        # TODO: Do we still use this?
        self.callbacks.append(cb)

    def notify(self, data):
        if not self.callbacks:
            return
        # Correct for filter delay
        d = data.copy()
        d['t0'] += self.filter_delay
        for cb in self.callbacks:
            cb(d)

    def _get_filter_delay(self):
        return self.target.filter_delay

    def _get_fs(self):
        return self.channel.fs

    def _get_calibration(self):
        return self.channel.calibration


class BaseAnalogOutput(HardwareOutput):

    # Current offset of sample generation for source. Should be set by
    # subclasses as needed. When first set, it represents the first sample of
    # the source but as samples from the source are consumed it is incremented.
    _offset = Int(0)

    def _observe_target(self, event):
        if self.target is not None:
            self.target.observe('fs', self._fs_updated)

    def _fs_updated(self, event):
        # Ensure that changes to the parent sampling rate get propagated down
        # the output hierarchy so that downstream nodes can respond accordingly
        # (e.g., plots can update their buffer sizes to accomodate data at the
        # correct sampling rate). Enaml has static and dynamic observers, so we
        # need to trigger both (this is an implementation quirk that is usually
        # invisible to most end-users).
        event = event.copy()
        event['value'] = self.fs
        self.notify('fs', event)
        member = self.get_member('fs')
        member.notify(self, event)

    def get_samples(self, offset, samples, out):
        '''
        Load new samples
        '''
        if samples == 0:
            # Not sure if we ever see this, but just in case.
            return
        if (offset + samples) < self._offset:
            # All samples occur before the output is supposed to start. Do
            # nothing. 
            return
        if offset > self._offset:
            delay = (self._offset - offset) / self.fs * 1e3
            raise ValueError(f'Missed chance to play signal. '
                             f'Start at {offset} but currently at {self._offset} ({delay:.0f} msec).')
        if offset < self._offset:
            # The output starts partway through the buffer section that is
            # requested.
            skip = self._offset - offset
            samples -= skip
            out[skip:] += self.get_next_samples(samples)
            self._offset += samples
        else:
            out[:] += self.get_next_samples(samples)
            self._offset += samples

    def get_next_samples(self, samples):
        raise NotImplementedError


class NullOutput(BaseAnalogOutput):

    def get_next_samples(self, samples):
        return np.zeros(samples, dtype=self.dtype)


class AnalogOutputWithSource(BaseAnalogOutput):

    #: This is managed by the manifest
    source = Typed(object).tag(metadata=True)
    source_md = Dict()
    active = Bool(False)
    paused = Bool(False)

    def activate(self, offset):
        log.info('Activating %s at %d', self.name, offset)
        self.active = True
        self.paused = False
        self._offset = offset

    def deactivate(self, offset=None):
        self.active = False
        self.source = None

    def pause(self, time):
        self.paused = True

    def resume(self, time):
        self.paused = False

    def get_duration(self):
        return self.source.get_duration()

    def is_ready(self):
        return self.source is not None

    def get_samples(self, offset, samples, out):
        if not self.active or self.paused:
            return
        super().get_samples(offset, samples, out)


class EpochOutput(AnalogOutputWithSource):

    # TODO: clean this up. it's sort of hackish.
    token = d_(Typed(Declarative)).tag(metadata=True)

    def set_waveform(self, waveform):
        # Ensure output is deactivated before new waveform is set, otherwise we
        # end up with edge conditions where the new waveform is played almost
        # immediately if the output is left in an active state from a previous
        # call to `start_waveform`.
        self.deactivate()
        self.source = FixedWaveform(self.fs, waveform)

    def start_waveform(self, ts):
        sample = round(int(self.fs * ts))
        self.activate(sample)

    def stop_waveform(self, ts):
        sample = round(int(self.fs * ts))
        self.deactivate(sample)

    def get_next_samples(self, samples):
        if self.source is None:
            return np.zeros(samples)
        return self.source.next(samples)


class RampedEpochOutput(EpochOutput):

    ramp = Value()

    def set_waveform(self, waveform, ramp_time=25e-3):
        # Generate an "inverse" ramp that attenuates the existing samples in
        # the output buffer. Generate it as if 0 means no attenuation
        # and 1 is full attenuation.
        if ramp_time is not None:
            ramp = cos2envelope(
                fs=self.fs,
                duration=waveform.shape[-1] / self.fs,
                rise_time=ramp_time,
            )
        else:
            ramp = np.zeros_like(waveform)
        self.ramp = FixedWaveform(self.fs, ramp)
        super().set_waveform(waveform)

    def get_samples(self, offset, samples, out):
        '''
        Load new samples
        '''
        if samples == 0:
            # Not sure if we ever see this, but just in case.
            return
        if (offset + samples) < self._offset:
            # All samples occur before the output is supposed to start. Do
            # nothing. 
            return
        if offset > self._offset:
            delay = (self._offset - offset) / self.fs * 1e3
            raise ValueError(f'Missed chance to play signal. '
                             f'Start at {self._offset} but currently at {offset} ({delay:.0f} msec).')
        if offset < self._offset:
            # The output starts partway through the buffer section that is
            # requested.
            skip = self._offset - offset
            samples -= skip
            self.write_next_samples(out[skip:])
            self._offset += samples
        else:
            self.write_next_samples(out)
            self._offset += samples

    def write_next_samples(self, out):
        if self.paused or not self.active:
            return

        samples = len(out)
        s = self.source.next(samples)
        # In the ramp, 1 means full attenuation. Calculate the scaling factor
        # needed to apply the ramp.
        r = self.ramp.next(samples)
        sf = 1 - r

        # out is the buffer that will be sent to the DAC . Write *into* the
        # buffer. Multiply by the attenuation factor (0 = full attentuation, 1
        # = no attenuation) and then add the stimulus waveform in.
        out[:] = out * sf + s


class MUXOutput(HardwareOutput):

    outputs = List().tag(metadata=True)

    def add_output(self, o):
        if o in self.outputs:
            return
        self.outputs.append(o)
        o.target = self

    def remove_output(self, o):
        if o not in self.outputs:
            return
        self.outputs.remove(o)
        o.target = None

    def get_samples(self, offset, samples, out):
        for output in self.outputs:
            output.get_samples(offset, samples, out)

    def get_next_samples(self, samples):
        s = np.zeros(samples, dtype=self.dtype)
        for output in self.outputs:
            s += output.get_next_samples(samples)
        return s


class QueuedEpochOutput(EpochOutput):

    queue = d_(Typed(AbstractSignalQueue))

    #: Automatically decrement the number of trials left to present? Set to
    #: False if you plan to handle this yourself (e.g., in the case of artifact
    #: reject).
    auto_decrement = d_(Bool(False)).tag(metadata=True)

    complete = d_(Event(), writable=False)
    removed_callbacks = List()

    def connect(self, cb, event='added'):
        if event == 'added':
            self.callbacks.append(cb)
        elif event == 'removed':
            self.removed_callbacks.append(cb)

    def notify_removed(self, data):
        if not self.removed_callbacks:
            return
        d = data.copy()
        d['t0'] += self.filter_delay
        for cb in self.removed_callbacks:
            cb(d)

    def rebuffer(self, time):
        self.queue.cancel(time)
        self.queue.rewind_samples(time)
        super().rebuffer(time)

    def pause(self, time):
        self.queue.pause(time)
        super().pause(time)

    def resume(self, time, delay=0):
        self.queue.resume(time)
        super().resume(time)

    def _observe_queue(self, event):
        self.source = self.queue
        self._update_queue()

    def _observe_target(self, event):
        super()._observe_target(event)
        self._update_queue()

    def _update_queue(self):
        if self.queue is not None and self.target is not None:
            self.queue.set_fs(self.fs)
            self.queue.connect(self.notify, 'added')
            self.queue.connect(self.notify_removed, 'removed')

    def get_next_samples(self, samples):
        if self.paused or not self.active:
            return np.zeros(samples, dtype=self.dtype)

        waveform = self.queue.pop_buffer(samples, self.auto_decrement)
        if self.queue.is_empty():
            self.complete = True
            self.active = False
            log.debug('Queue empty. Output %s no longer active.', self.name)
        return waveform

    def add_setting(self, setting, averages=None, iti_duration=None,
                    total_duration=None):
        if iti_duration is not None and total_duration is not None:
            raise ValueError('Cannot specify both iti_duration and total_duration')
        elif iti_duration is None and total_duration is None:
            raise ValueError('must specify either iti_duration or total_duration')

        with enaml.imports():
            # TODO: HACK ALERT!
            from .output_manifest import initialize_factory

        # Make a copy to ensure that we don't accidentally modify in-place
        context = setting.copy()

        # I'm not in love with this since it requires hooking into the
        # manifest system.
        factory = initialize_factory(self, self.token, context)

        sf = factory.max_amplitude() * 1.01
        if sf > self.channel.max_range[1]:
            from .channel import ChannelOutOfRange
            raise ChannelOutOfRange(sf, self.channel.max_range[1])

        duration = factory.get_duration()
        if total_duration is not None:
            iti_duration = total_duration - duration
            if iti_duration < 0:
                raise ValueError(f'ITI duratation cannot be negative. '
                                 f'Requested total duration of signal is {total_duration} '
                                 f'but minimum duration is {duration}.')
        key = self.queue.append(factory, averages, iti_duration, duration,
                                setting.copy())
        return key, factory

    def activate(self, offset):
        log.debug('Activating output at %d', offset)
        super().activate(offset)
        self.queue.set_t0(offset/self.fs)

    def deactivate(self, offset):
        raise NotImplementedError

    def get_duration(self):
        # TODO: add a method to get actual duration from queue.
        return np.inf


class ContinuousOutput(BaseAnalogOutput):

    # TODO: clean this up. it's sort of hackish. Do we even need to
    # differentiate between this and EpochOutput? 
    token = d_(Typed(Declarative)).tag(metadata=True)
    active = Bool(False)
    paused = Bool(False)

    def get_next_samples(self, samples):
        if self.paused or not self.active:
            return np.zeros(samples, dtype=self.dtype)
        return self.source.next(samples)

    def get_duration(self):
        return np.inf


class ContinuousCallbackOutput(BaseAnalogOutput):

    active = Bool(False)
    paused = Bool(False)
    callback = Callable()

    def get_next_samples(self, samples):
        if self.paused or not self.active:
            return np.zeros(samples, dtype=self.dtype)
        return self.callback(samples, self.name)


class ContinuousQueuedOutput(ContinuousOutput):

    notifiers = Dict()

    def _default_notifiers(self):
        return {
            'added': [],
            'removed': [],
            'decrement': [],
        }

    def connect(self, callback, event='added'):
        if event not in self.notifiers:
            raise KeyError(f'Event "{event}" not valid')
        self.notifiers[event].append(callback)

    def notify(self, event, info):
        for notifier in self.notifiers[event]:
            notifier(info)

    def pause(self, time):
        self.source.queue.pause(time)
        super().pause(time)

    def resume(self, time):
        self.source.queue.resume(time)
        super().resume(time)


class TimedTrigger(HardwareOutput):

    dtype = set_default('bool')

    #: First sample of TTL
    _start = Int(0)

    #: Last sample of TTL
    _stop = Int(0)

    #: Desired level of TTL on output.
    ttl_level = Float(1)

    def trigger(self, timestamp, duration):
        if timestamp is None:
            timestamp = 0
        self._start = int(round(self.fs * timestamp))
        self._stop = self._start + int(round(self.fs * duration))

    def get_samples(self, offset, samples, out):
        '''
        Load TTL into buffer
        '''
        # Check to see if we need to do anythying.
        if samples == 0:
            return
        if (offset + samples) < self._start:
            return
        if offset > self._stop:
            return

        # Clip bounds to range of samples needed in buffer
        lb = max(0, self._start - offset)
        ub = min(samples, self._stop - offset)
        out[lb:ub] += self.ttl_level


class Trigger(BaseOutput):

    #: Total number of triggers sent (useful for tracking things like the
    #: number of pellets dispensed from a triggered pellet dispenser).
    total_fired = d_(Int(0), writable=False)

    #: Specifies the default duration for the trigger. This can be overridden
    #: by specifying a duration when calling `fire`.
    duration = d_(Float(0.1))

    def fire(self, duration=None):
        if duration is None:
            duration = self.duration
        if self.engine.configured:
            self.engine.fire_sw_do(self.channel.name, duration=duration)
            self.total_fired += 1


class Toggle(BaseOutput):

    state = Bool(False)

    def _observe_state(self, event):
        if self.engine is not None and self.engine.configured:
            self.engine.set_sw_do(self.channel.name, event['value'])

    def set_high(self):
        self.state = True

    def set_low(self):
        self.state = False
