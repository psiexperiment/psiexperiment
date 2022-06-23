import logging
log = logging.getLogger(__name__)

import threading

import numpy as np

from atom.api import (Str, Float, Bool, observe, Property, Int, Typed,
                      Value)
from enaml.core.api import Declarative, d_

from psi.core.enaml.api import PSIContribution
from psi.util import copy_declarative, get_tagged_values
from .channel import (Channel, AnalogMixin, DigitalMixin, HardwareMixin,
                      SoftwareMixin, OutputMixin, InputMixin, CounterMixin)


class LogLock:

    def __init__(self, name):
        self.name = str(name)
        self.lock = threading.Lock()

    def acquire(self, blocking=True):
        return self.lock.acquire(blocking)

    def release(self):
        return self.lock.release()

    def __enter__(self):
        self.acquire()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False    # Do not swallow exceptions


class Engine(PSIContribution):
    '''
    Defines hardware-specific interface

    The user-defind attributes are ones set by the end-user of this library in
    their IO manifest. The IO manifest is system specific and describes the
    hardware they are using for data acquisition.

    User-defined attributes
    -----------------------
    name : string
        Name of the engine. Must be unique across all engines. This name is
        used for debugging and metadata purposes.
    master_clock : bool
        If true, this engine will provide a timestamp whenever it's requested
        via `get_ts`. This is typically used for software-timed events (events
        generated by the hardware will typically have a timestamp that's
        determined by the engine that controls that particular device).
    hw_ai_monitor_period : float (sec)
        Poll period (in seconds). This defines how quickly acquired (analog
        input) data is downloaded from the buffers (and made available to
        listeners). If you want to see data as soon as possible, set the poll
        period to a small value. If your application is stalling or freezing,
        set this to a larger value. This poll period is a suggestion, not a
        contract.
    hw_ao_monitor_period : float (sec)
        Poll period (in seconds). This defines how often callbacks for the
        analog outputs are notified (i.e., to generate additional samples for
        playout).  If the poll period is too long, then the analog output may
        run out of samples. This poll period is a suggestion, not a contract.

    Attributes
    ----------
    configured : bool
        True if the hardware has been configured.

    Notes
    -----
    When subclassing, you only need to implement the callbacks required by your
    hardware. For example, if your hardware only has analog inputs, you only
    need to implement the analog input methods.
    '''

    name = d_(Str()).tag(metadata=True)

    master_clock = d_(Bool(False)).tag(metadata=True)

    lock = Value()

    configured = Bool(False)

    hw_ai_monitor_period = d_(Float(0.1)).tag(metadata=True)
    hw_ao_monitor_period = d_(Float(1)).tag(metadata=True)

    def _default_lock(self):
        return LogLock(self.name)

    def get_channels(self, mode=None, direction=None, timing=None,
                     active=True):
        '''
        Return channels matching criteria

        Parameters
        ----------
        mode : {None, 'analog', 'digital'
            Type of channel
        direction : {None, 'input, 'output'}
            Direction
        timing : {None, 'hardware', 'software'}
            Hardware or software-timed channel. Hardware-timed channels have a
            sampling frequency greater than 0.
        active : bool
            If True, return only channels that have configured inputs or
            outputs.
        '''
        channels = [c for c in self.children if isinstance(c, Channel)]

        if active:
            channels = [c for c in channels if c.active]

        if timing is not None:
            if timing in ('hardware', 'hw'):
                channels = [c for c in channels if isinstance(c, HardwareMixin)]
            elif timing in ('software', 'sw'):
                channels = [c for c in channels if isinstance(c, SoftwareMixin)]
            else:
                raise ValueError('Unsupported timing')

        if direction is not None:
            if direction in ('input', 'in'):
                channels = [c for c in channels if isinstance(c, InputMixin)]
            elif direction in ('output', 'out'):
                channels = [c for c in channels if isinstance(c, OutputMixin)]
            else:
                raise ValueError('Unsupported direction')

        if mode is not None:
            if mode == 'analog':
                channels = [c for c in channels if isinstance(c, AnalogMixin)]
            elif mode == 'digital':
                channels = [c for c in channels if isinstance(c, DigitalMixin)]
            elif mode == 'counter':
                channels = [c for c in channels if isinstance(c, CounterMixin)]
            else:
                raise ValueError('Unsupported mode')

        return tuple(channels)

    def get_channel(self, channel_name):
        channels = self.get_channels(active=False)
        for channel in channels:
            if channel.name == channel_name:
                return channel
        m = '{} channel does not exist'.format(channel_name)
        raise AttributeError(m)

    def remove_channel(self, channel):
        channel.set_parent(None)

    def configure(self):
        for channel in self.get_channels():
            log.debug('Configuring channel {}'.format(channel.name))
            channel.configure()
        self.configured = True

    def register_ai_callback(self, callback, channel_name=None):
        raise NotImplementedError

    def register_et_callback(self, callback, channel_name=None):
        raise NotImplementedError

    def unregister_ai_callback(self, callback, channel_name=None):
        raise NotImplementedError

    def unregister_et_callback(self, callback, channel_name=None):
        raise NotImplementedError

    def register_done_callback(self, callback):
        raise NotImplementedError

    def write_hw_ao(self, data, offset, timeout=1):
        '''
        Write hardware-timed analog output data to the buffer

        Parameters
        ----------
        data : 2D array
            Data to write (format channel x time)
        offset : int
            Sample at which to start writing data. Sample is relative to
            beginning of data acquisition. This can overwrite data that has
            already been written to the buffer but not consumed by the
            hardware.
        timeout : float
            Time, in seconds, to keep trying to write the data before failing.

        Notes
        -----
        When subclassing, raise an exception if the system attempts to write
        data beginning at an offset that has already been consumed by the
        hardware and cannot be modified.
        '''
        raise NotImplementedError

    def get_ts(self):
        raise NotImplementedError

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def get_ts(self):
        raise NotImplementedError

    def get_buffer_size(self, channel_name):
        raise NotImplementedError

    def update_hw_ao_multiple(self, offsets, channel_names):
        raise NotImplementedError

    def update_hw_ao(self, offsets, channel_name, method):
        raise NotImplementedError

    def clone(self, channel_names=None):
        '''
        Return a copy of this engine with specified channels included
        This is intended as a utility function to assist various routines that
        may need to do a quick operation before starting the experiment. For
        example, calibration may only need to run a subset of the channels.
        '''
        new = copy_declarative(self)
        for channel in new.children:
            channel.set_parent(None)
        if channel_names is not None:
            for channel_name in channel_names:
                channel = self.get_channel(channel_name)
                new_channel = copy_declarative(channel, parent=new,
                                               exclude=['inputs', 'outputs'])
        return new
