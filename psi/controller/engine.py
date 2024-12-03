import logging
log = logging.getLogger(__name__)

import threading

import numpy as np

from atom.api import (Str, Float, Bool, observe, Property, Int, List, Typed,
                      Value)
from enaml.core.api import Declarative, d_

from psi.core.enaml.api import PSIContribution
from psi.util import copy_declarative, get_tagged_values
from .channel import (Channel, AnalogMixin, DigitalMixin, HardwareMixin,
                      SoftwareMixin, OutputMixin, InputMixin, CounterMixin)


class EngineStoppedException(IOError):
    '''
    Indicates that the engine has been stopped
    '''
    pass


class Engine(PSIContribution):
    '''
    Defines hardware-specific interface

    The user-defind attributes are ones set by the end-user of this library in
    their IO manifest. The IO manifest is system specific and describes the
    hardware they are using for data acquisition.

    Notes
    -----
    When subclassing, you only need to implement the callbacks required by your
    hardware. For example, if your hardware only has analog inputs, you only
    need to implement the analog input methods.
    '''

    #: Name of the engine. Must be unique across all engines. This name is used
    #: for debugging and metadata purposes.
    name = d_(Str()).tag(metadata=True)

    #: If True, this engine will provide a timestamp whenever it's requested
    #: via `get_ts`. This is typically used for software-timed events (events
    #: generated by the hardware will typically have a timestamp that's
    #: determined by the engine that controls that particular device).
    master_clock = d_(Bool(False)).tag(metadata=True)

    #: Indicates order in which engines should be started (higher numbers will
    #: be started last). Typically the engine with the master clock should be
    #: last. If no engine has a master clock, then the last engine will be used
    #: as the master clock source.
    weight = d_(Int()).tag(metadata=True)

    #: Used to ensure synchronization of threads.
    lock = Value()

    #: Indicates that engine has been stopped.
    stopped = Value()

    #: True if the hardware has been configured.
    configured = Bool(False)

    #: Poll period (in seconds). This defines how quickly acquired (analog
    #: input) data is downloaded from the buffers (and made available to
    #: listeners). If you want to see data as soon as possible, set the poll
    #: period to a small value. If your application is stalling or freezing,
    #: set this to a larger value. This poll period is a suggestion, not a
    #: contract.
    hw_ai_monitor_period = d_(Float(0.1)).tag(metadata=True)

    #: Poll period (in seconds). This defines how often callbacks for the
    #: analog outputs are notified (i.e., to generate additional samples for
    #: playout).  If the poll period is too long, then the analog output may
    #: run out of samples. This poll period is a suggestion, not a contract.
    hw_ao_monitor_period = d_(Float(1)).tag(metadata=True)

    channels = List(Typed(Channel))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # This is needed to give Conditional and Looper blocks a chance to
        # properly set up.
        self.initialize()

    def initialized(self):
        self.channels = [c for c in self.children if isinstance(c, Channel)]

    def _default_lock(self):
        return threading.RLock()

    def _default_stopped(self):
        return threading.Event()

    def add_channel(self, channel):
        self.channels.append(channel)
        channel.engine = self

    def get_channels(self, mode=None, direction=None, timing=None,
                     active=True):
        '''
        Return channels matching criteria

        Parameters
        ----------
        mode : {None, 'analog', 'digital', 'counter'}
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
        channels = self.channels[:]

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
        # Channel names are sometimes prefixed with the channel type (e.g.,
        # hw_ao) and will appear as "hw_ao::speaker_1" instead of "speaker_1".
        if '::' in channel_name:
            _, channel_name = channel_name.split('::')
        channels = self.get_channels(active=False)
        for channel in channels:
            if channel.name == channel_name:
                return channel
        valid = ', '.join(f'{c.name}' for c in channels)
        raise AttributeError(f'{channel_name} channel does not exist. '
                             f'Valid channels are {valid}.')

    def remove_channel(self, channel):
        channel.set_parent(None)
        self.channels.remove(channel)

    def configure(self):
        for channel in self.get_channels():
            log.debug('Configuring channel {}'.format(channel.name))
            channel.configure()
        self.configured = True

    def register_callback(self, callback, type_code, channel_name):
        '''
        Register callback given channel type code and name
        '''
        timing, ctype = type_code.split('_')
        if timing != 'hw':
            raise ValueError('Can only register callbacks for hardware-timed tasks')
        getattr(self, f'register_{ctype}_callback')(callback, channel_name)

    def register_ai_callback(self, callback, channel_name=None):
        raise NotImplementedError

    def register_ci_callback(self, callback, channel_name=None):
        raise NotImplementedError

    def unregister_ai_callback(self, callback, channel_name=None):
        raise NotImplementedError

    def unregister_ci_callback(self, callback, channel_name=None):
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

        Parameters
        ----------
        channel_names : {None, list of string}
            Names of channels to clone to new engine. If None, all channels
            will be cloned.
        '''
        new = copy_declarative(self)
        for channel in new.children[:]:
            channel.set_parent(None)
        if channel_names is None:
            channel_names = [c.name for c in self.get_channels(active=False)]
        if channel_names is not None:
            for channel_name in channel_names:
                channel = self.get_channel(channel_name)
                new_channel = copy_declarative(channel, parent=new,
                                               exclude=['inputs', 'outputs'])
        return new
