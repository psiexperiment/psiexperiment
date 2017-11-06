'''
Defines the NIDAQmx engine interface

General notes for developers
-----------------------------------------------------------------------------
This is a wraper around the NI-DAQmx C API. Refer to the NI-DAQmx C reference
(available as a Windows help file or as HTML documentation on the NI website).
Google can help you quicly find the online documentation).

This code is under heavy development and I may change the API in significant
ways. In general, the only portion of the code you should use in third-party
modules is the `Engine` class. This will serve as the sole communication layer
between the NI hardware and your application. By doing so, this ensures a
sufficient layer of abstraction that helps switch between DAQ hardware from
different vendors (including Measurement Computing and OpenElec).

Some parts of the code takes advantage of generators and coroutines. For details
on this topic, see the following resources:

    http://www.dabeaz.com/coroutines/
    http://www.python.org/dev/peps/pep-0342/

'''

import logging
log = logging.getLogger(__name__)

import types
import ctypes
from collections import OrderedDict
from threading import Timer

import numpy as np
import PyDAQmx as mx
from atom.api import Float, Typed, Unicode, Int, Bool
from enaml.core.api import Declarative, d_

from ..engine import Engine


################################################################################
# PSI utility
################################################################################
def get_channel_property(channels, property, allow_unique=False):
    values = [getattr(c, property) for c in channels]
    if allow_unique:
        return values
    elif len(set(values)) != 1:
        m = 'NIDAQEngine does not support per-channel {} as specified: {}' \
            .format(property, values)
        raise ValueError(m)
    else:
        return values[0]


################################################################################
# DAQmx utility
################################################################################
def read_digital_lines(task, size=1):
    nlines = ctypes.c_uint32()
    mx.DAQmxGetDINumLines(task, '', nlines)
    nsamp = ctypes.c_int32()
    nbytes = ctypes.c_int32()
    data = np.empty((size, nlines.value), dtype=np.uint8)
    mx.DAQmxReadDigitalLines(task, size, 0, mx.DAQmx_Val_GroupByChannel, data,
                             data.size, nsamp, nbytes, None)
    return data.T


def constant_lookup(value):
    for name in dir(mx.DAQmxConstants):
        if name in mx.DAQmxConstants.constant_list:
            if getattr(mx.DAQmxConstants, name) == value:
                return name
    raise ValueError('Constant {} does not exist'.format(value))


def channel_list(task):
    channels = ctypes.create_string_buffer(b'', 4096)
    mx.DAQmxGetTaskChannels(task, channels, len(channels))
    return [c.strip() for c in channels.value.split(b',')]


def verify_channel_names(task, names):
    lines = channel_list(task)
    if names is not None:
        if len(lines) != len(names):
            m = 'Number of names must match number of lines. ' \
                'Lines: {}, names: {}'
            raise ValueError(m.format(lines, names))
    else:
        names = lines
    return names


def device_list(task):
    devices = ctypes.create_string_buffer(b'', 4096)
    mx.DAQmxGetTaskDevices(task, devices, len(devices))
    return [d.strip() for d in devices.value.split(b',')]


################################################################################
# callback
################################################################################
class SamplesGeneratedCallbackHelper(object):

    def __init__(self, callback):
        self._callback = callback
        self._uint32 = ctypes.c_uint32()

    def __call__(self, task, event_type, callback_samples, callback_data):
        try:
            self._callback(callback_samples)
            return 0
        except Exception as e:
            log.exception(e)
            return -1


class SamplesAcquiredCallbackHelper(object):

    def __init__(self, callback, n_channels):
        self._callback = callback
        self._n_channels = n_channels
        self._int32 = ctypes.c_int32()
        self._uint32 = ctypes.c_uint32()

    def __call__(self, task, event_type, callback_samples, callback_data):
        try:
            while True:
                mx.DAQmxGetReadAvailSampPerChan(task, self._uint32)
                available_samples = self._uint32.value
                blocks = (available_samples//callback_samples)
                if blocks == 0:
                    break
                samples = blocks*callback_samples
                data_shape = self._n_channels, samples
                data = np.empty(data_shape, dtype=np.double)
                mx.DAQmxReadAnalogF64(task, samples, 0,
                                    mx.DAQmx_Val_GroupByChannel, data, data.size,
                                    self._int32, None)
                self._callback(data)
                #log.trace('Acquired {} samples'.format(data.shape))
            return 0
        except Exception as e:
            log.exception(e)
            return -1


class DigitalSamplesAcquiredCallbackHelper(object):

    def __init__(self, callback):
        self._callback = callback

    def __call__(self, task, event_type, callback_samples, callback_data):
        try:
            data = read_digital_lines(task, callback_samples)
            self._callback(data)
            return 0
        except Exception as e:
            raise
            log.exception(e)
            return -1


################################################################################
# Configuration functions
################################################################################
def setup_timing(task, fs, samples=np.inf, start_trigger=None):
    '''
    Configures timing for task

    Parameters
    ----------
    task : niDAQmx task handle
        Task to configure timing for
    fs : string or float
        If string, must be the name of a sample clock (e.g. ao/SampleClock) that
        the acquistion will be tied to. If float, the internal sample clock for
        that task will be used and the sample rate will be set to fs.
    samples :  float
        If infinite, the task will be set to continuous acquistion. If finite,
        the task will be set to acquire the specified number of samples.
    start_trigger : string
        Name of digtial line to start acquisition on. Can be set to any PFI line
        or to one of the analog sources (e.g., ao/StartTrigger or
        ai/StartTrigger).
    '''
    if start_trigger:
        mx.DAQmxCfgDigEdgeStartTrig(task, start_trigger, mx.DAQmx_Val_Rising)
    if isinstance(fs, str):
        sample_clock = fs
        fs = 200e3
    else:
        sample_clock = ''
    if samples is np.inf:
        sample_mode = mx.DAQmx_Val_ContSamps
        samples = int(fs)
    else:
        sample_mode = mx.DAQmx_Val_FiniteSamps
    mx.DAQmxCfgSampClkTiming(task, sample_clock, fs, mx.DAQmx_Val_Rising,
                             sample_mode, samples)


def create_task(name=None):
    '''
    Create niDAQmx task

    Parameters
    ----------
    name : {None, str}
        Task name (optional). Primarily useful only for debugging purposes
        (e.g., this is what's reported in NI error messages)

    Returns
    -------
    task : ctypes pointer
        Pointer to niDAQmx task
    '''
    if name is None:
        name = ''
    task = mx.TaskHandle(0)
    mx.DAQmxCreateTask(name, ctypes.byref(task))
    task._name = name
    return task


def setup_hw_ao(fs, lines, expected_range, callback, callback_samples,
                start_trigger=None, terminal_mode=None, buffer_samples=None,
                task_name='hw_ao'):

    # TODO: DAQmxSetAOTermCfg
    task = create_task(task_name)
    lb, ub = expected_range
    mx.DAQmxCreateAOVoltageChan(task, lines, '', lb, ub, mx.DAQmx_Val_Volts, '')
    setup_timing(task, fs, np.inf, start_trigger)

    if terminal_mode is not None:
        mx.DAQmxSetAOTermCfg(task, lines, terminal_mode)

    if start_trigger:
        mx.DAQmxCfgDigEdgeStartTrig(task, start_trigger, mx.DAQmx_Val_Rising)

    # If the write reaches the end of the buffer and no new data has been
    # provided, do not loop around to the beginning and start over.
    mx.DAQmxSetWriteRegenMode(task, mx.DAQmx_Val_DoNotAllowRegen)

    if buffer_samples is None:
        buffer_samples = int(callback_samples*10)
    log.debug('Setting output buffer size to %d samples', buffer_samples)
    mx.DAQmxSetBufOutputBufSize(task, buffer_samples)
    task._buffer_samples = buffer_samples

    result = ctypes.c_uint32()
    mx.DAQmxGetTaskNumChans(task, result)
    task._n_channels = result.value
    log.debug('%d channels in task', task._n_channels)

    #mx.DAQmxSetAOMemMapEnable(task, lines, True)
    mx.DAQmxSetAODataXferReqCond(task, lines, mx.DAQmx_Val_OnBrdMemHalfFullOrLess)

    # This controls how quickly we can update the buffer on the device. On some
    # devices it is not user-settable. On the X-series PCIe-6321 I am able to
    # change it. On the M-xeries PCI 6259 it appears to be fixed at 8191
    # samples. Haven't really been able to do much about this.
    mx.DAQmxGetBufOutputOnbrdBufSize(task, result)
    task._onboard_buffer_size = result.value
    log.debug('Onboard buffer size %d', task._onboard_buffer_size)

    result = ctypes.c_int32()
    mx.DAQmxGetAODataXferMech(task, lines, result)
    log.debug('Data transfer mechanism %d', result.value)
    mx.DAQmxGetAODataXferReqCond(task, lines, result)
    log.debug('Data transfer condition %d', result.value)
    result = ctypes.c_uint32()
    mx.DAQmxGetAOUseOnlyOnBrdMem(task, lines, result)
    log.debug('Use only onboard memory %d', result.value)
    mx.DAQmxGetAOMemMapEnable(task, lines, result)
    log.debug('Memory mapping enabled %d', result.value)


    #result = ctypes.c_int32()
    #mx.DAQmxGetAODataXferMech(task, result)
    #log.debug('DMA transfer mechanism %d', result.value)

    log.debug('Creating callback after every %d samples', callback_samples)
    callback_helper = SamplesGeneratedCallbackHelper(callback)
    cb_ptr = mx.DAQmxEveryNSamplesEventCallbackPtr(callback_helper)
    mx.DAQmxRegisterEveryNSamplesEvent(task,
                                       mx.DAQmx_Val_Transferred_From_Buffer,
                                       int(callback_samples), 0, cb_ptr, None)
    task._cb_ptr = cb_ptr
    mx.DAQmxTaskControl(task, mx.DAQmx_Val_Task_Verify)

    return task


def setup_hw_ai(fs, lines, expected_range, callback, callback_samples,
                start_trigger, terminal_mode, terminal_coupling,
                task_name='hw_ai'):

    task = create_task(task_name)
    lb, ub = expected_range
    mx.DAQmxCreateAIVoltageChan(task, lines, '', terminal_mode, lb, ub,
                                mx.DAQmx_Val_Volts, '')

    if terminal_coupling is not None:
        mx.DAQmxSetAICoupling(task, lines, terminal_coupling)

    if start_trigger:
        mx.DAQmxCfgDigEdgeStartTrig(task, start_trigger, mx.DAQmx_Val_Rising)

    mx.DAQmxCfgSampClkTiming(task, '', fs, mx.DAQmx_Val_Rising,
                             mx.DAQmx_Val_ContSamps, int(fs))

    result = ctypes.c_uint32()
    #mx.DAQmxGetBufInputBufSize(task, result)
    #buffer_size = result.value
    mx.DAQmxGetTaskNumChans(task, result)
    n_channels = result.value

    #log.debug('Buffer size for %s automatically allocated as %d samples',
    #          lines, buffer_size)
    #log.debug('%d channels in task', n_channels)

    #new_buffer_size = np.ceil(buffer_size/callback_samples)*callback_samples
    #mx.DAQmxSetBufInputBufSize(task, int(new_buffer_size))
    #n_channels = 1

    mx.DAQmxSetBufInputBufSize(task, callback_samples*100)
    mx.DAQmxGetBufInputBufSize(task, result)
    buffer_size = result.value
    log.debug('Buffer size for %s set to %d samples', lines, buffer_size)

    callback_helper = SamplesAcquiredCallbackHelper(callback, n_channels)
    cb_ptr = mx.DAQmxEveryNSamplesEventCallbackPtr(callback_helper)
    mx.DAQmxRegisterEveryNSamplesEvent(task, mx.DAQmx_Val_Acquired_Into_Buffer,
                                       int(callback_samples), 0, cb_ptr, None)

    info = ctypes.c_double()
    mx.DAQmxGetSampClkRate(task, info)
    log.debug('AI sample rate'.format(info.value))
    mx.DAQmxGetSampClkTimebaseRate(task, info)
    log.debug('AI timebase {}'.format(info.value))
    try:
        mx.DAQmxGetAIFilterDelay(task, lines, info)
        log.debug('AI filter delay {}'.format(info.value))
    except mx.DAQError:
        # Not a supported property
        pass
    task._cb_ptr = cb_ptr
    task._cb_helper = callback_helper

    mx.DAQmxTaskControl(task, mx.DAQmx_Val_Task_Verify)
    return task


def setup_hw_di(fs, lines, callback, callback_samples, start_trigger=None,
                clock=None, task_name='hw_di'):
    '''
    M series DAQ cards do not have onboard timing engines for digital IO.
    Therefore, we have to create one (e.g., using a counter or by using the
    analog input or output sample clock.
    '''
    task = create_task(task_name)
    mx.DAQmxCreateDIChan(task, lines, '', mx.DAQmx_Val_ChanForAllLines)

    # Get the current state of the lines so that we know what happened during
    # the first change detection event. Do this before configuring the timing
    # of the lines (otherwise we have to start the master clock as well)!
    mx.DAQmxStartTask(task)
    initial_state = read_digital_lines(task, 1)
    mx.DAQmxStopTask(task)

    # M-series acquisition boards don't have a dedicated engine for digital
    # acquisition. Use a clock to configure the acquisition.
    if clock is not None:
        clock_task = create_task('{}_clock'.format(task_name))
        mx.DAQmxCreateCOPulseChanFreq(clock_task, clock, '', mx.DAQmx_Val_Hz,
                                      mx.DAQmx_Val_Low, 0, fs, 0.5)
        mx.DAQmxCfgImplicitTiming(clock_task, mx.DAQmx_Val_ContSamps, int(fs))
        clock += 'InternalOutput'
        if start_trigger:
            mx.DAQmxCfgDigEdgeStartTrig(clock_task, start_trigger,
                                        mx.DAQmx_Val_Rising)
        setup_timing(task, clock, np.inf, None)
    else:
        setup_timing(task, fs, np.inf, start_trigger)

    callback_helper = DigitalSamplesAcquiredCallbackHelper(callback)
    cb_ptr = mx.DAQmxEveryNSamplesEventCallbackPtr(callback_helper)
    mx.DAQmxRegisterEveryNSamplesEvent(task, mx.DAQmx_Val_Acquired_Into_Buffer,
                                       int(callback_samples), 0, cb_ptr, None)

    task._cb_ptr = cb_ptr
    task._initial_state = initial_state

    #mx.DAQmxTaskControl(task, mx.DAQmx_Val_Task_Commit)
    rate = ctypes.c_double()
    mx.DAQmxGetSampClkRate(task, rate)

    mx.DAQmxTaskControl(task, mx.DAQmx_Val_Task_Verify)
    mx.DAQmxTaskControl(clock_task, mx.DAQmx_Val_Task_Verify)

    return [task, clock_task]


def setup_sw_ao(lines, expected_range, task_name='sw_ao'):
    # TODO: DAQmxSetAOTermCfg
    task = create_task(task_name)
    lb, ub = expected_range
    mx.DAQmxCreateAOVoltageChan(task, lines, '', lb, ub, mx.DAQmx_Val_Volts, '')
    mx.DAQmxTaskControl(task, mx.DAQmx_Val_Task_Verify)
    return task


def setup_sw_do(lines, task_name='sw_do'):
    task = create_task(task_name)
    mx.DAQmxCreateDOChan(task, lines, '', mx.DAQmx_Val_ChanForAllLines)
    mx.DAQmxTaskControl(task, mx.DAQmx_Val_Task_Verify)
    return task


################################################################################
# Engine
################################################################################
class NIDAQEngine(Engine):
    '''
    Hardware interface

    The tasks are started in the order they are configured. Most NI devices can
    only support a single hardware-timed task of a specified type (e.g., analog
    input, analog output, digital input, digital output are all unique task
    types).
    '''
    # TODO: Why is this relevant?
    engine_name = 'nidaq'

    # Flag indicating whether engine was configured
    _configured = Bool(False)

    # Poll period (in seconds). This defines how often callbacks for the analog
    # outputs are notified (i.e., to generate additional samples for playout).
    # If the poll period is too long, then the analog output may run out of
    # samples.
    hw_ao_monitor_period = d_(Float(1)).tag(metadata=True)

    # Size of buffer (in seconds). This defines how much data is pregenerated
    # for the buffer before starting acquisition. This is impotant because
    hw_ao_buffer_size = d_(Float(10)).tag(metadata=True)

    # Poll period (in seconds). This defines how quickly acquired (analog input)
    # data is downloaded from the buffers (and made available to listeners). If
    # you want to see data as soon as possible, set the poll period to a small
    # value. If your application is stalling or freezing, set this to a larger
    # value.
    hw_ai_monitor_period = d_(Float(0.1)).tag(metadata=True)

    # Even though data is written to the analog outputs, it is buffered in
    # computer memory until it's time to be transferred to the onboard buffer of
    # the NI acquisition card. NI-DAQmx handles this behind the scenes (i.e.,
    # when the acquisition card needs additional samples, NI-DAQmx will transfer
    # the next chunk of data from the computer memory). We can overwrite data
    # that's been buffered in computer memory (e.g., so we can insert a target
    # in response to a nose-poke). However, we cannot overwrite data that's
    # already been transfered to the onboard buffer. So, the onboard buffer size
    # determines how quickly we can change the analog output in response to an
    # event.
    # TODO: this is not configurable on some systems. How do we figure out if
    # it's configurable?
    hw_ao_onboard_buffer = d_(Int(4095)).tag(metadata=True)

    # Since any function call takes a small fraction of time (e.g., nanoseconds
    # to milliseconds), we can't simply overwrite data starting at
    # hw_ao_onboard_buffer+1. By the time the function calls are complete, the
    # DAQ probably has already transferred a couple hundred samples to the
    # buffer. This parameter will likely need some tweaking (i.e., only you can
    # determine an appropriate value for this based on the needs of your
    # program).
    hw_ao_min_writeahead = d_(Int(8191 + 1000)).tag(metadata=True)

    _tasks = Typed(dict)
    _callbacks = Typed(dict)
    _timers = Typed(dict)
    _uint32 = Typed(ctypes.c_uint32)
    _uint64 = Typed(ctypes.c_uint64)
    _int32 = Typed(ctypes.c_int32)

    ao_fs = Typed(float).tag(metadata=True)

    terminal_mode_map = {
        'differential': mx.DAQmx_Val_Diff,
        'pseudodifferential': mx.DAQmx_Val_PseudoDiff,
        'RSE': mx.DAQmx_Val_RSE,
        'NRSE': mx.DAQmx_Val_NRSE,
        'default': mx.DAQmx_Val_Cfg_Default,
    }

    terminal_coupling_map = {
        None: None,
        'AC': mx.DAQmx_Val_AC,
        'DC': mx.DAQmx_Val_DC,
        'ground': mx.DAQmx_Val_GND,
    }

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

        # Use an OrderedDict to ensure that when we loop through the tasks
        # stored in the dictionary, we process them in the order they were
        # configured.
        self._tasks = OrderedDict()
        self._callbacks = {}
        self._timers = {}
        self._configured = False

        # These are pointers to C datatypes that are required for communicating
        # with the NI-DAQmx library. When querying various properties of tasks,
        # channels and buffers, the NI-DAQmx function often requires an integer
        # of a specific type (e.g. unsigned 32-bit, unsigned 64-bit, etc.). This
        # integer must be passed by reference, allowing the NI-DAQmx function to
        # modify the value directly. For example:
        #
        #     mx.DAQmxGetWriteSpaceAvail(task, self._uint32)
        #     print(self._uint32.value)
        #
        # The ctypes library facilitates communicating with the NI-DAQmx C-API
        # by providing wrappers around C datatypes that can be passed by
        # reference.
        self._uint32 = ctypes.c_uint32()
        self._uint64 = ctypes.c_uint64()
        self._int32 = ctypes.c_int32()

    def configure(self, plugin=None):
        log.debug('Configuring {} engine'.format(self.name))
        if self.sw_do_channels:
            log.debug('Configuring SW DO channels')
            channels = self.sw_do_channels
            lines = ','.join(get_channel_property(channels, 'channel', True))
            names = get_channel_property(channels, 'name', True)
            self.configure_sw_do(lines, names)

        if self.hw_ai_channels:
            log.debug('Configuring HW AI channels')
            channels = self.hw_ai_channels
            lines = ','.join(get_channel_property(channels, 'channel', True))
            names = get_channel_property(channels, 'name', True)
            fs = get_channel_property(channels, 'fs')
            start_trigger = get_channel_property(channels, 'start_trigger')
            expected_range = get_channel_property(channels, 'expected_range')

            tmode = get_channel_property(channels, 'terminal_mode')
            tcoupling = get_channel_property(channels, 'terminal_coupling')
            terminal_mode = self.terminal_mode_map[tmode]
            terminal_coupling = self.terminal_coupling_map[tcoupling]

            self.configure_hw_ai(fs, lines, expected_range, names,
                                 start_trigger, terminal_mode,
                                 terminal_coupling)

        if self.hw_di_channels:
            log.debug('Configuring HW DI channels')
            channels = self.hw_di_channels
            lines = ','.join(get_channel_property(channels, 'channel', True))
            names = get_channel_property(channels, 'name', True)
            fs = get_channel_property(channels, 'fs')
            start_trigger = get_channel_property(channels, 'start_trigger')
            # Required for M-series to enable hardware-timed digital
            # acquisition. TODO: Make this a setting that can be configured
            # since X-series doesn't need this hack.
            device = channels[0].channel.strip('/').split('/')[0]
            clock = '/{}/Ctr0'.format(device)
            self.configure_hw_di(fs, lines, names, start_trigger, clock)

        # Configure the analog output last because acquisition is synced with
        # the analog output signal (i.e., when the analog output starts, the
        # analog input begins acquiring such that sample 0 of the input
        # corresponds with sample 0 of the output).
        # TODO: eventually we should be able to inspect the  'start_trigger'
        # property on the channel configuration to decide the order in which the
        # tasks are started.
        if self.hw_ao_channels:
            log.debug('Configuring HW AO channels')
            channels = self.hw_ao_channels
            lines = ','.join(get_channel_property(channels, 'channel', True))
            names = get_channel_property(channels, 'name', True)
            fs = get_channel_property(channels, 'fs')
            start_trigger = get_channel_property(channels, 'start_trigger')
            expected_range = get_channel_property(channels, 'expected_range')
            tmode = get_channel_property(channels, 'terminal_mode')
            terminal_mode = self.terminal_mode_map[tmode]
            self.configure_hw_ao(fs, lines, expected_range, names,
                                 start_trigger, terminal_mode)
            self.ao_fs = fs

        super().configure(plugin)

        # Required by start. This allows us to do the configuration
        # on the fly when starting the engines if the configure method hasn't
        # been called yet.
        self._configured = True
        log.debug('Completed engine configuration')

    def configure_hw_ao(self, fs, lines, expected_range, names=None,
                        start_trigger=None, terminal_mode=None):
        '''
        Initialize hardware-timed analog output

        Parameters
        ----------
        fs : float
            Sampling frequency of output (e.g., 100e3).
        lines : str
            Analog output lines to use (e.gk., 'Dev1/ao0:4' to specify a range of
            lines or 'Dev1/ao0,Dev1/ao4' to specify specific lines).
        expected_range : (float, float)
            Tuple of upper/lower end of expected range. The maximum range
            allowed by most NI devices is (-10, 10). Some devices (especially
            newer ones) will optimize the output resolution based on the
            expected range of the signal.
        '''
        callback_samples = int(self.hw_ao_monitor_period*fs)
        buffer_samples = int(self.hw_ao_buffer_size*fs)
        task_name = '{}_hw_ao'.format(self.name)
        task = setup_hw_ao(fs, lines, expected_range, self.hw_ao_callback,
                           callback_samples, start_trigger, terminal_mode,
                           buffer_samples, task_name)

        task._names = verify_channel_names(task, names)
        task._devices = device_list(task)

        self._tasks['hw_ao'] = task

    def configure_hw_ai(self, fs, lines, expected_range, names=None,
                        start_trigger=None, terminal_mode=None,
                        terminal_coupling=None):

        task_name = '{}_hw_ai'.format(self.name)
        callback_samples = int(self.hw_ai_monitor_period*fs)
        task = setup_hw_ai(fs, lines, expected_range, self._hw_ai_callback,
                           callback_samples, start_trigger, terminal_mode,
                           terminal_coupling, task_name)
        task._fs = fs
        task._names = verify_channel_names(task, names)
        task._devices = device_list(task)
        self._tasks['hw_ai'] = task

    def configure_sw_ao(self, lines, expected_range, names=None,
                        initial_state=None):
        if initial_state is None:
            initial_state = np.zeros(len(names), dtype=np.double)
        task_name = '{}_sw_ao'.format(self.name)
        task = setup_sw_ao(lines, expected_range, task_name)
        task._names = verify_channel_names(task, names)
        task._devices = device_list(task)
        self._tasks['sw_ao'] = task
        self.write_sw_ao(initial_state)

    def configure_hw_di(self, fs, lines, names=None, trigger=None, clock=None):
        callback_samples = int(self.hw_ai_monitor_period*fs)
        task_name = '{}_hw_di'.format(self.name)
        task, clock_task = setup_hw_di(fs, lines, self._hw_di_callback,
                                       callback_samples, trigger, clock,
                                       task_name)
        task._names = verify_channel_names(task, names)
        task._devices = device_list(task)
        task._fs = fs
        if clock_task is not None:
            self._tasks['hw_di_clock'] = clock_task
        self._tasks['hw_di'] = task

    def configure_hw_do(self, fs, lines, names):
        raise NotImplementedError

    def configure_sw_do(self, lines, names=None, initial_state=None):
        if initial_state is None:
            initial_state = np.zeros(len(names), dtype=np.uint8)
        task_name = '{}_sw_do'.format(self.name)
        task = setup_sw_do(lines, task_name)
        #task._names = verify_channel_names(task, names)
        task._names = names
        task._devices = device_list(task)
        self._tasks['sw_do'] = task
        self.write_sw_do(initial_state)

    def configure_et(self, lines, clock, names=None):
        '''
        Setup change detection with high-precision timestamps

        Anytime a rising or falling edge is detected on one of the specified
        lines, a timestamp based on the specified clock will be captured. For
        example, if the clock is 'ao/SampleClock', then the timestamp will be
        the number of samples played at the point when the line changed state.

        Parameters
        ----------
        lines : string
            Digital lines (in NI-DAQmx syntax, e.g., 'Dev1/port0/line0:4') to
            monitor.
        clock : string
            Reference clock from which timestamps will be drawn.
        names : string (optional)
            Aliases for the lines. When aliases are provided, registered
            callbacks will receive the alias for the line instead of the
            NI-DAQmx notation.

        Notes
        -----
        Be aware of the limitations of your device. All X-series devices support
        change detection on all ports; however, only some M-series devices do
        (and then, only on port 0).
        '''
        # Find out which device the lines are from. Use this to configure the
        # event timer. Right now we don't want to deal with multi-device event
        # timers. If there's more than one device, then we should configure each
        # separately.
        raise NotImplementedError

        # TODO: How to determine sampling rate of task?
        names = channel_names('digital', lines, names)
        devices = device_list(lines, 'digital')
        if len(devices) != 1:
            raise ValueError('Cannot configure multi-device event timer')

        trigger = '/{}/ChangeDetectionEvent'.format(devices[0])
        counter = '/{}/Ctr0'.format(devices[0])
        task_name = '{}_et'.format(self.name)
        et_task = setup_event_timer(trigger, counter, clock, task_name)
        task_name = '{}_cd'.format(self.name)
        cd_task = setup_change_detect_callback(lines, self._et_fired, et_task,
                                               names, task_name)
        cd_task._names = names
        self._tasks['et_task'] = et_task
        self._tasks['cd_task'] = cd_task

    def _get_channel_slice(self, task_name, channel_names):
        if channel_names is None:
            return Ellipsis
        else:
            return self._tasks[task_name]._names.index(channel_names)

    def register_ao_callback(self, callback, channel_name=None):
        s = self._get_channel_slice('hw_ao', channel_name)
        self._callbacks.setdefault('ao', []).append((channel_name, s, callback))

    def register_ai_callback(self, callback, channel_name=None):
        s = self._get_channel_slice('hw_ai', channel_name)
        self._callbacks.setdefault('ai', []).append((channel_name, s, callback))

    def register_di_callback(self, callback, channel_name=None):
        s = self._get_channel_slice('hw_di', channel_name)
        self._callbacks.setdefault('di', []).append((channel_name, s, callback))

    def register_et_callback(self, callback, channel_name=None):
        s = self._get_channel_slice('cd_task', channel_name)
        self._callbacks.setdefault('et', []).append((channel_name, s, callback))

    def unregister_ao_callback(self, callback, channel_name):
        try:
            s = self._get_channel_slice('hw_ao', channel_name)
            self._callbacks['ao'].remove((channel_name, s, callback))
        except (KeyError, AttributeError):
            log.warning('Callback no longer exists.')

    def unregister_ai_callback(self, callback, channel_name):
        try:
            s = self._get_channel_slice('hw_ai', channel_name)
            self._callbacks['ai'].remove((channel_name, s, callback))
        except (KeyError, AttributeError):
            log.warning('Callback no longer exists.')

    def unregister_di_callback(self, callback, channel_name):
        s = self._get_channel_slice('hw_di', channel_name)
        self._callbacks['di'].remove((channel_name, s, callback))

    def unregister_et_callback(self, callback, channel_name):
        s = self._get_channel_slice('cd_task', channel_name)
        self._callbacks['et'].remove((channel_name, s, callback))

    def write_sw_ao(self, state):
        task = self._tasks['sw_ao']
        state = np.array(state).astype(np.double)
        mx.DAQmxWriteAnalogF64(task, 1, True, 0, mx.DAQmx_Val_GroupByChannel,
                               state, self._int32, None)
        if self._int32.value != 1:
            raise ValueError('Unable to update software-timed AO')
        task._current_state = state

    def write_sw_do(self, state):
        task = self._tasks['sw_do']
        state = np.asarray(state).astype(np.uint8)
        mx.DAQmxWriteDigitalLines(task, 1, True, 0, mx.DAQmx_Val_GroupByChannel,
                                  state, self._int32, None)
        if self._int32.value != 1:
            raise ValueError('Problem writing data to software-timed DO')
        task._current_state = state

    def set_sw_do(self, name, state):
        task = self._tasks['sw_do']
        i = task._names.index(name)
        new_state = task._current_state.copy()
        new_state[i] = state
        self.write_sw_do(new_state)

    def set_sw_ao(self, name, state):
        task = self._tasks['sw_ao']
        i = task._names.index(name)
        new_state = task._current_state.copy()
        new_state[i] = state
        self.write_sw_ao(new_state)

    def fire_sw_do(self, name, duration=0.1):
        # TODO - Store reference to timer so that we can eventually track the
        # state of different timers and cancel pending timers when necessary.
        self.set_sw_do(name, 1)
        timer = Timer(duration, lambda: self.set_sw_do(name, 0))
        timer.start()

    def _et_fired(self, line_index, change, event_time):
        for i, cb in self._callbacks.get('et', []):
            if i == line_index:
                cb(change, event_time)

    def _hw_ai_callback(self, samples):
        for channel_name, s, cb in self._callbacks.get('ai', []):
            try:
                cb(samples[s])
            except StopIteration:
                log.warning('Callback no longer works. Removing.')
                self.unregister_ai_callback(cb, channel_name)

    def _hw_di_callback(self, samples):
        for i, cb in self._callbacks.get('di', []):
            cb(samples[i])

    def _get_hw_ao_samples(self, offset, samples):
        channels = len(self.hw_ao_channels)
        data = np.empty((channels, samples), dtype=np.double)
        for i, channel in enumerate(self.hw_ao_channels):
            data[i] = channel.get_samples(offset, samples)
        return data

    def get_offset(self, channel_name=None):
        # Doesn't matter. Offset is the same for all channels in the task.
        task = self._tasks['hw_ao']
        mx.DAQmxSetWriteRelativeTo(task, mx.DAQmx_Val_CurrWritePos)
        mx.DAQmxSetWriteOffset(task, 0)
        mx.DAQmxGetWriteCurrWritePos(task, self._uint64)
        return self._uint64.value

    def get_space_available(self, offset=None, channel_name=None):
        # It doesn't matter what the output channel is. Space will be the same
        # for all.
        task = self._tasks['hw_ao']
        mx.DAQmxGetWriteSpaceAvail(task, self._uint32)
        available = self._uint32.value
        log.trace('Current write space available %d', available)

        # Compensate for offset if specified.
        if offset is not None:
            write_position = self.ao_write_position()
            relative_offset = offset-write_position
            log.trace('Compensating write space for requested offset %d', offset)
            available -= relative_offset
        return available

    def ao_sample_clock(self):
        task = self._tasks['hw_ao']
        mx.DAQmxGetWriteTotalSampPerChanGenerated(task, self._uint64)
        log.trace('%d samples per channel generated', self._uint64.value)
        return self._uint64.value

    def hw_ao_callback(self, samples):
        # Get the next set of samples to upload to the buffer
        with self.lock:
            log.trace('Hardware AO callback for {}'.format(self.name))
            while True:
                offset = self.get_offset()
                available_samples = self.get_space_available(offset)
                if available_samples < samples:
                    log.trace('Not enough samples available for writing')
                    break
                data = self._get_hw_ao_samples(offset, samples)
                self.write_hw_ao(data, timeout=0)

    def update_hw_ao(self, offset, channel_name=None):
        # Get the next set of samples to upload to the buffer. Ignore the
        # channel name because we need to update all channels simultaneously.
        samples = self.get_space_available(offset)
        if samples <= 0:
            log.trace('No update of hw ao required')
            return
        log.trace('Updating hw ao at {} with {} samples'.format(offset, samples))
        data = self._get_hw_ao_samples(offset, samples)
        self.write_hw_ao(data, offset=offset, timeout=0)

    def ao_write_position(self):
        task = self._tasks['hw_ao']
        mx.DAQmxGetWriteCurrWritePos(task, self._uint64)
        log.trace('Current write position %d', self._uint64.value)
        return self._uint64.value

    def write_hw_ao(self, data, offset=None, timeout=1):
        # Due to historical limitations in the DAQmx API, the write offset is a
        # signed 32-bit integer. For long-running applications, we will have an
        # overflow if we attempt to set the offset relative to the first sample
        # written. Therefore, we compute the write offset relative to the last
        # sample written (for requested offsets it should be negative).
        log.trace('Writing {} samples at {}'.format(data.shape, offset))
        task = self._tasks['hw_ao']

        if offset is not None:
            write_position = self.ao_write_position()
            relative_offset = offset-write_position
            mx.DAQmxSetWriteOffset(task, relative_offset)
            m = 'Write position %d, requested offset %d, relative offset %d'
            log.trace(m, write_position, offset, relative_offset)
            log.trace('AO samples generated %d', self.ao_sample_clock())

        mx.DAQmxWriteAnalogF64(task, data.shape[-1], False, timeout,
                               mx.DAQmx_Val_GroupByChannel,
                               data.astype(np.float64), self._int32, None)

        # Now, reset it back to 0
        if offset is not None:
            log.trace('Resetting write offset')
            mx.DAQmxSetWriteOffset(task, 0)

        log.trace('Write complete')

    def get_ts(self):
        with self.lock:
            return self.ao_sample_clock()/self.ao_fs

    def start(self):
        if not self._configured:
            log.debug('Tasks were not configured yet')
            self.configure()

        log.debug('Reserving NIDAQmx task resources')
        for task in self._tasks.values():
            mx.DAQmxTaskControl(task, mx.DAQmx_Val_Task_Commit)

        if 'hw_ao' in self._tasks:
            log.debug('Calling HW ao callback before starting tasks')
            samples = self.get_space_available()
            self.hw_ao_callback(samples)

        log.debug('Starting NIDAQmx tasks')
        for task in self._tasks.values():
            log.debug('Starting task {}'.format(task._name))
            mx.DAQmxStartTask(task)

    def stop(self):
        # TODO: I would love to be able to stop a task and keep it in memory
        # without having to restart; however, this will require some thought as
        # to the optimal way to do this. For now, we just clear everything.
        # Configuration is generally fairly quick.
        log.debug('Stopping engine')
        for task in self._tasks.values():
            mx.DAQmxClearTask(task)
        self._callbacks = {}
        self._configured = False

    def ai_sample_clock(self):
        task = self._tasks['hw_ai']
        mx.DAQmxGetReadTotalSampPerChanAcquired(task, self._uint64)
        log.trace('%d samples per channel acquired', self._uint64.value)
        return self._uint64.value

    def get_buffer_size(self, channel_name):
        return self.hw_ao_buffer_size
