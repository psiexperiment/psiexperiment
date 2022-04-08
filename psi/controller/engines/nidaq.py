'''
Defines the NIDAQmx engine interface

General notes for developers
-----------------------------------------------------------------------------
This is a wraper around the NI-DAQmx C API. Refer to the NI-DAQmx C reference
(available as a Windows help file or as HTML documentation on the NI website).
Google can help you quickly find the online documentation).

This code is under heavy development and I may change the API in significant
ways. In general, the only portion of the code you should use in third-party
modules is the `Engine` class. This will serve as the sole communication layer
between the NI hardware and your application. By doing so, this ensures a
sufficient layer of abstraction that helps switch between DAQ hardware from
different vendors provided that the appropriate interface is written.
'''

# These are pointers to C datatypes that are required for communicating with
# the NI-DAQmx library. When querying various properties of tasks, channels and
# buffers, the NI-DAQmx function often requires an integer of a specific type
# (e.g. unsigned 32-bit, unsigned 64-bit, etc.). This integer must be passed by
# reference, allowing the NI-DAQmx function to modify the value directly. For
# example:
#
#   result = ctypes.c_uint32()
#   mx.DAQmxGetWriteSpaceAvail(task, result)
#   print(result.value)

import logging
log = logging.getLogger(__name__)
log_ai = logging.getLogger(__name__ + '.ai')
log_ao = logging.getLogger(__name__ + '.ao')

import types
import ctypes
from collections import OrderedDict
from functools import partial
from threading import Timer

import numpy as np
import PyDAQmx as mx
from atom.api import (Float, Typed, Str, Int, Bool, Callable, Enum,
                      Property, Value)
from enaml.core.api import Declarative, d_

from psiaudio.util import dbi
from ..engine import Engine
from ..channel import (CounterChannel,
                       HardwareAIChannel, HardwareAOChannel, HardwareDIChannel,
                       HardwareDOChannel, SoftwareDIChannel, SoftwareDOChannel)
from ..input import InputData


################################################################################
# Engine-specific channels
################################################################################
class NIDAQGeneralMixin(Declarative):

    # Channel identifier (e.g., /Dev1/ai0)
    channel = d_(Str()).tag(metadata=True)

    def __str__(self):
        return f'{self.label} ({self.channel})'

    def sync_start(self, channel):
        self.start_trigger = f'/{channel.device_name}/ao/StartTrigger'
        channel.start_trigger = ''


class NIDAQTimingMixin(Declarative):

    #: Specifies sampling clock for the channel. Even if specifying a sample
    #: clock, you still need to explicitly set the fs attribute.
    sample_clock = d_(Str().tag(metadata=True))

    #: Specifies the start trigger for the channel. If None, sampling begins
    #: when task is started.
    start_trigger = d_(Str().tag(metadata=True))

    #: Reference clock for the channel. If you aren't sure, a good value is
    #: `PXI_Clk10` if using a PXI chassis. This ensures that the sample clocks
    #: across all NI cards in the PXI chassis are synchronized.
    reference_clock = d_(Str()).tag(metadata=True)


class NIDAQCounterChannel(NIDAQGeneralMixin, CounterChannel):

    high_samples = d_(Int().tag(metadata=True))
    low_samples = d_(Int().tag(metadata=True))
    source_terminal = d_(Str().tag(metadata=True))


class NIDAQHardwareAOChannel(NIDAQGeneralMixin, NIDAQTimingMixin,
                             HardwareAOChannel):

    #: Available terminal modes. Not all terminal modes may be supported by a
    #: particular device
    TERMINAL_MODES = 'pseudodifferential', 'differential', 'RSE'

    #: Terminal mode
    terminal_mode = d_(Enum(*TERMINAL_MODES)).tag(metadata=True)

    device_name = Property().tag(metadata=False)

    def _get_device_name(self):
        return self.channel.strip('/').split('/')[0]


class NIDAQHardwareAOChannel4461(NIDAQHardwareAOChannel):
    '''
    Special channel that automatically compensates for filter delay of PXI 4461
    card.
    '''
    #: Filter delay lookup table for different sampling rates. The first column
    #: is the lower bound (exclusive) of the sampling rate (in samples/sec) for
    #: the filter delay (second column, in samples). The upper bound of the
    #: range (inclusive) for the sampling rate is denoted by the next row.
    #: e.g., if FILTER_DELAY[i, 0] < fs <= FILTER_DELAY[i+1, 0] is True, then
    #: the filter delay is FILTER_DELAY[i, 1].
    FILTER_DELAY = np.array([
        (  1.0e3, 36.6),
        (  1.6e3, 36.8),
        (  3.2e3, 37.4),
        (  6.4e3, 38.5),
        ( 12.8e3, 40.8),
        ( 25.6e3, 43.2),
        ( 51.2e3, 48.0),
        (102.4e3, 32.0),
    ])

    filter_delay = Property().tag(metadata=True)
    filter_delay_samples = Property().tag(metadata=True)

    def _get_filter_delay_samples(self):
        i = np.flatnonzero(self.fs > self.FILTER_DELAY[:, 0])[-1]
        return self.FILTER_DELAY[i, 1]

    def _get_filter_delay(self):
        return self.filter_delay_samples / self.fs


class NIDAQHardwareAIChannel(NIDAQGeneralMixin, NIDAQTimingMixin,
                             HardwareAIChannel):

    #: Available terminal modes. Not all terminal modes may be supported by a
    #: particular device
    TERMINAL_MODES = 'pseudodifferential', 'differential', 'RSE', 'NRSE'
    terminal_mode = d_(Enum(*TERMINAL_MODES)).tag(metadata=True)

    #: Terminal coupling to use. Not all terminal couplings may be supported by
    #: a particular device. Can be `None`, `'AC'`, `'DC'` or `'ground'`.
    terminal_coupling = d_(Enum(None, 'AC', 'DC', 'ground')).tag(metadata=True)


class NIDAQHardwareDIChannel(NIDAQGeneralMixin, NIDAQTimingMixin,
                             HardwareDIChannel):
    pass


class NIDAQHardwareDOChannel(NIDAQGeneralMixin, NIDAQTimingMixin,
                             HardwareDOChannel):
    pass


class NIDAQSoftwareDIChannel(NIDAQGeneralMixin, SoftwareDIChannel):
    pass


class NIDAQSoftwareDOChannel(NIDAQGeneralMixin, SoftwareDOChannel):
    pass



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


def read_hw_ai(task, available_samples=None, channels=1, block_size=1):
    if available_samples is None:
        uint32 = ctypes.c_uint32()
        mx.DAQmxGetReadAvailSampPerChan(task, uint32)
        available_samples = uint32.value

    blocks = (available_samples//block_size)
    if blocks == 0:
        return
    samples = blocks*block_size
    data = np.empty((channels, samples), dtype=np.double)
    int32 = ctypes.c_int32()
    mx.DAQmxReadAnalogF64(task, samples, 0, mx.DAQmx_Val_GroupByChannel, data,
                          data.size, int32, None)
    log_ai.trace('Read %d samples', samples)
    return data


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
def hw_ao_helper(cb, task, event_type, cb_samples, cb_data):
    cb(cb_samples)
    return 0


def hw_ai_helper(cb, channels, discard, fs, task, event_type=None,
                 cb_samples=None, cb_data=None):

    uint32 = ctypes.c_uint32()
    mx.DAQmxGetReadAvailSampPerChan(task, uint32)
    available_samples = uint32.value
    if available_samples == 0:
        return 0

    uint64 = ctypes.c_uint64()
    mx.DAQmxGetReadCurrReadPos(task, uint64)
    read_position = uint64.value

    log_ai.trace('Current read position %d, available samples %d',
                 read_position, available_samples)

    if read_position < discard:
        samples = min(discard-read_position, available_samples)
        read_hw_ai(task, samples, channels)
        available_samples -= samples
        log_ai.debug('Discarded %d samples from beginning, %d available',
                     samples, available_samples)

    if available_samples == 0:
        return 0

    data = read_hw_ai(task, available_samples, channels, cb_samples)
    if data is not None:
        metadata = {'t0_sample': read_position-discard, 'fs': fs}
        data = InputData(data, metadata=metadata)
        cb(data)
    return 0


################################################################################
# Configuration functions
################################################################################
def setup_timing(task, channels, delay=0):
    '''
    Configures timing for task

    Parameters
    ----------
    task : niDAQmx task handle
        Task to configure timing for
    channels : list of channels
        List of channels to configure

    References
    ----------
    http://www.ni.com/white-paper/11369/en/
    http://www.ni.com/pdf/manuals/371235h.pdf
    '''
    fs = get_channel_property(channels, 'fs')
    sample_clock = get_channel_property(channels, 'sample_clock')
    start_trigger = get_channel_property(channels, 'start_trigger')
    samples = get_channel_property(channels, 'samples')
    reference_clock = get_channel_property(channels, 'reference_clock')

    if reference_clock:
        mx.DAQmxSetRefClkSrc(task, reference_clock)

    if start_trigger:
        mx.DAQmxCfgDigEdgeStartTrig(task, start_trigger, mx.DAQmx_Val_Rising)

    if samples == 0:
        sample_mode = mx.DAQmx_Val_ContSamps
        samples = 2
    else:
        sample_mode = mx.DAQmx_Val_FiniteSamps
        samples += delay

    mx.DAQmxCfgSampClkTiming(task, sample_clock, fs, mx.DAQmx_Val_Rising,
                             sample_mode, samples)

    properties = get_timing_config(task)
    actual_fs = properties['sample clock rate']
    if round(actual_fs, 4) != fs:
        names = ', '.join(get_channel_property(channels, 'name', True))
        m = f'Actual sample clock rate of {actual_fs} does not match ' \
            f'requested sample clock rate of {fs} for {names}'
        raise ValueError(m)

    return properties


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


def setup_counters(channels, task_name='counter'):
    lines = get_channel_property(channels, 'channel', True)
    names = get_channel_property(channels, 'name', True)
    log.debug('Configuring lines {}'.format(lines))

    source_terminal = get_channel_property(channels, 'source_terminal')
    low_samples = get_channel_property(channels, 'low_samples')
    high_samples = get_channel_property(channels, 'high_samples')

    merged_lines = ','.join(lines)
    task = create_task(task_name)
    mx.DAQmxCreateCOPulseChanTicks(task, merged_lines, '', source_terminal,
                                   mx.DAQmx_Val_Low, 0, low_samples,
                                   high_samples)
    mx.DAQmxCfgSampClkTiming(task, source_terminal, 100, mx.DAQmx_Val_Rising,
                             mx.DAQmx_Val_HWTimedSinglePoint, 2)
    return task


def setup_hw_ao(channels, buffer_duration, callback_interval, callback,
                task_name='hw_ao'):

    lines = get_channel_property(channels, 'channel', True)
    names = get_channel_property(channels, 'name', True)
    expected_ranges = get_channel_property(channels, 'expected_range', True)
    start_trigger = get_channel_property(channels, 'start_trigger')
    terminal_mode = get_channel_property(channels, 'terminal_mode')
    terminal_mode = NIDAQEngine.terminal_mode_map[terminal_mode]
    task = create_task(task_name)
    merged_lines = ','.join(lines)

    for line, name, (vmin, vmax) in zip(lines, names, expected_ranges):
        log.debug(f'Configuring line %s (%s) with voltage range %f-%f', line, name, vmin, vmax)
        mx.DAQmxCreateAOVoltageChan(task, line, name, vmin, vmax, mx.DAQmx_Val_Volts, '')

    properties = setup_timing(task, channels)

    result = ctypes.c_double()
    try:
        for line in lines:
            mx.DAQmxGetAOGain(task, line, result)
            properties['{} AO gain'.format(line)] = result.value
    except:
        # This means that the gain is not settable
        properties['{} AO gain'.format(line)] = 0

    fs = properties['sample clock rate']
    log_ao.info('AO properties: %r', properties)

    if terminal_mode is not None:
        mx.DAQmxSetAOTermCfg(task, merged_lines, terminal_mode)

    # If the write reaches the end of the buffer and no new data has been
    # provided, do not loop around to the beginning and start over.
    mx.DAQmxSetWriteRegenMode(task, mx.DAQmx_Val_DoNotAllowRegen)
    mx.DAQmxSetWriteRelativeTo(task, mx.DAQmx_Val_CurrWritePos)

    callback_samples = int(round(fs*callback_interval))

    if buffer_duration is None:
        log_ao.debug('Buffer duration not provided. Setting to 10x callback.')
        buffer_samples = round(callback_samples*10)
    else:
        buffer_samples = round(buffer_duration*fs)

    # Now, make sure that buffer_samples is an integer multiple of callback_samples.
    log_ao.debug('Requested buffer samples %d', buffer_samples)
    n = int(round(buffer_samples / callback_samples))
    buffer_samples = callback_samples * n
    log_ao.debug('Coerced buffer samples to %d (%dx %d callback_samples)',
                 buffer_samples, n, callback_samples)

    log_ao.debug('Setting output buffer size to %d samples', buffer_samples)
    mx.DAQmxSetBufOutputBufSize(task, buffer_samples)
    task._buffer_samples = buffer_samples

    result = ctypes.c_uint32()
    mx.DAQmxGetTaskNumChans(task, result)
    task._n_channels = result.value
    log_ao.debug('%d channels in task', task._n_channels)

    #mx.DAQmxSetAOMemMapEnable(task, lines, True)
    #mx.DAQmxSetAODataXferReqCond(task, lines, mx.DAQmx_Val_OnBrdMemHalfFullOrLess)

    # This controls how quickly we can update the buffer on the device. On some
    # devices it is not user-settable. On the X-series PCIe-6321 I am able to
    # change it. On the M-xeries PCI 6259 it appears to be fixed at 8191
    # samples. Haven't really been able to do much about this.
    mx.DAQmxGetBufOutputOnbrdBufSize(task, result)
    task._onboard_buffer_size = result.value
    log_ao.debug('Onboard buffer size %d', task._onboard_buffer_size)

    result = ctypes.c_int32()
    mx.DAQmxGetAODataXferMech(task, merged_lines, result)
    log_ao.debug('Data transfer mechanism %d', result.value)
    mx.DAQmxGetAODataXferReqCond(task, merged_lines, result)
    log_ao.debug('Data transfer condition %d', result.value)
    #result = ctypes.c_uint32()
    #mx.DAQmxGetAOUseOnlyOnBrdMem(task, merged_lines, result)
    #log_ao.debug('Use only onboard memory %d', result.value)
    #mx.DAQmxGetAOMemMapEnable(task, merged_lines, result)
    #log_ao.debug('Memory mapping enabled %d', result.value)

    #mx.DAQmxGetAIFilterDelayUnits(task, merged_lines, result)
    #log_ao.debug('AI filter delay unit %d', result.value)

    #result = ctypes.c_int32()
    #mx.DAQmxGetAODataXferMech(task, result)
    #log_ao.debug('DMA transfer mechanism %d', result.value)

    log_ao.debug('Creating callback after every %d samples', callback_samples)
    task._cb = partial(hw_ao_helper, callback)
    task._cb_ptr = mx.DAQmxEveryNSamplesEventCallbackPtr(task._cb)
    mx.DAQmxRegisterEveryNSamplesEvent(
        task, mx.DAQmx_Val_Transferred_From_Buffer, int(callback_samples), 0,
        task._cb_ptr, None)

    mx.DAQmxTaskControl(task, mx.DAQmx_Val_Task_Reserve)
    task._names = verify_channel_names(task, names)
    task._devices = device_list(task)
    task._fs = fs
    return task


def get_timing_config(task):
    properties = {}

    info = ctypes.c_double()
    mx.DAQmxGetSampClkRate(task, info)
    properties['sample clock rate'] = info.value
    mx.DAQmxGetSampClkMaxRate(task, info)
    properties['sample clock maximum rate'] = info.value
    mx.DAQmxGetSampClkTimebaseRate(task, info)
    properties['sample clock timebase rate'] = info.value
    try:
        mx.DAQmxGetMasterTimebaseRate(task, info)
        properties['master timebase rate'] = info.value
    except:
        pass
    mx.DAQmxGetRefClkRate(task, info)
    properties['reference clock rate'] = info.value

    info = ctypes.c_buffer(256)
    mx.DAQmxGetSampClkSrc(task, info, len(info))
    properties['sample clock source'] = str(info.value)
    mx.DAQmxGetSampClkTimebaseSrc(task, info, len(info))
    properties['sample clock timebase source'] = str(info.value)
    mx.DAQmxGetSampClkTerm(task, info, len(info))
    properties['sample clock terminal'] = str(info.value)
    try:
        mx.DAQmxGetMasterTimebaseSrc(task, info, len(info))
        properties['master timebase source'] = str(info.value)
    except:
        pass
    mx.DAQmxGetRefClkSrc(task, info, len(info))
    properties['reference clock source'] = str(info.value)

    info = ctypes.c_int32()
    try:
        mx.DAQmxGetSampClkOverrunBehavior(task, info)
        properties['sample clock overrun behavior'] = info.value
    except:
        pass
    mx.DAQmxGetSampClkActiveEdge(task, info)
    properties['sample clock active edge'] = info.value

    info = ctypes.c_uint32()
    try:
        mx.DAQmxGetSampClkTimebaseDiv(task, info)
        properties['sample clock timebase divisor'] = info.value
    except:
        pass

    return properties


def setup_hw_ai(channels, callback_duration, callback, task_name='hw_ao'):
    log.debug('Configuring HW AI channels')

    # These properties can vary on a per-channel basis
    lines = get_channel_property(channels, 'channel', True)
    names = get_channel_property(channels, 'name', True)
    gains = get_channel_property(channels, 'gain', True)

    # These properties must be the same across all channels
    expected_range = get_channel_property(channels, 'expected_range')
    samples = get_channel_property(channels, 'samples')
    terminal_mode = get_channel_property(channels, 'terminal_mode')
    terminal_coupling = get_channel_property(channels, 'terminal_coupling')

    # Convert to representation required by NI functions
    lines = ','.join(lines)
    log.debug('Configuring lines {}'.format(lines))

    terminal_mode = NIDAQEngine.terminal_mode_map[terminal_mode]
    terminal_coupling = NIDAQEngine.terminal_coupling_map[terminal_coupling]

    task = create_task(task_name)
    mx.DAQmxCreateAIVoltageChan(task, lines, '', terminal_mode,
                                expected_range[0], expected_range[1],
                                mx.DAQmx_Val_Volts, '')

    if terminal_coupling is not None:
        mx.DAQmxSetAICoupling(task, lines, terminal_coupling)

    properties = setup_timing(task, channels)
    log_ai.info('AI timing properties: %r', properties)

    result = ctypes.c_uint32()
    mx.DAQmxGetTaskNumChans(task, result)
    n_channels = result.value

    fs = properties['sample clock rate']
    callback_samples = round(callback_duration * fs)
    mx.DAQmxSetReadOverWrite(task, mx.DAQmx_Val_DoNotOverwriteUnreadSamps)
    mx.DAQmxSetBufInputBufSize(task, callback_samples*100)
    mx.DAQmxGetBufInputBufSize(task, result)
    buffer_size = result.value
    log_ai.debug('Buffer size for %s set to %d samples', lines, buffer_size)

    try:
        info = ctypes.c_int32()
        mx.DAQmxSetAIFilterDelayUnits(task, lines,
                                      mx.DAQmx_Val_SampleClkPeriods)
        info = ctypes.c_double()
        mx.DAQmxGetAIFilterDelay(task, lines, info)
        log_ai.debug('AI filter delay {} samples'.format(info.value))
        filter_delay = int(info.value)

        # Ensure timing is compensated for the planned filter delay since these
        # samples will be discarded.
        if samples > 0:
            setup_timing(task, channels, filter_delay)

    except mx.DAQError:
        # Not a supported property. Set filter delay to 0 by default.
        filter_delay = 0

    task._cb = partial(hw_ai_helper, callback, n_channels, filter_delay, fs)
    task._cb_ptr = mx.DAQmxEveryNSamplesEventCallbackPtr(task._cb)
    mx.DAQmxRegisterEveryNSamplesEvent(
        task, mx.DAQmx_Val_Acquired_Into_Buffer, int(callback_samples), 0,
        task._cb_ptr, None)

    mx.DAQmxTaskControl(task, mx.DAQmx_Val_Task_Reserve)

    task._names = verify_channel_names(task, names)
    task._devices = device_list(task)
    task._sf = dbi(gains)[..., np.newaxis]
    task._fs = properties['sample clock rate']
    properties = get_timing_config(task)
    log_ai.info('AI timing properties: %r', properties)
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
        setup_timing(task, clock, -1, None)
    else:
        setup_timing(task, fs, -1, start_trigger)

    cb_helper = DigitalSamplesAcquiredCallbackHelper(callback)
    cb_ptr = mx.DAQmxEveryNSamplesEventCallbackPtr(cb_helper)
    mx.DAQmxRegisterEveryNSamplesEvent(task, mx.DAQmx_Val_Acquired_Into_Buffer,
                                       int(callback_samples), 0, cb_ptr, None)

    task._cb_ptr = cb_ptr
    task._cb_helper = cb_helper
    task._initial_state = initial_state

    rate = ctypes.c_double()
    mx.DAQmxGetSampClkRate(task, rate)

    mx.DAQmxTaskControl(task, mx.DAQmx_Val_Task_Reserve)
    mx.DAQmxTaskControl(clock_task, mx.DAQmx_Val_Task_Reserve)

    return [task, clock_task]


def setup_sw_ao(lines, expected_range, task_name='sw_ao'):
    # TODO: DAQmxSetAOTermCfg
    task = create_task(task_name)
    lb, ub = expected_range
    mx.DAQmxCreateAOVoltageChan(task, lines, '', lb, ub, mx.DAQmx_Val_Volts, '')
    mx.DAQmxTaskControl(task, mx.DAQmx_Val_Task_Reserve)
    return task


def setup_sw_do(channels, task_name='sw_do'):
    task = create_task(task_name)

    lines = get_channel_property(channels, 'channel', True)
    names = get_channel_property(channels, 'name', True)

    lines = ','.join(lines)
    mx.DAQmxCreateDOChan(task, lines, '', mx.DAQmx_Val_ChanForAllLines)
    mx.DAQmxTaskControl(task, mx.DAQmx_Val_Task_Reserve)

    task._names = names
    task._devices = device_list(task)

    return task


def halt_on_error(f):
    def wrapper(self, *args, **kwargs):
        try:
            f(self, *args, **kwargs)
        except Exception as e:
            log.exception(e)
            try:
                self.stop()
                for cb in self._callbacks.get('done', []):
                    cb()
            except:
                pass
            # Be sure to raise the exception so it can be recaptured by our
            # custom excepthook handler
            raise
    return wrapper


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
    #: Name of the engine. This is used for logging and configuration purposes
    #: (we can have multiple NIDAQ engines if we need to define separate sets
    #: of tasks (e.g., if we have more than one multifunction DAQ card).
    engine_name = 'nidaq'

    #: Flag indicating whether engine was configured
    _configured = Bool(False)

    #: Poll period (in seconds). This defines how often callbacks for the
    #: analog outputs are notified (i.e., to generate additional samples for
    #: playout).  If the poll period is too long, then the analog output may
    #: run out of samples.
    hw_ao_monitor_period = d_(Float(1)).tag(metadata=True)

    #: Size of buffer (in seconds). This defines how much data is pregenerated
    #: for the buffer before starting acquisition. This is impotant because
    hw_ao_buffer_size = d_(Float(10)).tag(metadata=True)

    #: Even though data is written to the analog outputs, it is buffered in
    #: computer memory until it's time to be transferred to the onboard buffer
    #: of the NI acquisition card. NI-DAQmx handles this behind the scenes
    #: (i.e., when the acquisition card needs additional samples, NI-DAQmx will
    #: transfer the next chunk of data from the computer memory). We can
    #: overwrite data that's been buffered in computer memory (e.g., so we can
    #: insert a target in response to a nose-poke). However, we cannot
    #: overwrite data that's already been transfered to the onboard buffer. So,
    #: the onboard buffer size determines how quickly we can change the analog
    #: output in response to an event.
    hw_ao_onboard_buffer = d_(Int(4095)).tag(metadata=True)
    # TODO: This is not configurable on every card. How do we know if it's
    # configurable?

    #: Since any function call takes a small fraction of time (e.g., nanoseconds
    #: to milliseconds), we can't simply overwrite data starting at
    #: hw_ao_onboard_buffer+1. By the time the function calls are complete, the
    #: DAQ probably has already transferred a couple hundred samples to the
    #: buffer. This parameter will likely need some tweaking (i.e., only you can
    #: determine an appropriate value for this based on the needs of your
    #: program).
    hw_ao_min_writeahead = d_(Int(8191 + 1000)).tag(metadata=True)

    #: Total samples written to the buffer.
    total_samples_written = Int(0)

    _tasks = Typed(dict)
    _task_done = Typed(dict)
    _callbacks = Typed(dict)
    _timers = Typed(dict)
    _uint32 = Typed(ctypes.c_uint32)
    _uint64 = Typed(ctypes.c_uint64)
    _int32 = Typed(ctypes.c_int32)

    ao_fs = Typed(float).tag(metadata=True)
    ai_fs = Typed(float).tag(metadata=True)

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

    # This defines the function for the clock that synchronizes the tasks.
    sample_time = Callable()

    instances = Value([])

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.instances.append(self)

        # Use an OrderedDict to ensure that when we loop through the tasks
        # stored in the dictionary, we process them in the order they were
        # configured.
        self._tasks = OrderedDict()
        self._callbacks = {}
        self._timers = {}
        self._configured = False

    def configure(self, active=True):
        log.debug('Configuring {} engine'.format(self.name))

        counter_channels = self.get_channels('counter', active=active)
        sw_do_channels = self.get_channels('digital', 'output', 'software',
                                           active=active)
        hw_ai_channels = self.get_channels('analog', 'input', 'hardware',
                                           active=active)
        hw_di_channels = self.get_channels('digital', 'input', 'hardware',
                                           active=active)
        hw_ao_channels = self.get_channels('analog', 'output', 'hardware',
                                           active=active)

        if counter_channels:
            log.debug('Configuring counter channels')
            self.configure_counters(counter_channels)

        if sw_do_channels:
            log.debug('Configuring SW DO channels')
            self.configure_sw_do(sw_do_channels)

        if hw_ai_channels:
            log.debug('Configuring HW AI channels')
            self.configure_hw_ai(hw_ai_channels)

        if hw_di_channels:
            raise NotImplementedError
            #log.debug('Configuring HW DI channels')
            #lines = ','.join(get_channel_property(hw_di_channels, 'channel', True))
            #names = get_channel_property(hw_di_channels, 'name', True)
            #fs = get_channel_property(hw_di_channels, 'fs')
            #start_trigger = get_channel_property(hw_di_channels, 'start_trigger')
            ## Required for M-series to enable hardware-timed digital
            ## acquisition. TODO: Make this a setting that can be configured
            ## since X-series doesn't need this hack.
            #device = hw_di_channels[0].channel.strip('/').split('/')[0]
            #clock = '/{}/Ctr0'.format(device)
            #self.configure_hw_di(fs, lines, names, start_trigger, clock)

        # Configure the analog output last because acquisition is synced with
        # the analog output signal (i.e., when the analog output starts, the
        # analog input begins acquiring such that sample 0 of the input
        # corresponds with sample 0 of the output).

        # TODO: eventually we should be able to inspect the  'start_trigger'
        # property on the channel configuration to decide the order in which the
        # tasks are started.
        if hw_ao_channels:
            log.debug('Configuring HW AO channels')
            self.configure_hw_ao(hw_ao_channels)

        # Choose sample clock based on what channels have been configured.
        if hw_ao_channels:
            self.sample_time = self.ao_sample_time
        elif hw_ai_channels:
            self.sample_time = self.ai_sample_time

        # Configure task done events so that we can fire a callback if
        # acquisition is done.
        self._task_done = {}
        for name, task in self._tasks.items():
            def cb(task, s, cb_data):
                nonlocal name
                self.task_complete(name)
                return 0
            cb_ptr = mx.DAQmxDoneEventCallbackPtr(cb)
            mx.DAQmxRegisterDoneEvent(task, 0, cb_ptr, None)
            task._done_cb_ptr_engine = cb_ptr
            self._task_done[name] = False

        super().configure()

        # Required by start. This allows us to do the configuration
        # on the fly when starting the engines if the configure method hasn't
        # been called yet.
        self._configured = True
        log.debug('Completed engine configuration')

    def task_complete(self, task_name):
        log.debug('Task %s complete', task_name)
        self._task_done[task_name] = True
        task = self._tasks[task_name]

        # We have frozen the initial arguments (in the case of hw_ai_helper,
        # that would be cb, channels, discard; in the case of hw_ao_helper,
        # that would be cb) using functools.partial and need to provide task,
        # cb_samples and cb_data. For hw_ai_helper, setting cb_samples to 1
        # means that we read all remaning samples, regardless of whether they
        # fit evenly into a block of samples. The other two arguments
        # (event_type and cb_data) are required of the function signature by
        # NIDAQmx but are unused.
        try:
            task._cb(task, None, 1, None)
        except Exception as e:
            log.exception(e)

        # Only check to see if hardware-timed tasks are complete.
        # Software-timed tasks must be explicitly canceled by the user.
        done = [v for t, v in self._task_done.items() if t.startswith('hw')]
        if all(done):
            for cb in self._callbacks.get('done', []):
                cb()

    def configure_counters(self, channels):
        task = setup_counters(channels)
        self._tasks['counter'] = task

    def configure_hw_ao(self, channels):
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
        task = setup_hw_ao(channels, self.hw_ao_buffer_size,
                           self.hw_ao_monitor_period, self.hw_ao_callback,
                           '{}_hw_ao'.format(self.name))
        self._tasks['hw_ao'] = task
        self.ao_fs = task._fs
        for channel in channels:
            channel.fs = task._fs
        self.total_samples_written = 0

    def configure_hw_ai(self, channels):
        task_name = '{}_hw_ai'.format(self.name)
        task = setup_hw_ai(channels, self.hw_ai_monitor_period,
                           self._hw_ai_callback, task_name)
        self._tasks['hw_ai'] = task
        self.ai_fs = task._fs

    def configure_sw_ao(self, lines, expected_range, names=None,
                        initial_state=None):
        raise NotImplementedError
        if initial_state is None:
            initial_state = np.zeros(len(names), dtype=np.double)
        task_name = '{}_sw_ao'.format(self.name)
        task = setup_sw_ao(lines, expected_range, task_name)
        task._names = verify_channel_names(task, names)
        task._devices = device_list(task)
        self._tasks['sw_ao'] = task
        self.write_sw_ao(initial_state)

    def configure_hw_di(self, fs, lines, names=None, trigger=None, clock=None):
        raise NotImplementedError
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

    def configure_sw_do(self, channels):
        task_name = '{}_sw_do'.format(self.name)
        task = setup_sw_do(channels, task_name)
        self._tasks['sw_do'] = task
        initial_state = np.zeros(len(channels), dtype=np.uint8)
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

    def register_done_callback(self, callback):
        self._callbacks.setdefault('done', []).append(callback)

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

    def unregister_done_callback(self, callback):
        try:
            self._callbacks['done'].remove(callback)
        except KeyError:
            log.warning('Callback no longer exists.')

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
        result = ctypes.c_int32()
        mx.DAQmxWriteAnalogF64(task, 1, True, 0, mx.DAQmx_Val_GroupByChannel,
                               state, result, None)
        if result.value != 1:
            raise ValueError('Unable to update software-timed AO')
        task._current_state = state

    def write_sw_do(self, state):
        task = self._tasks['sw_do']
        state = np.asarray(state).astype(np.uint8)
        result = ctypes.c_int32()
        mx.DAQmxWriteDigitalLines(task, 1, True, 0, mx.DAQmx_Val_GroupByChannel,
                                  state, result, None)
        if result.value != 1:
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

    @halt_on_error
    def _hw_ai_callback(self, samples):
        samples /= self._tasks['hw_ai']._sf
        for channel_name, s, cb in self._callbacks.get('ai', []):
            cb(samples[s])

    def _hw_di_callback(self, samples):
        for i, cb in self._callbacks.get('di', []):
            cb(samples[i])

    def _get_hw_ao_samples(self, offset, samples):
        channels = self.get_channels('analog', 'output', 'hardware')
        data = np.empty((len(channels), samples), dtype=np.double)
        for channel, ch_data in zip(channels, data):
            channel.get_samples(offset, samples, out=ch_data)
        return data

    def get_space_available(self, name=None):
        # It doesn't matter what the output channel is. Space will be the same
        # for all.
        result = ctypes.c_uint32()
        mx.DAQmxGetWriteSpaceAvail(self._tasks['hw_ao'], result)
        return result.value

    def hw_ao_callback(self, samples):
        # Get the next set of samples to upload to the buffer
        try:
            with self.lock:
                log_ao.trace('#> HW AO callback. Acquired lock for engine %s', self.name)
                available_samples = self.get_space_available()
                if available_samples < samples:
                    log_ao.trace('Not enough samples available for writing')
                else:
                    data = self._get_hw_ao_samples(self.total_samples_written, samples)
                    self.write_hw_ao(data, self.total_samples_written, timeout=0)
        except Exception as e:
            offset = ctypes.c_int32()
            curr_write = ctypes.c_uint64()
            mx.DAQmxGetWriteOffset(self._tasks['hw_ao'], offset)
            mx.DAQmxGetWriteCurrWritePos(self._tasks['hw_ao'], curr_write)
            raise

    def update_hw_ao(self, name, offset):
        # Get the next set of samples to upload to the buffer. Ignore the
        # channel name because we need to update all channels simultaneously.
        if offset > self.total_samples_written:
            return

        available = self.get_space_available()
        samples = available - (offset - self.total_samples_written)
        if samples <= 0:
            return

        log_ao.info('Updating hw ao at %d with %d samples', offset, samples)
        data = self._get_hw_ao_samples(offset, samples)
        self.write_hw_ao(data, offset=offset, timeout=0)

    def update_hw_ao_multiple(self, offsets, names):
        # This is really simple to implement since we have to update all
        # channels at once. So, we just pick the minimum offset and let
        # `update_hw_ao` do the work.
        self.update_hw_ao(None, min(offsets))

    @halt_on_error
    def write_hw_ao(self, data, offset, timeout=1):
        # TODO: add a safety-check to make sure waveform doesn't exceed limits.
        # This is a recoverable error unless the DAQmx API catches it instead.

        # Due to historical limitations in the DAQmx API, the write offset is a
        # signed 32-bit integer. For long-running applications, we will have an
        # overflow if we attempt to set the offset relative to the first sample
        # written. Therefore, we compute the write offset relative to the last
        # sample written (for requested offsets it should be negative).
        try:
            result = ctypes.c_int32()
            task = self._tasks['hw_ao']
            relative_offset = offset - self.total_samples_written
            mx.DAQmxSetWriteOffset(task, relative_offset)
            mx.DAQmxWriteAnalogF64(task, data.shape[-1], False, timeout, mx.DAQmx_Val_GroupByChannel, data.astype(np.float64), result, None)
            mx.DAQmxSetWriteOffset(task, 0)

            # Calculate total samples written
            self.total_samples_written = self.total_samples_written + relative_offset + data.shape[-1]

        except Exception as e:
            # If we log on every call, the logfile will get quite verbose.
            # Let's only log this information on an Exception.
            log_ao.info('Failed to write %r samples starting at %r', data.shape, offset)
            log_ao.info(' * Offset is %d samples relative to current write position %d', relative_offset, self.total_samples_written)
            log_ao.info(' * Current read position is %d and onboard buffer size is %d', self.ao_sample_clock(), task._onboard_buffer_size)
            raise

    def get_ts(self):
        try:
            return self.sample_time()
        except Exception as e:
            log.exception(e)
            return np.nan

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
        if not self._configured:
            return
        log.debug('Stopping engine')
        for task in self._tasks.values():
            mx.DAQmxClearTask(task)
        self._callbacks = {}
        self._configured = False

    def ai_sample_clock(self):
        task = self._tasks['hw_ai']
        result = ctypes.c_uint64()
        mx.DAQmxGetReadTotalSampPerChanAcquired(task, result)
        return result.value

    def ai_sample_time(self):
        return self.ai_sample_clock()/self.ai_fs

    def ao_sample_clock(self):
        try:
            task = self._tasks['hw_ao']
            result = ctypes.c_uint64()
            mx.DAQmxGetWriteTotalSampPerChanGenerated(task, result)
            return result.value
        except:
            return 0

    def ao_sample_time(self):
        return self.ao_sample_clock()/self.ao_fs

    def get_buffer_size(self, name):
        return self.hw_ao_buffer_size
