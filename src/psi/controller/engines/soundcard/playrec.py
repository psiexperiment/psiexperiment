import os
os.environ['SD_ENABLE_ASIO'] = '1'

import logging
log = logging.getLogger(__name__)

import threading
from time import perf_counter

import numpy as np
import sounddevice as sd

from psi.util import call_on_gui_thread

# PlayRec normally runs on psi's control-dispatcher thread (see
# ``psi.controller.dispatcher``), not the GUI thread. Some Windows audio
# backends require PortAudio stream lifecycle calls (open/start/stop) to run
# on the thread that owns the application's Windows message loop -- observed
# with an ASIO device, where PortAudio's WASAPI device-list refresh
# (triggered internally as a side effect of stream operations, regardless of
# host API) hangs indefinitely if invoked from a thread that isn't pumping
# messages. So every such call goes through `call_on_gui_thread`.


#: The problems PortAudio flags a callback with when the stream could not keep
#: up (see `BaseCallbackContext.status_counts`).
XRUN_FLAGS = ('output_underflow', 'output_overflow', 'input_underflow',
              'input_overflow')

#: Seconds over which `BaseCallbackContext.output_lag_floor` takes the lowest
#: value. Longer than the host API's bursts of callbacks (~15 ms in WASAPI
#: shared mode), short enough that the clock is corrected soon after a gap.
LAG_FLOOR_WINDOW = 0.25


class BaseCallbackContext:
    '''
    The callback PortAudio calls with each block. Subclasses fill in
    `process`; `__call__` also records how the stream is keeping up (read by
    `SoundcardEngine.log_timing`; written on the audio thread).
    '''

    def __init__(self, fs=None):
        self.ao_s0 = 0
        self.ai_s0 = 0

        #: Sampling rate, needed to track the output lag (see
        #: `output_lag_floor`). None turns that off.
        self.fs = fs

        #: Number of callbacks PortAudio flagged with each of `XRUN_FLAGS`.
        #: An output underflow means the device played silence because a
        #: block was late, so the samples fall behind the stream clock for
        #: good. The keys are fixed so the dict can be copied while the audio
        #: thread updates it. WASAPI shared mode does not flag these: the
        #: Windows audio engine plays the silence without telling PortAudio.
        self.status_counts = dict.fromkeys(XRUN_FLAGS, 0)

        #: How far the output samples are behind the stream clock, plus a
        #: constant: when each block reaches the DAC (stream time, as
        #: `stream.time`) minus its offset / fs. Silence the device played
        #: in place of late blocks raises it for good. The value from one
        #: block jitters by the host API's buffering (WASAPI shared mode
        #: hands over ~3 blocks at once), so this is the lowest value over
        #: each `LAG_FLOOR_WINDOW` seconds, updated once per window. NaN
        #: until the first window ends, or if the host API reports no DAC
        #: times (it passes 0). Read by `SoundcardEngine.output_lag`.
        self.output_lag_floor = np.nan
        self._window_end = None
        self._window_min = np.inf

        #: Longest time between the starts of two callbacks, and longest
        #: callback, in seconds, since `take_callback_stats` last read them.
        #: A gap longer than the audio the host API has queued is silence.
        self.max_callback_gap = 0.0
        self.max_callback_duration = 0.0
        self._last_callback_start = None

    def take_callback_stats(self):
        '''
        Return (max_callback_gap, max_callback_duration) and start over.
        '''
        stats = self.max_callback_gap, self.max_callback_duration
        self.max_callback_gap = self.max_callback_duration = 0.0
        return stats

    def record_status(self, status):
        if status:
            for flag in XRUN_FLAGS:
                if getattr(status, flag):
                    self.status_counts[flag] += 1

    def record_output_time(self, time):
        dac_time = time.outputBufferDacTime
        if not dac_time or self.fs is None:
            return
        self._window_min = min(self._window_min, dac_time - self.ao_s0 / self.fs)
        if self._window_end is None:
            self._window_end = dac_time + LAG_FLOOR_WINDOW
        elif dac_time >= self._window_end:
            self.output_lag_floor = self._window_min
            self._window_min = np.inf
            self._window_end = dac_time + LAG_FLOOR_WINDOW

    def process_ao(self, outdata, samples):
        buffer = np.zeros_like(outdata.T)
        self.ao_cb(self.ao_s0, samples, buffer)
        outdata[:] = buffer.T
        self.ao_s0 += samples

    def _process_ai_firstcall(self, indata, samples):
        # Drop first input buffer since we are priming the output buffer on the first call.
        self.process_ai = self._process_ai

    def _process_ai(self, indata, samples):
        self.ai_cb(self.ai_s0, samples, indata.T.copy())
        self.ai_s0 += samples

    process_ai = _process_ai_firstcall

    def __call__(self, *args):
        # PortAudio passes the data array(s), then samples, time and status.
        start = perf_counter()
        if self._last_callback_start is not None:
            self.max_callback_gap = max(self.max_callback_gap,
                                        start - self._last_callback_start)
        self._last_callback_start = start
        self.record_status(args[-1])
        self.process(*args[:-1])
        self.max_callback_duration = max(self.max_callback_duration,
                                         perf_counter() - start)

    def process(self, *args):
        raise NotImplementedError


class RecordCallbackContext(BaseCallbackContext):

    def __init__(self, ai_cb, ai_channels=None, fs=None):
        super().__init__(fs)
        self.ai_cb = ai_cb
        self.ai_channels = ai_channels

    def process(self, indata, samples, time):
        # Read the next segment to the input buffer
        if self.ai_channels is not None:
            self.process_ai(indata[:, self.ai_channels], samples)
        else:
            self.process_ai(indata, samples)


class PlayCallbackContext(BaseCallbackContext):

    def __init__(self, ao_cb, ao_channels=None, fs=None):
        super().__init__(fs)
        self.ao_cb = ao_cb
        self.ao_channels = ao_channels

    def process(self, outdata, samples, time):
        self.record_output_time(time)
        # Write the next segment to the output buffer
        if self.ao_channels is not None:
            self.process_ao(outdata[:, self.ao_channels], samples)
        else:
            self.process_ao(outdata, samples)


class PlayRecordCallbackContext(BaseCallbackContext):

    def __init__(self, ai_cb, ao_cb, ai_channels=None, ao_channels=None,
                 fs=None):
        super().__init__(fs)
        self.ai_cb = ai_cb
        self.ai_channels = ai_channels
        self.ao_cb = ao_cb
        self.ao_channels = ao_channels

    def process(self, indata, outdata, samples, time):
        self.record_output_time(time)
        if self.ai_channels is not None:
            self.process_ai(indata[:, self.ai_channels], samples)
        else:
            self.process_ai(indata, samples)
        if self.ao_channels is not None:
            self.process_ao(outdata[:, self.ao_channels], samples)
        else:
            self.process_ao(outdata, samples)


class PlayRec:
    '''
    Basic wrapper around an audio device that provides play and record
    functionality.

    Parameters
    ----------
    fs : float
        Sampling rate to set device to.
    device : int or string
        Device to open. Either a PortAudio device index, a device name, or --
        preferred, to avoid ambiguity when several drivers expose the same
        name -- a fully-qualified ``"<name>, <host API>"`` query string, which
        sounddevice matches exactly (see ``_get_device_id``). cftscal passes
        the fully-qualified form via ``PSI_SOUND_DEVICE_NAME``.
    ai_channels : {None, list of int}
        List of channels to record from. If None, do not record.
    ao_channels : {None, list of int}
        List of channels to play. If None, do not play.
    ai_cb : {None, callable}
        Callback that takes recorded samples.
    ao_cb : {None, callable}
        Callback that generates samples to play. First argument to the callback
        is the number of samples needed. Callback must return a 2D array with
        one row for each output channel.
    blocksize : {None, int}
        Blocksize of read/write. Smaller blocksizes generally have better
        latency. Set to None to let PortAudio decide what's best.

    All channel numbers are specified using 0 based numbering. To determine the
    channel number for RME devices, you can open up TotalMixFX and count from
    left to right starting at 0.
    '''

    def __init__(self, fs, device, ai_channels=None, ao_channels=None,
                 ai_cb=None, ao_cb=None, blocksize=4096):
        self.fs = fs
        self.device = device
        self.ai_channels = ai_channels
        self.ao_channels = ao_channels
        self.ai_cb = ai_cb
        self.ao_cb = ao_cb
        self.blocksize = blocksize

        log.warning('Requested device: %r', self.device)
        matches = [d for d in sd.query_devices() if d['name'] == self.device]
        log.warning('Devices matching that name: %r', matches)

        self.device_info = sd.query_devices(self.device)
        self.hostapi_info = sd.query_hostapis(self.device_info['hostapi'])
        log.warning('Resolved device: %r', self.device_info)
        log.warning('Resolved host API: %r', self.hostapi_info)

        self.event = threading.Event()
        self.configure()

    def configure(self):
        '''
        Configure the acquisition stream
        '''
        stream_kw = {
            'samplerate': self.fs,
            'blocksize': self.blocksize,
        }
        if self.ao_channels is not None and self.ai_channels is not None:
            stream_class = sd.Stream
            stream_kw['device'] = (self.device, self.device)
            stream_kw['channels'] = (len(self.ai_channels), len(self.ao_channels))
            stream_kw['callback'] = PlayRecordCallbackContext(self.ai_cb, self.ao_cb,
                                                             fs=self.fs)
            if self.hostapi_info['name'] == 'ASIO':
                stream_kw['extra_settings'] = (
                    sd.AsioSettings(self.ai_channels),
                    sd.AsioSettings(self.ao_channels),
                )
        elif self.ao_channels is not None:
            stream_class = sd.OutputStream
            stream_kw['device'] = self.device
            stream_kw['channels'] = len(self.ao_channels)
            stream_kw['callback'] = PlayCallbackContext(self.ao_cb, fs=self.fs)
            if self.hostapi_info['name'] == 'ASIO':
                stream_kw['extra_settings'] = sd.AsioSettings(self.ao_channels)
        elif self.ai_channels is not None:
            stream_class = sd.InputStream
            # Use the original selector (self.device), which may be a
            # fully-qualified "<name>, <host API>" string, rather than
            # device_info['name'] (the bare name) -- the bare name re-opens the
            # door to sounddevice's ambiguous substring matching that the
            # fully-qualified selector exists to avoid.
            stream_kw['device'] = self.device
            if self.hostapi_info['name'] == 'ASIO':
                stream_kw['channels'] = len(self.ai_channels)
                stream_kw['extra_settings'] = sd.AsioSettings(self.ai_channels)
                stream_kw['callback'] = RecordCallbackContext(self.ai_cb, fs=self.fs)
            elif self.hostapi_info['name'] == 'Windows WDM-KS':
                stream_kw['channels'] = self.device_info['max_input_channels']
                stream_kw['callback'] = RecordCallbackContext(self.ai_cb, self.ai_channels, fs=self.fs)
        else:
            raise ValueError('No input or output channels specified')

        log.warning('Opening stream with: %r', stream_kw)
        #: The callback context, which tracks xruns and output timing.
        self.callback = stream_kw['callback']
        self.stream = call_on_gui_thread(
            lambda: stream_class(**stream_kw, finished_callback=self.event.set)
        )
        if self.stream.samplerate != self.fs:
            raise ValueError('Could not get desired sample rate')
        if self.stream.blocksize != self.blocksize:
            raise ValueError('Could not get desired blocksize')

    def start(self):
        call_on_gui_thread(self.stream.start)

    def stop(self):
        call_on_gui_thread(self.stream.stop)
        call_on_gui_thread(self.stream.close)


if __name__ == '__main__':
    import argparse
    import time
    def printer(*args, **kwargs):
        print(args, kwargs)

    parser = argparse.ArgumentParser('playrec')
    parser.add_argument('fs', type=int)
    parser.add_argument('device')
    args = parser.parse_args()
    player = PlayRec(args.fs, args.device, ai_channels=[0], ai_cb=printer)
    player.start()
    time.sleep(1)
    player.stop()
