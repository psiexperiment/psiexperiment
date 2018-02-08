import logging
log = logging.getLogger(__name__)

from functools import partial
import time

import numpy as np

from .util import tone_power_conv, db
from ..util import acquire
from . import (FlatCalibration, PointCalibration, CalibrationTHDError,
               CalibrationNFError)
from ..queue import FIFOSignalQueue
from ..output import QueuedEpochOutput
from ..input import ExtractEpochs


from psi.token.primitives import (tone_factory, silence_factory,
                                  generate_waveform)


thd_err_mesg = 'Total harmonic distortion for {}Hz is {}%'
nf_err_mesg = 'Power at {}Hz has SNR of {}dB'


def _to_sens(output_spl, output_gain, vrms):
    # Convert SPL to value expected at 0 dB gain and 1 VRMS
    norm_spl = output_spl-output_gain-db(vrms)
    return -norm_spl-db(20e-6)


def process_tone(fs, signal, frequency, min_db=None, max_thd=None,
                 thd_harmonics=3, silence=None):
    '''
    Compute the RMS at the specified frequency. Check for distortion.

    Parameters
    ----------
    fs : float
        Sampling frequency of signal
    signal : ndarray
        Last dimension must be time. If more than one dimension, first
        dimension must be repetition.
    min_db : {None, float}
        If specified, must provide a noise floor measure (silence). The ratio,
        in dB, of signal RMS to silence RMS must be greater than min_db. If
        not, a CalibrationNFError is raised.
    max_thd : {None, float}
        If specified, ensures that the total harmonic distortion, as a
        percentage, is less than max_thd. If not, a CalibrationTHDError is
        raised.
    thd_harmonics : int
        Number of harmonics to compute. If you pick too many, some harmonics
        may be above the Nyquist frequency and you'll get an exception.
    thd_harmonics : int
        Number of harmonics to compute. If you pick too many, some harmonics
        may be above the Nyquist frequency and you'll get an exception.
    silence : {None, ndarray}
        Noise floor measurement. Required for min_db. Shape must match signal
        in all dimensions except the first and last.
    '''
    harmonics = frequency * (np.arange(thd_harmonics) + 1)

    # This returns an array of [harmonic, repetition, channel]. Here, rms[0] is
    # the rms at the fundamental frequency. rms[1:] is the rms at all the
    # harmonics.
    signal = np.atleast_2d(signal)
    rms = tone_power_conv(signal, fs, harmonics, 'flattop')

    # Compute the mean RMS at F0 across all repetitions
    rms = rms.mean(axis=1)
    freq_rms = rms[0]

    # Compute the harmonic distortion as a percent
    thd = np.sqrt(np.sum(rms[1:]**2))/freq_rms * 100

    if min_db is not None:
        silence = np.atleast_2d(silence)
        noise_rms = tone_power_conv(silence, fs, frequency, 'flattop')
        noise_rms = noise_rms.mean(axis=-1)
        freq_snr = db(freq_rms, noise_rms)
        if np.any(freq_snr < min_db):
            mesg = nf_err_mesg.format(frequency, freq_snr)
            raise CalibrationNFError(mesg)

    if max_thd is not None and np.any(thd > max_thd):
        mesg = thd_err_mesg.format(frequency, thd)
        raise CalibrationTHDError(mesg)

    return freq_rms


def tone_power(engine, frequencies, gain=0, vrms=1, repetitions=2, min_db=10,
               max_thd=0.1, thd_harmonics=3, duration=0.1, trim=0.01, iti=0.01):

    frequencies = np.asarray(frequencies)
    calibration = FlatCalibration.as_attenuation(vrms=vrms)
    hw_ai = engine.get_channels('analog', 'input', 'hardware', False)
    hw_ao = engine.get_channels('analog', 'output', 'hardware', False)
    ao_fs = hw_ao[0].fs
    ai_fs = hw_ai[0].fs
    queue = FIFOSignalQueue(ao_fs)

    for frequency in frequencies:
        factory = tone_factory(ao_fs, gain, frequency, 0, 1, calibration)
        waveform = generate_waveform(factory, int(duration*ao_fs))
        queue.append(waveform, repetitions, iti)

    factory = silence_factory(ao_fs, calibration)
    waveform = generate_waveform(factory, int(duration*ao_fs))
    queue.append(waveform, repetitions, iti)

    # Add the queue to the output channel
    hw_ao[0].add_queued_epoch_output(queue, auto_decrement=True)

    # Attach the input to the channel
    epochs = []

    def accumulate(epochs, epoch):
        epochs.extend(epoch)

    cb = partial(accumulate, epochs)
    epoch_input = ExtractEpochs(queue=queue, epoch_size=duration+iti)
    epoch_input.add_callback(cb)
    hw_ai[0].add_input(epoch_input)

    engine.start()
    while not epoch_input.is_complete():
        time.sleep(0.1)
    engine.stop()

    # Signal will be in the shape (frequency, repetition, channel, time). The
    # last "frequency" will be silence (i.e., the noise floor).
    signal = np.concatenate([e['signal'][np.newaxis, np.newaxis] for e in epochs])
    signal.shape = [-1, repetitions] + list(signal.shape[1:])
    print(signal.shape)

    # Loop through the frequency epochs.
    silence = signal[-1, :, :, :]
    signal = signal[:-1, :, :, :]
    rms = []
    for f, s in zip(frequencies, signal):
        r = process_tone(ai_fs, s, f, min_db, max_thd, thd_harmonics, silence)
        rms.append(r)
    return rms


def tone_spl(engine, frequencies, *args, **kwargs):
    rms = tone_power(engine, frequencies, *args, **kwargs)
    calibration = engine.hw_ai_channels[0].calibration
    return calibration.get_spl(frequencies, rms)


def tone_sens(engine, frequencies, gain=-40, vrms=1, *args, **kwargs):
    kwargs.update(dict(gain=gain, vrms=vrms))
    output_spl = tone_spl(engine, frequencies, *args, **kwargs)
    mesg = 'Output {}dB SPL at {}Hz, {}dB gain, {}Vrms'
    log.debug(mesg.format(output_spl, frequencies, gain, vrms))
    output_sens = _to_sens(output_spl, gain, vrms)
    return output_sens


def tone_calibration(engine, frequencies, *args, **kwargs):
    '''
    Single output calibration at a fixed frequency
    Returns
    -------
    sens : dB (V/Pa)
        Sensitivity of output in dB (V/Pa).
    '''
    output_sens = tone_sens(engine, frequencies, *args, **kwargs)[0]
    return PointCalibration(frequencies, output_sens)
