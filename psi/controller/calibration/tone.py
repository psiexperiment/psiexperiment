import logging
log = logging.getLogger(__name__)

import numpy as np

from .util import tone_power_conv, db
from ..util import acquire
from . import (FlatCalibration, PointCalibration, CalibrationTHDError,
               CalibrationNFError)
from ..queue import FIFOSignalQueue
from ..output import QueuedEpochOutput


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

    harmonics = frequency * (np.arange(thd_harmonics) + 1)

    # This returns an array of [harmonic, repetition, channel]. Here, rms[0] is
    # the rms at the fundamental frequency. rms[1:] is the rms at all the
    # harmonics.
    rms = tone_power_conv(signal, fs, harmonics, 'flattop')

    # Compute the mean RMS at F0 across all repetitions
    rms = rms.mean(axis=-2)
    freq_rms = rms[0]

    # Compute the harmonic distortion
    thd = np.sqrt(np.sum(rms[1:]**2))/freq_rms

    if min_db is not None:
        noise_rms = tone_power_conv(silence, fs, frequency, 'flattop')
        noise_rms = noise_rms.mean(axis=-1)
        freq_snr = db(freq_rms, noise_rms)
        if np.any(freq_snr < min_db):
            mesg = nf_err_mesg.format(frequency, freq_snr)
            raise CalibrationNFError(mesg)

    if max_thd is not None and np.any(thd > max_thd):
        mesg = thd_err_mesg.format(frequency, thd*100)
        raise CalibrationTHDError(mesg)

    return freq_rms


def tone_power(engine, frequencies, gain=0, vrms=1, repetitions=2, min_db=10,
               max_thd=0.1, thd_harmonics=3, duration=0.1, trim=0.01, iti=0.01):

    frequencies = np.asarray(frequencies)
    calibration = FlatCalibration.as_attenuation(vrms=vrms)
    ai_fs = engine.hw_ai_channels[0].fs
    ao_fs = engine.hw_ao_channels[0].fs
    queue = FIFOSignalQueue(ao_fs)

    for frequency in frequencies:
        factory = tone_factory(ao_fs, gain, frequency, 0, 1, calibration)
        waveform = generate_waveform(factory, int(duration*ao_fs))
        queue.append(waveform, repetitions, iti)

    factory = silence_factory(ao_fs, calibration)
    waveform = generate_waveform(factory, int(duration*ao_fs))
    queue.append(waveform, repetitions, iti)

    # Attach the output to the channel
    ao_channel = engine.hw_ao_channels[0]
    output = QueuedEpochOutput(parent=ao_channel, queue=queue,
                               auto_decrement=True)

    epochs = acquire(engine, queue, duration+iti)

    # Signal will be in the shape (frequency, repetition, channel, time). The
    # last "frequency" will be silence (i.e., the noise floor).
    signal = np.concatenate([e['signal'][np.newaxis] for e in epochs])
    signal.shape = [-1, repetitions] + list(signal.shape[1:])

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
