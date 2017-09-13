import logging
log = logging.getLogger(__name__)

import numpy as np

from .util import tone_power_conv, db
from ..util import acquire
from . import (FlatCalibration, PointCalibration, CalibrationTHDError,
               CalibrationNFError)
from ..queue import FIFOSignalQueue


from psi.token.primitives import (tone_factory, silence_factory,
                                  generate_waveform)



thd_err_mesg = 'Total harmonic distortion for {}Hz is {}%'
nf_err_mesg = 'Power at {}Hz has SNR of {}dB'


def _to_sens(output_spl, output_gain, vrms):
    # Convert SPL to value expected at 0 dB gain and 1 VRMS
    norm_spl = output_spl-output_gain-db(vrms)
    return -norm_spl-db(20e-6)


def process_tone(fs, signal, frequencies, min_db=None, max_thd=None,
                 thd_harmonics=3):

    frequencies = np.asarray(frequencies)
    harmonics = np.arange(thd_harmonics) + 1
    h_frequencies = frequencies * harmonics[..., np.newaxis]

    # This returns an array [harmonic, f0, wavform, repetition, channel]. Here,
    # rms[0] is the rms at the fundamental frequency. rms[1:] is the rms at all
    # the harmonics.
    rms = tone_power_conv(signal, fs, h_frequencies, 'flattop')

    # Compute the mean RMS across all repetitions
    mean_rms = rms[0].mean(axis=-2)

    # This gives us the RMS as [frequency, channel]
    freq_rms = mean_rms[..., 0].diagonal()

    # This gives us the noise floor as [frequency, channel]
    freq_noise = mean_rms[:, -1, 0]

    # Compute the harmonic distortion
    thd = np.sqrt(np.sum(rms[1:]**2, axis=0))/rms[0]
    mean_thd = thd.mean(axis=-2)
    freq_thd = mean_thd[..., 0].diagonal()

    # Compute the SNR
    freq_snr = db(freq_rms, freq_noise)

    if min_db is not None and np.any(freq_snr < min_db):
        m = freq_snr < min_db
        mesg = nf_err_mesg.format(frequencies[m], freq_snr[m])
        raise CalibrationNFError(mesg)

    if max_thd is not None and np.any(freq_thd > max_thd):
        m = thd > max_thd
        mesg = thd_err_mesg.format(frequencies[m], thd[m]*100)
        raise CalibrationTHDError(mesg)

    for info in zip(frequencies, freq_noise, freq_rms, freq_snr, freq_thd):
        mesg = '{:.1f}Hz: Noise floor {:.1f}Vrms, signal {:.1f}Vrms, SNR {:.1f}dB, THD {:.2f}%'
        log.debug(mesg.format(*info))

    return freq_rms


def tone_power(engine, frequencies, gain=0, vrms=1, repetitions=1, min_db=10,
               max_thd=0.1, thd_harmonics=3, duration=0.1, trim=0.01, iti=0.01):

    frequencies = np.asarray(frequencies)
    calibration = FlatCalibration.as_attenuation(vrms=vrms)
    ai_fs = engine.hw_ai_channels[0].fs
    ao_fs = engine.hw_ao_channels[0].fs
    queue = FIFOSignalQueue(ao_fs)

    for frequency in frequencies:
        factory = tone_factory(ao_fs, gain, frequency, 1, calibration)
        waveform = generate_waveform(factory, int(duration*ao_fs))
        queue.append(waveform, repetitions, iti)

    factory = silence_factory(ao_fs, calibration)
    waveform = generate_waveform(factory, int(duration*ao_fs))
    queue.append(waveform, repetitions, iti)
    epochs = acquire(engine, queue, duration+iti)

    signal = np.concatenate([e['signal'][np.newaxis] for e in epochs])
    signal.shape = [-1, repetitions] + list(signal.shape[1:])
    return process_tone(ai_fs, signal, frequencies, min_db, max_thd,
                        thd_harmonics)


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
    output_sens = tone_sens(engine, frequencies, *args, **kwargs)
    return PointCalibration(frequencies, output_sens)
