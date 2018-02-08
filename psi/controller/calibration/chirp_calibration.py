from ..util import acquire


def chirp_power(engine, freq_lb=50, freq_ub=100e3, attenuation=0, vrms=1,
                repetitions=32, duration=0.1, iti=0.01):

    calibration = FlatCalibration.as_attenuation(vrms=vrms)
    ai_fs = engine.hw_ai_channels[0].fs
    ao_fs = engine.hw_ao_channels[0].fs
    queue = FIFOSignalQueue(ao_fs)

    factory = chirp_factory(ao_fs, freq_lb, freq_ub, duration, attenuation,
                            calibration=calibration)
    waveform = generate_waveform(factory, int(duration*ao_fs))
    print(waveform)
    queue.append(waveform, repetitions, iti)

    ao_channel = engine.hw_ao_channels[0]
    output = QueuedEpochOutput(parent=ao_channel, queue=quee,
                               auto_decrement=True)
    epochs = acquire(engine, queue, duration+iti)


def tone_calibration(engine, *args, **kwargs):
    '''
    Single output calibration at a fixed frequency
    Returns
    -------
    sens : dB (V/Pa)
        Sensitivity of output in dB (V/Pa).
    '''
    output_sens = tone_sens(engine, frequencies, *args, **kwargs)[0]
    return PointCalibration(frequencies, output_sens)
