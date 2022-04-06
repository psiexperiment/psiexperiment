import logging
#logging.basicConfig(level='DEBUG')

import numpy as np
import pylab as plt


from psi.controller.api import FIFOSignalQueue
from psi.controller.util import acquire
from psi.controller.calibration.api import FlatCalibration
from psi.controller.calibration.util import load_calibration, psd
from psi.core.enaml.api import load_manifest_from_file
from psiaudio.stim import ChirpFactory


io_file = 'c:/psi/io/pika.enaml'
cal_file = 'c:/psi/io/pika/default.json'
io_manifest = load_manifest_from_file(io_file, 'IOManifest')
io = io_manifest()
audio_engine = io.find('NI_audio')

load_calibration(cal_file, audio_engine.get_channels(active=False))
mic_channel = audio_engine.get_channel('microphone_channel')
mic_channel.gain = 40

speaker_channel = audio_engine.get_channel('speaker_1')

factory = ChirpFactory(fs=speaker_channel.fs,
                       start_frequency=500,
                       end_frequency=50000,
                       duration=0.02,
                       level=-30,
                       calibration=FlatCalibration.as_attenuation())

n = factory.n_samples_remaining()
chirp_waveform = factory.next(n)

result = acquire(audio_engine, chirp_waveform, 'speaker_1',
               ['microphone_channel'], repetitions=64, trim=0)

waveform = result['microphone_channel'][0].mean(axis=0)
plt.plot(waveform)
plt.show()
