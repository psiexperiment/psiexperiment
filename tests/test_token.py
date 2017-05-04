import pytest

import numpy as np

import enaml
from enaml.workbench.api import Workbench

from psi.controller.calibration import InterpCalibration
from psi.controller.queue import InterleavedFIFOSignalQueue

with enaml.imports():
    from psi.token.manifest import TokenManifest


@pytest.fixture
def workbench():
    workbench = Workbench()
    workbench.register(TokenManifest())
    return workbench


@pytest.fixture
def tone_token(workbench):
    plugin = workbench.get_plugin('psi.token')
    return plugin.generate_epoch_token('tone_burst', 'target', 'target')


@pytest.fixture
def tone_context():
    calibration = InterpCalibration.as_attenuation()
    fs = 100e3
    frequency = 100
    context = {
        'target_tone_frequency': frequency,
        'target_tone_level': 0,
        'target_tone_burst_start_time': 0,
        'target_tone_burst_rise_time': 0,
        'target_tone_burst_duration': 1,
        'fs': fs,
        'calibration': calibration,
    }
    return context


def test_token_generation(tone_token, tone_context):
    # Request 2x as many samples as needed. Be sure it gets chopped down to the
    # actual length.
    fs = tone_context['fs']
    frequency = tone_context['target_tone_frequency']

    generator = tone_token.initialize_generator(tone_context)
    waveform, complete = generator.send({'samples': int(fs)*2})
    t = np.arange(fs, dtype=np.float32)/fs
    expected = np.cos(2*np.pi*t*frequency)
    np.testing.assert_array_equal(waveform, expected)
    assert complete == True


def test_queue_generation(tone_token, tone_context):
    queue = InterleavedFIFOSignalQueue()

    fs = tone_context['fs']
    iti_samples = int(fs*0.1)
    tone_context['target_tone_burst_rise_time'] = 0.25
    tone_context['target_tone_burst_duration'] = 1
    factory = tone_token.initialize_factory(tone_context)
    queue.append(factory, 1, iti_samples)

    tone_context['target_tone_burst_duration'] = 2
    factory = tone_token.initialize_factory(tone_context)
    queue.append(factory, 2, iti_samples+13)

    w, empty = queue.pop_buffer(50e3)
    waveforms = [w]
    while not empty:
        w, empty = queue.pop_buffer(10e3)
        waveforms.append(w)

    import pylab as pl
    offset = 0
    for w in waveforms:
        i = np.arange(len(w)) + offset
        pl.plot(i, w)
        offset += len(w)
    pl.show()
