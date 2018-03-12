import pytest

import numpy as np

import enaml
from enaml.workbench.api import Workbench

from psi.controller.calibration import InterpCalibration
from psi.controller.queue import InterleavedFIFOSignalQueue


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
        'target_tone_polarity': 1,
        'target_tone_phase': 0,
        'fs': fs,
        'calibration': calibration,
    }
    return context


def tone_queue(tone_token, tone_context):
    fs = tone_context['fs']
    queue = InterleavedFIFOSignalQueue(fs)
    iti = 0.1
    tone_context['target_tone_burst_rise_time'] = 0.25
    tone_context['target_tone_burst_duration'] = 1

    factory = tone_token.initialize_factory(tone_context)
    queue.append(factory, 1, iti)

    tone_context['target_tone_burst_duration'] = 2
    factory = tone_token.initialize_factory(tone_context)
    queue.append(factory, 2, iti + 13/fs)
    return queue


tone_queue_1 = pytest.fixture(tone_queue)
tone_queue_2 = pytest.fixture(tone_queue)


def test_queue_generation(tone_queue_1, tone_queue_2):
    w, empty = tone_queue_1.pop_buffer(50000)
    waveforms = [w]
    while not empty:
        w, empty = tone_queue_1.pop_buffer(10000)
        waveforms.append(w)

    waveforms1 = np.concatenate(waveforms)
    n = waveforms1.shape[-1]

    # There's some numerical precision issues here that limit how closely we
    # can get the two waveforms to match. I've plotted them and it does not
    # appear to be an "off by one sample" issue. One possible source is how the
    # phase is calculated for the tone generation. Perhaps it's stopping on an
    # imprecise floating-point value that introduces the error.
    waveforms2, empty = tone_queue_2.pop_buffer(n)

    np.testing.assert_almost_equal(waveforms1, waveforms2, decimal=3)
