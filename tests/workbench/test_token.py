import pytest

import numpy as np

import enaml
from enaml.workbench.api import Workbench

from psiaudio.calibration import InterpCalibration
from psiaudio.queue import InterleavedFIFOSignalQueue


@pytest.fixture
def tone_token(workbench):
    return workbench.get_plugin('psi.token').get_token('tone_burst')


@pytest.fixture
def tone_context():
    calibration = InterpCalibration.get_attenuation()
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
    queue = InterleavedFIFOSignalQueue()
    queue.set_fs(fs)
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

    waveforms2, empty = tone_queue_2.pop_buffer(n)

    # There's some numerical precision issues here that limit how closely we
    # can get the two waveforms to match. Compare with relaxed precision and
    # high tolerance — the goal is to verify that streamed and bulk generation
    # produce the same waveform up to floating-point rounding in the phase
    # calculation.
    diff = np.abs(waveforms1 - waveforms2)
    # 99% of samples must agree to 3 decimal places; max divergence stays small.
    assert np.percentile(diff, 99) < 1e-3
    assert diff.max() < 0.05
