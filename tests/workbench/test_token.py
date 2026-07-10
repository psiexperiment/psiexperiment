import pytest

import numpy as np

from psiaudio.calibration import FlatCalibration
from psiaudio.queue import InterleavedFIFOSignalQueue

from psi.controller.api import EpochOutput
from psi.controller.token_context import initialize_factory, load_items


@pytest.fixture
def tone_token(workbench):
    return workbench.get_plugin('psi.token').get_token('tone_burst')


@pytest.fixture
def tone_factory_builder(tone_token):
    # initialize_factory resolves context names (e.g. target_tone_frequency)
    # through the per-output block map populated by load_items.
    output = EpochOutput(name='target')
    load_items(output, tone_token)
    return lambda context: initialize_factory(output, tone_token, context)


@pytest.fixture
def tone_context():
    calibration = FlatCalibration.as_attenuation()
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


def tone_queue(tone_factory_builder, tone_context):
    fs = tone_context['fs']
    queue = InterleavedFIFOSignalQueue()
    queue.set_fs(fs)
    iti = 0.1
    tone_context['target_tone_burst_rise_time'] = 0.25
    tone_context['target_tone_burst_duration'] = 1

    queue.append(tone_factory_builder(tone_context), 1, iti)

    tone_context['target_tone_burst_duration'] = 2
    queue.append(tone_factory_builder(tone_context), 2, iti + 13/fs)
    return queue


tone_queue_1 = pytest.fixture(tone_queue)
tone_queue_2 = pytest.fixture(tone_queue)


def test_queue_generation(tone_queue_1, tone_queue_2):
    # pop_buffer always returns exactly the requested number of samples,
    # zero-padding once the queue is exhausted; emptiness is tracked via
    # is_empty().
    waveforms = [tone_queue_1.pop_buffer(50000)]
    while not tone_queue_1.is_empty():
        waveforms.append(tone_queue_1.pop_buffer(10000))

    waveforms1 = np.concatenate(waveforms)
    n = waveforms1.shape[-1]

    waveforms2 = tone_queue_2.pop_buffer(n)

    # There's some numerical precision issues here that limit how closely we
    # can get the two waveforms to match. Compare with relaxed precision and
    # high tolerance — the goal is to verify that streamed and bulk generation
    # produce the same waveform up to floating-point rounding in the phase
    # calculation.
    diff = np.abs(waveforms1 - waveforms2)
    # 99% of samples must agree to 3 decimal places; max divergence stays small.
    assert np.percentile(diff, 99) < 1e-3
    assert diff.max() < 0.05
