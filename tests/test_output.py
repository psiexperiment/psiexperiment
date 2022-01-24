import pytest

import numpy as np

from psiaudio.stim import Cos2EnvelopeFactory, ToneFactory


@pytest.fixture()
def tb1(epoch_output):
    tone = ToneFactory(fs=epoch_output.fs, level=0, frequency=100,
                       calibration=epoch_output.calibration)
    envelope = Cos2EnvelopeFactory(fs=epoch_output.fs, start_time=0,
                                   rise_time=0.5, duration=5,
                                   input_factory=tone)
    return envelope


@pytest.fixture()
def tb2(epoch_output):
    tone = ToneFactory(fs=epoch_output.fs, level=0, frequency=250,
                       calibration=epoch_output.calibration)
    envelope = Cos2EnvelopeFactory(fs=epoch_output.fs, start_time=0,
                                   rise_time=0.25, duration=5,
                                   input_factory=tone)
    return envelope


def test_epoch_output_buffer(epoch_output, tb1, tb2):
    full_waveform1 = tb1.get_samples_remaining()
    tb1.reset()

    full_waveform2 = tb2.get_samples_remaining()
    tb2.reset()

    epoch_output.source = tb1
    epoch_output.activate(0)

    out = np.empty(1000)
    epoch_output.get_samples(0, 1000, out)
    assert np.all(out == full_waveform1[:1000])

    with pytest.raises(SystemError):
        epoch_output.get_samples(1001, 1000, out)

    epoch_output.get_samples(1000, 1000, out)
    assert np.all(out == full_waveform1[1000:2000])

    epoch_output.get_samples(0, 1000, out)
    assert np.all(out == full_waveform1[0:1000])

    epoch_output.get_samples(500, 1000, out)
    assert np.all(out == full_waveform1[500:1500])

    epoch_output.get_samples(1500, 1000, out)
    assert np.all(out == full_waveform1[1500:2500])

    epoch_output.get_samples(2500, 13, out[:13])
    assert np.all(out[:13] == full_waveform1[2500:2513])

    epoch_output.get_samples(2513, 13, out[:13])
    assert np.all(out[:13] == full_waveform1[2513:2526])

    epoch_output.source = tb2
    epoch_output.activate(2000)

    with pytest.raises(SystemError):
        epoch_output.get_samples(2001, 1000, out)

    epoch_output.get_samples(1500, 1000, out)
    expected = np.concatenate((full_waveform1[1500:2000],
                               full_waveform2[:500]), axis=-1)
    assert np.all(out == expected)

    epoch_output.get_samples(2000, 1000, out)
    assert np.all(out == full_waveform2[:1000])

    epoch_output.deactivate(2500)
    with pytest.raises(SystemError):
        epoch_output.get_samples(2600, 1000, out)

    epoch_output.get_samples(2000, 1000, out)
    assert np.all(out[:500] == full_waveform2[:500])
    assert np.all(out[500:] == 0)

    epoch_output.deactivate(500)
    epoch_output.get_samples(250, 1000, out)
    assert np.all(out[:250] == full_waveform1[250:500])
    assert np.all(out[250:] == 0)

    temp = np.empty(1000*25)
    epoch_output.get_samples(500, 1000*25, temp)
    epoch_output.get_samples(500, 1000, out)
    assert np.all(out == 0)
