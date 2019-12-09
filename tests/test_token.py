import pytest
import numpy as np
import matplotlib.pyplot as plt

import enaml
with enaml.imports():
    from psi.token.primitives import (cos2envelope, Cos2EnvelopeFactory,
                                      SilenceFactory, ToneFactory)
    from psi.controller.calibration.api import InterpCalibration


def test_cos2envelope():
    fs = 10e3
    duration = 10e-3
    tone_duration = 5e-3
    rise_time = 0.5e-3
    samples = round(duration*fs)

    # Make sure that the envelope is identical even if we delay the start
    y0 = cos2envelope(fs, 0, samples, 0, rise_time, tone_duration)
    for offset in (0.1e-3, 0.2e-3, 0.3e-3):
        y1 = cos2envelope(fs, 0, samples, offset, rise_time, tone_duration)
        n = round(offset * fs)
        np.testing.assert_allclose(y0[:-n], y1[n:])

    # Now, test piecemeal generation
    partition_size = 0.1e-3
    partition_samples = round(partition_size * fs)
    n_partitions = round(samples / partition_samples)
    for offset in (0, 0.1e-3, 0.2e-3, 0.3e-3):
        print(offset)
        env = Cos2EnvelopeFactory(fs, offset, rise_time, tone_duration,
                                SilenceFactory(fill_value=1))
        y1 = [env.next(partition_samples) for i in range(n_partitions)]
        y1 = np.concatenate(y1)
        n = round(offset * fs)
        if n > 0:
            np.testing.assert_allclose(y0[:-n], y1[n:])
        else:
            np.testing.assert_allclose(y0, y1)


def test_token_generation():
    calibration = InterpCalibration.as_attenuation()

    fs = 100e3

    tone_factory = ToneFactory(fs=fs, level=0, frequency=100,
                               calibration=calibration)
    factory = Cos2EnvelopeFactory(fs=fs, start_time=0, rise_time=0, duration=1,
                                  input_factory=tone_factory)

    assert tone_factory.get_duration() == np.inf
    assert factory.get_duration() == 1

    samples = int(fs)

    # test what happens when we get more samples than are required to generate
    # the token (should be zero-padded).
    waveform = factory.next(samples*2)
    assert tone_factory.is_complete() is False
    assert factory.is_complete() is True
    assert waveform.shape == (samples*2,)
    rms1 = np.mean(waveform[:samples]**2)**0.5
    rms2 = np.mean(waveform[samples:]**2)**0.5
    assert rms1 == pytest.approx(1)
    assert rms2 == 0

    # test what happens when we request even more samples (should continue to
    # be zero-padded)
    waveform = factory.next(samples*2)
    rms1 = np.mean(waveform[:samples]**2)**0.5
    rms2 = np.mean(waveform[samples:]**2)**0.5
    assert rms1 == 0
    assert rms2 == 0

    # now reset the token and start over
    factory.reset()
    assert tone_factory.is_complete() is False
    assert factory.is_complete() is False
    waveform = factory.next(samples)
    assert tone_factory.is_complete() is False
    assert factory.is_complete() is True
    assert waveform.shape == (samples,)
    rms = np.mean(waveform**2)**0.5
    assert rms == pytest.approx(1)

    t = np.arange(fs, dtype=np.float32)/fs
    expected = np.cos(2*np.pi*t*100)*np.sqrt(2)
    np.testing.assert_allclose(waveform, expected, atol=2e-4)
