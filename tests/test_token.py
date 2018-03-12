import pytest
import numpy as np

import enaml
with enaml.imports():
    from psi.token.primitives import ToneFactory, Cos2EnvelopeFactory
    from psi.controller.calibration import InterpCalibration


def test_token_generation():
    calibration = InterpCalibration.as_attenuation()

    fs = 100e3

    tone_factory = ToneFactory(fs=fs, level=0, frequency=100,
                               calibration=calibration)
    factory = Cos2EnvelopeFactory(fs=fs, start_time=0, rise_time=0, duration=1,
                                  calibration=calibration,
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
    np.testing.assert_array_equal(waveform, expected)
