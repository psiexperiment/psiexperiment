import pytest

import numpy as np
import enaml


with enaml.imports():
    from psi.controller.calibration.tone import process_tone
    from psi.controller.calibration import CalibrationNFError
    from psi.controller.calibration import CalibrationTHDError


def make_tone(fs, f0, duration):
    n = int(duration*fs)
    t = np.arange(n, dtype=np.double)/fs
    y = np.cos(2*np.pi*f0*t)
    return y


def test_process_tone():
    fs = 100e3
    f1 = 1e3
    f2 = 500

    t1 = make_tone(fs, f1, 1)
    t2 = make_tone(fs, f2, 1)

    rms = 1/np.sqrt(2)

    # Build a 3D array of repetition x channel x time with two repetitions of
    # t1. The RMS power should be np.sqrt(2) by definition (e.g., if a tone's
    # peak to peak amplitude is X, then the RMS amplitude is X/np.sqrt(2)).
    signal = np.concatenate((t1[np.newaxis], t1[np.newaxis]))
    signal.shape = (2, 1, -1)
    result = process_tone(fs, signal, f1)
    assert np.allclose(result['rms'], rms)
    assert result.shape == (1,3)

    # Build a 3D array of repetition x channel x time with two repetitions, but
    # designed such that the second repetition is t2 (and therefore will have 0
    # power at f1). This means that the average RMS power should be half.
    signal = np.concatenate((t1[np.newaxis], t2[np.newaxis]))
    signal.shape = (2, 1, -1)
    result = process_tone(fs, signal, f1)
    assert np.allclose(result['rms'], 0.5*rms)
    assert result.shape == (1,3)

    # Build a 3D array of repetition x channel x time with one repetition and
    # two channels (with channel 1 containing t1 and channel 2 containing t2).
    # This should return *two* numbers (one for each channel).
    signal = np.concatenate((t1[np.newaxis], t2[np.newaxis]))
    signal.shape = (1, 2, -1)
    result = process_tone(fs, signal, f1)
    assert np.allclose(result['rms'], [rms, 0])
    print(result)
    assert result.shape == (2,3)
    result = process_tone(fs, signal, f2)
    assert np.allclose(result['rms'], [0, rms])
    assert result.shape == (2,3)

    # Now, test the most simple case (e.g., single repetition, single channel).
    result = process_tone(fs, t1, f1)
    assert result['rms'] == pytest.approx(rms)

    # Now, test silence
    silence = np.random.normal(scale=1e-12, size=t1.shape)
    result = process_tone(fs, silence, f1)
    assert result['rms'] == pytest.approx(0)

    # Now, make sure we get an error for the noise floor.
    with pytest.raises(CalibrationNFError):
        result = process_tone(fs, silence, f1, min_snr=3, silence=silence)

    # Now, create a harmonic for t1 at 2*f1. This harmonic will have 0.1% the
    # power of t1.
    t1_harmonic = 1e-2*make_tone(fs, f1*2, 1)
    signal = t1 + t1_harmonic
    result = process_tone(fs, signal, f1, max_thd=2)
    assert result['rms'] == pytest.approx(rms)

    with pytest.raises(CalibrationTHDError):
        result = process_tone(fs, signal, f1, max_thd=1)
