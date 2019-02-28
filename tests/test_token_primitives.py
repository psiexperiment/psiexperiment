import pytest
import enaml
import numpy as np

__builtins__['profile'] = lambda x: x

with enaml.imports():
    from psi.token import primitives


def test_cos2envelope(benchmark):
    fs = 100e3
    offset = 0
    samples = 400000
    start_time = 0
    rise_time = 0.25
    duration = 4.0

    expected = np.ones(400000)
    t_samples = round(rise_time*fs)
    t_env = np.linspace(0, rise_time, t_samples, endpoint=False)
    ramp = np.sin(2*np.pi*t_env*1.0/rise_time*0.25)**2
    expected[:t_samples] = ramp
    ramp = np.sin(2*np.pi*t_env*1.0/rise_time*0.25 + np.pi/2)**2
    expected[-t_samples:] = ramp

    actual = benchmark(primitives.cos2envelope, fs, offset, samples,
                       start_time, rise_time, duration)

    np.testing.assert_almost_equal(actual, expected)


def test_sam_envelope(benchmark):
    offset = 0
    samples = 400000
    fs = 100000
    depth = 1
    fm = 5
    delay = 1
    eq_phase = 0
    eq_power = 1

    actual = benchmark(primitives.sam_envelope, offset, samples, fs, depth, fm,
                       delay, eq_phase, eq_power)


def test_square_wave(benchmark):
    fs = 100e3
    level = 5
    frequency = 10
    duty_cycle = 0.5
    samples = round(fs / frequency)

    #expected = np.zeros(samples)

    for duty_cycle in np.arange(0, 1.0, 0.2):
        factory = primitives.SquareWaveFactory(fs, level, frequency, duty_cycle)
        waveform = factory.next(samples)
        assert waveform.mean() == pytest.approx(level * duty_cycle)

    factory = primitives.SquareWaveFactory(fs, level, frequency, duty_cycle)
    actual = benchmark(factory.next, samples)
