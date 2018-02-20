import pytest
import enaml
import numpy as np

with enaml.imports():
    from psi.controller.calibration import FlatCalibration
    from psi.controller.queue import FIFOSignalQueue
    from psi.token.primitives import tone_factory


def test_queue():
    '''
    Test ability to work with continuous tones and move to next
    '''
    calibration = FlatCalibration.as_attenuation()

    queue = FIFOSignalQueue(fs=100e3)
    f1 = lambda: tone_factory(100e3, 0, 5, 0, 1, calibration)
    queue.append(f1, 1)
    g1 = f1()
    next(g1)

    f2 = lambda: tone_factory(100e3, 0, 10, 0, 1, calibration)
    queue.append(f2, 1)
    g2 = f2()
    next(g2)

    s1 = queue.pop_buffer(100e3)[0]
    gs1 = g1.send({'samples': 100e3})[0]
    assert np.allclose(s1, gs1)

    s1 = queue.pop_buffer(100e3)[0]
    gs1 = g1.send({'samples': 100e3})[0]
    assert np.allclose(s1, gs1)

    queue.next_trial()

    s2 = queue.pop_buffer(100e3)[0]
    gs2 = g2.send({'samples': 100e3})[0]
    assert np.allclose(s2, gs2)

    s2 = queue.pop_buffer(100e3)[0]
    gs2 = g2.send({'samples': 100e3})[0]
    assert np.allclose(s2, gs2)
