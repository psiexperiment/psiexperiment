from copy import copy

import pytest
import enaml
import numpy as np

with enaml.imports():
    from psi.controller.calibration import FlatCalibration
    from psi.controller.queue import FIFOSignalQueue
    from psi.token.primitives import ToneFactory


def test_queue():
    '''
    Test ability to work with continuous tones and move to next
    '''
    calibration = FlatCalibration.as_attenuation()

    queue = FIFOSignalQueue(fs=100e3)
    f1 = ToneFactory(100e3, 0, 5, 0, 1, calibration)
    gf1 = copy(f1)
    queue.append(f1, 1)

    f2 = ToneFactory(100e3, 0, 10, 0, 1, calibration)
    gf2 = copy(f2)
    queue.append(f2, 1)

    assert queue.get_max_duration() is np.inf

    s1 = queue.pop_buffer(100e3)[0]
    gs1 = gf1.next(100e3)
    assert np.allclose(s1, gs1)

    s1 = queue.pop_buffer(100e3)[0]
    gs1 = gf1.next(100e3)
    assert np.allclose(s1, gs1)

    queue.next_trial()

    s2 = queue.pop_buffer(100e3)[0]
    gs2 = gf2.next(100e3)
    assert np.allclose(s2, gs2)

    s2 = queue.pop_buffer(100e3)[0]
    gs2 = gf2.next(100e3)
    assert np.allclose(s2, gs2)
