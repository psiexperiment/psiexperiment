from atom.api import Float, Str
from enaml.core.api import Declarative, d_

from psi import SimpleState


class Calibration(SimpleState, Declarative):
    pass


class SensCalibration(Calibration):

    sensitivity = Float()


class FileCalibration(Calibration):

    filename = Str()
