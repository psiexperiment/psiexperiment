from atom.api import Bool, Dict, Float, Int, List, Unicode, Value
from enaml.core.api import d_

from psi.core.enaml.api import PSIContribution


class ToneCalibrate(PSIContribution):

    outputs = d_(Dict())
    input_name = d_(Unicode())
    gain = d_(Float(-40))
    duration = d_(Float(100e3))
    iti = d_(Float(0))
    trim = d_(Float(10e-3))
    max_thd = d_(Value(None))
    min_snr = d_(Value(None))
    selector_name = d_(Unicode('default'))
    show_widget = d_(Bool(True))

    result = Value()


class ChirpCalibrate(PSIContribution):

    outputs = d_(List())
    input_name = d_(Unicode())
    gain = d_(Float(-30))
    duration = d_(Float(20e-3))
    iti = d_(Float(1e-3))
    repetitions = d_(Int(64))

    result = Value()
