import ctypes

from atom.api import Float, Typed
from enaml.core.api import Declarative, d_

from ..engine import Engine
from daqengine import ni


################################################################################
# engine
################################################################################
class NIDAQEngine(Engine, ni.Engine):
    # Even though data is written to the analog outputs, it is buffered in
    # computer memory until it's time to be transferred to the onboard buffer of
    # the NI acquisition card. NI-DAQmx handles this behind the scenes (i.e.,
    # when the acquisition card needs additional samples, NI-DAQmx will transfer
    # the next chunk of data from the computer memory). We can overwrite data
    # that's been buffered in computer memory (e.g., so we can insert a target
    # in response to a nose-poke). However, we cannot overwrite data that's
    # already been transfered to the onboard buffer. So, the onboard buffer size
    # determines how quickly we can change the analog output in response to an
    # event.

    # TODO: this is not configurable on some systems. How do we figure out if
    # it's configurable?
    #hw_ao_onboard_buffer = d_(Float(8191))
    hw_ao_onboard_buffer = d_(Float(4095))

    # Since any function call takes a small fraction of time (e.g., nanoseconds
    # to milliseconds), we can't simply overwrite data starting at
    # hw_ao_onboard_buffer+1. By the time the function calls are complete, the
    # DAQ probably has already transferred a couple hundred samples to the
    # buffer. This parameter will likely need some tweaking (i.e., only you can
    # determine an appropriate value for this based on the needs of your
    # program).
    hw_ao_min_writeahead = d_(Float(8191 + 1000))

    _tasks = Typed(dict, {})
    _callbacks = Typed(dict, {})
    _timers = Typed(dict, {})
    _uint32 = Typed(ctypes.c_uint32)
    _uint64 = Typed(ctypes.c_uint64)
    _int32 = Typed(ctypes.c_int32)

    def __init__(self, *args, **kwargs):
        ni.Engine.__init__(self)
        Engine.__init__(self, *args, **kwargs)
