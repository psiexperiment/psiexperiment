import numpy as np

from threading import Timer
import time

from atom.api import Typed, Int, Float, Bool

from ..engine import Engine


class DummyEngine(Engine):

    _names = Typed(dict, {})
    _callbacks = Typed(dict, {})
    _hw_ao_buffer = Typed(np.ndarray)

    _ao_offset = Int(0)
    _timer = Typed(object)
    _last_tick = Float(0)

    _timer_interval = Float(1)
    _stop = Bool(False)

    _start_time = Float()

    buffer_size = Float(30)
    buffer_samples = Int()

    def start(self):
        self.buffer_samples = int(self.buffer_size*self.ao_fs)
        for cb in self._callbacks['ao']:
            cb(self._names['hw_ao'], self._ao_offset, self.buffer_samples)

        self._timer = Timer(self._timer_interval, self._tick)
        self._timer.start()
        self._start_time = time.time()

    def stop(self):
        self._stop = True

    def _tick(self):
        if self._stop:
            return

        current_tick = self.get_ts()

        samples = int((current_tick-self._last_tick)*self.ao_fs)
        self._last_tick = current_tick

        for cb_type, callbacks in self._callbacks.items():
            if cb_type == 'ao':
                for cb in callbacks:
                    cb(self._names['hw_ao'], self._ao_offset, samples)

        self._timer = Timer(self._timer_interval, self._tick)
        self._timer.start()

    def configure(self, configuration):
        super(DummyEngine, self).configure(configuration)
        for key in ['hw_ao', 'hw_ai']:
            self._names[key] = [c.name for c in configuration.get('hw_ao', [])]

    def register_ao_callback(self, callback):
        cb = self._callbacks.setdefault('ao', [])
        cb.append(callback)

    def register_ai_callback(self, callback):
        cb = self._callbacks.setdefault('ai', [])
        cb.append(callback)

    def register_et_callback(self, callback):
        cb = self._callbacks.setdefault('et', [])
        cb.append(callback)

    def write_hw_ao(self, waveforms, offset=None):
        if offset is None:
            if self._hw_ao_buffer is not None:
                self._hw_ao_buffer = \
                    np.concatenate((self._hw_ao_buffer, waveforms), axis=-1)
            else:
                self._hw_ao_buffer = waveforms
            self._ao_offset = self._hw_ao_buffer.shape[-1]
        else:
            lb, ub = offset, offset+waveforms.shape[-1]
            self._hw_ao_buffer[..., lb:ub] = waveforms
            self._hw_ao_buffer = self._hw_ao_buffer[..., :ub]
        self._ao_offset = self._hw_ao_buffer.shape[-1]

    def get_ts(self):
        return time.time()-self._start_time
