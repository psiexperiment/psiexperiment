from atom.api import Typed

from ..engine import Engine


class DummyEngine(Engine):

    _callbacks = Typed(dict, {})
    _hw_ao_buffer = Typed(list, [])

    def start(self):
        pass

    def configure(self, configuration):
        if 'hw_ao' in configuration:
            pass

    def register_ao_callback(self, callback):
        cb = self._callbacks.setdefault('ao', [])
        cb.append(callback)

    def register_ai_callback(self, callback):
        cb = self._callbacks.setdefault('ai', [])
        cb.append(callback)

    def register_et_callback(self, callback):
        cb = self._callbacks.setdefault('et', [])
        cb.append(callback)
