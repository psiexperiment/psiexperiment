from types import SimpleNamespace

import pytest

from atom.api import Value

from psi.controller.api import NullOutput
from psi.controller.engines.null import NullEngine
from psi.controller.io_manager import IOManager, find_engines, find_outputs


def _fake_point(*extensions):
    return SimpleNamespace(extensions=list(extensions))


def _fake_extension(children):
    return SimpleNamespace(get_children=lambda cls: list(children))


def test_find_engines_duplicate_name_raises():
    # Regression: this error path used to raise NameError (undefined
    # `engine_error`) instead of the intended descriptive ValueError.
    e1 = NullEngine(name='dup', buffer_size=10)
    e2 = NullEngine(name='dup', buffer_size=10)
    point = _fake_point(_fake_extension([e1, e2]))
    with pytest.raises(ValueError, match='More than one engine named "dup"'):
        find_engines(point)


def test_find_engines_last_engine_is_default_master():
    e1 = NullEngine(name='a', buffer_size=10, weight=0)
    e2 = NullEngine(name='b', buffer_size=10, weight=1)
    point = _fake_point(_fake_extension([e1, e2]))
    engines, master = find_engines(point)
    assert list(engines) == ['a', 'b']
    assert master is e2


def test_find_outputs_duplicate_name_raises(ao_channel):
    # Regression: this error path used to raise NameError (undefined
    # `output_error`) instead of the intended descriptive ValueError.
    NullOutput(name='dup', parent=ao_channel)
    NullOutput(name='dup', parent=ao_channel)
    point = _fake_point()
    with pytest.raises(ValueError, match='More than one output named "dup"'):
        find_outputs({'speaker': ao_channel}, point)


class _RecordingEngine(NullEngine):
    '''NullEngine that records lifecycle calls into a shared journal.'''

    journal = Value()
    done_callback = Value()

    def get_channels(self, *args, **kwargs):
        # Pretend the engine has active channels so the lifecycle methods
        # do not skip it.
        return ['fake_channel']

    def configure(self, active=True):
        self.journal.append((self.name, 'configure'))

    def start(self):
        self.journal.append((self.name, 'start'))

    def stop(self):
        self.journal.append((self.name, 'stop'))

    def register_done_callback(self, callback):
        self.done_callback = callback


def _io_with_engines():
    journal = []
    e1 = _RecordingEngine(name='a', buffer_size=10, journal=journal)
    e2 = _RecordingEngine(name='b', buffer_size=10, journal=journal)
    io = IOManager(engines={'a': e1, 'b': e2}, master_engine=e2)
    return io, e1, e2, journal


def test_io_manager_configure_engines_registers_done_callbacks():
    io, e1, e2, journal = _io_with_engines()
    io.configure_engines(lambda engine: lambda: engine.name)
    assert journal == [('a', 'configure'), ('b', 'configure')]
    assert e1.done_callback() == 'a'
    assert e2.done_callback() == 'b'


def test_io_manager_engine_lifecycle():
    io, e1, e2, journal = _io_with_engines()

    io.start_engines()
    assert io.engines_running
    # The master engine (b) must be started last.
    assert journal == [('a', 'start'), ('b', 'start')]
    with pytest.raises(ValueError, match='already running'):
        io.start_engines()

    io.stop_engines()
    assert not io.engines_running
    assert journal[-2:] == [('a', 'stop'), ('b', 'stop')]
    with pytest.raises(ValueError, match='not running'):
        io.stop_engines()


def test_io_manager_connect_output_unknown_target_raises():
    io = IOManager(outputs={'o1': NullOutput(name='o1')})
    with pytest.raises(ValueError, match='Unknown target'):
        io.connect_output('o1', 'nope')


def test_io_manager_unresolvable_outputs_raise():
    # Two outputs targeting each other can never resolve to a channel; the
    # wiring loop must fail loudly instead of spinning forever.
    o1 = NullOutput(name='o1', target_name='o2')
    o2 = NullOutput(name='o2', target_name='o1')
    io = IOManager(outputs={'o1': o1, 'o2': o2})
    with pytest.raises(ValueError, match='Unable to configure outputs'):
        io.connect_outputs()


def test_io_manager_connect_output_to_channel(ao_channel):
    o = NullOutput(name='o1', target_name=ao_channel.reference)
    io = IOManager(channels={ao_channel.reference: ao_channel},
                   outputs={'o1': o})
    io.connect_outputs()
    assert o.target is ao_channel
