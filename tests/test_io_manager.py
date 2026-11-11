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

    #: What get_channels returns. Pretend the engine has active channels so
    #: the lifecycle methods do not skip it, unless told otherwise.
    used_channels = Value()

    def _default_used_channels(self):
        return ['fake_channel']

    def get_channels(self, *args, **kwargs):
        return self.used_channels

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


def test_io_manager_appoints_master_in_use():
    # E.g., an EEG monitor on a keyboard loopback: the keyboard engine is in
    # use, the declared master (the sound card) is not. The last engine in
    # use becomes the master; engines not in use never run.
    journal = []
    nidaq = _RecordingEngine(name='nidaq', buffer_size=10, journal=journal,
                             weight=0)
    keyboard = _RecordingEngine(name='keyboard', buffer_size=10,
                                journal=journal, weight=1)
    sound = _RecordingEngine(name='sound', buffer_size=10, journal=journal,
                             used_channels=[], weight=2)
    io = IOManager(engines={'nidaq': nidaq, 'keyboard': keyboard,
                            'sound': sound},
                   master_engine=sound)

    io.configure_engines()
    assert io.master_engine is keyboard
    io.start_engines()
    io.stop_engines()
    assert journal == [
        ('nidaq', 'configure'), ('keyboard', 'configure'),
        ('nidaq', 'start'), ('keyboard', 'start'),
        ('nidaq', 'stop'), ('keyboard', 'stop'),
    ]


def test_io_manager_keeps_master_in_use():
    # A master in use (declared, or set by NIDAQ's start-trigger setup) is
    # kept.
    io, e1, e2, journal = _io_with_engines()
    io.master_engine = e1
    io.configure_engines()
    assert io.master_engine is e1


def test_io_manager_no_engines_in_use():
    journal = []
    sound = _RecordingEngine(name='sound', buffer_size=10, journal=journal,
                             used_channels=[])
    io = IOManager(engines={'sound': sound}, master_engine=sound)
    io.configure_engines()
    io.start_engines()
    io.stop_engines()
    assert io.master_engine is sound
    assert journal == []


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


################################################################################
# Waiting for the inputs before stopping
################################################################################
class FakeClock:
    '''
    Stands in for the `time` module in `psi.controller.engine`. Each wait
    advances the clock by the wait, and every engine's input by `rate` s per
    s of waiting (0 for a stalled input).
    '''

    def __init__(self, journal=None):
        self.now = 0.0
        self.engines = []
        self.journal = journal

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds
        for engine in self.engines:
            engine.delivered += engine.rate * seconds
        if self.journal is not None:
            self.journal.append(('clock', 'sleep'))


class FakeInputEngine:
    '''
    An engine whose clock is at `ts` and whose inputs have been delivered up
    to `delivered`, both in s. `delivered` None means it has no inputs.
    '''

    def __init__(self, name, ts, delivered, rate=1.0, timeout=0.2):
        self.name = name
        self.ts = ts
        self.delivered = delivered
        self.rate = rate
        self.timeout = timeout

    def get_ts(self):
        return self.ts

    def input_time(self):
        return self.delivered

    def input_wait_timeout(self):
        return self.timeout


@pytest.fixture
def clock(monkeypatch):
    from psi.controller import engine as engine_module
    clock = FakeClock()
    monkeypatch.setattr(engine_module, 'time', clock)
    return clock


def wait(clock, *engines):
    from psi.controller.engine import wait_for_inputs
    clock.engines = [e for e in engines if e.delivered is not None]
    return wait_for_inputs(engines)


def test_wait_for_inputs_until_each_catches_up(clock):
    # 50 ms and 120 ms behind their clocks.
    a = FakeInputEngine('a', 10.0, 9.95)
    b = FakeInputEngine('b', 3.0, 2.88)
    assert wait(clock, a, b) == []
    assert a.delivered >= 10.0 and b.delivered >= 3.0
    # Until the slower one caught up, and no longer.
    assert 0.12 <= clock.now < 0.13


def test_wait_for_inputs_gives_up_on_a_stalled_input(clock, caplog):
    '''
    An input that stops arriving can't hang the stop: it is waited for until
    its timeout, with a warning, while the others still catch up.
    '''
    stalled = FakeInputEngine('stalled', 10.0, 9.9, rate=0, timeout=0.2)
    fine = FakeInputEngine('fine', 10.0, 9.95)
    assert wait(clock, stalled, fine) == [stalled]
    assert 0.2 <= clock.now < 0.21
    assert fine.delivered >= 10.0
    assert 'Stopping stalled 100 ms before its inputs caught up' in caplog.text


def test_wait_for_inputs_skips_engines_without_inputs(clock):
    assert wait(clock, FakeInputEngine('output only', 10.0, None)) == []
    assert clock.now == 0


def test_wait_for_inputs_skips_an_unreadable_clock(clock):
    assert wait(clock, FakeInputEngine('a', float('nan'), 9.0)) == []
    assert clock.now == 0


def test_io_manager_waits_for_inputs_before_stopping(monkeypatch):
    '''
    Every engine keeps running until all the inputs have caught up, and only
    then are the engines stopped.
    '''
    from psi.controller import engine as engine_module
    import psi.controller.io_manager as io_module
    io, e1, e2, journal = _io_with_engines()
    clock = FakeClock(journal)
    monkeypatch.setattr(engine_module, 'time', clock)
    # The recording engines are null engines, which have no inputs; add one
    # that is 10 ms behind to what the IO manager waits for.
    inputs = FakeInputEngine('inputs', 10.0, 9.99)
    clock.engines = [inputs]
    real_wait = engine_module.wait_for_inputs
    monkeypatch.setattr(io_module, 'wait_for_inputs',
                        lambda engines: real_wait([*engines, inputs]))
    io.start_engines()
    io.stop_engines()
    stops = journal.index(('a', 'stop'))
    assert ('clock', 'sleep') in journal[:stops]
    assert ('clock', 'sleep') not in journal[stops:]
    assert inputs.delivered >= 10.0
