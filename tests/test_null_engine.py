"""Direct unit tests for psi.controller.engines.null.NullEngine."""
import time

import pytest

from psiaudio.calibration import FlatCalibration

from psi.controller.api import (
    HardwareAIChannel, HardwareAOChannel, HardwareDIChannel,
    SoftwareAIChannel,
)
from psi.controller.engines.null import NullEngine


@pytest.fixture
def populated_engine():
    engine = NullEngine(buffer_size=10.0)
    cal = FlatCalibration.as_attenuation()
    engine.add_channel(HardwareAIChannel(name='ai0', fs=100e3, calibration=cal))
    engine.add_channel(HardwareAOChannel(name='ao0', fs=100e3, calibration=cal))
    engine.add_channel(HardwareDIChannel(name='di0', fs=100e3, calibration=cal))
    engine.add_channel(SoftwareAIChannel(name='swai0', calibration=cal))
    return engine


def test_add_channel_records_engine(populated_engine):
    for c in populated_engine.channels:
        assert c.engine is populated_engine


def test_get_buffer_size(populated_engine):
    ch = populated_engine.get_channel('ai0')
    assert populated_engine.get_buffer_size(ch) == 10.0


def test_get_ts_monotonic_after_start():
    engine = NullEngine(buffer_size=10.0)
    engine.start()
    t0 = engine.get_ts()
    # Sleep briefly and verify time advances.
    time.sleep(0.01)
    t1 = engine.get_ts()
    assert t1 > t0


def test_get_channels_filter_by_direction(populated_engine):
    inputs = populated_engine.get_channels(direction='input', active=False)
    outputs = populated_engine.get_channels(direction='output', active=False)
    in_names = {c.name for c in inputs}
    out_names = {c.name for c in outputs}
    assert 'ai0' in in_names and 'di0' in in_names and 'swai0' in in_names
    assert 'ao0' in out_names
    assert in_names.isdisjoint(out_names)


def test_get_channels_filter_by_timing(populated_engine):
    hw = populated_engine.get_channels(timing='hardware', active=False)
    sw = populated_engine.get_channels(timing='software', active=False)
    assert {c.name for c in hw} == {'ai0', 'ao0', 'di0'}
    assert {c.name for c in sw} == {'swai0'}


def test_get_channels_filter_by_mode(populated_engine):
    analog = populated_engine.get_channels(mode='analog', active=False)
    digital = populated_engine.get_channels(mode='digital', active=False)
    assert {c.name for c in analog} == {'ai0', 'ao0', 'swai0'}
    assert {c.name for c in digital} == {'di0'}


def test_get_channels_filter_combined(populated_engine):
    hw_ai = populated_engine.get_channels(
        mode='analog', direction='input', timing='hardware', active=False
    )
    assert {c.name for c in hw_ai} == {'ai0'}


def test_get_channels_invalid_timing_raises(populated_engine):
    with pytest.raises(ValueError, match='Unsupported timing'):
        populated_engine.get_channels(timing='bogus', active=False)


def test_get_channels_invalid_direction_raises(populated_engine):
    with pytest.raises(ValueError, match='Unsupported direction'):
        populated_engine.get_channels(direction='bogus', active=False)


def test_get_channels_invalid_mode_raises(populated_engine):
    with pytest.raises(ValueError, match='Unsupported mode'):
        populated_engine.get_channels(mode='bogus', active=False)


def test_get_channel_with_prefix(populated_engine):
    # `hw_ai::ai0` form should resolve as just `ai0`.
    ch = populated_engine.get_channel('hw_ai::ai0')
    assert ch.name == 'ai0'


def test_get_channel_unknown_raises(populated_engine):
    with pytest.raises(AttributeError, match='does not exist'):
        populated_engine.get_channel('not_a_channel')


def test_remove_channel(populated_engine):
    ch = populated_engine.get_channel('ai0')
    populated_engine.remove_channel(ch)
    assert ch not in populated_engine.channels
    assert ch.engine is None
