'''
Tests for `SoundcardEngine` that need no sound card: `PlayRec`, which opens
the device, is replaced by a stand-in that records what it was asked for.
'''
from threading import Thread

import numpy as np
import pytest

import psi.controller.engines.soundcard as soundcard_module
from psi.controller.engines.soundcard import (
    SoundcardEngine, SoundcardHardwareAOChannel,
)


class FakePlayRec:

    def __init__(self, **kw):
        self.kw = kw


class FakeThread(Thread):

    def __init__(self, *args):
        super().__init__(daemon=True)

    def start(self):
        pass


@pytest.fixture
def fake_device(monkeypatch):
    monkeypatch.setattr(soundcard_module, 'PlayRec', FakePlayRec)
    monkeypatch.setattr(soundcard_module, 'SDAIThread', FakeThread)


def test_idle_engine_plays_silence(fake_device):
    # No channel in use (e.g., an EEG monitor on a keyboard loopback), but
    # the engine is the master clock, so it still opens the device: one
    # output, playing silence.
    engine = SoundcardEngine(name='sound', fs=48000, device_name='device')
    engine.add_channel(
        SoundcardHardwareAOChannel(name='speaker_2', channel=1))
    engine.configure()
    kw = engine._stream.kw
    assert kw['ao_channels'] == [1]
    assert 'ai_channels' not in kw

    # PlayRec hands the callback a zeroed buffer.
    buffer = np.zeros((1, 512))
    kw['ao_cb'](0, 512, buffer)
    assert np.all(buffer == 0)


def test_clone_keeps_settings():
    # The in-ear calibration runs on clones (`psi.controller.calibration.
    # acquire`). A clone left at the default 96 kHz failed to open an output
    # running at 44.1 kHz.
    engine = SoundcardEngine(name='sound', fs=44100, device_name='device',
                             blocksize=512, latency_samples=10)
    engine.add_channel(
        SoundcardHardwareAOChannel(name='speaker_1', channel=0))
    clone = engine.clone()
    assert (clone.fs, clone.blocksize, clone.latency_samples) == \
        (44100, 512, 10)
    assert clone.device_name == 'device'
    assert [c.name for c in clone.get_channels(active=False)] == \
        ['speaker_1']


def test_idle_engine_without_outputs_raises(fake_device):
    engine = SoundcardEngine(name='sound', fs=48000, device_name='device')
    with pytest.raises(ValueError, match='cannot keep time'):
        engine.configure()
