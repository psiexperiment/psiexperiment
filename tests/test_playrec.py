"""Unit tests for psi.controller.engines.soundcard.playrec."""
from types import SimpleNamespace

import psi.controller.engines.soundcard.playrec as playrec_module
import psi.util as util_module


class _FakeStream:
    def __init__(self, **kw):
        self.kw = kw
        self.samplerate = kw['samplerate']
        self.blocksize = kw['blocksize']


class _FakeSD:
    '''
    Minimal ``sounddevice`` stand-in for exercising ``PlayRec.configure``
    without real hardware. Records the kwargs the stream is opened with.
    '''

    def __init__(self):
        self.opened = {}
        self._device = {
            'name': 'RME Babyface', 'hostapi': 0,
            'max_input_channels': 8, 'max_output_channels': 8,
        }

    def query_devices(self, device=None):
        if device is None:
            return [dict(self._device)]
        return dict(self._device, index=0)

    def query_hostapis(self, index):
        return {'name': 'ASIO'}

    def AsioSettings(self, channels):
        return ('asio', tuple(channels))

    def InputStream(self, **kw):
        self.opened = dict(kw)
        return _FakeStream(**kw)


def test_input_stream_opened_with_fully_qualified_selector(monkeypatch):
    # Regression: the input (record-only) path must open the stream with the
    # *original* device selector it was handed -- a fully-qualified
    # "<name>, <host API>" string -- not the bare device name resolved from
    # it. The bare name re-opens sounddevice's ambiguous substring matching
    # that the fully-qualified selector exists to avoid (multiple drivers
    # exposing the same name). See PlayRec.configure.
    fake = _FakeSD()
    monkeypatch.setattr(playrec_module, 'sd', fake)
    monkeypatch.setattr(util_module, 'Application',
                        SimpleNamespace(instance=lambda: None))

    selector = 'RME Babyface, ASIO'
    playrec_module.PlayRec(
        fs=96000, device=selector, ai_channels=[0, 1],
        ai_cb=lambda *a: None, blocksize=4096,
    )
    assert fake.opened['device'] == selector
