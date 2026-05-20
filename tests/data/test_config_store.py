"""Tests for the `_save_io` helper in psi.data.sinks.config_store."""
import json
from types import SimpleNamespace

import enaml


with enaml.imports():
    from psi.data.sinks.config_store import _save_io


class _FakeChannel:
    """Minimal stand-in for psi.controller.channel.Channel for io.json."""
    def __init__(self, name):
        self.name = name


class _FakeController:
    def __init__(self):
        self._channels_by_direction = {'output': [], 'input': []}

    def get_channels(self, direction=None, active=True):
        return self._channels_by_direction[direction]


def test_save_io_writes_input_and_output_channel_keys(tmp_path, monkeypatch):
    # _save_io calls declarative_to_dict on each channel; we replace it with a
    # trivial serializer so we don't need real declarative metadata.
    monkeypatch.setattr(
        'psi.data.sinks.config_store.declarative_to_dict',
        lambda c, tag, include_dunder: {'name': c.name},
    )
    controller = _FakeController()
    controller._channels_by_direction['output'].append(_FakeChannel('speaker_1'))
    controller._channels_by_direction['input'].append(_FakeChannel('mic_1'))

    out = tmp_path / 'io.json'
    _save_io(controller, out)

    payload = json.loads(out.read_text())
    assert 'output' in payload and 'input' in payload
    assert payload['output']['speaker_1'] == {'name': 'speaker_1'}
    assert payload['input']['mic_1'] == {'name': 'mic_1'}
