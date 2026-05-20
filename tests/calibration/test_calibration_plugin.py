"""Pure-Python test of CalibrationPlugin.will_calibrate.

We do not start a real workbench — we just instantiate the plugin and set
its calibrators list directly.
"""
from types import SimpleNamespace

from psi.controller.calibration.plugin import CalibrationPlugin


def test_will_calibrate_finds_matching_output():
    plugin = CalibrationPlugin()
    plugin._calibrators = [
        SimpleNamespace(outputs={'speaker_1': []}),
        SimpleNamespace(outputs={'speaker_2': []}),
    ]
    assert plugin.will_calibrate('speaker_1') is True
    assert plugin.will_calibrate('speaker_2') is True


def test_will_calibrate_returns_false_for_unknown_output():
    plugin = CalibrationPlugin()
    plugin._calibrators = [SimpleNamespace(outputs={'speaker_1': []})]
    assert plugin.will_calibrate('not_a_speaker') is False


def test_will_calibrate_empty_calibrators():
    plugin = CalibrationPlugin()
    plugin._calibrators = []
    assert plugin.will_calibrate('any') is False
