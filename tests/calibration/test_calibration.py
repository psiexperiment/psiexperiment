from pathlib import Path

from psi.controller.calibration.api import load_calibration_data


def test_load_calibration():
    # This is a simple test to make sure that we can continue to load
    # calibration formats.
    path = Path(__file__).parent / 'default.json'
    calibrations = load_calibration_data(path)
    assert 'reference_microphone_channel' in calibrations
    assert 'microphone_channel' in calibrations
