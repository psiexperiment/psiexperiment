"""Tests for psi.controller.calibration.calibrate.BaseCalibrate."""
import threading
from types import SimpleNamespace

import pandas as pd

import psi.controller.calibration.calibrate as calibrate_module
from psi.controller.calibration.calibrate import BaseCalibrate


class FakeChannel:

    name = 'speaker_1'


class FakeCalibrate(BaseCalibrate):

    def get_config(self, controller, core):
        return {FakeChannel(): {}}

    def run_calibration(self, ao_channel, ai_channel, kwargs):
        return pd.DataFrame({'spl': [60]}, index=pd.Index([1000], name='freq'))


def test_result_set_on_gui_thread(monkeypatch):
    # Calibration runs on the control dispatcher thread, but the result
    # table and plots redraw when `result` changes, which Qt allows only on
    # the GUI thread (a click calibration's plot failed with "Timers cannot
    # be started from another thread").
    calls = []
    monkeypatch.setattr(calibrate_module, 'deferred_call',
                        lambda fn, *args: calls.append((fn, args)))

    calibrator = FakeCalibrate(input_name='hw_ai::microphone')
    seen = {}
    calibrator.observe(
        'result', lambda e: seen.setdefault('thread', threading.current_thread()))
    controller = SimpleNamespace(get_channel=lambda name: None)

    def run_on_dispatcher():
        seen['returned'] = calibrator.calibrate(controller, None)

    dispatcher = threading.Thread(target=run_on_dispatcher, name='dispatcher')
    dispatcher.start()
    dispatcher.join()

    # Not set on the dispatcher thread. The result is returned instead, for
    # `save`.
    assert 'thread' not in seen
    result = seen['returned']
    assert result['calibration'].loc[('speaker_1', 1000), 'spl'] == 60

    # Set when the GUI thread runs the deferred call.
    for fn, args in calls:
        fn(*args)
    assert seen['thread'] is threading.main_thread()
    assert calibrator.result['calibration'].equals(result['calibration'])
