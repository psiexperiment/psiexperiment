"""Tests for psi.controller.calibration.calibrate.BaseCalibrate."""
import threading
from types import SimpleNamespace

import pandas as pd

import psi.util as util_module
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
    app = SimpleNamespace(is_main_thread=lambda: False)
    monkeypatch.setattr(util_module, 'Application',
                        SimpleNamespace(instance=lambda: app))

    def fake_deferred_call(fn, *args, **kwargs):
        threading.Thread(target=fn, args=args, kwargs=kwargs,
                         name='fake-gui-thread', daemon=True).start()

    monkeypatch.setattr(util_module, 'deferred_call', fake_deferred_call)

    calibrator = FakeCalibrate(input_name='hw_ai::microphone')
    seen = {}
    calibrator.observe(
        'result', lambda e: seen.setdefault('thread', threading.current_thread()))
    controller = SimpleNamespace(get_channel=lambda name: None)
    calibrator.calibrate(controller, None)

    # Set (on the GUI thread) by the time calibrate returns, for `save`.
    assert seen['thread'].name == 'fake-gui-thread'
    assert calibrator.result['calibration'].loc[('speaker_1', 1000), 'spl'] == 60
