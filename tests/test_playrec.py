"""Unit tests for psi.controller.engines.soundcard.playrec."""
from types import SimpleNamespace

import numpy as np
import pytest

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


def _time(dac=0, current=0, adc=0):
    '''
    Stand-in for the time info PortAudio passes the callback.
    '''
    return SimpleNamespace(outputBufferDacTime=dac, currentTime=current,
                           inputBufferAdcTime=adc)


class _Status:
    '''
    Stand-in for ``sounddevice.CallbackFlags``: false when no flag is set.
    '''

    def __init__(self, **flags):
        for flag in playrec_module.XRUN_FLAGS:
            setattr(self, flag, flags.get(flag, False))

    def __bool__(self):
        return any(getattr(self, f) for f in playrec_module.XRUN_FLAGS)


def test_play_callback_counts_xruns():
    # Underflows used to be ignored, so a device playing silence (which
    # delayed every later stimulus for good) went unnoticed.
    context = playrec_module.PlayCallbackContext(lambda *args: None)
    outdata = np.zeros((512, 1))
    time = _time()
    for status in [_Status(), _Status(output_underflow=True), _Status()]:
        context(outdata, 512, time, status)
    assert context.status_counts['output_underflow'] == 1
    assert context.status_counts['input_overflow'] == 0


def test_record_callback_counts_xruns():
    context = playrec_module.RecordCallbackContext(lambda *args: None)
    indata = np.zeros((512, 1))
    time = _time()
    context(indata, 512, time, _Status(input_overflow=True))
    assert context.status_counts['input_overflow'] == 1


def test_output_lag_floor():
    # 10 ms blocks at 1 kHz. Each reaches the DAC 1 s after its offset says
    # it should, give or take the host API's buffering (here every other
    # block is 4 ms later), which the floor leaves out.
    context = playrec_module.PlayCallbackContext(lambda *args: None, fs=1000)
    outdata = np.zeros((10, 1))

    def play(blocks, silence=0):
        for _ in range(blocks):
            k = context.ao_s0 // 10
            dac_time = 1 + k * 0.01 + 0.004 * (k % 2) + silence
            context(outdata, 10, _time(dac_time),
                    _Status())

    play(10)
    # Not measured until the first window has passed.
    assert np.isnan(context.output_lag_floor)
    play(30)
    assert context.output_lag_floor == pytest.approx(1)

    # The device played 30 ms of silence: from then on every block reaches
    # the DAC 30 ms later.
    play(60, silence=0.03)
    assert context.output_lag_floor == pytest.approx(1.03)


def test_output_lag_needs_dac_times():
    # Host APIs that do not report DAC times pass 0.
    context = playrec_module.PlayCallbackContext(lambda *args: None, fs=1000)
    for _ in range(100):
        context(np.zeros((10, 1)), 10, _time(),
                _Status())
    assert np.isnan(context.output_lag_floor)


def test_callback_gap_and_duration(monkeypatch):
    # WASAPI shared mode flags no underflows, so a callback that runs late
    # (e.g., waiting for the GIL) shows only as a long gap.
    clock = iter([0.0, 0.001, 0.0053, 0.0063, 0.0303, 0.0313]).__next__
    monkeypatch.setattr(playrec_module, 'perf_counter', clock)
    context = playrec_module.PlayCallbackContext(lambda *args: None)
    outdata = np.zeros((512, 1))
    time = _time()
    for _ in range(3):
        context(outdata, 512, time, _Status())
    gap, duration = context.take_callback_stats()
    assert gap == pytest.approx(0.025)
    assert duration == pytest.approx(0.001)
    assert context.take_callback_stats() == (0, 0)


def test_callback_record(monkeypatch, tmp_path):
    # Each callback's timing is kept, so late ones can be found afterwards:
    # the second block below reaches the DAC 2 ms after the callback that
    # made it began.
    clock = iter([10.0, 10.001, 10.1, 10.102]).__next__
    monkeypatch.setattr(playrec_module, 'perf_counter', clock)
    context = playrec_module.PlayCallbackContext(lambda *args: None)
    outdata = np.zeros((512, 1))
    context(outdata, 512, _time(dac=1.05, current=1.0), _Status())
    context(outdata, 512, _time(dac=1.102, current=1.1), _Status())
    rows = context.record.get()
    assert rows.shape == (2, len(playrec_module.CallbackRecord.columns))
    entered, current, dac, adc, duration, ao_s0, ai_s0 = rows.T
    np.testing.assert_allclose(entered, [10.0, 10.1])
    np.testing.assert_allclose(dac - (entered - 9), [0.05, 0.002])
    np.testing.assert_allclose(duration, [0.001, 0.002])
    np.testing.assert_array_equal(ao_s0, [0, 512])

    path = tmp_path / 'callbacks.csv'
    context.record.save(path)
    lines = path.read_text().splitlines()
    assert lines[0].startswith('# entered - current median 9.0')
    assert lines[1] == ','.join(playrec_module.CallbackRecord.columns)
    assert len(lines) == 4


def test_callback_record_keeps_the_most_recent():
    record = playrec_module.CallbackRecord(size=3)
    for i in range(5):
        record.add(i, 0, 0, 0, 0, i, 0)
    np.testing.assert_array_equal(record.get()[:, 0], [2, 3, 4])
