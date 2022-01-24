import pytest

import enaml
with enaml.imports():
    from .helper_manifest import EVENT_RESULTS


def test_invoke_action(benchmark, controller):
    result = benchmark(controller.invoke_actions, 'dispense')


def test_actions(controller):
    EVENT_RESULTS[:] = []
    controller.invoke_actions('dispense')
    controller.invoke_actions('trial_start')
    controller.invoke_actions('dispense')
    controller.invoke_actions('trial_end')
    controller.invoke_actions('dispense')
    expected = [
        'dispense',
        'not trial_active and dispense',

        'trial_start',

        'dispense',
        'trial_active and dispense',

        'trial_end',

        'dispense',
        'not trial_active and dispense',
    ]
    assert expected == EVENT_RESULTS


def test_default_name(controller):
    i = controller.get_input('microphone_filtered')
    assert i.name == 'microphone_filtered'
    assert i.source.name == 'microphone_blocked_downsample'


def test_input_metadata(controller):
    from psi.util import declarative_to_dict
    i = controller.get_input('microphone_filtered')
    result = declarative_to_dict(i, 'metadata')
    assert result['btype'] == 'bandpass'
    assert result['source']['fs'] == 100e3
    channel = result['source']['source']['source']
    assert channel['calibration']['frequency'] == [1000, 2000]


def test_input_active(controller):
    mic = controller.get_input('microphone_filtered')
    assert mic.active == False
    blocked = controller.get_input('microphone_blocked')
    assert blocked.active == False
    dc = controller.get_input('microphone_dc')
    assert blocked.active == False

    mic.add_callback(lambda x: x)
    assert mic.active == True
    assert blocked.active == True
    assert dc.active == False


def test_filter_delay(controller):
    # Only the NIDAQ channels support the filter delay property right now. This
    # is because only the NDIAQ engine makes provisions to correct for the
    # filter delay.
    channel = controller.get_channel('hw_ao::speaker_1')
    assert channel.filter_delay == 0
    channel = controller.get_channel('hw_ao::speaker_2')
    assert channel.filter_delay == 1e-3
