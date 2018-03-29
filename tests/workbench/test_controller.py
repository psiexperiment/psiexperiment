import enaml
with enaml.imports():
    from .helper_manifest import EVENT_RESULTS


def test_actions(controller):
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
    print(EVENT_RESULTS)
    assert expected == EVENT_RESULTS


def test_default_name(controller):
    i = controller.get_input('microphone_filtered')
    assert i.name == 'microphone_filtered'
    assert i.source.name == 'microphone_blocked_downsample'


def test_input_metadata(controller):
    from psi.util import declarative_to_dict
    mic = controller.get_input('microphone_filtered')
    result = declarative_to_dict(mic, 'metadata')
    assert result['btype'] == 'bandpass'
    assert result['source']['fs'] == 100e3
    channel = result['source']['source']['source']
    assert channel['calibration']['frequency'] == [1000, 2000]
