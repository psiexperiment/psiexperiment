import pytest

import enaml
with enaml.imports():
    from .helper_manifest import EVENT_RESULTS


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
    assert i.source.name == 'microphone_blocked_Downsample'


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


def test_get_input_missing_raises(controller):
    with pytest.raises(KeyError, match='valid inputs are'):
        controller.get_input('nope')


def test_get_channel_missing_raises(controller):
    with pytest.raises(ValueError, match='No such channel'):
        controller.get_channel('hw_ai::not_a_real_channel')


def test_get_output_missing_raises(controller):
    with pytest.raises(ValueError, match='No such output'):
        controller.get_output('not_a_real_output')


def test_event_used_detects_bound_events(controller):
    # 'dispense' is bound via ExperimentAction in helper_manifest.
    assert controller.event_used('dispense') is True
    # 'not_an_event' is not bound to any action.
    assert controller.event_used('not_an_event') is False


def test_invoke_actions_updates_state(controller):
    controller.invoke_actions('trial_start')
    assert controller._action_context['trial_active'] is True
    controller.invoke_actions('trial_end')
    assert controller._action_context['trial_active'] is False


def test_template_methods_raise_actionable_errors():
    # The base controller has no trial/pause concept; the template methods
    # must fail with errors that tell a subclass author what to implement.
    from psi.controller.plugin import ControllerPlugin
    plugin = ControllerPlugin()
    with pytest.raises(NotImplementedError, match='next_trial'):
        plugin.end_trial()
    with pytest.raises(NotImplementedError, match='pause_experiment'):
        plugin.pause_experiment()
    with pytest.raises(NotImplementedError, match='resume_experiment'):
        plugin.resume_experiment()


def test_request_latches_flag_when_deferred(monkeypatch):
    # A falsy return from the template method means "not safe right now";
    # request_* must latch the corresponding flag for the subclass to
    # consume between trials. A truthy return must not latch.
    from psi.controller import plugin as plugin_module
    monkeypatch.setattr(plugin_module, 'deferred_call', lambda cb, *a: cb(*a))

    class PausableController(plugin_module.ControllerPlugin):

        def pause_experiment(self):
            return False

        def resume_experiment(self):
            return True

        def apply_changes(self):
            return False

    controller = PausableController()
    controller.request_pause()
    assert controller._pause_requested is True
    controller.request_apply()
    assert controller._apply_requested is True
    # Handled immediately: nothing to latch.
    controller.request_resume()
    assert controller._resume_requested is False
