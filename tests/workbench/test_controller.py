import enaml 
with enaml.imports():
    from .helper_manifest import EVENT_RESULTS


def test_actions(workbench):
    controller = workbench.get_plugin('psi.controller')
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
