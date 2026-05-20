"""Workbench-level tests for engine wiring on the controller plugin."""


def test_engine_registered(workbench):
    controller = workbench.get_plugin('psi.controller')
    # TestControllerManifest registers exactly one NullEngine.
    assert len(controller._engines) == 1


def test_master_engine_defaults_to_last(workbench):
    controller = workbench.get_plugin('psi.controller')
    # With only one engine and no explicit master_clock=True, that engine
    # becomes the master by the find_engines fallback.
    assert controller._master_engine is list(controller._engines.values())[-1]


def test_get_channels_filters_by_direction(workbench, controller):
    inputs = controller.get_channels(direction='input', active=False)
    outputs = controller.get_channels(direction='output', active=False)
    # Helper manifest defines one AI channel and three AO channels.
    assert len(inputs) == 1
    assert len(outputs) == 3
    assert {c.name for c in outputs} == {'speaker_0', 'speaker_1', 'speaker_2'}


def test_get_channel_returns_by_reference(controller):
    ch = controller.get_channel('hw_ao::speaker_1')
    assert ch.name == 'speaker_1'
