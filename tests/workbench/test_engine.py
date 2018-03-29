def test_channel_durations(workbench):
    controller = workbench.get_plugin('psi.controller')
    engine = list(controller._engines.values())[0]
    print(engine)
