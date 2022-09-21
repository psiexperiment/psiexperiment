from psi import set_config


def main():
    import logging
    logging.basicConfig(level='ERROR')

    from psi.experiment.api import PSIWorkbench

    workbench = PSIWorkbench()

    set_config('EXPERIMENT', 'demo_experiment')
    io_manifest = 'io_manifest.IOManifest'
    manifests = ['startstop_controller.ControllerManifest']
    workbench.register_core_plugins(io_manifest, manifests)
    workbench.start_workspace('demo')


if __name__ == '__main__':
    main()
