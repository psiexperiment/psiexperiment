from psi import set_config


def main():
    import logging
    logging.basicConfig(level='ERROR')

    from psi.experiment.api import PSIWorkbench
    workbench = PSIWorkbench()

    set_config('EXPERIMENT', 'demo_experiment')
    io_manifest = 'io_manifest.IOManifest'
    controller_manifests = ['simple_experiment.ControllerManifest']
    workbench.register_core_plugins(io_manifest, controller_manifests)
    workbench.start_workspace('demo')


if __name__ == '__main__':
    main()
