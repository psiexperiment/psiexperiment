import enaml

from experiments import initialize_default, configure_logging


if __name__ == '__main__':
    with enaml.imports():
        from experiments.appetitive.manifest \
            import ControllerManifest
        from psi.controller.actions.NE1000.manifest \
            import NE1000Manifest
        from psi.controller.actions.pellet_dispenser.manifest \
            import PelletDispenserManifest
        from psi.controller.actions.room_light.manifest \
            import RoomLightManifest
        from psi.data.trial_log.manifest import TrialLogManifest
        from psi.data.event_log.manifest import EventLogManifest
        from psi.data.sdt_analysis.manifest import SDTAnalysisManifest
        from psi.data.hdf_store.manifest import HDFStoreManifest

        extra_manifests = [
            ControllerManifest,
            NE1000Manifest,
            PelletDispenserManifest,
            RoomLightManifest,
            TrialLogManifest,
            SDTAnalysisManifest,
            HDFStoreManifest,
            EventLogManifest,
        ]
        workbench = initialize_default(extra_manifests)

        core = workbench.get_plugin('enaml.workbench.core')
        core.invoke_command('enaml.workbench.ui.select_workspace',
                            {'workspace': 'psi.experiment.workspace'})

    ui = workbench.get_plugin('enaml.workbench.ui')
    ui.show_window()

    configure_logging()
    ui.start_application()
