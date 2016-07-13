import warnings
import tables as tb

import enaml

from experiments import initialize_default, configure_logging


if __name__ == '__main__':
    with enaml.imports():
        from experiments.appetitive.manifest \
            import ControllerManifest
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

    with tb.open_file('c:/users/bburan/desktop/test.h5', 'w') as fh:
        core.invoke_command('psi.data.hdf_store.set_node', {'node': fh.root})

        ui = workbench.get_plugin('enaml.workbench.ui')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ui.show_window()

        configure_logging()
        ui.start_application()
