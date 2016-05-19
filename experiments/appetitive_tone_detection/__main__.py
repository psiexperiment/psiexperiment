import enaml

from experiments import initialize_default


if __name__ == '__main__':
    with enaml.imports():
        from experiments.appetitive_tone_detection.manifest import ControllerManifest
        from psiexperiment.controller.reward.NE1000.manifest \
            import NE1000Manifest
        from psiexperiment.data.trial_log.manifest import TrialLogManifest
        from psiexperiment.data.sdt_analysis.manifest import SDTAnalysisManifest
        from psiexperiment.data.hdf_store.manifest import HDFStoreManifest

        extra_manifests = [
            ControllerManifest,
            NE1000Manifest,
            TrialLogManifest,
            SDTAnalysisManifest,
            #HDFStoreManifest,
        ]
        workbench = initialize_default(extra_manifests)
        ui = workbench.get_plugin('enaml.workbench.ui')
        ui.start_application()
