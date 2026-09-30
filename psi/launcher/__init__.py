'''
Framework for building launchers: GUIs that queue and run psi experiments.

A launcher lets the operator pick paradigms, preferences and plugins, then
runs each experiment as a separate `psi` subprocess and tracks its progress
over a websocket. The pieces are:

* `Experiment` / `FrozenExperiment` (`psi.launcher.experiment`): one row of
  the launcher, before and after its run mode is fixed.
* `ProcessManager` (`psi.launcher.process_manager`): runs the `psi`
  subprocesses and relays their events.
* `LauncherSettings` (`psi.launcher.settings`): launcher state; subclass it
  and implement `build_experiment`.
* Widgets (`psi.launcher.widgets`): `ExperimentSequence`, `AddRemoveCombo`,
  `AddItem`.
* Icons (`psi.launcher.icons`): the shared look of psi program icons.

Everything is importable from `psi.launcher.api` (with the enaml import hook
active). `set_app_id`, which a launcher's `main` calls before building its GUI,
lives in `psi.application`.

This package is provisional: its API may change between minor releases while
the first few launchers built on it settle.
'''
