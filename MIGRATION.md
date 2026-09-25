# Migrating third-party code to the refactored psiexperiment

Audience: anyone (including a future Claude session) updating packages that
depend on psiexperiment — e.g. `cfts`, `abr`, `psilbhb`, lab-specific
paradigm repositories, or analysis scripts. This documents every change from
the 2026-07 architectural cleanup that can affect downstream code, with
mechanical steps to find and fix each one.

Relevant psiexperiment commits: `7ef8573` (bug fixes + lint), `f69be36`
(logic extracted from manifests), `ee36a02` (package layering).

## TL;DR

Most code keeps working because `psi.controller.api` and
`psi.experiment.api` still re-export the moved names. You must act if the
downstream package does any of the following:

1. Imports `PSIWorkbench` (moved, **not** re-exported).
2. Imports directly from a *module path* that moved (e.g.
   `psi.controller.experiment_action`, `psi.experiment.preferences`,
   `psi.controller.output_manifest`).
3. Uses the removed `CONTEXT_MAP` global or `Block.initialize_factory`.
4. Runs on Python < 3.10 (no longer supported).
5. Uses hardware-timed digital input on NIDAQ (now raises
   `NotImplementedError`).

## Step 1 — Find affected code

Run these from the root of the downstream repository (they cover `.py` and
`.enaml` files):

```sh
grep -rn "PSIWorkbench" --include="*.py" --include="*.enaml" .
grep -rn "psi.controller.experiment_action" --include="*.py" --include="*.enaml" .
grep -rn "psi.experiment.preferences\|psi.experiment.status_item\|psi.experiment.metadata_item\|psi.experiment.workbench" --include="*.py" --include="*.enaml" .
grep -rn "psi.controller.output_manifest\|psi.controller.manifest import\|psi.experiment.manifest import" --include="*.py" --include="*.enaml" .
grep -rn "CONTEXT_MAP\|initialize_factory\|load_items\|generate_waveform" --include="*.py" --include="*.enaml" .
```

If all greps come back empty, no changes are needed.

## Step 2 — Apply the import renames

| Old import | New import | Notes |
|---|---|---|
| `from psi.experiment.api import PSIWorkbench` | `from psi.application.workbench import PSIWorkbench` | **Not re-exported anywhere else.** The workbench is the application bootstrapper and now lives at the application layer. |
| `from psi.experiment.workbench import PSIWorkbench` | `from psi.application.workbench import PSIWorkbench` | Same move. |
| `from psi.controller.experiment_action import ExperimentAction, ...` | `from psi.core.api import ExperimentAction, ...` | `from psi.controller.api import ExperimentAction` also still works. |
| `from psi.experiment.preferences import Preferences, ItemPreferences, PluginPreferences` | `from psi.core.api import ...` | `psi.experiment.api` re-exports these, so `from psi.experiment.api import Preferences` still works. |
| `from psi.experiment.status_item import StatusItem` | `from psi.core.api import StatusItem` | Ditto re-export via `psi.experiment.api`. |
| `from psi.experiment.metadata_item import MetadataItem` | `from psi.core.api import MetadataItem` | Ditto. |
| `from psi.controller.output_manifest import initialize_factory, load_items, get_parameters, generate_waveform` | `from psi.controller.token_context import ...` | Plain module now — no `enaml.imports()` block needed. `generate_waveform` is also re-exported by `psi.controller.api`. |
| `from psi.controller.output_manifest import prepare_output, start_output, ...` (any command handler) | `from psi.controller.output_commands import ...` | |
| `from psi.controller.manifest import get_hw_ao_choices, get_hw_ai_choices` | `from psi.controller.controller_commands import ...` | Also re-exported by `psi.controller.api`. |
| `from psi.experiment.manifest import _save_preferences` (or `save_layout`, `load_preferences`, etc.) | `from psi.experiment.experiment_commands import ...` | Better: invoke the workbench command (`psi.save_preferences`, etc.) instead of importing the handler. |
| `from psi.application import list_preferences` | *(still works)* | Implementation moved to `psi.experiment.util`; prefer importing from there in new code. |

When a name is available from `psi.core.api`, prefer that in manifests that
*contribute* actions/events/preferences/status items — it is the canonical
home and keeps your package compatible with the layering direction.

## Step 3 — API/behavior changes beyond renames

- **`CONTEXT_MAP` is gone.** The token-parameter map is now stored per
  output in `BaseOutput._block_context_map` and is populated by
  `psi.controller.token_context.load_items(output, block)`. If code peeked
  at `CONTEXT_MAP[output, block]`, use
  `output._block_context_map[block]` after calling `load_items`. Calling
  `get_parameters`/`initialize_factory` before `load_items` now raises a
  `KeyError` with an explanatory message (previously a bare `KeyError`).
- **`Block.initialize_factory(context)` does not exist** (removed upstream
  before this refactor). Use
  `token_context.initialize_factory(output, block, context)` after
  `token_context.load_items(output, block)`.
- **Python >= 3.10 is required** (`requires-python` was previously a stale
  `>=3.7` while the code already used 3.8+ syntax).
- **NIDAQ hardware-timed digital input** (`setup_hw_di`) now raises
  `NotImplementedError` with guidance. It had bit-rotted (undefined helper
  class + stale `setup_timing` signature) and could not have worked; if a
  rig config declares a hardware-timed DI channel on a NIDAQ engine it was
  already broken, but it now fails at configure time with a clear message.
- **`stop_experiment` wrapup messages now propagate.** End-of-experiment
  messages returned by `experiment_end` actions (other than "Saved data to
  disk", which is suppressed) will now actually appear in the result popup.
  Previously a double-invocation bug discarded them. If a downstream action
  returns a string from its `experiment_end` handler, expect it to be shown.
- **The `Logger` data sink** now locates the logfile via
  `get_config('LOG_FILENAME')`, which `psi.application.configure_logging`
  publishes. If downstream code configured file logging by hand (without
  `configure_logging`) and relied on setting
  `psi.application.exception_handler.logfile`, also call
  `psi.set_config('LOG_FILENAME', filename)`.
- **`psi.data.plugin.DataPlugin.find_source`** no longer swallows arbitrary
  exceptions from sinks; only `AttributeError`/`NotImplementedError` mean
  "not in this sink". A sink whose `get_source` raises something else will
  now propagate that error instead of reporting "Could not find source".
- **`merge_results(results, names=None)`** — the `names` default changed
  from a mutable `['ao_channel']` literal to `None` (same effective
  default). Only affects callers introspecting the signature.

## Step 4 — Rules for downstream *contributions* (manifests)

psiexperiment now enforces an internal layer order (`util → core → context
→ token → controller → data → experiment → application`) via import-linter
and `tools/check_enaml_layering.py`. Downstream packages are consumers at
the top of the stack, so they may import from any psi layer — no constraint
applies to them. However:

- Do not import from `psi.controller.output_manifest`,
  `psi.controller.manifest`, or `psi.experiment.manifest` for *functions*;
  those files now contain only `enamldef`s and may shed remaining
  module-level names without notice. Import functions from the
  `*_commands.py` / `token_context.py` modules or the `api` modules.
- Extension point IDs are unchanged (`psi.controller.actions`,
  `psi.data.sinks`, `psi.experiment.preferences`, etc.). Only Python import
  paths moved.

## Step 5 — Verify

1. `python -c "import <downstream_package>"` (or import its api module).
2. Launch one paradigm per package with the Null/soundcard engine if no
   hardware is attached, or run the package's test suite.
3. If the package has CI, add `pip install "psiexperiment @ <new rev>"` to
   its matrix before merging.

## Phase 4 changes (concurrency contract)

Commit: see "Introduce control-plane dispatcher". Full contract in
`docs/threading.md`.

- **Action failures raise `psi.core.exceptions.ActionError`** (a
  `PSIException`) instead of `RuntimeError`. Code catching `RuntimeError`
  around `invoke_actions`/`invoke_command` chains must catch `ActionError`
  (or `PSIException`). The original exception remains available as
  `__cause__`.
- **Actions now run on a dedicated control-dispatcher thread**, not the
  caller's thread. `invoke_actions(wait=True)` (the default) still blocks
  until completion, returns results, and propagates exceptions — call sites
  usually need no change. But actions that assumed they run on the GUI
  thread must marshal GUI work through `deferred_call` (which was already
  the convention).
- **New rule: never call `invoke_actions` while holding an engine lock or
  the controller lock.** Downstream command handlers that do
  `with output.engine.lock: ... invoke_actions(...)` must move the
  invocation outside the `with` block, or they can deadlock against the
  dispatcher.
- **Data-plane callbacks should pass `wait=False`** to `invoke_actions`
  (fire-and-forget; errors logged) so acquisition threads never block on
  control work.
- `ControllerPlugin.start_timer/stop_timer` still exist but are now backed
  by the dispatcher: timer callbacks execute on the dispatcher thread
  (previously a raw `threading.Timer` thread). `stop_all_timers()` is new.
  The `_timers` dict attribute is gone.
- Downstream tests that monkeypatch `threading.Timer` in the controller
  should target `psi.controller.dispatcher` instead.

## Phase 5 changes (no side effects at import time)

- **`import psi` no longer loads the configuration.** The config (including
  execution of the user's `config.py`) loads lazily on the first
  `get_config`/`set_config` call. Code that relied on `psi._config` being
  populated immediately after import must call `psi.get_config()` (or
  `psi.reload_config()`) first. `psi.DEFAULT_CONFIG` (a module-level dict)
  no longer exists; defaults are computed inside `load_config`.
- **`import psi.application` no longer installs `sys.excepthook`** and no
  longer flips the Windows console quick-edit mode. `launch_experiment` and
  the `psi` / `psi-config` CLI entry points do both automatically, so
  normal launches are unaffected. Custom launchers that bypass
  `launch_experiment` and want the graceful-shutdown hook must call
  `psi.application.install_exception_handler()` (and optionally
  `psi.application.setup_windows_console()`) themselves.
- `configure_logging` still installs the exception handler as before.

## Fail-fast validation (post-0.7.0)

Two validation passes now convert previously-silent misconfigurations into
hard errors with descriptive messages. **Paradigms with latent typos that
"worked" before (because the broken piece silently never fired) will now
refuse to start** — this is intentional; fix the typo the error names.

- **Experiment actions**: at experiment start, every `ExperimentAction`'s
  event expression is checked against the registered events and state
  flags. Unknown names raise `ActionError` with close-match suggestions.
  Actions bound to events generated dynamically at runtime must set
  `allow_unregistered = True` on the action.
- **Context expressions**: `apply_changes` (run when the user clicks Apply
  and at experiment initialization) now verifies that every context-item
  expression parses and references only known context items, symbols, or
  builtins. Errors raise `ValueError` naming the parameter, the unknown
  name, and suggestions. If a paradigm injects non-context-item names into
  the expression namespace at runtime via `ExpressionNamespace.set_value`,
  register those names as context items or symbols instead.

## Grouped epoch plots: incremental averaging (post-0.7.0)

Grouped epoch plots (`GroupedEpochAveragePlot`, `GroupedEpochFFTPlot`,
`GroupedEpochPhasePlot`, `StackedEpochAveragePlot`) now fold each epoch
into a per-group running mean as it arrives instead of re-averaging the
full epoch stack on every redraw. Redraw cost no longer grows with the
number of epochs acquired, and raw epochs are no longer retained by the
plot (memory is per-group, not per-epoch).

- **The `_y(epoch_stack)` override hook is gone.** Subclasses that
  customized it must override `_fold(epoch)` (per-epoch transform applied
  before averaging) and/or `_render_mean(mean)` (running mean -> plotted y
  values) instead. A stale `_y` override raises `TypeError` at source
  wiring rather than being silently ignored. Linear post-processing (e.g.,
  a referencing/diff matrix) belongs in `_render_mean`; nonlinear
  per-epoch math (e.g., dB-PSD) belongs in `_fold`.
- Parameters used inside `_fold` (fs, channel count, waveform averages)
  must not change once epochs have been folded; call `_reset_plots()` if
  they do.
- Plots that need the raw epochs for other purposes must retain them
  themselves (see `BiosemiEpochPlot.epochs` for the pattern).

## Plot thread ownership (post-0.7.0)

Plot redraws now always execute on the GUI thread. Data-plane callbacks
(and `data_range` observers) trigger redraws via the new
`BasePlot.request_update()`, which marshals to the GUI thread and coalesces
bursts of data into one redraw per event-loop pass.

- Custom plot classes with callbacks that receive acquired data must call
  `self.request_update()` from the callback instead of `self.update()`.
  Calling `update()` from an acquisition thread creates/mutates Qt objects
  off the GUI thread (previously undefined behavior that mostly worked).
- Observers wired as `data_range.observe('current_time', self.update)`
  should target `self.request_update` instead.
- Redraw coalescing also means high-rate sources no longer redraw once per
  chunk; expect lower GUI CPU with identical visuals.

## Controller IO manager and subclass contract (post-0.7.0)

The hardware-configuration half of `ControllerPlugin` (engine/channel/
output/input registries, wiring, engine lifecycle) now lives in
`psi.controller.io_manager.IOManager`, available as `controller.io`. The
plugin's **public method surface is unchanged** — `get_channel`,
`get_output`, `get_input`, `get_channels`, `get_ts`, `connect_output`,
`connect_input`, `configure/start/stop/reset_engines`, `finalize_io` all
still exist on the plugin and delegate — so most downstream code needs no
changes. Code that touched the private attributes must update:

| Old (on `controller`) | New |
|---|---|
| `controller._engines` | `controller.io.engines` |
| `controller._channels` | `controller.io.channels` |
| `controller._outputs` | `controller.io.outputs` |
| `controller._inputs` | `controller.io.inputs` |
| `controller._supporting` | `controller.io.supporting` |
| `controller._master_engine` | `controller.io.master_engine` |
| `controller.engines_running` | `controller.io.engines_running` |
| `controller._lock` | `controller.io._lock` |
| `psi.controller.plugin.find_engines` (etc.) | `psi.controller.io_manager.find_engines` |

Reassigning the master engine during setup (as the NIDAQ start-trigger
helpers do) is supported: set `controller.io.master_engine`.

The controller subclass contract is now explicit:

- `end_trial` is declared on the base class (it backs the
  `psi.controller.next_trial` command) and raises a descriptive
  `NotImplementedError` instead of the previous `AttributeError` from deep
  in the rpc plumbing. Controllers exposing next-trial UI must override it.
- `apply_changes` / `pause_experiment` / `resume_experiment` share a
  documented protocol: return True if handled immediately, falsy to defer
  (the corresponding `request_*` latches `_apply_requested` /
  `_pause_requested` / `_resume_requested` for the subclass to consume —
  act on, then clear — between trials). This is what existing controllers
  already did; it is now written down in the method docstrings.

## Configuration rework (2026-09)

**Breaking, with no compatibility shim.** A legacy setting name is not read,
and nothing warns you — deliberate, on the basis that every affected machine
is under one administrator's control. Audit a machine *before* upgrading it.

### What changed

1. **A setting has one spelling.** The name in code is the name in the
   configuration file and the name of the environment variable, package
   prefix included: `get_config('PSI_DATA_ROOT')`, `PSI_DATA_ROOT = "..."`,
   `set PSI_DATA_ROOT=...`. Previously `DATA_ROOT` and `PSI_DATA_ROOT` were
   one setting under two names.

2. **The environment now overrides the configuration file.** It used to be
   the reverse: `PSI_*` variables were applied to the *defaults* and then
   `config.py` was loaded on top, so on a machine that had a configuration
   file, setting `PSI_DATA_ROOT` did nothing at all, silently.

3. **`config.py` became `config.toml`.** The old file was executable Python;
   the new one is data, because the GUI writes settings too and machine-
   writing a source file is a bad trade. Needs Python 3.11 (`tomllib`) and
   adds a `tomlkit` dependency.

4. **`get_config_folder()` and `PSI_CONFIG` are gone.** `PSI_CONFIG_FILE`
   names the file outright and is the only bootstrap variable. No directory
   is inferred from the configuration file's location — every directory is
   its own named setting (`PSI_BASE_DIRECTORY`, `PSI_DATA_ROOT`, `CFTS_ROOT`,
   `CFTSCAL_ROOT`, …). Most derive from `PSI_BASE_DIRECTORY`, so a
   configuration file usually sets only that one.

5. **`set_config` is gone.** The four values it carried — `EXPERIMENT`,
   `LOG_FILENAME`, `ARGS`, `PROFILE` — are written by the launcher at runtime
   and were never settings. They moved to `psi.runtime` (`get_runtime` /
   `set_runtime`), which has no environment or file path at all. Reading one
   through `get_config` no longer works.

6. **Defaults live in code**, one `config_defaults.py` table per package,
   registered through `psi.config.register_defaults`. Every setting has a
   default, so an installation with no configuration file still runs.

### Renames

| Old | New |
| --- | --- |
| `LOG_ROOT`, `DATA_ROOT`, `PROCESSED_ROOT`, `PREFERENCES_ROOT`, `LAYOUT_ROOT`, `IO_ROOT`, `HOSTNAME`, `STANDARD_IO`, `PARADIGM_DESCRIPTIONS`, `WEBSOCKETS_URI` | the same with a `PSI_` prefix |
| `CAL_ROOT` | `CFTSCAL_ROOT` — ownership moved to cftscal, which already read that variable. The `CAL_ROOT` key `psi-config` used to emit was read by nothing. |
| `RAW_DATA_DIR`, `PROC_DATA_DIR` | `PSIDATA_RAW_DIR`, `PSIDATA_PROC_DIR` |
| every `CFTS_*` handoff variable except `CFTS_ROOT` | `CFTSCAL_*` |

`CFTS_ROOT` is unchanged: it is a cfts setting (where the launcher keeps its
saved presets), not a cftscal handoff variable.

psidata and cftsdata sit *below* psiexperiment in the dependency graph and
cannot import `psi`. They took the rename but still read `os.environ`
directly, so those two settings have no configuration-file spelling.

### Upgrading a machine

1. **Audit first**, before installing anything:

   ```bash
   python tools/audit_legacy_config.py
   ```

   Stand-alone: standard library only, no `psi` import, and it parses
   configuration files without executing them, so it runs on a machine that
   is not yet — or only half — upgraded. It exits non-zero when it finds
   anything, so it can gate a deployment script. `--format json` for scripted
   use.

2. **Upgrade psiexperiment and every dependent package together.** There is
   no half-migrated state that runs: these are separate repositories, so
   upgrading psiexperiment without the matching cftscal / cfts / noise-exp
   will fail at import.

3. **Convert the configuration:**

   ```bash
   psi-config migrate path/to/config.py
   ```

   Executes the old file once to capture computed values (`BASE_DIRECTORY /
   'data'` and the like), maps the names, and folds in cftscal's
   `workspace.json` and its per-plugin `cfts/calibration/*.json` files when
   present. `--dry-run` shows what it would write; `PSI_CONFIG_FILE` controls
   where the result goes.

4. **Verify:**

   ```bash
   psi-config show
   ```

   Every setting, its resolved value, and the layer that supplied it. Check
   the paths are what the rig actually uses, and that nothing you expected
   from the file is being shadowed by a stale environment variable.

5. **Delete the leftovers** once the rig runs: the old `config.py`,
   `cfts/workspace.json`, `cfts/calibration/*.json`, and any
   `noise-exp/default.json`. Nothing reads them, but leaving them invites
   confusion later about which file is live.

### Finding affected code

Restrict these to source files — a stale `__pycache__` or `__enamlcache__`
still contains the old strings and will match everything otherwise.

```bash
SRC='--include=*.py --include=*.enaml'

# Legacy setting names
grep -rnE $SRC "get_config\(['\"](LOG|DATA|PROCESSED|PREFERENCES|LAYOUT|IO|CAL)_ROOT" .

# Removed API
grep -rn $SRC "get_config_folder\|psi.set_config\|PSI_CONFIG\b" .
grep -rn $SRC "from psi import.*set_config" .

# Handoff variables. CFTS_ROOT is a cfts setting and CFTS_PATH is a module
# path constant; neither is one of these.
grep -rn $SRC "CFTS_[A-Z]" . | grep -v "CFTSCAL_\|CFTS_ROOT\|CFTS_PATH"
```

Note that `CalibrationSettings.get_config()` / `set_config()` in cftscal are
unrelated methods that serialize Atom members — they are not psi's
configuration API and did not change.

## Known-unchanged surfaces (no action needed)

- `psi.controller.api`, `psi.context.api`, `psi.data.api`,
  `psi.data.sinks.api`, `psi.token.api`, `psi.core.enaml.api` exports.
- `ExperimentManifest`, `ParadigmDescription`/`paradigm_manager`.
  (`psi.get_config` still exists but its setting names changed, and
  `psi.set_config` is gone — see the configuration rework above.)
- IO manifest format and engine classes (`NIDAQEngine`, TDT, Biosemi,
  soundcard), except the NIDAQ hardware-timed DI path noted above.
- All workbench command IDs and extension point IDs.
