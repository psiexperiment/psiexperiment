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

## Known-unchanged surfaces (no action needed)

- `psi.controller.api`, `psi.context.api`, `psi.data.api`,
  `psi.data.sinks.api`, `psi.token.api`, `psi.core.enaml.api` exports.
- `ExperimentManifest`, `ParadigmDescription`/`paradigm_manager`,
  `psi.get_config`/`set_config`.
- IO manifest format and engine classes (`NIDAQEngine`, TDT, Biosemi,
  soundcard), except the NIDAQ hardware-timed DI path noted above.
- All workbench command IDs and extension point IDs.
