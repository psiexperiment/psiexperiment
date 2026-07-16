# Ideas: clone contract and restartable experiments

Status: **proposed, not scheduled** (discussion notes, 2026-07-16).

## The problem, in one paragraph

psiexperiment's declarative objects (Engine, Channel, Input, Output) are
also the runtime instances — the declaration is consumed at instantiation
and the same object graph is then progressively mutated through wiring
(`finalize_io`), configuration (hardware tasks, primed coroutine
pipelines), and the run itself (consumed iterators, drained queues, opened
files, latched state flags). Most of these mutations have no inverse, so
the lifecycle is one-way. This shows up at two scales:

- **Calibration must clone engines** (`Engine.clone` +
  `psi.util.copy_declarative`) because configuring the same channel
  instances twice — once for calibration, once for the experiment — would
  collide. The clone is a hand-maintained approximation of "re-instantiate
  from the declaration."
- **An experiment cannot be restarted** without closing the program. The
  cfts launcher works around this by running every experiment as a
  subprocess; the OS process is currently the only instance boundary
  (and the only "destructor").

Both are the same disease: there is no first-class notion of *a run* as an
instance created from (and disposable independently of) the declarations.

Rejected alternatives, for the record:

- **Enaml-style declarative/proxy split** (Engine + EngineProxy): wrong
  fix. That split exists to substitute widget backends (Qt/wx); engines
  have no second backend — the subclass *is* the backend. It would add
  indirection everywhere and buy nothing.
- **Rewind-in-place** (add `reset()` to everything, wire a Restart
  button): whack-a-mole where every miss is silent state leakage between
  runs of a data-acquisition system (run 1's epochs in run 2's average, a
  stale queue entry played to hardware). Every future plugin author
  inherits an invisible obligation. Do not do this.

## Item 1: configuration/runtime contract + class-owned cloning

One focused session. Fixes the calibration-clone fragility on its own
merits and is the enabling investment for item 2.

**The contract:** `d_`-tagged members are *configuration* (the
declaration; safe to copy into a new instance). Anything mutated after
wiring/configure — pipeline attachments, task handles, buffers, backrefs —
is *runtime state* and must never survive a clone. Adding a member to a
Channel/Engine subclass should force the author to decide, visibly, which
side it is on.

Known weaknesses of the current mechanism this must fix:

- `copy_declarative` copies **current values**, not declared values — a
  clone taken after `configure_experiment` has run inherits mutated state
  (e.g., `channel.expected_range`, runtime-attached calibrations).
  Decide and document which semantics clone-for-calibration actually
  wants (it may legitimately want the current calibration but not the
  pipeline).
- Enaml `<<` bindings collapse to static snapshots in the clone,
  silently.
- The knowledge of which graph edges are structural vs. runtime lives at
  the call site (`exclude=['inputs', 'outputs']` in `Engine.clone`)
  instead of in the classes.

Tasks:

- [ ] Document the contract on `PSIContribution` (or the engine/channel
      bases): d_ = declaration; post-configure mutations = runtime state.
- [ ] Move cloning into the classes: `Channel.clone()` knows not to copy
      `inputs`/`outputs`/`engine`; `Engine.clone()` composes channel
      clones. Kill the ad-hoc exclude list at the call site.
- [ ] Per-engine clone round-trip tests: clone with a channel subset →
      configure → mutate the clone → assert the source is untouched (and
      vice versa). Base-class version runs on NullEngine; hardware
      variants marked for bench runs.
- [ ] Remove the debug `print(f'removing {channel}')` in
      `Engine.clone` (psi/controller/engine.py:266).
- [ ] Stretch: consider whether calibration needs a clone at all — a
      scoped-configuration path (configure the *same* engine with only the
      calibration channels, run, reset, reconfigure for the experiment)
      would eliminate the second instance. More invasive to the engine
      state machine; treat as follow-on if the clone contract still feels
      brittle.

## Item 2: restartable experiments via per-run instance graphs

Phase-6 sized; touches every plugin. Only worth starting after item 1.

**The shape:** the workbench and manifests stay loaded as the template;
*Start* builds the run. Restart = discard the run's instance graph and
build a fresh one — never rewind. Most per-run state is already built
late (pipelines at configure, context iterators at initialize); the work
is moving the remaining construction out of plugin `start()`/manifest
load into an explicit "begin run" step, plugin by plugin.

Inventory of one-way mutations that define the work (each needs to move
into per-run construction or be made disposable):

| Where | What accumulates/freezes |
|---|---|
| `finalize_io` | inputs/outputs attached to channels (declaration tree → run graph) |
| Engine `configure` | hardware tasks (`NIDAQEngine._tasks`), done-callbacks registered |
| Input pipelines | primed coroutine chains — consumed generators can only be rebuilt, not rewound |
| Outputs | token queues drained; per-output block context maps |
| Context plugin | selector iterators consumed (rebuilt at `initialize` already — verify) |
| Data sinks | files opened under the experiment path (need per-run naming or close/reopen) |
| Controller | `experiment_state` Enum has no edge out of `'stopped'`; `_action_context` flags latched; `register_action` appends to `_registered_actions` (a second start would duplicate actions) |
| Plots | buffers/accumulators hold old data (reset machinery exists post-plots-refactor: `_reset_plots`, `EpochGroupAccumulator.reset`) |
| Engines (misc) | `Engine.reset` not uniformly implemented across engine subclasses |

Tasks (rough order):

- [ ] Define the run boundary: an explicit `begin_run`/`end_run` step (or
      equivalent) with a documented per-plugin obligation: "all per-run
      state is created here and disposable at end_run."
- [ ] Re-instantiate the IO graph per run using the item-1 clone
      contract (the run wires and configures *clones*; the declared
      objects are never mutated).
- [ ] Sinks: per-run output directories / file lifecycle.
- [ ] Controller: add the `stopped → initialized` transition; rebuild
      `_action_context`; clear or re-register `_registered_actions` per
      run; make engine done-callback registration per-run.
- [ ] Plots/data: wire the existing reset machinery into the run
      boundary.
- [ ] Regression tests: two full runs back-to-back in one process on the
      NullEngine paradigm (workbench tests), asserting no state leakage
      (fresh files, fresh averages, correct action counts).
- [ ] GUI: Start button enabled again after stop.

**Design note kept from the discussion:** even with in-process restart
working, subprocess-per-run retains real advantages for long rig sessions
(crash isolation, leak hygiene, guaranteed-clean state). Consider keeping
it the default and reserving in-process restart for quick repeat runs —
after this work, that's a choice rather than an architectural constraint.

**Payoff:** after item 2, calibration can simply build its own small run
against the same declarations — no cloning workarounds at all. Same
disease, one cure.
