# Threading contract

psiexperiment runs three kinds of threads. Every piece of code belongs to
exactly one of them, and the rules below define how they interact.

## The three planes

### 1. Data plane (multiple threads, high rate)

Hardware acquisition callbacks (NIDAQ/TDT/soundcard driver threads), the
input processing pipelines they feed (filtering, decimation, epoch
extraction), and sink writes. This is the CPU-intensive path and it is
deliberately multi-threaded so acquisition and signal processing never
block the GUI.

**Rules:**
- Data-plane code must not make control decisions or mutate plugin state.
- To raise a control event, call
  `controller.invoke_actions(name, ..., wait=False)` — never `wait=True`,
  which would block acquisition on control work.
- To update the GUI (plots, labels), use `enaml.application.deferred_call`.
- Engine locks (`engine.lock`) protect hardware buffer state and belong to
  this plane.

### 2. Control plane (one thread: the control dispatcher)

Experiment-level decisions: action matching and invocation
(`ControllerPlugin.invoke_actions`), the action context
(`_action_context`), experiment state transitions, and delayed events
(timers). All of this executes on a single dedicated thread owned by
`psi.controller.dispatcher.ControlDispatcher`, serialized in FIFO order.

`invoke_actions` may be called from any thread:

- `wait=True` (default): the caller blocks until the actions complete,
  receives their results, and sees their exceptions (raised as
  `psi.core.exceptions.ActionError` chained to the original). Calls *from*
  the dispatcher thread run inline, so actions may recursively trigger
  actions without deadlock.
- `wait=False`: fire-and-forget; failures are logged. Required from
  data-plane callbacks.
- `delayed=True`: scheduled by name; a later invocation of the same event
  cancels the pending one (`cancel_existing`). The callback runs on the
  dispatcher thread.

**Rules:**
- `_action_context` and the delayed-event registry are owned by the
  dispatcher. Never touch them from another thread.
- **Never invoke actions while holding a lock** (engine lock or the
  controller's `_lock`). Actions may acquire those locks themselves;
  invoking under a lock is a deadlock. Compute what you need under the
  lock, release, then invoke. (`configure/start/stop_engines`,
  `output_pause`/`output_resume`, `start_output`, `clear_output`, and the
  synchronized-output commands all follow this pattern.)
- Actions must not *block* on the GUI thread (fire-and-forget
  `deferred_call` is fine; waiting for the GUI is not), since the GUI may
  itself be blocked waiting for the dispatcher.

### 3. GUI thread (Qt event loop)

Widget updates, dock layout, toolbar state. Reached exclusively via
`deferred_call`/`timed_call`. GUI event handlers (button clicks) invoke
workbench commands, which typically call `invoke_actions(wait=True)` —
the GUI briefly blocks on the dispatcher, which is acceptable because
control work is short. Heavy work (waveform generation during
`experiment_prepare`) also runs here today via `start_experiment`'s
`deferred_call`; that predates this contract and is unchanged.

## Event logging

`_invoke_actions` marshals event logging to the GUI thread via
`deferred_call` (event loggers may touch GUI-adjacent state). This is
fire-and-forget and does not violate the rules above.

## Request flags

`_apply_requested` / `_pause_requested` / `_resume_requested` are set via
`deferred_call` (GUI-owned) and read by controller subclasses between
trials. This is a low-rate, latch-style handshake and is unchanged.

## Adding new code: a checklist

1. Does it run in response to acquired data? It's data plane: no state
   mutations, `wait=False` for events, `deferred_call` for GUI.
2. Is it an action/command making experiment decisions? It runs on the
   dispatcher (via `invoke_actions`); never hold a lock across an
   `invoke_actions` call.
3. Is it a widget update? `deferred_call` only.
