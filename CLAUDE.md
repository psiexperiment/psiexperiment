# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**psiexperiment** ("psi") is a framework for running trial-based experiments (auditory physiology/psychoacoustics), built on **Enaml** (declarative Qt) with **Atom** for the object model. It's launched via the `psi` CLI, which reads an experiment paradigm (`.enaml`) plus an IO manifest (hardware configuration) and runs a Qt-based acquisition GUI. Full architecture docs live in `docs/source/` (`architecture.rst`, `io_manifest.rst`, `plugin_reference.rst`, etc.) — this file is a Claude-Code-specific quick-reference, not a replacement for those.

**cftscal** (sibling repo, `../cftscal`) is the primary consumer relevant to Claude Code work here: it's a calibration GUI that shells out to `psi <paradigm> <output_dir> --io <manifest_class>` as a subprocess for every calibration run. Bugs reported "from cftscal" (device hangs, channel mismatches, wrong sample rate, etc.) are very often actually here, in psiexperiment's engine/IO code — see cftscal's own `CLAUDE.md` for the cross-repo debugging note.

## Commands

```bash
pip install -e .        # from repo root
python -m pytest        # run the test suite (tests/)
```

## Threading Architecture — read before touching `psi/controller/`

All experiment-control work (`experiment_prepare`, `experiment_start`, engine `configure()`/`start()`/`stop()`, action dispatch) is serialized onto one background thread, the **control dispatcher** (`psi/controller/dispatcher.py`, `ControlDispatcher`). The GUI/main thread requests work via `ControllerPlugin.invoke_actions(..., wait=True)` → `ControlDispatcher.submit_sync()`, which blocks the calling thread until the dispatcher thread finishes.

**This blocking is normally fine, except for one real, hard-won gotcha**: opening a sound-card stream via `sounddevice`/PortAudio for an **ASIO** device must happen on a thread that's actively pumping Windows messages — not because of a simple "COM must be initialized" issue (we tried `pythoncom.CoInitialize()` on the dispatcher thread first; it did *not* fix it), but because PortAudio's `Pa_OpenStream()` internally triggers a WASAPI device-list refresh (`PaWasapi_UpdateDeviceList`) that does COM cross-apartment activation via `SendMessageW` — confirmed with `py-spy dump --native` against a real hang. Verified empirically (see `tests/test_playrec.py`, `tests/test_dispatcher.py`): the exact same `sd.InputStream(...)` call succeeds every time on the process's main thread and hangs every time on a plain background thread, regardless of `CoInitialize`/call ordering.

The fix, already implemented — **don't re-derive this from scratch if you hit an ASIO hang**:
- `psi/controller/engines/soundcard/playrec.py`: `_on_main_thread(fn)` marshals a zero-arg callable onto the GUI thread via `enaml.application.deferred_call` and blocks (locally, via a `threading.Event`) until it completes, re-raising any exception. `PlayRec.configure()`/`.start()`/`.stop()` all route their actual `sd.Stream`/`InputStream`/`OutputStream` open/start/stop calls through it.
- `psi/controller/dispatcher.py`: `ControlDispatcher.submit_sync()` gained an optional `pump` callback — when the *caller* is itself the GUI thread waiting on dispatched work (e.g. `_start_experiment` waiting on `experiment_prepare`, which needs to call back into the GUI thread via the mechanism above), the naive blocking wait would deadlock (GUI thread not pumping while the dispatcher thread's marshaled call waits for it). `ControllerPlugin.invoke_actions()` auto-supplies a pump (`_gui_pump`, `psi/controller/plugin.py`) that calls `QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)` when the caller is on the GUI thread — excluding user input so this doesn't let e.g. a Stop-button click get serviced mid-`experiment_prepare`.
- `ControlDispatcher._run()` also does `pythoncom.CoInitialize()`/`CoUninitialize()` around its whole lifetime (Windows-only, harmless elsewhere) — not sufficient by itself for the hang above, but still correct/recommended per PortAudio's own issue tracker (`CO_E_NOTINITIALIZED when calling Pa_OpenStream() from a different thread than Pa_Initialize()`).

If a *new* kind of hang shows up in this area, get a real stack trace before guessing further: `py-spy dump --pid <psi.exe PID> --native` (install via `pip install py-spy`) shows the actual native call each thread is blocked in — this is what actually cracked the case above, after several plausible-but-wrong theories (COM init, sample-rate mismatch, message-pump-only) each failed to fix it.

## Data Sinks — zarr version fragility

`psi/data/sinks/zarr_store.enaml`: `BinaryStore` is an alias for `ZarrStore` (`psi/data/sinks/api.py`) — if you're chasing a "BinaryStore" bug, it's actually `ZarrStore`. `zarr` is **unpinned** in `pyproject.toml`, so different installs can resolve meaningfully different zarr versions with different array-creation code paths. Multi-channel array creation used to pass a bare `None` inside the `chunks` tuple to mean "one chunk spanning the whole channel axis" — a zarr-v2-era convention that isn't honored consistently across zarr's newer array-creation internals, and surfaced as a confusing `'NoneType' object is not iterable` deep in `zarr/core/chunk_grids.py` on some installs. Fixed by passing the channel count explicitly instead of relying on `None`/auto-inference (see `ZarrStore._create_array`) — if you touch chunk-shape logic here again, avoid reintroducing implicit/auto-inferred chunk dimensions.

## Experiment Plugin — layout persistence format

`psi/experiment/experiment_commands.py`: `.layout` files (dock/window layout) are now saved as **YAML**, not pickle — `_save_layout` uses `psi/experiment/dock_layout_serializer.py` to convert the `enaml.layout.dock_layout` Atom tree (`ItemLayout`/`TabLayout`/`SplitLayout`/`DockBarLayout`/`AreaLayout`/`DockLayout`) to plain dict/list data before handing it to `yaml.dump`. `_load_layout` auto-detects old binary files by peeking pickle's magic byte (`0x80`) and falls back to `pickle.load` for them, so pre-existing `.layout` files keep loading with no forced migration. `tools/convert_layout.py` migrates old pickled files to the new format in place (`--recursive` to walk subfolders, `--dry-run` to preview; keeps a `.bak` of the original). Preferences (`.preferences` files) were already YAML — layout was the last pickle-based persistence format in the experiment plugin.

## Key Conventions

- **Enaml files (`.enaml`)**: use `enaml.imports()` context manager before importing from Python (`with enaml.imports(): from enaml.stdlib.fields import RegexField`). Plain `.py` submodules of `enaml` (e.g. `enaml.application`) don't need this.
- Windows-only functionality (COM, `pywin32`) must be platform-guarded (`if sys.platform == 'win32':`) — CI runs on both `ubuntu-latest` and `windows-latest` (`.github/workflows/test.yml`).
