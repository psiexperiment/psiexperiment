# Ideas: how psi picks the IO manifest when `--io` is not given

Status: **proposed, not scheduled** (discussion notes, 2026-10-01).

## Where things stand

`psi <paradigm>` without `--io` falls back to `get_default_io()`
(`src/psi/application/__init__.py`), which looks for this machine's
manifest in `PSI_IO_ROOT` by hostname. It is also what cftscal's
`CFTSCAL_IO = default` means: cftscal's `resolve_io` calls it, and so do
the unset default of `CFTSCAL_IO` and the workspace dialog's "This
computer's IO manifest". cftscal and the launchers built on it (cfts,
abts, noise-exp) always pass `--io` from `CFTSCAL_IO`, so for them the
fallback only matters through that `default` value.

## Problems with `get_default_io` / `list_io` today

- **The hostname match is too loose.** It tests `hostname in str(io)`
  against the *full path* of each file `list_io` globs, so:
  - when the hostname appears anywhere in `PSI_IO_ROOT`
    (`D:/rigs/sable/io`), the first `.enaml` the glob returns matches,
    whatever it is called;
  - a hostname that is a prefix of another file matches it (`rig1`
    matches `rig10.enaml`).

  Its own error message says it wants `{hostname}.enaml`, so an exact
  match was the intent.
- `method='hostname'` is its only accepted value -- a dead parameter.
- It globs the whole folder to find one file it could check for directly.
- `list_io`'s only remaining real use is the list of available manifests
  in the IO error message (`format_io_manifest_error`). Its one outside
  caller, psilbhb's launcher, is already broken against current
  psiexperiment (it imports `list_calibrations` from `psi.application`,
  which no longer exists; last touched 2025-03).

## Proposed fix (small)

1. `get_default_io()` checks exactly `<PSI_IO_ROOT>/<PSI_HOSTNAME>.enaml`
   (case-insensitively), drops `method`, and raises the same `ValueError`
   when it is missing.
2. `list_io` becomes a private helper of the error message and leaves the
   public API.
3. psilbhb is updated whenever it is revived.

Keep `get_default_io` itself: removing it would make `--io` mandatory for
every direct `psi` run, and cftscal would have to re-implement the
hostname rule.

## Optional further step: a `PSI_IO` setting

Give psi its own setting, `PSI_IO`, whose default is the hostname file,
and have `CFTSCAL_IO = default` mean "whatever psi would use". A rig could
then point plain `psi` at a manifest with `psi-config set PSI_IO ...`
instead of naming a file after the machine, and there would be one
psi-level answer to "which hardware is this machine" that cftscal defers
to.

Worth weighing against it: the hostname convention lets one
network-shared `PSI_IO_ROOT` serve several rigs with no per-machine
configuration, which a per-machine `config.toml` setting does not.
