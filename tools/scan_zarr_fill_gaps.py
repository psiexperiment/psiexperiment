'''
Scan saved recordings for the fill-value block left by the zarr append-retry
bug, without relying on the experiment log.

Before the fix in psi/data/sinks/zarr_store.enaml, a transient PermissionError
during a chunk write retried `zarr.Array.append` as a whole. The resize from
the failed attempt was already committed, so the retry grew the array a second
time. The block that failed was never written, so it stays at the array's fill
value, and every sample after it is stored one block late.

That fill block is a signature the data carries on its own: a run of samples
that are exactly the fill value on every channel. Acquired analog data has
noise on it, so a run of exact zeros hundreds of samples long does not happen
by chance -- the DAQ would have to be reading precisely 0.0 for milliseconds.
Recordings made before the NIDAQ_DATA_GAP logging was added can therefore be
checked directly, and tools/repair_zarr_append_retry.py can repair the ones
whose ranges this finds.

Two further signs are reported when present:

- If the failed write reached disk in part (its block spanned two chunks and
  only one failed), the samples just before the run are real and appear again
  at the start of the retried block. The run then understates the damage by
  that many samples, which is reported as a partial prefix.
- Continuous arrays recorded by the same engine should end up the same length.
  A mismatch is a cheap hint that something was inserted, and is reported from
  the array metadata alone (--quick skips the sample scan entirely).

Exits with status 1 if any recording looks damaged, so it can be used over a
whole data directory in a shell loop or CI-style check.
'''
import argparse
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import zarr
from zarr.storage import LocalStore, ZipStore


# Runs shorter than this are reported only with --min-samples. At 100 kHz a
# real acquisition block is ~12500 samples, and 500 samples is 5 ms of the
# signal sitting at exactly 0.0 on every channel.
DEFAULT_MIN_SAMPLES = 500

# How far back to look for a partially written block ahead of a fill run.
MAX_PARTIAL_PREFIX = 2 ** 16

# Quantized channels repeat sample values often (~1e-4 of the time for the
# microphone monitor), so over a block-sized search window a one- or
# two-sample "match" is expected by chance. Only longer runs mean anything.
DEFAULT_MIN_PREFIX = 8


def open_recording(path):
    '''
    Return (store, [array name, ...]) for a recording zip or directory.
    '''
    if path.is_dir():
        names = sorted(p.stem for p in path.glob('*.zarr')
                       if (p / 'zarr.json').exists() or (p / '.zarray').exists())
        return LocalStore(path), names
    with zipfile.ZipFile(path) as zf:
        names = sorted({n.split('.zarr/')[0] for n in zf.namelist()
                        if n.endswith(('.zarr/zarr.json', '.zarr/.zarray'))})
    return ZipStore(path, mode='r'), names


def iter_fill_runs(array, min_samples, block):
    '''
    Yield (start, stop) for each run of at least `min_samples` consecutive
    samples that equal the fill value on every channel.
    '''
    fill = 0.0 if array.fill_value is None else array.fill_value
    n = array.shape[-1]
    run_start = None
    for i in range(0, n, block):
        data = array[..., i:min(i + block, n)]
        is_fill = data == fill
        while is_fill.ndim > 1:
            is_fill = is_fill.all(axis=0)
        # Bracket the block with False so edges show up as transitions, then
        # carry an unfinished run across block boundaries via run_start.
        edges = np.diff(np.concatenate(([False], is_fill, [False])).astype(np.int8))
        starts = np.flatnonzero(edges == 1) + i
        stops = np.flatnonzero(edges == -1) + i
        if run_start is not None:
            starts = np.concatenate(([run_start], starts))
            run_start = None
        if len(stops) < len(starts):
            run_start = starts[-1]
            starts = starts[:-1]
        for start, stop in zip(starts, stops):
            if stop - start >= min_samples:
                yield int(start), int(stop)
    if run_start is not None and n - run_start >= min_samples:
        yield int(run_start), n


def find_partial_prefix(array, start, stop, min_prefix=DEFAULT_MIN_PREFIX,
                        max_prefix=MAX_PARTIAL_PREFIX):
    '''
    Return how many real samples precede a fill run as part of the same
    damaged block, i.e. how many samples just before `start` reappear at
    `stop` as the retried block's copy of them.
    '''
    n = array.shape[-1]
    limit = min(max_prefix, start, n - stop)
    if limit <= 0:
        return 0
    tail = array[..., stop:stop + limit]
    last = array[..., start - 1]
    match = tail == np.asarray(last)[..., np.newaxis]
    while match.ndim > 1:
        match = match.all(axis=0)
    for i in np.flatnonzero(match)[::-1]:
        k = int(i) + 1
        if k < min_prefix:
            break
        if np.array_equal(array[..., start - k:start], array[..., stop:stop + k]):
            return k
    return 0


def scan_recording(path, min_samples, block, quick, min_prefix=DEFAULT_MIN_PREFIX):
    store, names = open_recording(path)
    report = {'recording': str(path), 'arrays': {}, 'suspect': False}
    try:
        for name in names:
            array = zarr.open_array(store, path=f'{name}.zarr', mode='r')
            fs = array.attrs.get('fs')
            info = {
                'fs': fs,
                'length': array.shape[-1],
                'n_channels': int(np.prod(array.shape[:-1])) if array.ndim > 1 else 1,
                'fill_runs': [],
            }
            report['arrays'][name] = info
            if quick:
                continue
            for start, stop in iter_fill_runs(array, min_samples, block):
                prefix = find_partial_prefix(array, start, stop, min_prefix)
                info['fill_runs'].append({
                    'start': start,
                    'stop': stop,
                    'n_samples': stop - start,
                    'partial_prefix': prefix,
                    'damaged_start': start - prefix,
                    'seconds': None if not fs else (start - prefix) / fs,
                    'duration': None if not fs else (stop - start + prefix) / fs,
                })
                report['suspect'] = True
    finally:
        store.close()

    # Arrays sharing a sample rate come off the same clock, so they should be
    # the same length.
    by_fs = {}
    for name, info in report['arrays'].items():
        by_fs.setdefault(info['fs'], set()).add(info['length'])
    report['length_mismatch'] = sorted(fs for fs, lengths in by_fs.items()
                                       if fs and len(lengths) > 1)
    if report['length_mismatch']:
        report['suspect'] = True
    return report


def print_report(report, verbose):
    path = report['recording']
    if not report['suspect'] and not verbose:
        return
    print(path)
    for name, info in report['arrays'].items():
        fs = info['fs']
        duration = f'{info["length"] / fs:.1f} s' if fs else 'unknown duration'
        if info['fill_runs'] or verbose:
            print(f'  {name}.zarr: {info["length"]} samples ({duration})')
        for run in info['fill_runs']:
            where = '' if run['seconds'] is None else f' at t={run["seconds"]:.3f} s'
            extra = '' if not run['partial_prefix'] else \
                f', plus {run["partial_prefix"]} partly written samples before it'
            print(f'    fill run [{run["damaged_start"]}, {run["stop"]}): '
                  f'{run["n_samples"]} fill samples{where}{extra}')
    for fs in report['length_mismatch']:
        lengths = {name: info['length'] for name, info in report['arrays'].items()
                   if info['fs'] == fs}
        print(f'  arrays at {fs} Hz have different lengths: {lengths}')


def iter_recordings(paths, recursive):
    for path in paths:
        if path.is_file() or (path / 'experiment_log.txt').exists() \
                or any(path.glob('*.zarr')):
            yield path
        elif recursive:
            yield from sorted(p for p in path.rglob('*.zip'))
        else:
            yield from sorted(p for p in path.glob('*.zip'))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('paths', nargs='+', type=Path,
                        help='Recording zips or directories, or directories '
                        'of recordings')
    parser.add_argument('--recursive', action='store_true',
                        help='Search directories for recordings recursively')
    parser.add_argument('--min-samples', type=int, default=DEFAULT_MIN_SAMPLES,
                        help='Shortest fill run to report (default: %(default)s)')
    parser.add_argument('--min-prefix', type=int, default=DEFAULT_MIN_PREFIX,
                        help='Shortest partially written prefix to believe '
                        '(default: %(default)s)')
    parser.add_argument('--block', type=int, default=4_000_000,
                        help='Samples to read at a time (default: %(default)s)')
    parser.add_argument('--quick', action='store_true',
                        help='Only compare array lengths; do not read samples')
    parser.add_argument('--verbose', action='store_true',
                        help='Report every recording, not just suspect ones')
    parser.add_argument('--json', type=Path,
                        help='Also write the full report to this file')
    args = parser.parse_args(argv)

    reports = []
    for path in iter_recordings(args.paths, args.recursive):
        try:
            report = scan_recording(path, args.min_samples, args.block,
                                    args.quick, args.min_prefix)
        except Exception as e:
            print(f'{path}\n  ERROR: {e}')
            reports.append({'recording': str(path), 'error': str(e),
                            'suspect': False})
            continue
        reports.append(report)
        print_report(report, args.verbose)
        sys.stdout.flush()

    if args.json:
        args.json.write_text(json.dumps(reports, indent=2))
    suspect = [r for r in reports if r['suspect']]
    print(f'\n{len(suspect)} of {len(reports)} recordings look damaged.')
    if suspect:
        print('Repair them with tools/repair_zarr_append_retry.py (it needs '
              'NIDAQ_DATA_GAP lines in the recording log).')
    return 1 if suspect else 0


if __name__ == '__main__':
    sys.exit(main())
