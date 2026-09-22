'''
Repair continuous zarr arrays shifted by the old zarr append-retry bug.

Before the fix in psi/data/sinks/zarr_store.enaml, a transient PermissionError
during a chunk write caused `zarr.Array.append` to be retried as a whole. The
resize from the failed attempt had already been committed, so the retry grew
the array a second time. That left one block of fill values in the file and
stored every later sample one block late.

Each occurrence can be located from experiment_log.txt. From the next append
onward, every block logs a NIDAQ_DATA_GAP line, and gap_samples jumps by the
length of the bad block, d. On the first such line, expected_s0 is the file
length L just after the bad append. That append wrote its data to [L - d, L),
so the fill block is [L - 2d, L - d). This script removes those ranges and
writes a repaired copy of the recording zip. The input is never modified.

Each range is verified before anything is written. Every sample in it must be
either the fill value or an exact copy of the matching sample in the retried
block that follows it. If any range fails, no output is written. Without
--output, the script only reports the ranges and checks.
'''
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import zipfile

import numpy as np
import zarr
from zarr.storage import ZipStore


GAP_RE = re.compile(r'NIDAQ_DATA_GAP store=\S+ input=(\S+) expected_s0=(\d+) '
                    r'got_s0=(\d+) gap_samples=(-?\d+)')
RETRY_RE = re.compile(r'Transient PermissionError writing zarr chunk; '
                      r'retrying \((\d+)/\d+\)')

# Samples after each join used to estimate the typical sample-to-sample step.
JUMP_CONTEXT = 1000


def find_fill_blocks(log_text):
    '''
    Return ({input name: [(start, stop), ...]}, number of retries) from the
    experiment log. Ranges are in the coordinates of the damaged file.
    '''
    retries = 0
    for m in RETRY_RE.finditer(log_text):
        if int(m.group(1)) != 1:
            # With k failed attempts the array grew k extra times, and the
            # fill block's position depends on k. That case isn't handled.
            raise ValueError('Log contains a chunk write that failed more '
                             'than once; this script only handles single '
                             'retries.')
        retries += 1

    last_gap = {}
    blocks = {}
    for m in GAP_RE.finditer(log_text):
        name = m.group(1)
        expected, got, gap = (int(g) for g in m.group(2, 3, 4))
        prior = last_gap.get(name, 0)
        if gap == prior:
            continue
        d = prior - gap
        if d <= 0 or expected - got != -gap:
            raise ValueError(f'NIDAQ_DATA_GAP for {name} does not match the '
                             f'append-retry signature: {m.group(0)}')
        blocks.setdefault(name, []).append((expected - 2 * d, expected - d))
        last_gap[name] = gap
    return blocks, retries


def check_range(src, start, stop):
    fill = src[..., start:stop]
    is_fill = fill == src.fill_value
    zero_fraction = float(np.mean(is_fill))
    # If part of the failed write reached disk (the block spanned two chunks
    # and only one failed), those samples are real, and they equal the
    # retry's copy of the same block, which directly follows the range.
    retry = src[..., stop:stop + (stop - start)]
    verified = retry.shape == fill.shape and bool(np.all(is_fill | (fill == retry)))
    before = src[..., start - 1]
    after = src[..., stop:stop + JUMP_CONTEXT]
    typical_step = np.median(np.abs(np.diff(after, axis=-1)), axis=-1)
    jump = np.abs(after[..., 0] - before)
    jump_ratio = float(np.max(jump / np.where(typical_step > 0, typical_step, np.nan)))
    return {
        'start': start,
        'stop': stop,
        'n_samples': stop - start,
        'zero_fraction': zero_fraction,
        'verified': verified,
        'join_jump_vs_typical_step': jump_ratio,
    }


def kept_segments(n, remove):
    segments = []
    pos = 0
    for start, stop in sorted(remove):
        segments.append((pos, start))
        pos = stop
    segments.append((pos, n))
    return segments


def copy_without(src, dst, remove, batch):
    '''
    Copy `src` into `dst` along the last axis, skipping the `remove` ranges.
    Writes are aligned to `batch` (a multiple of the chunk length) so each
    output chunk is written exactly once.
    '''
    pending = []
    n_pending = 0
    out_pos = 0

    def flush(n):
        nonlocal pending, n_pending, out_pos
        data = np.concatenate(pending, axis=-1)
        dst[..., out_pos:out_pos + n] = data[..., :n]
        pending = [data[..., n:]]
        n_pending -= n
        out_pos += n

    for start, stop in kept_segments(src.shape[-1], remove):
        for i in range(start, stop, batch):
            pending.append(src[..., i:min(i + batch, stop)])
            n_pending += pending[-1].shape[-1]
            if n_pending >= batch:
                flush(batch)
    if n_pending:
        flush(n_pending)
    assert out_pos == dst.shape[-1]


def zip_copy_entry(zin, zout, info):
    new_info = zipfile.ZipInfo(info.filename, info.date_time)
    new_info.compress_type = info.compress_type
    new_info.external_attr = info.external_attr
    if info.is_dir():
        zout.writestr(new_info, b'')
        return
    with zin.open(info) as fh_in, zout.open(new_info, 'w', force_zip64=True) as fh_out:
        shutil.copyfileobj(fh_in, fh_out, 16 * 2**20)


def zip_add_tree(zout, root, arc_root, compress_type):
    now = dt.datetime.now().timetuple()[:6]
    for dirpath, dirnames, filenames in os.walk(root):
        rel = Path(dirpath).relative_to(root).as_posix()
        arc_dir = arc_root if rel == '.' else f'{arc_root}{rel}/'
        zout.writestr(zipfile.ZipInfo(arc_dir, now), b'')
        for filename in filenames:
            info = zipfile.ZipInfo(arc_dir + filename, now)
            info.compress_type = compress_type
            with open(Path(dirpath) / filename, 'rb') as fh_in, \
                    zout.open(info, 'w', force_zip64=True) as fh_out:
                shutil.copyfileobj(fh_in, fh_out, 16 * 2**20)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('recording', type=Path,
                        help='Recording zip containing experiment_log.txt')
    parser.add_argument('-o', '--output', type=Path,
                        help='Where to write the repaired zip. If omitted, '
                        'only report what would be removed.')
    parser.add_argument('--batch-chunks', type=int, default=32,
                        help='Chunks to copy per write (default: %(default)s)')
    args = parser.parse_args(argv)

    if args.output is not None:
        if args.output.resolve() == args.recording.resolve():
            parser.error('--output must differ from the input recording')
        if args.output.exists():
            parser.error(f'{args.output} already exists')

    with zipfile.ZipFile(args.recording) as zin:
        log_text = zin.read('experiment_log.txt').decode('utf-8', 'replace')
        entry_names = {i.filename for i in zin.infolist()}
    blocks, retries = find_fill_blocks(log_text)

    if not blocks:
        print('No append-retry shifts found in the log; nothing to repair.')
        return
    n_shifts = sum(len(r) for r in blocks.values())
    if n_shifts != retries:
        # e.g. a retry on an epoch store, which has no continuity check, or
        # on the final block, so there is no later append to log a gap.
        print(f'WARNING: log has {retries} retries but only {n_shifts} '
              f'detected shifts. The others cannot be repaired from the log.')

    src_store = ZipStore(args.recording, mode='r')
    report = {
        'source': str(args.recording),
        'repaired_at': dt.datetime.now().isoformat(timespec='seconds'),
        'script': Path(__file__).name,
        'arrays': {},
    }
    sources = {}
    for name, remove in blocks.items():
        if f'{name}.zarr/zarr.json' not in entry_names:
            raise ValueError(f'Log mentions {name}, but {name}.zarr is not '
                             f'in the recording')
        src = zarr.open_array(src_store, path=f'{name}.zarr', mode='r')
        sources[name] = src
        checks = [check_range(src, start, stop) for start, stop in remove]
        n_removed = sum(stop - start for start, stop in remove)
        report['arrays'][name] = {
            'fs': src.attrs.get('fs'),
            'original_length': src.shape[-1],
            'repaired_length': src.shape[-1] - n_removed,
            'removed': checks,
        }
        print(f'{name}.zarr: {src.shape[-1]} -> {src.shape[-1] - n_removed} samples')
        for c in checks:
            status = 'ok' if c['verified'] else 'FAILED'
            print(f'  remove [{c["start"]}, {c["stop"]}) ({c["n_samples"]} samples): '
                  f'{status}, {c["zero_fraction"]:.1%} fill values, join jump = '
                  f'{c["join_jump_vs_typical_step"]:.1f}x typical step')

    failed = [(name, c['start'], c['stop'])
              for name, info in report['arrays'].items()
              for c in info['removed'] if not c['verified']]
    for name, start, stop in failed:
        print(f'ERROR: {name}.zarr [{start}, {stop}) is not all fill values '
              f'or copies of the retried block, so it may not be the '
              f'damaged range.')

    lengths = {r['repaired_length'] for r in report['arrays'].values()}
    if len(lengths) > 1:
        print('NOTE: repaired arrays have different lengths. That is expected '
              'only if they come from different engines or sample rates.')

    if args.output is None:
        print('Dry run only; pass --output to write a repaired copy.')
        return
    if failed:
        raise SystemExit('Refusing to write a repaired copy.')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as tmp, \
            zipfile.ZipFile(args.recording) as zin, \
            zipfile.ZipFile(args.output, 'x', allowZip64=True) as zout:
        tmp = Path(tmp)
        repaired_prefixes = tuple(f'{name}.zarr/' for name in blocks)
        compress_type = zipfile.ZIP_DEFLATED
        for info in zin.infolist():
            if info.filename.startswith(repaired_prefixes):
                if not info.is_dir():
                    compress_type = info.compress_type
                continue
            zip_copy_entry(zin, zout, info)

        for name, remove in blocks.items():
            src = sources[name]
            print(f'Writing repaired {name}.zarr')
            shape = src.shape[:-1] + (report['arrays'][name]['repaired_length'],)
            dst = zarr.create_array(
                store=str(tmp / f'{name}.zarr'), shape=shape, chunks=src.chunks,
                dtype=src.dtype, fill_value=src.fill_value,
                serializer=src.serializer, compressors=src.compressors,
                filters=src.filters, attributes=src.attrs.asdict())
            copy_without(src, dst, remove, src.chunks[-1] * args.batch_chunks)
            zip_add_tree(zout, tmp / f'{name}.zarr', f'{name}.zarr/', compress_type)

        info = zipfile.ZipInfo('repair_log.json', dt.datetime.now().timetuple()[:6])
        info.compress_type = zipfile.ZIP_DEFLATED
        zout.writestr(info, json.dumps(report, indent=2))
    src_store.close()
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
