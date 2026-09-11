'''
Scan a saved continuous-AI zarr array for exact duplicated sample runs.

Motivation
----------
psi/controller/engines/nidaq.py's hw_input_helper (the callback that reads
newly-acquired AI samples off the DAQmx task and pushes them into the
pipeline) used to have no locking around its access to the DAQmx task
handle. NI-DAQmx documents that it is possible -- rare, but real -- for the
driver to invoke this callback again for the same task before a slow
previous invocation (e.g. one stuck in zarr_store's _retry_write, waiting
out a Windows file-lock from antivirus/search-indexer) has returned. If that
happens, two threads can end up calling DAQmxGetReadCurrReadPos /
DAQmxReadAnalogF64 on the same task without synchronization.

One concrete way that can corrupt the saved data: if the two threads'
DAQmxReadAnalogF64 calls are not actually serialized by the driver at the
moment they execute, both can read starting from the same (stale) buffer
position, so the *same* physical samples get pulled twice and both get
appended to the on-disk array -- producing a short, exact, contiguous run of
duplicated samples, immediately or shortly after another copy of itself, in
the file. (A related but distinct failure mode -- two blocks landing in the
zarr array in the wrong order because the earlier-read block was the slow
one to reach .append() -- would show up as a discontinuity/reordering
instead of a duplication, and is not what this script looks for.)

This script scans a saved continuous AI array for that specific signature:
a contiguous run of samples (across all channels stored in the array, if
more than one) that reappears verbatim a short distance later. It does NOT
prove the reentrancy race is the cause of a match -- only that duplicated
content exists at that location -- but a match is strong, concrete evidence
in favor of the hypothesis, and the absence of any match across a dataset
known to exhibit the "timing suddenly shifts" symptom is evidence against
this specific mechanism (as opposed to, e.g., the reordering variant, or an
unrelated cause).

How it works
------------
The array is scanned in overlapping blocks (so no duplicate pair separated
by up to --search-radius samples is ever split across two reads). Within
each in-memory block, a rolling (sum, sum-of-squares) fingerprint of
--window samples is computed at *every* start position (all channels at
once, so a coincidental match requires *every* channel to agree --
extremely unlikely for real, independent noisy signals), via an O(length)
cumulative-sum trick rather than a naive O(length * window) approach. This
checks every possible alignment, not just a sparse grid -- a sparse grid
(e.g. only checking starts at 0, window, 2*window, ...) would silently miss
any duplicate pair whose separation isn't a multiple of the grid spacing,
which real acquisition-driven duplicates have no reason to respect.
Positions whose fingerprint collides are verified with a direct array
comparison (fingerprint collisions between genuinely different content are
possible in principle but caught here) and then grown outward in both
directions to find the full duplicated run.

Near-constant windows (e.g. silence, a railed/grounded channel) are
expected to "duplicate" trivially and are reported separately, clearly
marked, rather than mixed in with genuine matches.

Usage
-----
    python tools/scan_ai_duplicates.py PATH [PATH ...] [options]

PATH may be:
  * a single continuous-AI zarr array's directory (e.g. microphone.zarr)
  * a directory containing one or more *.zarr arrays (e.g. an experiment's
    output folder) -- each is discovered and scanned in turn

Options:
    --window N          Anchor window length in samples (default: 64).
                         Smaller catches shorter duplicated runs but is
                         slower and more prone to trivial constant-window
                         matches; the true run length is recovered by
                         extension regardless of this value.
    --search-radius N   Only pair up anchors within this many samples of
                         each other (default: 200000). Should comfortably
                         exceed one AI callback's worth of samples
                         (hw_ai_monitor_period * fs) for your acquisition.
    --chunk-size N       Samples read into memory per pass (default: 2000000).
    --min-std X          Windows whose values span less than this are
                         treated as "near-constant" and reported separately
                         (default: 0, i.e. only exactly-constant windows are
                         separated out; raise this for noisy-but-flat
                         channels).
    --channel N          Restrict to a single channel of a multi-channel
                         array (0-based). Default: use all channels.
    --self-test          Run a synthetic sanity check (a known-good array
                         with an injected duplicate, and a clean array with
                         none) and report whether detection works, then
                         exit. Use this once to build confidence in the
                         tool before pointing it at real data.

Example
-------
    python tools/scan_ai_duplicates.py D:\\data\\some_experiment --search-radius 500000
'''
import argparse
from pathlib import Path
import sys

import numpy as np
import zarr


def _is_zarr_array_dir(path):
    return (path / 'zarr.json').exists() or (path / '.zarray').exists()


def discover_arrays(path):
    '''
    Given a path, return a list of (label, Path) for every zarr array to
    scan: `path` itself if it is one, else every *.zarr entry directly
    inside it.
    '''
    path = Path(path)
    if _is_zarr_array_dir(path):
        return [(path.name, path)]
    found = []
    for child in sorted(path.glob('*.zarr')):
        if _is_zarr_array_dir(child):
            found.append((child.name, child))
    if not found:
        raise ValueError(
            f'{path} is neither a zarr array nor a directory containing '
            'any *.zarr arrays.')
    return found


def _windows_equal(block, p1, p2, window):
    return np.array_equal(block[:, p1:p1 + window], block[:, p2:p2 + window])


def _window_slice(block, p, window):
    return block[:, p:p + window]


def _first_mismatch(eq):
    '''Index of the first False in boolean array `eq`, or eq.size if none.'''
    bad = np.flatnonzero(~eq)
    return int(bad[0]) if bad.size else int(eq.size)


def _extend_match(block, p1, p2, window):
    '''
    Given two block-local starting positions p1 < p2 whose `window`-sample
    windows are known to be equal, grow the match outward in both
    directions as far as it continues to hold. Returns
    (start1, end1, start2, end2) block-local bounds of the two matching
    (non-overlapping) regions.

    Grows via a single vectorized comparison over the whole candidate
    extension range in each direction rather than a Python loop stepping
    one sample at a time -- the latter is correct but was measured to be
    the dominant cost (tens of seconds) when a match sits inside a long
    near-constant/silent stretch, where a single match can legitimately
    extend across tens of thousands of samples.
    '''
    length = block.shape[-1]

    # Left extension: p1 can't go below 0. (p2-p1 is invariant under a
    # simultaneous left shift, so it can never close to zero on its own.)
    max_left = p1
    if max_left > 0:
        seg1 = block[:, p1 - max_left:p1]
        seg2 = block[:, p2 - max_left:p2]
        # eq[k] compares position (p1-1-k) vs (p2-1-k) -- nearest-to-window
        # first -- so the first mismatch scanning from the window outward
        # is the first False in the *reversed* comparison.
        eq = np.all(seg1 == seg2, axis=0)[::-1]
        left_ext = _first_mismatch(eq)
    else:
        left_ext = 0
    start1, start2 = p1 - left_ext, p2 - left_ext

    # Right extension: end1 can't reach start2 (regions must stay
    # non-overlapping), end2 can't run past the end of the block.
    end1, end2 = p1 + window, p2 + window
    max_right = min(length - end2, start2 - end1)
    if max_right > 0:
        seg1 = block[:, end1:end1 + max_right]
        seg2 = block[:, end2:end2 + max_right]
        eq = np.all(seg1 == seg2, axis=0)
        right_ext = _first_mismatch(eq)
    else:
        right_ext = 0
    end1, end2 = end1 + right_ext, end2 + right_ext

    return start1, end1, start2, end2


def _is_near_constant(values, min_std):
    lo = float(np.min(values))
    hi = float(np.max(values))
    if min_std <= 0:
        return hi == lo
    return (hi - lo) <= min_std


def _rolling_fingerprint(block, window):
    '''
    Compute an O(length) rolling (sum, sum-of-squares) fingerprint per
    channel for *every* possible window start position (not just a sparse
    grid of anchors spaced by `window`).

    A sparse, non-overlapping anchor grid (checking only starts at
    0, window, 2*window, ...) sounds like a natural way to speed this up,
    but it is not just slower to use a coarse grid -- it is *wrong*: it can
    only ever detect a duplicate pair whose separation happens to be a
    multiple of the grid spacing (verified empirically while building this
    script -- a synthetic duplicate offset by a non-multiple of the anchor
    spacing was silently missed). Real acquisition-driven duplicates have
    no reason to land on such a special alignment, so a sparse grid would
    silently miss the vast majority of genuine matches. Checking every
    start position removes that blind spot; the O(length) cumulative-sum
    trick below keeps it cheap (no O(length * window) blowup).

    The rolling statistics are computed on the exact bit pattern of each
    float64 sample reinterpreted as uint64 (`.view(np.uint64)`), using
    integer sum (mod 2**64) and integer XOR -- both are exactly
    reversible/updatable over a sliding window via a running total with no
    loss of precision. An earlier version of this function used a
    *floating-point* rolling sum/sum-of-squares (accumulated via
    np.cumsum on float64 values); that is NOT exact -- np.cumsum
    accumulates rounding error as it runs, so subtracting two nearby
    cumulative sums to get a local total can differ by ~1e-11 between two
    truly identical windows located at very different absolute positions
    (confirmed empirically: a synthetic exact duplicate ~50000 samples into
    a 200000-sample test signal was silently missed by the float version).
    Integer bit-pattern arithmetic has no such issue: it is either exactly
    equal or it isn't.

    Returns an (n_positions, 2*channels) uint64 array (one fingerprint row
    per start position 0 .. length-window) suitable for np.unique to group
    by exact equality. Two positions can only be *reported* as duplicates
    if the raw sample values also compare exactly equal (see
    `_windows_equal`, called before any match is yielded), so a
    fingerprint collision between genuinely different content only costs
    a little extra verification work, never a wrong result; genuine
    duplicates always produce identical fingerprints, so this cannot
    produce a false negative.
    '''
    channels, length = block.shape
    n_positions = length - window + 1
    fp = np.empty((n_positions, 2 * channels), dtype=np.uint64)
    for c in range(channels):
        xu = block[c].astype(np.float64, copy=False).view(np.uint64)
        cs = np.concatenate(([np.uint64(0)], np.cumsum(xu, dtype=np.uint64)))
        cx = np.concatenate(([np.uint64(0)], np.bitwise_xor.accumulate(xu)))
        # cs/cx have length `length + 1`; cs[window:]/cx[window:] and
        # cs[:n_positions]/cx[:n_positions] are both exactly `n_positions`
        # long (n_positions = length - window + 1).
        fp[:, 2 * c] = cs[window:] - cs[:n_positions]
        fp[:, 2 * c + 1] = np.bitwise_xor(cx[window:], cx[:n_positions])
    return fp


def _grouped_positions(fingerprints):
    '''
    Group row indices of `fingerprints` by exact row equality. Returns an
    iterator of sorted int arrays of positions sharing a fingerprint, for
    every group with more than one member.

    Uses np.lexsort + an adjacent-row diff rather than
    `np.unique(fingerprints, axis=0, ...)` -- the latter is correct but,
    for large row counts, dramatically slower in practice (axis=0 uniquing
    does extra bookkeeping this doesn't need): ~9x slower measured on a
    5,000,000-row fingerprint array during development of this script.
    '''
    order = np.lexsort(fingerprints.T[::-1])
    sorted_fp = fingerprints[order]
    same_as_prev = np.all(sorted_fp[1:] == sorted_fp[:-1], axis=1)
    boundaries = np.flatnonzero(~same_as_prev) + 1
    for group in np.split(order, boundaries):
        if group.size > 1:
            yield np.sort(group)


#: Cap on how many adjacent pairs within one fingerprint group get fully
#: verified/extended/reported. A long constant or silent stretch produces
#: one huge group (every position within it "matches" every other), which
#: is expected and uninteresting -- report one representative match plus a
#: count rather than flooding the output.
_MAX_PAIRS_PER_GROUP = 200


def scan_array(array, window=64, search_radius=200_000, chunk_size=2_000_000,
               min_std=0.0, channel=None, progress=sys.stderr):
    '''
    Scan `array` (a zarr array, 1D time or 2D channel-by-time) for exact
    duplicated contiguous runs. Yields dicts describing each match, in
    ascending order of the earlier occurrence's start sample.
    '''
    n = array.shape[-1]
    overlap = search_radius + window
    seen = set()
    start = 0
    while start < n:
        end = min(n, start + chunk_size + overlap)
        if array.ndim == 1:
            block = np.asarray(array[start:end])[np.newaxis, :]
        elif channel is not None:
            block = np.asarray(array[channel, start:end])[np.newaxis, :]
        else:
            block = np.asarray(array[:, start:end])

        if block.shape[-1] >= window:
            fp = _rolling_fingerprint(block, window)
            for positions in _grouped_positions(fp):
                # positions are already sorted; only pair up those within
                # search_radius of each other, adjacent-first.
                n_checked = 0
                for a, b in zip(positions, positions[1:]):
                    if n_checked >= _MAX_PAIRS_PER_GROUP:
                        break
                    if (b - a) > search_radius:
                        continue
                    n_checked += 1
                    if not _windows_equal(block, a, b, window):
                        continue  # fingerprint collision, not a real match
                    s1, e1, s2, e2 = _extend_match(block, a, b, window)
                    g1, g2 = s1 + start, s2 + start
                    key = (g1, g2)
                    if key in seen:
                        continue
                    seen.add(key)

                    content = _window_slice(block, s1, e1 - s1)
                    near_const = _is_near_constant(content, min_std)

                    yield {
                        'first_start': g1,
                        'first_end': e1 + start,
                        'second_start': g2,
                        'second_end': e2 + start,
                        'run_length': e1 - s1,
                        'gap': g2 - (e1 + start),
                        'near_constant': near_const,
                    }

        if progress is not None:
            pct = 100 * min(end, n) / n
            print(f'\r  scanned {min(end, n):,} / {n:,} samples ({pct:5.1f}%)',
                  end='', file=progress)

        if end >= n:
            break
        start += chunk_size

    if progress is not None:
        print(file=progress)


def _format_match(m, fs):
    tag = ' [near-constant -- likely coincidental, not the race]' if m['near_constant'] else ''
    run_s = m['run_length'] / fs
    gap_s = m['gap'] / fs
    return (
        f"  samples [{m['first_start']:,}:{m['first_end']:,}) duplicated at "
        f"[{m['second_start']:,}:{m['second_end']:,})  "
        f"run={m['run_length']} samples ({run_s * 1e3:.2f} ms)  "
        f"gap={m['gap']} samples ({gap_s * 1e3:.2f} ms){tag}"
    )


def scan_path(path, label, window, search_radius, chunk_size, min_std, channel):
    array = zarr.open(store=str(path), mode='r')
    fs = array.attrs.get('fs', None)
    print(f'{label}: shape={array.shape} dtype={array.dtype} fs={fs}')
    if fs is None:
        print('  (no fs attribute found -- reporting durations as N/A)')
        fs = float('nan')

    matches = list(scan_array(
        array, window=window, search_radius=search_radius,
        chunk_size=chunk_size, min_std=min_std, channel=channel))

    real = [m for m in matches if not m['near_constant']]
    trivial = [m for m in matches if m['near_constant']]

    if real:
        print(f'  {len(real)} candidate duplicate run(s) found:')
        for m in real:
            print(_format_match(m, fs))
    else:
        print('  no duplicate runs found')

    if trivial:
        print(f'  ({len(trivial)} additional near-constant match(es) '
              'suppressed from the count above -- likely coincidental '
              'flat/silent segments, not the race; shown for transparency)')
        for m in trivial:
            print(_format_match(m, fs))

    return real


def _make_signal(rng, n_channels, n_samples, seed_offset=0):
    # Smooth-ish, non-repeating, multi-channel test signal -- band-limited
    # noise so no two windows coincidentally match by chance.
    t = np.arange(n_samples)
    sig = np.zeros((n_channels, n_samples))
    for c in range(n_channels):
        freqs = rng.uniform(1, 50, size=5)
        phases = rng.uniform(0, 2 * np.pi, size=5)
        for f, p in zip(freqs, phases):
            sig[c] += np.sin(2 * np.pi * f * t / 1000 + p)
        sig[c] += rng.normal(scale=0.01, size=n_samples)
    return sig


def self_test():
    import tempfile
    print('Running self-test...')
    rng = np.random.RandomState(0)
    n_channels, n_samples = 3, 200_000
    sig = _make_signal(rng, n_channels, n_samples)

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)

        # Clean array: no injected duplication.
        clean_path = d / 'clean.zarr'
        z = zarr.create((n_channels, 0), store=str(clean_path), dtype='float64',
                        chunks=(n_channels, 100_000))
        z.append(sig, axis=1)
        z.attrs['fs'] = 1000.0

        # Buggy array: re-append a 500-sample chunk a little further on,
        # simulating the reentrancy race duplicating one AI callback's data.
        buggy = sig.copy()
        dup_start, dup_len, shift = 50_000, 500, 3_000
        duplicated = np.concatenate([
            buggy[:, :dup_start + dup_len + shift],
            buggy[:, dup_start:dup_start + dup_len],
            buggy[:, dup_start + dup_len + shift:],
        ], axis=1)
        buggy_path = d / 'buggy.zarr'
        z2 = zarr.create((n_channels, 0), store=str(buggy_path), dtype='float64',
                         chunks=(n_channels, 100_000))
        z2.append(duplicated, axis=1)
        z2.attrs['fs'] = 1000.0

        print('\n[clean array -- expect no matches]')
        clean_hits = scan_path(clean_path, 'clean.zarr', window=64,
                               search_radius=200_000, chunk_size=2_000_000,
                               min_std=0.0, channel=None)

        print('\n[buggy array -- expect one match near sample '
              f'{dup_start + dup_len + shift}, run length {dup_len}]')
        buggy_hits = scan_path(buggy_path, 'buggy.zarr', window=64,
                               search_radius=200_000, chunk_size=2_000_000,
                               min_std=0.0, channel=None)

        ok = True
        if clean_hits:
            print('\nSELF-TEST FAILED: found spurious match(es) in the clean array')
            ok = False
        if len(buggy_hits) != 1:
            print(f'\nSELF-TEST FAILED: expected exactly 1 match in the buggy '
                 f'array, found {len(buggy_hits)}')
            ok = False
        elif buggy_hits[0]['run_length'] != dup_len:
            print('\nSELF-TEST FAILED: matched run length '
                 f"{buggy_hits[0]['run_length']} != injected {dup_len}")
            ok = False

        print('\nSELF-TEST PASSED' if ok else '\nSELF-TEST FAILED')
        return 0 if ok else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('paths', nargs='*', type=Path,
                        help='zarr array(s) or directories containing them')
    parser.add_argument('--window', type=int, default=64)
    parser.add_argument('--search-radius', type=int, default=200_000)
    parser.add_argument('--chunk-size', type=int, default=2_000_000)
    parser.add_argument('--min-std', type=float, default=0.0)
    parser.add_argument('--channel', type=int, default=None)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    if not args.paths:
        parser.error('at least one PATH is required (or pass --self-test)')

    any_hits = False
    for path in args.paths:
        for label, array_path in discover_arrays(path):
            hits = scan_path(array_path, label, args.window, args.search_radius,
                             args.chunk_size, args.min_std, args.channel)
            any_hits = any_hits or bool(hits)
            print()

    return 1 if any_hits else 0


if __name__ == '__main__':
    sys.exit(main())
