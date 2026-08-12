'''
Convert legacy pickled .layout files to the new YAML format.

psi/experiment/experiment_commands.py switched workspace layout persistence
from pickle to YAML (see psi/experiment/dock_layout_serializer.py). Old
files still load transparently (experiment_commands._load_layout falls back
to pickle when it detects one), so running this script is optional -- it
just gets existing files onto the new, diffable, non-binary format.

Usage:
    python tools/convert_layout.py PATH [PATH ...] [--recursive] [--suffix .bak] [--dry-run]

PATH may be a .layout file or a directory (searched for *.layout files;
add --recursive to also search subfolders). Each converted file is
overwritten in place; the original pickled bytes are preserved alongside
it with --suffix appended (default ".bak") unless --dry-run is given.
'''
import argparse
import pickle
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from psi.experiment.dock_layout_serializer import workspace_layout_to_dict  # noqa: E402

# Pickle's default protocol (>= 2) always starts a stream with this opcode
# byte -- see the matching check in experiment_commands.py.
_PICKLE_MAGIC = b'\x80'


def is_legacy_pickle(path):
    with open(path, 'rb') as fh:
        return fh.read(len(_PICKLE_MAGIC)) == _PICKLE_MAGIC


def convert_file(path, backup_suffix, dry_run):
    if not is_legacy_pickle(path):
        print(f'skip (already converted): {path}')
        return False

    with open(path, 'rb') as fh:
        layout = pickle.load(fh)
    text = yaml.dump(workspace_layout_to_dict(layout), default_flow_style=False)

    if dry_run:
        print(f'would convert: {path}')
        return True

    backup_path = path.with_name(path.name + backup_suffix)
    path.rename(backup_path)
    path.write_text(text)
    print(f'converted: {path} (original backed up to {backup_path})')
    return True


def iter_layout_files(paths, recursive):
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            glob = path.rglob if recursive else path.glob
            yield from sorted(glob('*.layout'))
        else:
            yield path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('paths', nargs='+',
                         help='.layout files or directories to convert')
    parser.add_argument('--recursive', action='store_true',
                         help='also search subfolders of any directory '
                              'given in PATH')
    parser.add_argument('--suffix', default='.bak',
                         help='suffix appended to back up the original '
                              'pickled file (default: .bak)')
    parser.add_argument('--dry-run', action='store_true',
                         help='report what would be converted without '
                              'writing any files')
    args = parser.parse_args(argv)

    converted = 0
    for path in iter_layout_files(args.paths, args.recursive):
        if not path.is_file():
            print(f'skip (not a file): {path}')
            continue
        if convert_file(path, args.suffix, args.dry_run):
            converted += 1

    verb = 'Would convert' if args.dry_run else 'Converted'
    print(f'{verb} {converted} file(s).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
