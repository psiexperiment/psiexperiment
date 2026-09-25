#!/usr/bin/env python
'''
Report legacy psi/cfts configuration names found on this machine.

The configuration system was reworked so that a setting has exactly one
spelling: the config key *is* the environment variable name, package
prefix included, and the environment overrides `config.py` rather than the
other way round. The core packages carry no backwards compatibility --
a legacy name is simply not read any more, silently.

This script finds those names so they can be fixed before the upgrade. It
is deliberately stand-alone: stdlib only, no `psi` import, no dependency
on the reworked packages. Copy it onto a rig and run it with any Python
3.8+, before or after upgrading.

    python audit_legacy_config.py
    python audit_legacy_config.py --config path/to/config.py
    python audit_legacy_config.py --format json

Exit status is 1 if anything needs attention, 0 if the machine is clean,
so it can gate a deployment script.

`config.py` is read with `ast`, never executed: it may import modules that
are not installed yet, and executing a config file to audit it is a bad
habit to build into a deployment tool.
'''
import argparse
import ast
import json
import os
import socket
import sys
from pathlib import Path


#: Settings that were renamed. Old name -> new name. These are both config
#: keys (in `config.py`) and environment variables, since the rework made
#: those the same namespace.
RENAMED = {
    # psiexperiment: every setting gains the PSI_ prefix it already used
    # for its environment twin.
    'LOG_ROOT': 'PSI_LOG_ROOT',
    'DATA_ROOT': 'PSI_DATA_ROOT',
    'PROCESSED_ROOT': 'PSI_PROCESSED_ROOT',
    'PREFERENCES_ROOT': 'PSI_PREFERENCES_ROOT',
    'LAYOUT_ROOT': 'PSI_LAYOUT_ROOT',
    'IO_ROOT': 'PSI_IO_ROOT',
    'HOSTNAME': 'PSI_HOSTNAME',
    'STANDARD_IO': 'PSI_STANDARD_IO',
    'PARADIGM_DESCRIPTIONS': 'PSI_PARADIGM_DESCRIPTIONS',
    'WEBSOCKETS_URI': 'PSI_WEBSOCKETS_URI',

    # Ownership move: where calibration files are stored belongs to
    # cftscal, which already read CFTSCAL_ROOT from the environment. The
    # CAL_ROOT key that `psi-config` used to emit was read by nothing.
    'CAL_ROOT': 'CFTSCAL_ROOT',

    # psidata / cftsdata read these straight from the environment with no
    # prefix at all -- the highest collision risk in the tree.
    'RAW_DATA_DIR': 'PSIDATA_RAW_DIR',
    'PROC_DATA_DIR': 'PSIDATA_PROC_DIR',
}


#: Settings that are gone entirely, as opposed to renamed. Name ->
#: explanation. `CFTS_ROOT` is NOT listed here -- it remains valid, as
#: the folder where the cfts launcher keeps its saved experiment and
#: hardware presets.
REMOVED = {
    # There is no "config folder" any more. Every directory is its own
    # named setting, and the config file is named outright.
    'PSI_CONFIG': 'the config folder variable is gone; name the file '
                  'directly with PSI_CONFIG_FILE, and set each directory '
                  'through its own setting (PSI_BASE_DIRECTORY, '
                  'PSI_DATA_ROOT, CFTS_ROOT, CFTSCAL_ROOT, ...)',
}


#: Keys that are written at runtime by the application and must never
#: appear in a config file or the environment. Name -> what sets it.
RUNTIME_ONLY = {
    'EXPERIMENT': 'set by the psi launcher from the command line',
    'LOG_FILENAME': 'set by the psi launcher when logging is configured',
    'ARGS': 'set by the psi launcher; holds parsed arguments',
    'PROFILE': 'set by the psi launcher from --profile',
}


#: Handoff variables: written into the environment of the `psi`
#: subprocess by cftscal (and the launchers built on it) and read by the
#: paradigm manifests. Every one of these moves from the CFTS_ prefix to
#: CFTSCAL_, because cftscal owns the format and exports the manifests
#: that read it. `CFTS_ROOT` is a genuine cfts setting and is excluded.
HANDOFF_PREFIX_OLD = 'CFTS_'
HANDOFF_PREFIX_NEW = 'CFTSCAL_'
HANDOFF_KEEP = {'CFTS_ROOT'}


#: Names that belong to third-party libraries and must be left exactly as
#: they are.
THIRD_PARTY = {
    'SD_ENABLE_ASIO',       # sounddevice / PortAudio
    'SPHINX_APIDOC_OPTIONS',
    'LINE_PROFILE',
}


#: Names that are already correct. Listed so the report can say so
#: explicitly rather than leaving the user wondering.
CURRENT = {
    # PSI_CONFIG is deliberately absent: it is in REMOVED. This script
    # still *reads* it below to locate a legacy config.py, because that
    # is where the old world kept one -- understanding the old layout is
    # a migration tool's job even when the variable itself is going away.
    'PSI_CONFIG_FILE',
    'PSI_SOUND_DEVICE_NAME',
    'PSI_SOUND_DEVICE_FS',
    'CFTS_ROOT',
    'CFTSCAL_ROOT',
    'NOISE_EXP_MAX_ANIMALS',
}


#: Prefixes a setting is allowed to carry after the rework.
KNOWN_PREFIXES = ('PSI_', 'PSIDATA_', 'CFTS_', 'CFTSCAL_', 'NOISE_EXP_')


#: Legacy names that are also, and much more often, set in a normal
#: environment for reasons that have nothing to do with psi. These are
#: still reported when they appear in a *config file*, where they can only
#: mean the psi setting, but are ignored in the environment scan.
#: `HOSTNAME` in particular is set by bash, msys and most CI runners, so
#: flagging it would put a false positive in every single report.
ENV_SCAN_IGNORE = {
    'HOSTNAME',
}


#: Sort order for the `action` column. Errors first within a file, then
#: the mechanical edits, then the judgement calls.
ACTION_ORDER = {'error': 0, 'rename': 1, 'remove': 2, 'review': 3}


class Finding:

    def __init__(self, source, name, action, detail, new_name=None,
                 lineno=None):
        self.source = source
        self.name = name
        self.action = action
        self.detail = detail
        self.new_name = new_name
        self.lineno = lineno

    @property
    def location(self):
        '''
        Where to go to fix it: `path:lineno` for a file, so the terminal
        can link straight to the line.
        '''
        if self.lineno is None:
            return self.source
        return f'{self.source}:{self.lineno}'

    @property
    def sort_key(self):
        # Line number numerically, not as text: otherwise line 10 sorts
        # ahead of line 7 and a config file's keys come out shuffled.
        return (self.lineno if self.lineno is not None else 0,
                ACTION_ORDER.get(self.action, 9),
                self.name)

    def as_dict(self):
        return {
            'source': self.source,
            'lineno': self.lineno,
            'location': self.location,
            'name': self.name,
            'action': self.action,
            'new_name': self.new_name,
            'detail': self.detail,
        }


def classify(name, source, lineno=None):
    '''
    Decide what, if anything, has to happen to `name`.

    Returns a `Finding`, or None if the name is already correct or is
    none of our business.
    '''
    if name in THIRD_PARTY or name in CURRENT:
        return None

    if name in RUNTIME_ONLY:
        return Finding(source, name, 'remove',
                       f'written at runtime ({RUNTIME_ONLY[name]}); '
                       'setting it by hand has no effect and may confuse '
                       'the launcher', None, lineno)

    if name in REMOVED:
        return Finding(source, name, 'remove', REMOVED[name], None, lineno)

    if name in RENAMED:
        new = RENAMED[name]
        return Finding(source, name, 'rename',
                       f'renamed to {new}', new, lineno)

    if name.startswith(HANDOFF_PREFIX_OLD) and name not in HANDOFF_KEEP:
        new = HANDOFF_PREFIX_NEW + name[len(HANDOFF_PREFIX_OLD):]
        return Finding(source, name, 'rename',
                       'cftscal handoff variable; the whole namespace moved '
                       f'to {HANDOFF_PREFIX_NEW}', new, lineno)

    return None


def scan_environment():
    '''
    Legacy names set in the current environment.

    Note that the handoff variables are normally set by the launcher for
    the lifetime of one subprocess. Finding one set *persistently* here
    usually means somebody pinned it by hand, which is worth knowing
    either way.
    '''
    findings = []
    for name in sorted(os.environ):
        if name in ENV_SCAN_IGNORE:
            continue
        finding = classify(name, 'environment')
        if finding is not None:
            findings.append(finding)
    return findings


def default_config_paths():
    '''
    Where psi would look for a config file, using the same rules the
    package uses -- reimplemented rather than imported so the script runs
    on a machine where psi is absent or half-upgraded.
    '''
    explicit = os.environ.get('PSI_CONFIG_FILE')
    if explicit:
        return [Path(explicit)]
    folder = os.environ.get('PSI_CONFIG', Path('~') / 'psi')
    return [Path(folder).expanduser() / 'config.py']


def scan_config_file(path):
    '''
    Legacy keys assigned at module level in a config file.

    Parsed, never executed: a config file may import modules that are not
    installed, and executing one to audit it is a bad habit to build into
    a deployment tool.
    '''
    findings = []
    try:
        # utf-8-sig, not the locale default: a config file edited in
        # Notepad or written by PowerShell's Out-File carries a BOM, which
        # ast.parse rejects as a stray character on line 1.
        tree = ast.parse(path.read_text(encoding='utf-8-sig'),
                         filename=str(path))
    except SyntaxError as e:
        findings.append(Finding(str(path), '<file>', 'error',
                                f'could not be parsed: {e}'))
        return findings
    except OSError as e:
        findings.append(Finding(str(path), '<file>', 'error',
                                f'could not be read: {e}'))
        return findings

    assigned = []
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id == target.id.upper():
                assigned.append((target.id, getattr(node, 'lineno', 0)))

    source = str(path)
    for name, lineno in assigned:
        finding = classify(name, source, lineno)
        if finding is not None:
            findings.append(finding)
            continue
        # A key that is neither legacy nor recognised is worth surfacing:
        # after the rework an unprefixed key is read by nothing at all.
        if name in CURRENT or name in THIRD_PARTY:
            continue
        if not name.startswith(KNOWN_PREFIXES):
            # BASE_DIRECTORY and SYSTEM are scaffolding in the generated
            # template, used to build other values rather than read by psi.
            if name in ('BASE_DIRECTORY', 'SYSTEM'):
                continue
            findings.append(Finding(
                source, name, 'review',
                'not a recognised setting and carries no package prefix; '
                'after the rework nothing reads an unprefixed key',
                None, lineno))
    return findings


def scan_cftscal_workspace(path):
    '''
    The cftscal GUI stores its calibration folder in workspace.json, which
    used to override the CFTSCAL_ROOT environment variable. Under the new
    precedence the environment wins, so a rig where the two disagree will
    change behaviour on upgrade.
    '''
    findings = []
    if not path.exists():
        return findings
    try:
        config = json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError) as e:
        findings.append(Finding(str(path), '<file>', 'error',
                                f'could not be read: {e}'))
        return findings
    data_path = config.get('data_path')
    if not data_path:
        return findings
    env_value = os.environ.get('CFTSCAL_ROOT')
    if env_value and str(env_value) != str(data_path):
        findings.append(Finding(
            str(path), 'data_path', 'review',
            f'workspace.json says {data_path!r} but CFTSCAL_ROOT says '
            f'{env_value!r}. The file used to win; the environment now '
            'does, so calibrations will be read from a different folder '
            'after the upgrade'))
    return findings


def report_text(findings, scanned, stream):
    hostname = socket.gethostname()
    print(f'Legacy configuration audit for {hostname}', file=stream)
    print(f'Python {sys.version.split()[0]} at {sys.executable}', file=stream)
    print('', file=stream)

    print('Scanned:', file=stream)
    for item in scanned:
        print(f'  {item}', file=stream)
    print('', file=stream)

    if not findings:
        print('No legacy configuration found. This machine is ready.',
              file=stream)
        return

    # Group by file so the report reads as a work list -- open one file,
    # fix everything listed under it, move on.
    by_source = {}
    for finding in findings:
        by_source.setdefault(finding.source, []).append(finding)

    width = max(len(f.name) for f in findings)
    for source in sorted(by_source, key=lambda s: (s != 'environment', s)):
        print(f'{source}', file=stream)
        for finding in sorted(by_source[source], key=lambda f: f.sort_key):
            where = f'{finding.lineno}: ' if finding.lineno is not None else ''
            arrow = f' -> {finding.new_name}' if finding.new_name else ''
            print(f'  {where}[{finding.action}] '
                  f'{finding.name:<{width}}{arrow}', file=stream)
            print(f'      {finding.detail}', file=stream)
        print('', file=stream)

    counts = {}
    for finding in findings:
        counts[finding.action] = counts.get(finding.action, 0) + 1
    summary = ', '.join(f'{v} to {k}' for k, v in sorted(counts.items()))
    print(f'{len(findings)} item(s) need attention: {summary}.', file=stream)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Report legacy psi/cfts configuration names on this '
                    'machine.')
    parser.add_argument('--config', action='append', type=Path, default=None,
                        help='Config file to scan. May be repeated. Defaults '
                             'to the file psi itself would load.')
    parser.add_argument('--workspace', type=Path, default=None,
                        help='Path to the cftscal workspace.json. Defaults '
                             'to <config folder>/cfts/workspace.json.')
    parser.add_argument('--no-environment', action='store_true',
                        help='Skip the environment; scan files only.')
    parser.add_argument('--format', choices=['text', 'json'], default='text')
    args = parser.parse_args(argv)

    findings = []
    scanned = []

    if not args.no_environment:
        scanned.append('the current environment')
        findings.extend(scan_environment())

    paths = args.config if args.config else default_config_paths()
    for path in paths:
        path = path.expanduser()
        if path.exists():
            scanned.append(f'{path}')
            findings.extend(scan_config_file(path))
        else:
            scanned.append(f'{path} (does not exist)')

    if args.workspace is not None:
        workspace = args.workspace.expanduser()
    else:
        folder = os.environ.get('PSI_CONFIG', Path('~') / 'psi')
        workspace = Path(folder).expanduser() / 'cfts' / 'workspace.json'
    if workspace.exists():
        scanned.append(f'{workspace}')
        findings.extend(scan_cftscal_workspace(workspace))
    else:
        scanned.append(f'{workspace} (does not exist)')

    if args.format == 'json':
        json.dump({
            'hostname': socket.gethostname(),
            'scanned': scanned,
            'findings': [f.as_dict() for f in findings],
        }, sys.stdout, indent=2)
        sys.stdout.write('\n')
    else:
        report_text(findings, scanned, sys.stdout)

    return 1 if findings else 0


if __name__ == '__main__':
    sys.exit(main())
