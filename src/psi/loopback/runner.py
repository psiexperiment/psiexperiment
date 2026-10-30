'''
Run a paradigm on the test rig in a subprocess, unattended, and report how it
ended.
'''
from dataclasses import dataclass
import functools
import json
import os
from pathlib import Path
import re
import subprocess
import sys


@dataclass
class RunResult:
    path: Path
    log: Path
    returncode: int
    #: What psi reported when it wrapped up the experiment (None if it never
    #: got that far).
    wrapup: dict

    @property
    def completed(self):
        '''
        True if the experiment ran to the end (or was stopped by the
        autorun's `duration`), rather than being stopped by an error or never
        starting.
        '''
        return self.wrapup is not None and not self.wrapup['stop_reason'] \
            and not self.wrapup['error_message']

    def describe(self):
        if self.wrapup is None:
            return f'psi exited ({self.returncode}) before wrapping up; ' \
                f'see {self.log}'
        if self.wrapup['error_message']:
            return f'psi stopped with an error: ' \
                f'{self.wrapup["error_message"]}\n{self.wrapup["traceback"]}'
        return f'psi stopped early: {self.wrapup["stop_reason"]}'


def run_paradigm(rig, paradigm, path, settings, modules=(), env=None,
                 plugins=None, manifests=None, duration=None, timeout=300,
                 cwd=None):
    '''
    Run `paradigm` on `rig` until it finishes, saving its data to `path`.

    Parameters
    ----------
    rig : `psi.loopback.rig.Rig`
        Supplies the IO manifest.
    paradigm : str
        Name of the paradigm (e.g., 'abr_io').
    path : Path
        Folder for the data. Must not exist yet. The settings, psi's log and
        the outcome go in a sibling folder, `<path> run`.
    settings : dict
        Context settings, see `autorun_manifest.apply_settings`.
    modules : list of str
        Modules to import so the paradigm is registered (e.g.,
        'cfts.paradigms').
    env : dict
        Extra environment variables.
    plugins : list of str
        Optional plugins of the paradigm to enable.
    manifests : dict
        Extra manifests to load, as {'module.Manifest': {attribute: value}}
        (e.g., `psi.loopback.pilot.PilotManifest`). The attributes must be
        JSON serializable.
    duration : float
        If set, stop the experiment this many seconds after it starts, for
        paradigms that otherwise run until the user stops them.
    timeout : float
        Seconds to wait before killing a run that hasn't finished.
    cwd : Path
        Folder to run in. It is on the path, so `rig.io_manifest` and
        `manifests` can name modules in it (e.g., the hardware test folder).
    '''
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    run_dir = path.parent / f'{path.name} run'
    run_dir.mkdir()
    run_file = run_dir / 'settings.json'
    run_file.write_text(json.dumps({
        'settings': settings,
        'manifests': manifests or {},
        'duration': duration,
    }, indent=2))
    result_file = run_dir / 'result.json'
    log_file = run_dir / 'psi.log'

    cmd = [
        sys.executable, '-m', 'psi.loopback.autorun', paradigm, str(path),
        '--io', rig.io_manifest,
        '--import', *modules,
        '--plugins', *(plugins or []),
        '--settings', str(run_file),
        '--result', str(result_file),
        '--log', str(log_file),
    ]
    full_env = {**os.environ, **(env or {})}
    # Not connected to a launcher, and on the sound card the IO manifest
    # names rather than whichever one the launcher last selected.
    for name in ('PSI_WEBSOCKETS_URI', 'PSI_SOUND_DEVICE_NAME',
                 'PSI_SOUND_DEVICE_FS'):
        full_env.pop(name, None)

    try:
        proc = subprocess.run(cmd, env=full_env, timeout=timeout, cwd=cwd,
                              capture_output=True, text=True)
        returncode = proc.returncode
        (run_dir / 'stdout.txt').write_text(proc.stdout + proc.stderr)
    except subprocess.TimeoutExpired as e:
        returncode = None
        out = [o.decode(errors='replace') if isinstance(o, bytes) else o
               for o in (e.stdout, e.stderr) if o]
        (run_dir / 'stdout.txt').write_text(''.join(out) + '\nTIMED OUT\n')

    wrapup = json.loads(result_file.read_text()) \
        if result_file.exists() else None
    return RunResult(path, log_file, returncode, wrapup)


@functools.cache
def paradigm_available(paradigm, modules=(), cwd=None):
    '''
    True if `paradigm` is registered once `modules` are imported (in a
    subprocess, so it sees the same packages the runs do).
    '''
    lines = [f'import {m}' for m in modules] + [
        'from psi.experiment.api import paradigm_manager',
        f'paradigm_manager.get_paradigm({paradigm!r})',
    ]
    proc = subprocess.run([sys.executable, '-c', '\n'.join(lines)], cwd=cwd,
                          capture_output=True)
    return proc.returncode == 0


#: A record in psi's log file: "<date> <time> <LEVEL> : <thread> - <logger>
#: :: <message>". Lines that don't match continue the record above (e.g., a
#: traceback).
LOG_RECORD = re.compile(r'^\S+ \S+ (\w+)\s+:')

LOG_LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']


def log_problems(log_file, level='ERROR', ignore=()):
    '''
    Records in psi's log at `level` or above, other than those whose first
    line contains any of `ignore`. Each is returned as one string, including
    any traceback.
    '''
    threshold = LOG_LEVELS.index(level)
    problems = []
    keep = False
    for line in Path(log_file).read_text(errors='replace').splitlines():
        match = LOG_RECORD.match(line)
        if match is None:
            if keep:
                problems[-1] += '\n' + line
            continue
        record_level = match.group(1)
        keep = record_level in LOG_LEVELS \
            and LOG_LEVELS.index(record_level) >= threshold \
            and not any(i in line for i in ignore)
        if keep:
            problems.append(line)
    return problems
