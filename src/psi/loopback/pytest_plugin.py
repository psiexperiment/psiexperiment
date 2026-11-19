'''
pytest options, hooks and fixtures for loopback tests.

Load it as a plugin for every test run, from the project's pyproject.toml:

    [tool.pytest.ini_options]
    addopts = "-p psi.loopback.pytest_plugin"

and define a session-scoped `rig` fixture returning a `psi.loopback.rig.Rig`
in the conftest of the hardware tests. Tests in any folder named `hardware`
are then skipped unless pytest is run with `--fireface`, whether it is given
the whole suite or just the hardware tests:

    pytest tests --fireface
    pytest tests/hardware --fireface

Don't import this module's hooks into a conftest instead. pytest reads
options only from the conftests next to (or above) the paths it is given,
so `pytest tests --fireface` would fail with "unrecognized arguments"; and
importing them as well as loading the plugin adds the options twice. Options
of a project's own belong in `tests/conftest.py` for the same reason.

Each run writes its data, psi's log, and a report with plots under
`tests/hardware/results/<date time>/` (or `--fireface-results`); open
`index.html` there to look through them.
'''
import datetime as dt
import os
from pathlib import Path
import sys

import pytest

from .report import write_index


def pytest_addoption(parser):
    parser.addoption('--fireface', action='store_true',
                     help='Run the paradigm tests through the loopback '
                     'sound card')
    parser.addoption('--fireface-results', default=None,
                     help='Folder for the run data and reports')
    parser.addoption('--src', action='append', default=[],
                     help='Put this source folder ahead of the installed '
                     'packages when running paradigms (e.g., a worktree of '
                     'another branch, or psiexperiment/src). May be given '
                     'more than once. Write it as --src=PATH: given as two '
                     'arguments, pytest also takes PATH as a candidate root '
                     'directory.')


def pytest_configure(config):
    src = config.getoption('--src')
    if src:
        # The paradigms run in subprocesses, which inherit this.
        resolved = [str(Path(s).resolve()) for s in src]
        paths = resolved + [os.environ.get('PYTHONPATH')]
        os.environ['PYTHONPATH'] = os.pathsep.join(p for p in paths if p)
        # Some tests also call paradigm code directly.
        sys.path[:0] = resolved


def pytest_collection_modifyitems(config, items):
    if config.getoption('--fireface'):
        return
    skip = pytest.mark.skip(reason='needs the loopback sound card; run with '
                            '--fireface')
    for item in items:
        if 'hardware' in item.path.parts:
            item.add_marker(skip)


@pytest.fixture(scope='session')
def results_dir(request):
    path = request.config.getoption('--fireface-results')
    if path is None:
        stamp = dt.datetime.now().strftime('%Y-%m-%d %H%M%S')
        path = request.config.rootpath / 'tests' / 'hardware' / 'results' / stamp
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope='session')
def routing(rig, results_dir):
    '''
    How the sound card's mixer currently routes outputs to inputs.
    '''
    from .plots import plot_routing
    from .rig import measure_routing
    routing = measure_routing(rig)
    plot_routing(routing, results_dir / 'routing.png')
    write_index(results_dir)
    return routing


def require_routes(routing, required):
    '''
    Skip unless the mixer has every route in `required` ({input: [outputs]}).
    '''
    missing = routing.missing(required)
    if missing:
        pytest.skip('The mixer is missing ' + ', '.join(missing))
