'''
Default values for every psiexperiment setting.

Defaults live in code rather than in a shipped configuration file, so a
bare install runs with no `config.toml` at all. They live *here*, in one
table per package, rather than at each `get_config` call site: a setting
read from several places would otherwise need its default repeated at
each one, and nothing would keep the copies in agreement. That is not
hypothetical -- before this table existed, ``LOG_FILENAME`` defaulted to
``''`` in one module and ``None`` in another.

Every entry is a zero-argument callable, not a value. Two reasons:

- The derived roots are defined in terms of ``PSI_BASE_DIRECTORY``, which
  is itself configurable. Resolving them at import would freeze the
  built-in base directory into the defaults, so setting
  ``PSI_BASE_DIRECTORY`` in ``config.toml`` would silently fail to move
  the other six roots.
- Nothing is computed unless it is actually read, which keeps importing
  `psi` free of side effects (no environment reads, no hostname lookup).

Downstream packages (cftscal, cfts, psidata) each own an equivalent
module and register it via :func:`psi.config.register_defaults`.
'''
import os
import socket
from pathlib import Path

from .config import get_config


def _local_state_directory():
    '''
    Per-user directory for files that must stay on the local disk.

    Logs and profiling output are written continuously while an
    experiment runs, so they must not follow PSI_BASE_DIRECTORY onto a
    network share -- which is exactly where a rig points it, since that
    is where the data belongs.
    '''
    if os.name == 'nt':
        local = os.environ.get('LOCALAPPDATA')
        if local:
            return Path(local) / 'psi'
    return Path('~/.local/state/psi').expanduser()


DEFAULTS = {
    #: Root of the psiexperiment paths that belong with the data.
    #: Overriding this alone moves data, settings and IO manifests with
    #: it. Logs deliberately do not follow it; see PSI_LOG_ROOT.
    'PSI_BASE_DIRECTORY': lambda: Path('~/Documents/psi').expanduser(),

    #: Log files and profiling output. Local by default rather than
    #: derived from PSI_BASE_DIRECTORY, so that pointing the base
    #: directory at a network share does not send continuous log writes
    #: over the network as well.
    'PSI_LOG_ROOT': lambda: _local_state_directory() / 'logs',

    'PSI_DATA_ROOT': lambda: get_config('PSI_BASE_DIRECTORY') / 'data',
    'PSI_IO_ROOT': lambda: get_config('PSI_BASE_DIRECTORY') / 'io',

    #: Per-paradigm saved layouts and preferences, as
    #: <root>/layout/<paradigm> and <root>/preferences/<paradigm>. One
    #: setting rather than two: they have the same lifecycle, the same
    #: structure, and the same readers.
    'PSI_SETTINGS_ROOT': lambda: get_config('PSI_BASE_DIRECTORY') / 'settings',

    #: Name of this machine. Used to pick a hostname-specific IO manifest.
    'PSI_HOSTNAME': socket.gethostname,

    #: Address of the websocket server an experiment reports back to.
    #: This is a handoff rather than a setting: the cfts launcher runs the
    #: server and writes the address into the environment of each `psi`
    #: subprocess it starts, the way it does the CFTSCAL_ variables.
    #:
    #: None, not '', because absence is the whole meaning of the default
    #: -- nobody supplied an address, so there is no server to talk to.
    #: TOML has no null, so None can only ever come from here; a value in
    #: the file or the environment is always a string.
    'PSI_WEBSOCKETS_URI': lambda: None,
}
