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
import socket
from pathlib import Path

from .config import get_config


def _shipped_paradigm_descriptions():
    '''
    Paradigm description modules that ship with psi.

    Imported inside the function rather than at module scope: psi.config
    is imported very early, and psi.application pulls in a large part of
    the package. Defaults are only ever called on demand, so the cost is
    paid by whoever actually reads this setting.
    '''
    from .application import list_paradigm_descriptions
    return list_paradigm_descriptions()


DEFAULTS = {
    #: Root of every other psiexperiment path. Overriding this alone moves
    #: logs, data, preferences and the rest with it.
    'PSI_BASE_DIRECTORY': lambda: Path('~/Documents/psi').expanduser(),

    'PSI_LOG_ROOT': lambda: get_config('PSI_BASE_DIRECTORY') / 'logs',
    'PSI_DATA_ROOT': lambda: get_config('PSI_BASE_DIRECTORY') / 'data',
    'PSI_PROCESSED_ROOT': lambda: get_config('PSI_BASE_DIRECTORY') / 'processed',
    'PSI_PREFERENCES_ROOT': lambda: get_config('PSI_BASE_DIRECTORY') / 'preferences',
    'PSI_LAYOUT_ROOT': lambda: get_config('PSI_BASE_DIRECTORY') / 'layout',
    'PSI_IO_ROOT': lambda: get_config('PSI_BASE_DIRECTORY') / 'io',

    #: Name of this machine. Used to pick a hostname-specific IO manifest.
    'PSI_HOSTNAME': socket.gethostname,

    #: Hardware configurations offered in addition to whatever is found in
    #: PSI_IO_ROOT.
    'PSI_STANDARD_IO': lambda: [],

    #: Modules containing paradigm descriptions to import at startup. An
    #: experiment is only discoverable when its module is listed here.
    #: Defaults to the descriptions shipped with psi, so an install that
    #: configures nothing still offers the built-in experiments.
    'PSI_PARADIGM_DESCRIPTIONS': lambda: _shipped_paradigm_descriptions(),

    #: Websocket endpoint for the websocket paradigm mixins.
    'PSI_WEBSOCKETS_URI': lambda: '',
}
