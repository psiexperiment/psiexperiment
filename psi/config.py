'''
Configuration for psiexperiment and the packages built on it.

A setting has exactly one spelling. The name used in code is the name in
``config.toml`` and the name of the environment variable, package prefix
included::

    get_config('PSI_DATA_ROOT')     # code
    PSI_DATA_ROOT = "D:/data"       # config.toml
    set PSI_DATA_ROOT=D:/data       # environment

Resolution order, last wins::

    default (psi.config_defaults)  ->  config.toml  ->  environment

The environment overriding the file is the ordinary arrangement (pip, git,
conda and the AWS CLI all work this way) and it is the reverse of what
this package did before: the old implementation applied environment
variables to the *defaults* and then let ``config.py`` overwrite them, so
setting ``PSI_DATA_ROOT`` on a machine with a config file did nothing at
all, silently.

Two things deliberately live outside this system:

- ``PSI_CONFIG_FILE`` selects the config file, so it cannot be a setting
  inside the file it selects. It is the only bootstrap variable; there is
  no separate "config folder" variable, and no path is ever inferred from
  the config file's location -- every directory is its own named setting.
- Values the application writes while it runs (the experiment name, the
  log filename) are not configuration. They live in :mod:`psi.runtime`,
  which has no environment or file path at all.
'''
import logging
import os
import tomllib
from pathlib import Path

log = logging.getLogger(__name__)


#: Sentinel distinguishing "no default supplied" from an explicit default
#: of None. Compared by identity, never with ``!=`` -- a default may be a
#: numpy array or anything else without sane equality.
NoDefault = object()


#: Default for every known setting, as {name: zero-argument callable}.
#: Populated by `register_defaults`; psi registers its own table on
#: import, and each downstream package registers its own.
_defaults = {}

#: Parsed contents of the config file. None until first read.
_config = None


CFG_ERR_MESG = '''
"{}" is not a known setting and was not found in the configuration file.

The configuration file was looked for at:
    {}

Run `psi-config create` to generate one, or set the PSI_CONFIG_FILE
environment variable to point at an existing file.
'''


def register_defaults(defaults):
    '''
    Register a package's default values.

    Parameters
    ----------
    defaults : dict
        Maps setting name to a zero-argument callable returning the
        default. Callables rather than values so that defaults derived
        from other settings (most of the psi roots are defined relative
        to ``PSI_BASE_DIRECTORY``) resolve *after* the config file is
        read rather than being frozen at import.

    Raises
    ------
    ValueError
        If a setting is registered twice with a different callable. Two
        packages claiming one name is a bug, not a merge.
    '''
    for name, factory in defaults.items():
        if not callable(factory):
            raise ValueError(
                f'Default for {name} must be a zero-argument callable, not '
                f'{type(factory).__name__}. Defaults are called after the '
                'config file is read so that derived values pick up '
                'overrides.')
        existing = _defaults.get(name)
        if existing is not None and existing is not factory:
            raise ValueError(f'{name} already has a registered default.')
        _defaults[name] = factory


def get_config_file():
    '''
    Path to the configuration file.

    ``PSI_CONFIG_FILE`` names it outright; otherwise it is
    ``~/psi/config.toml``. Per-environment configuration is a separate
    file per environment, selected by setting this variable before
    launching -- there is no layering and no auto-detection.
    '''
    explicit = os.environ.get('PSI_CONFIG_FILE')
    if explicit:
        return Path(explicit)
    return Path('~/psi/config.toml').expanduser()


def load_config():
    '''
    Read and return the config file, or an empty dict if there is none.

    Missing is not an error: every setting has a default in code, so a
    bare install runs with no config file at all.
    '''
    path = get_config_file()
    if not path.exists():
        log.debug('No configuration file at %s', path)
        return {}
    with path.open('rb') as fh:
        config = tomllib.load(fh)
    log.debug('Loaded configuration from %s', path)
    return config


def reload_config():
    '''
    Discard the cached config file so the next read picks up changes.
    '''
    global _config
    _config = None


def _ensure_config():
    global _config
    if _config is None:
        _config = load_config()
    return _config


def _coerce(value, template):
    '''
    Convert `value` to the type of `template`.

    Environment variables are always strings, and TOML has no path type,
    so a setting must be converted to whatever its default is -- callers
    do ``get_config('PSI_DATA_ROOT') / filename`` and are entitled to a
    Path regardless of where the value came from.
    '''
    if template is None or not isinstance(value, str):
        return value

    # bool before int: bool is a subclass of int, and bool('false') is
    # True, which would make every spelling of "off" mean "on".
    if isinstance(template, bool):
        lowered = value.strip().lower()
        if lowered in ('1', 'true', 'yes', 'on'):
            return True
        if lowered in ('0', 'false', 'no', 'off'):
            return False
        raise ValueError(
            f'Cannot interpret {value!r} as true/false. Use one of '
            '1/0, true/false, yes/no, on/off.')

    if isinstance(template, Path):
        return Path(value)
    if isinstance(template, int):
        return int(value)
    if isinstance(template, float):
        return float(value)
    if isinstance(template, (list, tuple)):
        # TOML gives a real array; only an environment variable arrives as
        # a string, where comma is the only separator that does not
        # collide with Windows paths.
        return [v.strip() for v in value.split(',') if v.strip()]
    return value


def _resolve_default(setting, default=NoDefault):
    '''
    The registered default for `setting`, else the caller's `default`.

    The registered table wins. A call site passing its own default for a
    setting that has a registered one is a bug -- it is how the same
    setting ends up with two different defaults in two modules -- and is
    caught by a test rather than silently honoured here.
    '''
    factory = _defaults.get(setting)
    if factory is not None:
        return factory()
    return default


def get_config(setting=None, default=NoDefault):
    '''
    Value of `setting`, resolved from defaults, the config file and the
    environment.

    Parameters
    ----------
    setting : {None, str}
        Name of the setting. If None, returns every known setting.
    default : object
        Used only for settings with no registered default (an ad-hoc key
        not in any package's defaults table).

    Raises
    ------
    KeyError
        If the setting is unknown, absent from the config file, and no
        default was supplied.
    '''
    if setting is None:
        return get_all_config()

    fallback = _resolve_default(setting, default)

    value = os.environ.get(setting)
    if value is not None:
        return _coerce(value, fallback if fallback is not NoDefault else None)

    config = _ensure_config()
    if setting in config:
        return _coerce(config[setting],
                       fallback if fallback is not NoDefault else None)

    if fallback is NoDefault:
        raise KeyError(CFG_ERR_MESG.strip().format(setting, get_config_file()))
    return fallback


def config_source(setting):
    '''
    Which layer supplied the current value of `setting`.

    Returns 'environment', 'config file' or 'default'. This is what makes
    a surprising value diagnosable -- it is the question "why is this
    setting not what I put in the file?" answered directly, and it is the
    only diagnostic available to a user running a frozen application with
    no CLI.
    '''
    if setting in os.environ:
        return 'environment'
    if setting in _ensure_config():
        return 'config file'
    return 'default'


def get_all_config():
    '''
    Every known setting and its resolved value.

    Covers everything with a registered default plus anything present in
    the config file, so a key nobody registered still shows up.
    '''
    config = _ensure_config()
    names = set(_defaults) | set(config)
    result = {}
    for name in sorted(names):
        try:
            result[name] = get_config(name)
        except Exception as e:
            # One unresolvable setting must not blind the user to the
            # rest; `psi-config show` is often what they are running to
            # find out why something is broken.
            result[name] = f'<error: {e}>'
    return result


def save_config(updates):
    '''
    Write `updates` into the config file, creating it if necessary.

    Uses tomlkit so that comments and formatting in a hand-edited file
    survive a programmatic write, and replaces the file atomically so an
    interrupted write cannot leave a truncated config behind -- this is
    called from GUI save buttons, not just from tooling.

    Note that a setting overridden in the environment will not change
    value as a result of this call. Callers with a user interface should
    check :func:`config_source` and say so.
    '''
    import tomlkit

    path = get_config_file()
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        document = tomlkit.parse(path.read_text(encoding='utf-8'))
    else:
        document = tomlkit.document()

    for name, value in updates.items():
        # TOML has no path type and no None. A path becomes a string; a
        # None means "stop overriding this" and removes the key.
        if value is None:
            document.pop(name, None)
        elif isinstance(value, Path):
            document[name] = str(value)
        elif isinstance(value, tuple):
            document[name] = list(value)
        else:
            document[name] = value

    temp = path.with_name(path.name + '.tmp')
    temp.write_text(tomlkit.dumps(document), encoding='utf-8')
    os.replace(temp, path)

    reload_config()
    log.debug('Wrote %d setting(s) to %s', len(updates), path)


def create_config_dirs():
    '''
    Create the directory named by every registered ``*_ROOT`` setting,
    plus the base directory.
    '''
    for name in sorted(_defaults):
        if not (name.endswith('_ROOT') or name.endswith('_DIRECTORY')):
            continue
        value = get_config(name)
        if isinstance(value, (str, Path)):
            Path(value).mkdir(parents=True, exist_ok=True)
