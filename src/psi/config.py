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
import copy
import logging
import os
import tomllib
from pathlib import Path

log = logging.getLogger(__name__)


#: Sentinel distinguishing "no default supplied" from an explicit default
#: of None. Compared by identity, never with ``!=`` -- a default may be a
#: numpy array or anything else without sane equality.
NoDefault = object()


#: The types a setting can be declared as: what TOML and an environment
#: variable can both spell.
SETTING_TYPES = (bool, int, float, str, Path, list, dict)


class Setting:
    '''
    Declaration of one setting: its type, its default and what it is for.

    The type is stated rather than inferred from the default. Inferring
    it meant a setting whose default is None had no type at all, that an
    int default made a configured 96000.5 an error, and that reading any
    configured value first had to compute the default just to learn what
    to convert to.

    Parameters
    ----------
    type : type
        One of `SETTING_TYPES`. A value from the configuration file or
        the environment is converted to it.
    default : object
        The value when nothing is configured, or a zero-argument callable
        returning it. A callable is needed for a default derived from
        another setting (most psi roots derive from
        ``PSI_BASE_DIRECTORY``), so that it resolves after the
        configuration file is read rather than being frozen at import.
        None means the setting is unset unless configured.
    doc : str
        One line saying what the setting is for.
    '''

    def __init__(self, type, default=None, doc=''):
        if type not in SETTING_TYPES:
            names = ', '.join(t.__name__ for t in SETTING_TYPES)
            raise ValueError(f'A setting must be one of {names}, not {type!r}.')
        self.type = type
        self._default = default
        self.doc = doc

    def default(self):
        if callable(self._default):
            return self._default()
        # A copy, so a caller appending to a list default cannot change
        # the default for everybody after it.
        return copy.deepcopy(self._default)

    def __repr__(self):
        return f'Setting({self.type.__name__}, {self._default!r})'


#: Every known setting, as {name: Setting}. Populated by
#: `register_defaults`; psi registers its own table on import, and each
#: downstream package registers its own.
_defaults = {}

#: Parsed contents of the config file. None until first read.
_config = None

#: 'no layer supplied a value', which a stored None must not be
#: mistaken for.
_MISSING = object()


CFG_ERR_MESG = '''
"{}" is not a known setting and was not found in the configuration file.

The configuration file was looked for at:
    {}

Run `psi-config create` to generate one, or set the PSI_CONFIG_FILE
environment variable to point at an existing file.
'''


def register_defaults(defaults):
    '''
    Register a package's settings.

    Parameters
    ----------
    defaults : dict
        Maps setting name to a `Setting`.

    Raises
    ------
    ValueError
        If a value is not a `Setting`, or a setting is registered twice
        with a different declaration. Two packages claiming one name is a
        bug, not a merge; registering the same table twice (on import and
        again through the entry point) is not.
    '''
    for name, setting in defaults.items():
        if not isinstance(setting, Setting):
            raise ValueError(
                f'{name} must be declared with Setting(type, default), not '
                f'{type(setting).__name__}.')
        existing = _defaults.get(name)
        if existing is not None and existing is not setting:
            raise ValueError(f'{name} already has a registered default.')
        _defaults[name] = setting


def get_setting(name):
    '''
    The registered declaration for `name`, or None if nothing registered
    it.
    '''
    return _defaults.get(name)


def setting_names():
    '''
    Names of every registered setting, sorted.
    '''
    return sorted(_defaults)


def setting_type(name, default=NoDefault):
    '''
    The type a value of `name` is converted to, or None if it has none.

    A registered setting declares it. An unregistered one goes by the type
    of the caller's `default`, if any.
    '''
    setting = _defaults.get(name)
    if setting is not None:
        return setting.type
    return _type_of(default)


def _type_of(value):
    # bool before int: bool is a subclass of int.
    for t in (bool, Path, int, float, str, dict):
        if isinstance(value, t):
            return t
    if isinstance(value, (list, tuple)):
        return list
    return None


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
    # Read as text with utf-8-sig rather than handing tomllib a binary
    # handle: a configuration file saved by Notepad, by PowerShell's
    # Out-File, or by VS Code with "UTF-8 with BOM" starts with a byte
    # order mark, and tomllib rejects it as "Invalid statement (at line 1,
    # column 1)" -- naming neither the BOM nor the file.
    config = tomllib.loads(path.read_text(encoding='utf-8-sig'))
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


def _coerce(value, type_):
    '''
    Convert `value` to `type_`.

    Environment variables are always strings, and TOML has no path type,
    so a setting must be converted to its type -- callers do
    ``get_config('PSI_DATA_ROOT') / filename`` and are entitled to a Path
    regardless of where the value came from. A `type_` of None leaves the
    value as it is.
    '''
    if type_ is None:
        return value
    if isinstance(value, str):
        return _parse(value, type_)

    # A TOML value of the wrong shape -- `PSI_DATA_ROOT = 5` -- would
    # otherwise surface much later as a TypeError at the `/` operator,
    # naming neither the setting nor the file.
    if type_ is Path:
        if isinstance(value, Path):
            return value
        raise ValueError(
            f'Expected a path, but the configured value is {value!r} '
            f'({type(value).__name__}). Quote it: a path has to be a '
            'string in TOML.')
    # bool is a subclass of int, so it has to be excluded from the
    # numeric types by hand: `true` is not a sample rate.
    if type_ is float and isinstance(value, (int, float)) \
            and not isinstance(value, bool):
        return float(value)
    if type_ is int and isinstance(value, int) \
            and not isinstance(value, bool):
        return value
    if type_ is list and isinstance(value, (list, tuple)):
        return list(value)
    if type_ in (bool, dict) and isinstance(value, type_):
        return value
    raise ValueError(
        f'Expected {_describe(type_)}, but the configured value is '
        f'{value!r} ({type(value).__name__}).')


def _parse(text, type_):
    '''
    Convert the text of an environment variable (or a quoted TOML value)
    to `type_`.
    '''
    if type_ is str:
        return text
    # bool('false') is True, which would make every spelling of "off"
    # mean "on".
    if type_ is bool:
        lowered = text.strip().lower()
        if lowered in ('1', 'true', 'yes', 'on'):
            return True
        if lowered in ('0', 'false', 'no', 'off'):
            return False
        raise ValueError(
            f'Cannot interpret {text!r} as true/false. Use one of '
            '1/0, true/false, yes/no, on/off.')
    if type_ is Path:
        if not text.strip():
            raise ValueError(
                'An empty value cannot be a path. Path("") is the '
                'working directory, which is never what was meant.')
        return Path(text)
    if type_ is int:
        return int(text)
    if type_ is float:
        return float(text)
    if type_ is list:
        # TOML gives a real array; only an environment variable arrives as
        # a string, where comma is the only separator that does not
        # collide with Windows paths.
        return [v.strip() for v in text.split(',') if v.strip()]
    raise ValueError(
        f'Expected {_describe(type_)}, which cannot be written as text. '
        'Set it in the configuration file.')


def _describe(type_):
    return {
        bool: 'true or false', int: 'a whole number', float: 'a number',
        str: 'a string', Path: 'a path', list: 'a list', dict: 'a table',
    }[type_]


def parse_setting(name, text):
    '''
    Convert `text` to the type of setting `name`, as the environment would.

    For a tool taking a value on the command line, so that it can be
    checked -- and written to the file as a real number, switch or
    array -- before anything is saved.

    Raises
    ------
    ValueError
        If `text` is not a valid value for the setting.
    '''
    return _coerce(text, setting_type(name))


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

    value = os.environ.get(setting)
    # An environment variable set to nothing is treated as not set. A
    # launcher building `env[name] = str(field)` from an empty field
    # would otherwise win over both the file and the default, and for a
    # path setting `_coerce` turns '' into Path('.') -- silently moving a
    # data root to the working directory.
    if value is not None and value.strip() == '':
        log.debug('%s is set to an empty string in the environment; '
                  'treating it as unset', setting)
        value = None

    if value is None:
        config = _ensure_config()
        value = config[setting] if setting in config else _MISSING

    registered = _defaults.get(setting)
    if value is _MISSING:
        # The registered default wins over the caller's. A call site
        # passing its own default for a registered setting is a bug -- it
        # is how one setting ends up with two defaults in two modules --
        # and is caught by a test rather than silently honoured here.
        if registered is not None:
            return registered.default()
        if default is NoDefault:
            raise KeyError(
                CFG_ERR_MESG.strip().format(setting, get_config_file()))
        return default

    try:
        return _coerce(value, setting_type(setting, default))
    except ValueError as e:
        raise ValueError(f'{setting}: {e}') from None


def config_source(setting):
    '''
    Which layer supplied the current value of `setting`.

    Returns 'environment', 'config file' or 'default'. This is what makes
    a surprising value diagnosable -- it is the question "why is this
    setting not what I put in the file?" answered directly, and it is the
    only diagnostic available to a user running a frozen application with
    no CLI.
    '''
    # Same emptiness rule as get_config, or this reports a layer that was
    # skipped -- `psi-config show` would blame the environment for a
    # value that actually came from the file or the default.
    if os.environ.get(setting, '').strip():
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


def _tomlify(value, name, _path=()):
    '''
    Convert a value into something TOML can represent.

    Paths become strings and tuples become arrays. A None *inside* a
    table is dropped, for the same reason a None at the top level removes
    a key: TOML has no null, and absence is what None means. Dropping it
    is safe for a table -- the reader sees a missing key -- but not for
    an array, where it would shift everything after it, so that raises.

    Without this, a table holding a None reached tomlkit and came back as
    `ConvertError: Unable to convert an object of <class 'NoneType'>`,
    naming neither the setting nor the key. That is what a launcher
    writes on a fresh install, where a device has not been chosen yet.
    '''
    where = '.'.join((name,) + _path)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _tomlify(v, name, _path + (str(k),))
                for k, v in value.items() if v is not None}
    if isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            if v is None:
                raise ValueError(
                    f'Cannot save {where}: element {i} is None, and TOML '
                    'has no null. Removing it would renumber the rest, so '
                    'the caller has to decide what the empty slot means.')
        return [_tomlify(v, name, _path + (str(i),))
                for i, v in enumerate(value)]
    return value


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
        document = tomlkit.parse(path.read_text(encoding='utf-8-sig'))
    else:
        document = tomlkit.document()

    for name, value in updates.items():
        # TOML has no None. At the top level it means "stop overriding
        # this", so the key is removed.
        if value is None:
            document.pop(name, None)
        else:
            document[name] = _tomlify(value, name)

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
