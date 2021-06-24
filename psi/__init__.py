import logging
import os
from pathlib import Path

from atom.api import Event

_config = {}


# Set up a verbose debugger level for tracing
TRACE_LEVEL_NUM = 5
logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")
def trace(self, message, *args, **kws):
    # Yes, logger takes its '*args' as 'args'.
    if self.isEnabledFor(TRACE_LEVEL_NUM):
        self._log(TRACE_LEVEL_NUM, message, args, **kws)
logging.Logger.trace = trace

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


# Flag indicating whether user configuration file was loaded.
CONFIG_LOADED = False


exclude = ['_d_storage', '_d_engine', '_flags', '_parent', '_children']


class SimpleState(object):

    def __getstate__(self):
        state = super(SimpleState, self).__getstate__()
        for k, v in self.members().items():
            if isinstance(v, Event):
                del state[k]
            elif k in exclude:
                del state[k]
            elif v.metadata and v.metadata.get('transient', False):
                del state[k]
        return state

    def __setstate__(self, state):
        for key, value in state.items():
            setattr(self, key, value)


def get_config_folder():
    '''
    Return configuration folder
    '''
    default = Path('~') / 'psi'
    default = default.expanduser()
    return Path(os.environ.get('PSI_CONFIG', default))


def get_config_file():
    '''
    Return configuration file
    '''
    default = get_config_folder() / 'config.py'
    return Path(os.environ.get('PSI_CONFIG_FILE', default))


def create_config(base_directory=None, log=None, data=None, processed=None,
                  cal=None, preferences=None, layout=None, io=None):

    # This approach allows code inspection to show valid function parameters
    # without hiding it behind an anonymous **kwargs definition.
    kwargs = locals()
    kwargs.pop('base_directory')

    # Figure out where to save everything
    target = get_config_file()
    target.parent.mkdir(exist_ok=True, parents=True)

    if base_directory is None:
        base_directory = str(target.parent)

    defaults = {
        'LOG_ROOT': "BASE_DIRECTORY / 'logs'",
        'DATA_ROOT': "BASE_DIRECTORY / 'data'",
        'PROCESSED_ROOT': "BASE_DIRECTORY / 'processed'",
        'CAL_ROOT': "BASE_DIRECTORY / 'calibration'",
        'PREFERENCES_ROOT': "BASE_DIRECTORY / 'settings' / 'preferences'",
        'LAYOUT_ROOT': "BASE_DIRECTORY / 'settings' / 'layout'",
        'IO_ROOT': "BASE_DIRECTORY / 'io'",
    }

    for key, value in kwargs.items():
        if value is None:
            continue
        config_key = f"{key.upper()}_ROOT"
        config_value = f"Path(r'{value}')"
        defaults[config_key] = config_value

    paths = '\n'.join(f'{k} = {v}' for k, v in defaults.items())
    config_template = Path(__file__).parent / 'templates' / 'config.txt'
    config_text = config_template.read_text()
    config_text = config_text.format(base_directory, paths)
    target.write_text(config_text)


def create_io_manifest(template):
    io_template = Path(__file__).parent / 'templates' / 'io' / template
    io_template = io_template.with_suffix('.enaml')
    io = Path(get_config('IO_ROOT')) / template
    io = io.with_suffix('.enaml')
    io.parent.mkdir(exist_ok=True, parents=True)
    io_text = io_template.read_text()
    io.write_text(io_text)


def create_config_dirs():
    config = load_config()
    for name, value in vars(config).items():
        if name.endswith('_ROOT'):
            Path(value).mkdir(exist_ok=True, parents=True)


def load_config():
    # Load the default settings
    global CONFIG_LOADED
    import importlib.util
    from os import environ
    from . import config

    config_path = get_config_file()
    if config_path.exists():
        try:
            spec = importlib.util.spec_from_file_location('settings', config_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for name, value in vars(module).items():
                if name == name.upper():
                    setattr(config, name, value)
            CONFIG_LOADED = True
        except Exception as e:
            log.exception(e)

    for name, value in vars(config).items():
        if name == name.upper():
            log.debug('CONFIG %s : %r', name, value)

    return config


def reload_config():
    global _config
    _config = load_config()


def set_config(setting, value):
    '''
    Set value of setting
    '''
    setattr(_config, setting, value)


CFG_ERR_MESG = '''
Could not find setting "{}" in configuration. This may be because the
configuration file is missing. Please run psi-config to create it.
'''


def get_config(setting=None):
    '''
    Get value of setting
    '''
    if setting is not None:
        try:
            return getattr(_config, setting)
        except AttributeError as e:
            if CONFIG_LOADED:
                raise
            mesg = CFG_ERR_MESG.strip().format(setting)
            raise SystemError(mesg) from e
    else:
        setting_names = [s for s in dir(_config) if s.upper() == s]
        setting_values = [getattr(_config, s) for s in setting_names]
        return dict(zip(setting_names, setting_values))


reload_config()

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
