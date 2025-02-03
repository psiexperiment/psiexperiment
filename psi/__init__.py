# This adds the TRACE logging level
import psiaudio

import logging
import importlib.util
import os
from pathlib import Path

# This automatically introduces the TRACE logging level
import psiaudio

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


# Flag indicating whether user configuration file was loaded.
CONFIG_LOADED = False


DEFAULT_CONFIG = {
    'LOG_ROOT': os.path.expanduser('~/Documents/psi/logs'),
    'DATA_ROOT': os.path.expanduser('~/Documents/psi/data'),
    'PROCESSED_ROOT': os.path.expanduser('~/Documents/psi/processed'),
    'PREFERENCES_ROOT': os.path.expanduser('~/Documents/psi/preferences'),
    'LAYOUT_ROOT': os.path.expanduser('~/Documents/psi/layout'),
    'CFTS_ROOT': os.path.expanduser('~/Documents/psi/cfts'),
    'IO_ROOT': os.path.expanduser('~/Documents/psi/io'),
}


# Container for configuration variables
_config = {}


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
                  cal=None, preferences=None, layout=None, io=None,
                  standard_io=None, paradigm_descriptions=None):

    # This approach allows code inspection to show valid function parameters
    # without hiding it behind an anonymous **kwargs definition.
    kwargs = locals()
    kwargs.pop('base_directory')
    kwargs.pop('standard_io')
    kwargs.pop('paradigm_descriptions')

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

    help_text = {
        'LOG_ROOT': 'Location of log files for psiexperiment',
        'DATA_ROOT': 'Location where experiment data is saved',
        'PROCESSED_ROOT': 'Location where post-processed experiment data is saved',
        'CAL_ROOT': 'Location where calibration data is saved',
        'PREFERENCES_ROOT': 'Location where experiment-specific defaults are saved',
        'LAYOUT_ROOT': 'Location where experiment-specific layouts are saved',
        'IO_ROOT': 'Location where custom hardware configurations are saved',
        'STANDARD_IO': 'List of standard hardware configurations the user can select from'
    }

    if standard_io is None:
        defaults['STANDARD_IO'] = "[]"
    else:
        engine_string = ',\n'.join(f"    '{e}'" for e in standard_io)
        standard_io = f"[\n{engine_string}\n]"
        defaults['STANDARD_IO'] = standard_io

    for key, value in kwargs.items():
        if value is None:
            continue
        config_key = f"{key.upper()}_ROOT"
        config_value = f"Path(r'{value}')"
        defaults[config_key] = config_value

    lines = []
    for k, v in defaults.items():
        if k in help_text:
            lines.append(f'# {help_text[k]}')
        lines.append(f'{k} = {v}\n')

    if paradigm_descriptions is not None:
        lines.append('# List of module paths containing experiment paradigm descriptions')
        lines.append('PARADIGM_DESCRIPTIONS = [')
        for description in paradigm_descriptions:
            lines.append(f'    "{description}",')
        lines.append(']\n')

    paths = '\n'.join(lines)
    config_template = Path(__file__).parent / 'templates' / 'config.txt'
    config_text = config_template.read_text()
    config_text = config_text.format(base_directory, paths)
    target.write_text(config_text)


def create_io_manifest(template):
    io_template = Path(__file__).parent / 'templates' / 'io' / template
    io_template = io_template.with_suffix('.enaml')
    io = Path(get_config('IO_ROOT')) / template.lstrip('_')
    io = io.with_suffix('.enaml')
    io.parent.mkdir(exist_ok=True, parents=True)
    io_text = io_template.read_text()
    io.write_text(io_text)


def create_config_dirs():
    config = load_config()
    for name, value in config.items():
        if name.endswith('_ROOT'):
            Path(value).mkdir(exist_ok=True, parents=True)


def load_config():
    # Load the default settings
    global CONFIG_LOADED

    config = DEFAULT_CONFIG.copy()
    config_path = get_config_file()
    if config_path.exists():
        spec = importlib.util.spec_from_file_location('settings', config_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config.update({n: v for n, v in vars(module).items() if n == n.upper()})
        CONFIG_LOADED = True

    log.debug('CONFIG: %r', config)
    return config


def reload_config():
    global _config
    _config = load_config()


def set_config(setting, value):
    '''
    Set value of setting
    '''
    _config[setting] = value


CFG_ERR_MESG = f'''
Could not find setting "{{}}" in the configuration file. This may be because
the configuration file is missing. The configuration file was expected to be
found at {get_config_file()}.

You can either create this file manually or run `psi-config` to create it. A
different location for the file can be specified by setting the environment
variable PSI_CONFIG_FILE.
'''

# Special singleton value that enables us to use `get_config` with None as the
# default value. This value is a "flag" that, if used, indicates that the user
# did not provide a default value.
NoDefault = object()


def get_config(setting=None, default_value=NoDefault):
    '''
    Get value of setting
    '''
    if setting is not None:
        try:
            if default_value != NoDefault:
                return _config.get(setting, default_value)
            else:
                return _config[setting]
        except KeyError as e:
            if CONFIG_LOADED:
                raise
            mesg = CFG_ERR_MESG.strip().format(setting)
            raise SystemError(mesg) from e
    else:
        return _config.copy()


reload_config()

from .version import __version__
