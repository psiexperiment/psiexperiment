import logging
from pathlib import Path

from atom.api import Event


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


def get_config_path():
    user_path = Path('~') / 'psi' / 'config.py'
    return user_path.expanduser()


def create_config(base_directory=None):
    config_template = Path(__file__).parent / 'templates' / 'config.txt'
    target = get_config_path()
    target.parent.mkdir(exist_ok=True, parents=True)

    if base_directory is None:
        base_directory = str(target.parent)

    with open(target, 'w') as fh:
        config_text = config_template.read_text()
        config_text = config_text.format(base_directory)
        fh.write(config_text)


def create_config_dirs():
    config = load_config()
    for name, value in vars(config).items():
        if name.endswith('_ROOT'):
            Path(value).mkdir(exist_ok=True, parents=True)


def load_config():
    # Load the default settings
    import importlib.util
    from os import environ
    from . import config

    config_path = get_config_path()
    if config_path.exists():
        try:
            spec = importlib.util.spec_from_file_location('settings', config_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for name, value in vars(module).items():
                if name == name.upper():
                    setattr(config, name, value)
        except Exception as e:
            log.exception(e)

    for name, value in vars(config).items():
        if name == name.upper():
            log.debug('CONFIG %s : %r', name, value)

    return config

_config = load_config()


def set_config(setting, value):
    '''
    Set value of setting
    '''
    setattr(_config, setting, value)


def get_config(setting=None):
    '''
    Get value of setting
    '''
    if setting is not None:
        return getattr(_config, setting)
    else:
        setting_names = [s for s in dir(_config) if s.upper() == s]
        setting_values = [getattr(_config, s) for s in setting_names]
        return dict(zip(setting_names, setting_values))


# Monkeypatch built-in JSON library to better handle special types. The
# json-tricks library handles quite a few different types of Python objects
# fairly well. This ensures that third-party libraries (e.g., bcolz) that see
# psiexperiment data structures can properly deal with them.
import json
import json_tricks

for fn_name in ('dump', 'dumps', 'load', 'loads'):
    fn = getattr(json_tricks, fn_name)
    setattr(json, fn_name, fn)
log.debug('Monkeypatched system JSON')
