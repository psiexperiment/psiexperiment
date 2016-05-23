import logging
from atom.api import Event

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


def load_config():
    # Load the default settings
    from os import environ
    from . import config

    try:
        # Load the computer-specific settings
        path = environ['PSIEXPERIMENT_SETTINGS']
        import imp
        from os.path import dirname
        extra_settings = imp.load_module('settings', open(path), dirname(path),
                                         ('.py', 'r', imp.PY_SOURCE))
        # Update the setting defaults with the computer-specific settings
        for setting in dir(extra_settings):
            value = getattr(extra_settings, setting)
            setattr(config, setting, value)
    except KeyError:
        log.debug('No PSIEXPERIMENT_SETTINGS defined')
    except IOError:
        log.debug('%s file defined by PSIEXPERIMENT_SETTINGS is missing', path)
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
