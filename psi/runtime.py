'''
Values the application writes while it runs.

These are not configuration. The experiment name, the log filename, the
parsed command-line arguments and the profiling flag are all decided by
the launcher after the process starts; nobody sets them in a file or the
environment, and setting them there would either do nothing or confuse
the launcher.

They used to live in the same dictionary as the settings, which made that
dictionary two things at once: a user would reasonably conclude from
``get_config('LOG_FILENAME')`` that ``PSI_LOG_FILENAME`` was a variable
they could set, and it is not. Keeping them here means the configuration
system can state without exceptions that every setting has a file
spelling and an environment spelling.
'''
import logging

log = logging.getLogger(__name__)


NoDefault = object()


_runtime = {}


def set_runtime(name, value):
    '''
    Record a runtime value.
    '''
    _runtime[name] = value
    log.debug('Runtime value %s set', name)


def get_runtime(name, default=NoDefault):
    '''
    Read a runtime value.

    Raises
    ------
    KeyError
        If the value has not been set and no default was supplied. This
        means the launcher has not run the step that sets it.
    '''
    if name in _runtime:
        return _runtime[name]
    if default is NoDefault:
        raise KeyError(
            f'{name} has not been set. Runtime values are written by the '
            'launcher while the application starts; they are not settings '
            'and cannot be supplied through the configuration file or the '
            'environment.')
    return default


def clear_runtime():
    '''
    Forget every runtime value. For tests.
    '''
    _runtime.clear()
