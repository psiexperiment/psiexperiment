'''
Configuration file

Variable names in capital letters indicate that this is a setting that can be
overridden in a custom settings.py file that PSIEXPERIMENT_CONFIG environment
variable points to.
'''
import os, re, logging, tempfile, socket
log = logging.getLogger(__name__)

SYSTEM = socket.gethostname()

try:
    BASE_DIRECTORY  = os.environ['PSIEXPERIMENT_BASE']
except KeyError:
    import warnings
    import textwrap
    # Default to the user's home directory and raise a warning.
    BASE_DIRECTORY = os.path.join(os.path.expanduser('~'), '.config',
                                  'psi')
    mesg = '''
    No PSIEXPERIMENT_BASE environment variable defined.  Defaulting to the
    user's home directory, {}.  In the future, it is recommended that you create
    a base directory where the paradigm settings, calibration data, log files
    and data files can be stored.  Once this directory is created, create the
    environment variable, PSIEXPERIMENT_BASE, with the path to the directory as
    the value.'''
    mesg = textwrap.dedent(mesg.format(BASE_DIRECTORY))
    mesg = mesg.replace('\n', ' ').strip()
    log.warn(mesg)
    print(mesg)


LOG_ROOT = os.path.join(BASE_DIRECTORY, 'logs')
DATA_ROOT = os.path.join(BASE_DIRECTORY, 'data')
CAL_ROOT = os.path.join(BASE_DIRECTORY, 'calibration')
LAYOUT_ROOT = os.path.join(BASE_DIRECTORY, 'layout')
PREFERENCES_ROOT = os.path.join(BASE_DIRECTORY, 'preferences')
TEMP_ROOT = os.path.join(BASE_DIRECTORY, 'temp')

# Ensure the folders exist
for setting_name, setting_value in globals().items():
    if setting_name.endswith('_ROOT') and not os.path.exists(setting_value):
        os.makedirs(setting_value)

# Default filename extensions used by the FileBrowser dialog to open/save files.
PREFERENCES_WILDCARD = 'Preferences (*.preferences)'
LAYOUT_WILDCARD = 'Workspace layout (*.layout)'
CAL_WILDCARD = 'Cal (*.cal)'

# Options for pump syringe
SYRINGE_DEFAULT = 'Popper 20cc (glass)'
SYRINGE_DATA = {
        'B-D 10cc (plastic)'    : 14.43,
        'B-D 20cc (plastic)'    : 19.05,
        'B-D 30cc (plastic)'    : 21.59,
        'B-D 60cc (plastic)'    : 26.59,
        'Popper 20cc (glass)'   : 19.58,
        'B-D 10cc (glass)'      : 14.20,
        }

H5_COMPRESSION = {
    'complevel': 1,
    'complib': 'blosc',
    'shuffle': True,
    'fletcher32': True,
}

# Log the settings to facilitate debugging. By convention, settings are in all
# caps, so only log those variables.
log = logging.getLogger()
for k, v in sorted(globals().items()):
    if k == k.upper():
        log.debug("CONFIG %s : %r", k, v)

