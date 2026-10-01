# This import automatically adds the TRACE logging level
import psiaudio

import logging
from pathlib import Path

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


from .config import (  # noqa: E402,F401
    NoDefault, Setting, config_source, create_config_dirs, get_all_config,
    get_config, get_config_file, get_setting, load_config, parse_setting,
    register_defaults, reload_config, save_config, setting_names,
    setting_type
)
from .config_defaults import DEFAULTS as _PSI_DEFAULTS  # noqa: E402
from .runtime import get_runtime, set_runtime  # noqa: E402,F401

register_defaults(_PSI_DEFAULTS)


def create_config(base_directory=None, **roots):
    '''
    Write a new configuration file.

    Only ``PSI_BASE_DIRECTORY`` is written unless a root is overridden
    explicitly. The other roots derive from it in
    :mod:`psi.config_defaults`, so writing them out here would freeze
    today's layout into the file and mean that moving the base directory
    later required editing several lines instead of one.

    Parameters
    ----------
    base_directory : {None, str, Path}
        Root beneath which data, settings and IO manifests live.
    **roots
        Any other setting to write, by its full name (e.g.
        ``PSI_DATA_ROOT='D:/data'``).
    '''
    updates = {}
    if base_directory is not None:
        updates['PSI_BASE_DIRECTORY'] = str(base_directory).rstrip('\\')
    updates.update(roots)
    save_config(updates)


def create_io_manifest(template):
    '''
    Copy one of the shipped IO manifest templates into ``PSI_IO_ROOT``.
    '''
    io_template = Path(__file__).parent / 'templates' / 'io' / template
    io_template = io_template.with_suffix('.enaml')
    io = Path(get_config('PSI_IO_ROOT')) / template.lstrip('_')
    io = io.with_suffix('.enaml')
    io.parent.mkdir(exist_ok=True, parents=True)
    io_text = io_template.read_text()
    io.write_text(io_text)


from .version import __version__  # noqa: E402,F401
