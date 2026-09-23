import logging
from importlib import resources
from pathlib import Path

from enaml.icon import Icon, IconImage
from enaml.image import Image

from psi import get_config


log = logging.getLogger(__name__)


PREFERENCES_WILDCARD = 'Preferences (*.preferences)'
LAYOUT_WILDCARD = 'Workspace layout (*.layout)'


def list_preferences(experiment, include_default=False):
    if not isinstance(experiment, str):
        experiment = experiment.name
    p_root = Path(get_config('PREFERENCES_ROOT')) / experiment
    p_glob = PREFERENCES_WILDCARD[:-1].split('(')[1]
    matches = p_root.glob(p_glob)
    if not include_default:
        matches = [p for p in matches if not p.stem == 'default']
    return sorted(Path(p) for p in matches)


def load_icon():
    # Use importlib.resources rather than a `__file__`-relative path so this
    # keeps working if psi is ever installed/accessed via a loader that
    # doesn't expose package contents as plain files on disk (e.g., a zipped
    # install). This does *not* by itself guarantee the file is bundled in a
    # frozen (PyInstaller) build -- that's handled by psi/__pyinstaller's
    # hook, which declares this data file to PyInstaller's static analysis.
    # If, despite that, the icon is missing at runtime, fail soft: a missing
    # window icon is cosmetic and shouldn't prevent the app from starting.
    try:
        data = resources.files('psi.experiment').joinpath('psi-logo.png').read_bytes()
    except (FileNotFoundError, ModuleNotFoundError):
        log.warning('Unable to load window icon psi-logo.png', exc_info=True)
        return Icon(images=[])
    image = Image(data=data)
    icon_image = IconImage(image=image)
    return Icon(images=[icon_image])


main_icon = load_icon()


def set_application_icon(icon=None):
    '''
    Make an icon the default for every window of the running application.

    Setting the icon on the main window alone is not enough. A dialog built
    without a parent -- such as the popup reporting how an experiment ended,
    or the error shown when an experiment fails to start -- is a top-level
    window with a taskbar button of its own, and without an application-wide
    default it shows Qt's generic window icon. On Windows the taskbar can
    then show that generic icon for the whole group of psi windows.

    Parameters
    ----------
    icon : enaml Icon, optional
        The icon to use. Defaults to `main_icon`, psi's own icon.

    Notes
    -----
    Call once the Qt application exists (in psi, as soon as the enaml UI
    plugin has started). Does nothing if there is no application yet.
    Windows that set their own icon keep it.
    '''
    from enaml.qt.QtWidgets import QApplication
    from enaml.qt.q_resource_helpers import get_cached_qicon

    app = QApplication.instance()
    if app is None:
        return
    app.setWindowIcon(get_cached_qicon(main_icon if icon is None else icon))
