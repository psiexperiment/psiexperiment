from pathlib import Path

from enaml.icon import Icon, IconImage
from enaml.image import Image

from psi import get_config


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
    path = Path(__file__).parent / 'psi-logo.png'
    image = Image(data=path.read_bytes())
    icon_image = IconImage(image=image)
    return Icon(images=[icon_image])


main_icon = load_icon()
