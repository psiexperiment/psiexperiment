from pathlib import Path

from enaml.icon import Icon, IconImage
from enaml.image import Image


def load_icon():
    path = Path(__file__).parent / 'psi-logo.png'
    image = Image(data=path.read_bytes())
    icon_image = IconImage(image=image)
    return Icon(images=[icon_image])


main_icon = load_icon()
