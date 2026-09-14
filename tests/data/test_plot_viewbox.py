import enaml
from enaml.qt.QtWidgets import QApplication

with enaml.imports():
    # tests/data is not a package, so import by bare name.
    from plot_legend_helper import LegendContainer


def build(**viewbox_attrs):
    pc = LegendContainer()
    viewbox = pc.viewboxes[0]
    for name, value in viewbox_attrs.items():
        setattr(viewbox, name, value)
    pc.initialize()
    pc.container
    for _ in range(10):
        QApplication.processEvents()
    return pc, viewbox


def test_y_axis_linked_to_viewbox(app):
    # The axis forwards scroll and drag to its linked viewbox, so an
    # unlinked axis can no longer zoom or pan.
    pc, viewbox = build()
    assert viewbox.y_axis.linkedView() is viewbox.viewbox


def test_y_axis_linked_when_limits_set_before_viewbox_exists(app):
    pc, viewbox = build(y_min=-10, y_max=10)
    assert viewbox.y_axis.linkedView() is viewbox.viewbox
    assert viewbox.viewbox.viewRange()[1] == [-10, 10]


def test_y_axis_still_linked_after_refresh(app):
    pc, viewbox = build()
    pc._update_container()
    assert viewbox.y_axis.linkedView() is viewbox.viewbox
