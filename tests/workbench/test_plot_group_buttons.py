import enaml
import pytest
from enaml.qt.QtWidgets import QPushButton

from psi.data.plots import PlotContainer

with enaml.imports():
    from psi.data.plots_manifest import PlotGroupSelectButtons


@pytest.fixture
def container():
    pc = PlotContainer(name='container', fmt_button_cb=lambda key, sep: str(key[0]))
    pc.buttons = [(1000,), (2000,)]
    return pc


def build(container):
    view = PlotGroupSelectButtons(contribution=container)
    view.initialize()
    view.activate_proxy()
    return {b.text(): b for b in view.proxy.widget.findChildren(QPushButton)}


def test_clicking_current_button_keeps_it_selected(app, container):
    container.auto_select = True
    container.current_button = (1000,)
    buttons = build(container)

    buttons['1000'].click()
    assert not container.auto_select
    assert container.current_button == (1000,)
    assert buttons['1000'].isChecked()
    assert not buttons['2000'].isChecked()


def test_clicking_other_button_selects_it(app, container):
    container.auto_select = True
    container.current_button = (1000,)
    buttons = build(container)

    buttons['2000'].click()
    assert not container.auto_select
    assert container.current_button == (2000,)
    assert buttons['2000'].isChecked()
    assert not buttons['1000'].isChecked()
