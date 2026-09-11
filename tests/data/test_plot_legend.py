'''
Tests for what ends up in a plot container's legend.

`ViewBox._configure_viewbox` runs again every time the `psi.data.plots`
extension point changes (`DataPlugin._refresh_plots` calls
`_update_container` for every container), and plots get reset whenever
the data they group by changes. Neither `pg.ViewBox.addItem` nor
`pg.LegendItem.addItem` ignores an item it already holds, and
`LegendItem` has no way to notice that a plot it lists has been dropped,
so both paths have to be handled here.
'''
import enaml
import pytest
from enaml.qt.QtWidgets import QApplication

with enaml.imports():
    # tests/data is not a package, so pytest puts this directory on
    # sys.path and the helper imports by bare name.
    from plot_legend_helper import LegendContainer


@pytest.fixture
def container(app):
    pc = LegendContainer()
    # Building the Qt layout is what the dock item does when the view is
    # created, and is the first of the two paths that add plots.
    pc.container
    pump()
    return pc


def pump():
    for _ in range(10):
        QApplication.processEvents()


def labels(container):
    return [label.text for _, label in container.legend.items]


LABELS = ['F2', 'F1', 'Noise Floor', 'DPOAE']


class TestRefresh:

    def test_plots_are_listed_once(self, container):
        assert labels(container) == LABELS

    def test_refresh_does_not_duplicate_entries(self, container):
        # Each refresh used to add a full set of entries, so an experiment
        # whose plots are contributed by three manifests showed every
        # label three times.
        container._update_container()
        container._update_container()
        pump()
        assert labels(container) == LABELS

    def test_refresh_does_not_duplicate_plots(self, container):
        # The same duplication in the viewbox is invisible -- each copy is
        # drawn on top of itself -- but every redraw does the work N times.
        container._update_container()
        pump()
        assert len(container.viewboxes[0].viewbox.addedItems) == len(LABELS)


class TestAddRemove:

    def test_removing_a_plot_removes_its_entry(self, container):
        viewbox = container.viewboxes[0]
        plot = viewbox.children[0].plot

        viewbox.remove_plot(plot)
        assert labels(container) == LABELS[1:]

    def test_a_removed_plot_can_be_added_again(self, container):
        viewbox = container.viewboxes[0]
        plot = viewbox.children[0].plot

        viewbox.remove_plot(plot)
        viewbox.add_plot(plot, 'F2')
        assert labels(container) == LABELS[1:] + ['F2']
