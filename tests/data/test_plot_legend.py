'''
Tests for the entries shown in a plot container's legend.
'''
import enaml
import pytest
from enaml.qt.QtCore import QRectF
from enaml.qt.QtWidgets import QApplication

with enaml.imports():
    # tests/data is not a package, so import by bare name.
    from plot_legend_helper import LegendContainer


@pytest.fixture
def container(app):
    pc = LegendContainer()
    # As done by manifest registration and then view creation.
    pc.initialize()
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
        # _update_container runs on every refresh of the plot extension
        # point, and used to re-add every plot each time.
        container._update_container()
        container._update_container()
        pump()
        assert labels(container) == LABELS

    def test_refresh_does_not_duplicate_plots(self, container):
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


class TestPlacement:

    def test_legend_above_plot_without_overlap(self, container):
        container.container.setGeometry(QRectF(0, 0, 600, 400))
        pump()
        viewbox = container.viewboxes[0]
        legend = container.legend.sceneBoundingRect()
        assert legend.height() > 0
        assert legend.bottom() <= viewbox.viewbox.sceneBoundingRect().top()
        assert legend.bottom() <= viewbox.y_axis.sceneBoundingRect().top()

    def test_columns_capped_at_four(self, container):
        viewbox = container.viewboxes[0]
        assert container.legend.columnCount == 4
        viewbox.remove_plot(viewbox.children[0].plot)
        assert container.legend.columnCount == 3

    def test_removal_does_not_leave_holes(self, container):
        viewbox = container.viewboxes[0]
        viewbox.remove_plot(viewbox.children[1].plot)
        layout = container.legend.layout
        labels = [layout.itemAt(0, col).text for col in (1, 3, 5)]
        assert labels == ['F2', 'Noise Floor', 'DPOAE']

    def test_empty_legend_takes_no_space(self, container):
        viewbox = container.viewboxes[0]
        for child in viewbox.children:
            viewbox.remove_plot(child.plot)
        assert not container.legend.isVisible()
        assert container.legend.maximumHeight() == 0


def test_marker_matches_plot_before_data(app):
    # The DPOAE plots register (and join the legend) before any data arrives.
    import pyqtgraph as pg
    from psi.data.plots import ContainerLegend, ResultPlot
    plot = ResultPlot(color='indianred', symbol='square', symbol_size=12).plot
    ContainerLegend().addItem(plot, 'F2')
    scatter = plot.scatter.opts
    assert scatter['brush'].color().name() == pg.mkColor('indianred').name()
    assert scatter['pen'].color().name() == pg.mkColor('indianred').name()
    assert scatter['symbol'] == 's'
    assert scatter['size'] == 12
