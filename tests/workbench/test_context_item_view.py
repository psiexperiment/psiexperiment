'''
Tests for how a context item is drawn (`psi/context/context_item_view.enaml`).

An item that cannot be edited -- because its value was set for the user
(e.g., from an environment variable) or because the experiment is
running -- must gray out the widget that would change the value, but not
the label naming it.
'''
import enaml
import pytest
from enaml.widgets.api import Label

from psi.context.api import ContextRow, EnumParameter, Parameter

with enaml.imports():
    from .context_item_view_helper import (ContextItemViewHarness,
                                           ContextItemViewWindow)


def render(workbench, context_item):
    '''
    Build the view for one context item and return its Qt widgets.
    '''
    harness = ContextItemViewHarness(workbench=workbench,
                                     context_item=context_item)
    harness.initialize()
    harness.activate_proxy()
    return harness


def find(harness, widget_type):
    return [w for w in harness.traverse() if isinstance(w, widget_type)]


def label_widget(harness):
    return find(harness, Label)[0].proxy.widget


def value_widget(harness, widget_type):
    return find(harness, widget_type)[0].proxy.widget


@pytest.fixture
def editable_item():
    return Parameter(name='level', label='Level (dB SPL)', dtype='float64',
                     default=60, scope='experiment')


@pytest.fixture
def enum_item():
    return EnumParameter(name='microphone', label='Microphone',
                         choices={'mic_1': '"mic_1"'}, scope='experiment')


class TestReadOnlyItem:

    def test_label_stays_enabled(self, workbench, app, editable_item):
        from enaml.widgets.api import Field
        editable_item.editable = False
        harness = render(workbench, editable_item)
        # isEnabled() is the effective state, so this also catches the
        # label being disabled by an ancestor.
        assert label_widget(harness).isEnabled()
        assert not value_widget(harness, Field).isEnabled()

    def test_enum_dropdown_is_disabled(self, workbench, app, enum_item):
        from enaml.widgets.api import ObjectCombo
        enum_item.editable = False
        harness = render(workbench, enum_item)
        assert label_widget(harness).isEnabled()
        assert not value_widget(harness, ObjectCombo).isEnabled()


class TestEditableItem:

    def test_everything_enabled(self, workbench, app, editable_item):
        from enaml.widgets.api import Field
        harness = render(workbench, editable_item)
        assert label_widget(harness).isEnabled()
        assert value_widget(harness, Field).isEnabled()

    def test_value_locked_while_experiment_runs(self, workbench, app,
                                                editable_item):
        # An experiment-scope value cannot change mid-experiment, but the
        # label should stay readable while it runs.
        from enaml.widgets.api import Field
        controller = workbench.get_plugin('psi.controller')
        harness = render(workbench, editable_item)
        controller.experiment_state = 'running'
        assert label_widget(harness).isEnabled()
        assert not value_widget(harness, Field).isEnabled()


class TestRowFieldWidth:
    '''
    In a ContextRow, each value's field is `field_width` pixels wide, 45 unless
    the item says otherwise.
    '''

    def field_widths(self, workbench, app, row):
        from enaml.qt.QtWidgets import QApplication
        from enaml.widgets.api import Field
        window = ContextItemViewWindow(workbench=workbench, context_item=row)
        window.show()
        try:
            QApplication.processEvents()
            return [f.proxy.widget.width() for f in window.traverse()
                    if isinstance(f, Field)]
        finally:
            window.close()

    def test_default_width(self, workbench, app):
        level = Parameter(name='level', default=60, scope='experiment')
        row = ContextRow(name='row', fmt=['Level', level, 'dB SPL'])
        assert level.field_width == 45
        assert self.field_widths(workbench, app, row) == [45]

    def test_item_width(self, workbench, app):
        duration = Parameter(name='duration', default=7200, field_width=90,
                             scope='experiment')
        ramp = Parameter(name='ramp', default=1, scope='experiment')
        row = ContextRow(name='row', fmt=['Duration', duration, 's with',
                                          ramp, 's ramp'])
        assert self.field_widths(workbench, app, row) == [90, 45]

    def test_width_not_saved(self):
        '''
        The width is layout, not a setting: saved preferences don't change
        with it.
        '''
        default = Parameter(name='a', default=1)
        wide = Parameter(name='a', default=1, field_width=90)
        assert wide.get_preferences() == default.get_preferences()
