from atom.api import (Bool, List, observe, set_default, Unicode, Enum, Int, Typed, Event)

from enaml.widgets.api import RawWidget
from enaml.core.declarative import d_, d_func
from enaml.qt.QtWidgets import QListWidget, QListWidgetItem, QAbstractItemView
from enaml.qt.QtCore import Qt
from .event_filter import EventFilter


class ListView(RawWidget):

    hug_width = set_default('weak')
    hug_height = set_default('weak')

    __slots__ = '__weakref__'

    # The list being edited by the widget
    items = d_(List())
    selected_rows = d_(List())
    selected_items = d_(List())

    # Whether or not the items should be editable
    editable = d_(Bool(True))
    updated = d_(Event())

    # Filter for capturing the delete key
    _event_filter = Typed(EventFilter)

    selection_mode = Enum('single', 'contiguous', 'extended', 'multi', None)

    SELECTION_MODE_MAP = {
        'single': QAbstractItemView.SingleSelection,
        'contiguous': QAbstractItemView.ContiguousSelection,
        'extended': QAbstractItemView.ExtendedSelection,
        'multi': QAbstractItemView.MultiSelection,
        None: QAbstractItemView.NoSelection,
    }

    def create_widget(self, parent):
        # Create the list model and accompanying controls:
        widget = QListWidget(parent)
        widget.itemChanged.connect(self.on_edit)
        widget.currentItemChanged.connect(self._selected)
        widget.setEditTriggers(QAbstractItemView.AnyKeyPressed)
        mode = self.SELECTION_MODE_MAP[self.selection_mode]
        widget.setSelectionMode(mode)
        self._event_filter = EventFilter(self)
        widget.installEventFilter(self._event_filter)
        self.set_items(self.items, widget, select_first=True)
        return widget

    def get_selected_rows(self):
        widget = self.get_widget()
        return [widget.row(wi) for wi in widget.selectedItems()]

    def _selected(self, selected=None, deselected=None):
        self.selected_rows = self.get_selected_rows()
        self.selected_items = [self.items[r] for r in self.selected_rows]

    def add_item(self, item='', widget=None):
        if widget is None:
            widget = self.get_widget()
        text = self.to_string(item)
        wi = QListWidgetItem(text)
        if self.editable:
            flags = wi.flags() | Qt.ItemIsEditable
            wi.setFlags(flags)
        widget.addItem(wi)
        return wi

    def select_next(self):
        widget = self.get_widget()
        row = widget.currentRow()
        prev_wi = widget.item(row)
        next_wi = widget.item(row+1)
        if next_wi is not None:
            prev_wi.setSelected(False)
            next_wi.setSelected(False)
            widget.setCurrentItem(next_wi)

    def remove_selected_rows(self):
        widget = self.get_widget()
        rows = self.get_selected_rows()
        for row in sorted(rows, reverse=True):
            if row < len(self.items):
                del self.items[row]
                widget.takeItem(row)
        self.updated = True

    def on_edit(self, item, widget=None):
        if widget is None:
            widget = self.get_widget()
        value = self.from_string(item.text())
        row = widget.currentRow()
        try:
            self.items[row] = value
            next_wi = widget.item(row+1)
        except IndexError:
            self.items.append(value)
            self.add_item('', widget)
        self.select_next()
        self.updated = True

    def set_items(self, items, widget=None, select_first=False):
        if widget is None:
            widget = self.get_widget()
        widget.clear()
        widget_items = [self.add_item(i, widget) for i in items]
        if self.editable:
            self.add_item('', widget)
        if select_first:
            widget_items[0].setSelected(True)
            self._selected()

    @observe('items')
    def _update_proxy(self, change):
        name = change['name']
        if name == 'items':
            if self.get_widget():
                self.set_items(self.items[:])

    @observe('updated')
    def _update_items(self, change):
        self.items = self.items[:]

    @d_func
    def to_string(self, value):
        return str(value)

    @d_func
    def from_string(self, text):
        return text
