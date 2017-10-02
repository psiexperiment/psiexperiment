from atom.api import (Bool, List, observe, set_default, Unicode, Enum, Int, Typed, Event)

from enaml.widgets.api import RawWidget
from enaml.core.declarative import d_, d_func
from enaml.qt.QtWidgets import QListWidget, QListWidgetItem, QAbstractItemView
from enaml.qt.QtCore import *


class DeleteFilter(QObject):

    def __init__(self, widget, *args, **kwargs):
        self.widget = widget
        super().__init__(*args, **kwargs)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Delete:
                self.widget.remove_selected_items()
                return True
        return super().eventFilter(obj, event)


class ListView(RawWidget):

    __slots__ = '__weakref__'

    # The list being edited by the widget
    items = d_(List())

    # Whether or not the items should be editable
    editable = d_(Bool(True))

    updated = d_(Event())

    hug_width = set_default('weak')
    hug_height = set_default('weak')

    # Filter for capturing the delete key
    _delete_filter = Typed(QObject)
    _widget = Typed(QObject)

    def create_widget(self, parent):
        # Create the list model and accompanying controls:
        #self.set_items(self.items)
        #widget = QListWidget(parent)
        view = QListView(parent)
        widget.itemChanged.connect(self.on_edit)
        widget.setEditTriggers(QAbstractItemView.AnyKeyPressed)
        widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._delete_filter = DeleteFilter(self)
        widget.installEventFilter(self._delete_filter)
        self.set_items(self.items, widget)
        return widget

    def add_item(self, item='', widget=None):
        if widget is None:
            widget = self.get_widget()
        text = self.to_string(item)
        wi = QListWidgetItem(text)
        if self.editable:
            flags = wi.flags() | Qt.ItemIsEditable
            wi.setFlags(flags)
        widget.addItem(wi)

    def select_next(self):
        widget = self.get_widget()
        row = widget.currentRow()
        prev_wi = widget.item(row)
        next_wi = widget.item(row+1)
        if next_wi is not None:
            prev_wi.setSelected(False)
            next_wi.setSelected(False)
            widget.setCurrentItem(next_wi)

    def remove_selected_items(self):
        widget = self.get_widget()
        rows = [widget.row(wi) for wi in widget.selectedItems()]
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

    def set_items(self, items, widget=None):
        if widget is None:
            widget = self.get_widget()
        widget.clear()
        for item in items:
            self.add_item(item, widget)
        self.add_item('', widget)

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
