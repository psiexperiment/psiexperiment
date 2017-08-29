""" Enaml widget for editing a list of string
"""

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------
from atom.api import (Bool, List, observe, set_default, Unicode, Enum, Int, Typed)

from enaml.widgets.api import RawWidget
from enaml.core.declarative import d_, d_func
from enaml.qt.QtWidgets import QListWidget, QListWidgetItem, QAbstractItemView
from enaml.qt.QtCore import *


class DeleteFilter(QObject):

    def __init__(self, widget, *args, **kwargs):
        self.widget = widget
        super().__init__(*args, **kwargs)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress and event.key() == Qt.Key_Delete:
            self.widget.remove_item()
            return True
        else:
            return super().eventFilter(obj, event)


class List(RawWidget):

    __slots__ = '__weakref__'

    # The list of str being viewed
    items = d_(List())
    
    #: Whether or not the items should be editable
    editable = d_(Bool(True))

    #hug_width = set_default('weak')

    delete_filter = Typed(QObject)
    
    def create_widget(self, parent):
        # Create the list model and accompanying controls:
        widget = QListWidget(parent)
        self.set_items(self.items, widget)
        widget.itemChanged.connect(self.on_edit)

        widget.setEditTriggers(QAbstractItemView.AnyKeyPressed)
        widget.setSelectionMode(QAbstractItemView.ExtendedSelection)

        self.delete_filter = DeleteFilter(self)
        widget.installEventFilter(self.delete_filter)

        return widget

    def eventFilter(self, source, event):
        print('event filter occured')
        return super().eventFilter(source, event)

    def add_item(self, widget, item):
        text = self.to_view(item)
        itemWidget = QListWidgetItem(text)
        if self.editable:
            flags = itemWidget.flags() | Qt.ItemIsEditable
            itemWidget.setFlags(flags)

        current_item = widget.currentItem()
        if current_item is not None:
            current_item.setSelected(False)
        widget.addItem(itemWidget)
        widget.setCurrentItem(itemWidget)
        return itemWidget

    def remove_item(self, widget=None, item=None):
        widget = self.get_widget()
        item_widgets = widget.selectedItems()
        rows = [widget.row(item) for item in item_widgets]
        items = self.items[:]
        for row in sorted(rows, reverse=True):
            del items[row]
        self.items = items

    def on_edit(self, item):
        """ The signal handler for the item changed signal.
        """
        widget = self.get_widget()
        value = self.from_view(item.text())
        try:
            self.items[widget.currentRow()] = value
        except IndexError:
            self.items.append(value)
            item = self.add_item(widget, '')

    def set_items(self, items, widget=None):
        if widget is None:
            widget = self.get_widget()
        count = widget.count()
        nitems = len(items)
        for idx, item in enumerate(items[:count]):
            itemWidget = widget.item(idx)
            itemWidget.setText(self.to_view(item))
        if nitems > count:
            for item in items[count:]:
                self.add_item(widget, item)
        elif nitems < count:
            for idx in reversed(range(nitems, count)):
                w = widget.takeItem(idx)
                del w
        self.add_item(widget, '')

    @observe('items')
    def _update_proxy(self, change):
        name = change['name']
        if self.get_widget():
            if name == 'items':
                self.set_items(self.items)       

    @d_func
    def to_view(self, value):
        return str(value)

    @d_func
    def from_view(self, text):
        return text
