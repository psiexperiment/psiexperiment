#----------------------------------------------------------------------------
#  Adapted from BSD-licensed module used by Enthought, Inc.
#----------------------------------------------------------------------------

# TODO - make this faster for simple row append. Do not redraw the entire
# dataframe on each iteration. For now, it's fast enough.

import numpy as np
import pandas as pd

from atom.api import (Typed, set_default, observe, Value, Event, Property,
                      ContainerList, Bool, Signal)
from atom.api import Signal as AtomSignal
from enaml.core.declarative import d_, d_func
from enaml.widgets.api import RawWidget, Menu

from enaml.qt.QtCore import QAbstractTableModel, QModelIndex, Qt, Slot, Signal
from enaml.qt.QtWidgets import QTableView, QHeaderView, QAbstractItemView, QItemDelegate, QMenu
from enaml.qt.QtGui import QFont, QColor


class QDelegate(QItemDelegate):

    def setEditorData(self, editor, index):
        text = index.data(Qt.EditRole) or index.data(Qt.DisplayRole)
        editor.setText(text)


class QEditableTableModel(QAbstractTableModel):

    data_changed = Signal(int, str)

    def __init__(self, data, columns, **kw):
        self.data = data
        self.columns = columns
        super().__init__(**kw)

    def headerData(self, section, orientation, role):
        if role == Qt.TextAlignmentRole:
            if orientation == Qt.Horizontal:
                return int(Qt.AlignHCenter | Qt.AlignVCenter)
            return int(Qt.AlignRight | Qt.AlignVCenter)
        elif role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return self.columns[section]
            else:
                return str(section+1)

    def flags(self, index):
        return Qt.ItemIsEnabled | Qt.ItemIsEditable | Qt.ItemIsSelectable

    def data(self, index, role):
        if role != Qt.DisplayRole:
            return
        if not index.isValid():
            return
        if not (0 <= index.row() < self.rowCount()):
            return

        r = index.row()
        c = self.columns[index.column()]
        return str(self.data[r][c])

    def setData(self, index, value, role):
        r = index.row()
        c = self.columns[index.column()]
        self.data[r][c] = value
        self.dataChanged.emit(index, index)
        self.data_changed.emit(r, c)
        return True

    def columnCount(self, index=QModelIndex()):
        return len(self.columns)

    def rowCount(self, index=QModelIndex()):
        return len(self.data) if self.data is not None else 0


class QEditableTableView(QTableView):

    def __init__(self, model, parent=None, **kwds):
        super().__init__(parent=parent, **kwds)
        self.model = model
        self.setModel(model)
        self._setup_scrolling()
        self._setup_hheader()
        self._setup_vheader()
        self.setSelectionBehavior(self.SelectRows)

    def _setup_scrolling(self):
        self.setVerticalScrollMode(QAbstractItemView.ScrollPerItem)
        self.setHorizontalScrollMode(QAbstractItemView.ScrollPerItem)

    def _setup_vheader(self):
        self.vheader = QHeaderView(Qt.Vertical)
        self.setVerticalHeader(self.vheader)
        self.vheader.setSectionResizeMode(QHeaderView.Fixed)
        self.vheader.setDefaultSectionSize(20)
        self.vheader.setSectionsMovable(False)

    def set_vheader_menu(self, menu):
        if menu is not None:
            self._vheader_menu = menu
            self.vheader.setContextMenuPolicy(Qt.CustomContextMenu)
            self.vheader.customContextMenuRequested.connect(self._show_vheader_menu)

    def _setup_hheader(self):
        self.hheader = self.horizontalHeader()
        self.hheader.setSectionsMovable(False)

    def _show_vheader_menu(self, point):
        #self._vheader_menu.popup(self.vheader.mapToGlobal(point))
        self._vheader_menu.popup()
        #row = self.vheader.logicalIndexAt(point.y())
        #menu = QMenu(self)
        #menu.addAction('Remove setting')
        #menu.addAction('Insert setting before')
        #menu.addAction('Insert setting after')
        #menu.popup(self.vheader.mapToGlobal(point))


class EditableTable(RawWidget):

    # Expand the table by default
    hug_width = set_default('weak')
    hug_height = set_default('weak')

    data = d_(Typed(list))
    columns = d_(Typed(list))
    updated = d_(AtomSignal())

    data_changed = d_(Event())

    model = Typed(QEditableTableModel)
    view = Typed(QEditableTableView)

    row_menu = d_(Typed(Menu))

    def create_widget(self, parent):
        self.model = QEditableTableModel(self.data, self.columns)
        self.view = QEditableTableView(self.model, parent=parent)
        self.view.set_vheader_menu(self.row_menu)
        delegate = QDelegate()
        self.view.setItemDelegate(delegate)
        self.model.data_changed.connect(self._data_changed)
        return self.view

    @Slot(int, str)
    def _data_changed(self, row, column):
        print('data changed', row, column)

    def _observe_data(self, event):
        if self.model is None:
            return
        self.model.data = event['value']
        self._refresh_view()

    def _observe_columns(self, event):
        if self.model is None:
            return
        self.model.columns = event['value']
        self._refresh_view()

    def _observe_updated(self):
        print('observed updated')

    def _refresh_view(self):
        #self.model.dataChanged.emit(QModelIndex(), QModelIndex())
        self.model.layoutChanged.emit()
