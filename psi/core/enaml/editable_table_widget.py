#----------------------------------------------------------------------------
#  Adapted from BSD-licensed module used by Enthought, Inc.
#----------------------------------------------------------------------------

# TODO - make this faster for simple row append. Do not redraw the entire
# dataframe on each iteration. For now, it's fast enough.

import numpy as np
import pandas as pd

from atom.api import (Typed, set_default, observe, Value, Event, Property,
                      ContainerList, Bool, Signal, List, Dict, Unicode)
from atom.api import Signal as AtomSignal
from enaml.core.declarative import d_, d_func
from enaml.widgets.api import RawWidget, Menu

from enaml.qt.QtCore import QAbstractTableModel, QModelIndex, Qt, Slot, Signal, QObject, QEvent
from enaml.qt.QtWidgets import QTableView, QHeaderView, QAbstractItemView, QItemDelegate, QMenu
from enaml.qt.QtGui import QFont, QColor


class DeleteFilter(QObject):

    def __init__(self, widget, *args, **kwargs):
        self.widget = widget
        super().__init__(*args, **kwargs)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Delete:
                self.widget.remove_selected_rows()
                return True
            if event.key() == Qt.Key_Plus:
                if (event.modifiers() & Qt.ControlModifier):
                    self.widget.insert_row()
                    return True
        return super().eventFilter(obj, event)


class QEditableTableModel(QAbstractTableModel):

    def __init__(self, interface, **kw):
        self.interface = interface
        super().__init__(**kw)

    def headerData(self, section, orientation, role):
        if role == Qt.TextAlignmentRole:
            if orientation == Qt.Horizontal:
                return int(Qt.AlignHCenter | Qt.AlignVCenter)
            return int(Qt.AlignRight | Qt.AlignVCenter)
        elif role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self.interface.get_column_label(section))
            else:
                return str(self.interface.get_row_label(section))

    def flags(self, index):
        flags = Qt.ItemIsEnabled
        if self.interface.editable:
            flags = flags | Qt.ItemIsEditable | Qt.ItemIsSelectable
        return flags

    def data(self, index, role):
        if not index.isValid():
            return
        if role == Qt.BackgroundRole:
            r = index.row()
            c = index.column()
            color_name = self.interface.get_cell_color(r, c)
            color = QColor()
            color.setNamedColor(color_name)
            return color
        elif role in (Qt.DisplayRole, Qt.EditRole):
            r = index.row()
            c = index.column()
            return self.interface.get_data(r, c)

    def setData(self, index, value, role):
        r = index.row()
        c = index.column()
        self.interface.set_data(r, c, value)
        self.dataChanged.emit(index, index)
        return True

    def removeRows(self, row, count, index):
        self.beginRemoveRows(index, row, row)
        self.interface.remove_row(row)
        self.endRemoveRows()
        return True

    def insertRows(self, row, count, index):
        self.beginInsertRows(index, row, row)
        self.interface.insert_row(row)
        self.endInsertRows()
        return True

    def columnCount(self, index=QModelIndex()):
        if self.interface is None:
            return 0
        return len(self.interface.get_columns())

    def rowCount(self, index=QModelIndex()):
        if self.interface is None:
            return 0
        return len(self.interface.get_rows())


class QEditableTableView(QTableView):

    def __init__(self, model, parent=None, **kwds):
        super().__init__(parent=parent, **kwds)
        self.model = model
        self.setModel(model)
        self._setup_hheader()
        self._setup_vheader()
        self.setSelectionBehavior(self.SelectRows)
        self.setVerticalScrollMode(QAbstractItemView.ScrollPerItem)
        self.setHorizontalScrollMode(QAbstractItemView.ScrollPerItem)

    def _setup_vheader(self):
        self.vheader = QHeaderView(Qt.Vertical)
        self.setVerticalHeader(self.vheader)
        self.vheader.setSectionResizeMode(QHeaderView.Fixed)
        self.vheader.setDefaultSectionSize(20)
        #self.vheader.setSectionsMovable(True)

    def _setup_hheader(self):
        self.hheader = self.horizontalHeader()
        self.hheader.setSectionsMovable(True)

    def remove_selected_rows(self):
        selection_model = self.selectionModel()
        rows = [index.row() for index in selection_model.selectedRows()]
        for row in sorted(rows, reverse=True):
            self.model.removeRow(row)

    def insert_row(self):
        selection_model = self.selectionModel()
        rows = [index.row() for index in selection_model.selectedRows()]
        if len(rows) == 0:
            self.model.insertRow(0)
        for row in rows:
            self.model.insertRow(row)


class EditableTable(RawWidget):

    # Expand the table by default
    hug_width = set_default('weak')
    hug_height = set_default('weak')

    model = Typed(QEditableTableModel)
    view = Typed(QEditableTableView)
    delete_filter = Typed(DeleteFilter)

    editable = d_(Bool(False))
    updated = d_(Event())

    column_info = d_(Dict(Unicode(), Typed(object), {}))

    data = d_(Typed(object))

    def get_column_attribute(self, column_name, attribute, default):
        column = self.column_info.get(column_name, {})
        try:
            return column.get(attribute, default)
        except AttributeError:
            return getattr(column, attribute, default)

    @d_func
    def get_cell_color(self, row_index, column_index):
        # This must return one of the SVG color names (see
        # http://www.december.com/html/spec/colorsvg.html) or a hex color code.
        return 'white'

    @d_func
    def get_row_label(self, row_index):
        return str(row_index+1)

    @d_func
    def get_column_label(self, column_index):
        column = self.get_columns()[column_index]
        try:
            return self.get_column_attribute(column, 'compact_label', column)
        except AttributeError:
            return self.get_column_attribute(column, 'label', column)

    @d_func
    def get_rows(self):
        if self.data is None:
            return []
        return range(len(self.data))

    @d_func
    def get_columns(self):
        raise NotImplementedError

    @d_func
    def get_data(self, row_index, column_index):
        raise NotImplementedError

    @d_func
    def set_data(self, row_index, column_index, value):
        raise NotImplementedError

    @d_func
    def remove_row(self, row):
        raise NotImplementedError

    @d_func
    def insert_row(self, row=None):
        raise NotImplementedError

    @d_func
    def get_default_row(self):
        values = []
        for column in self.get_columns():
            default = self.column_info.get(column, {}).get('default', None)
            values.append(default)
        return values

    @d_func
    def coerce_to_type(self, column_index, value):
        column = self.get_columns()[column_index]
        func = self.column_info.get(column, {}).get('coerce', lambda x: x)
        return func(value)

    def create_widget(self, parent):
        self.model = QEditableTableModel(self)
        self.view = QEditableTableView(self.model, parent=parent)
        if self.editable:
            self.delete_filter = DeleteFilter(self.view)
            self.view.installEventFilter(self.delete_filter)
        return self.view

    def _observe_data(self, event):
        # TODO: for lists does not reset if the values are equivalent. We then
        # lose a reference to the actual list.
        self._reset_model()

    def _reset_model(self):
        # Forces a reset of the model and view
        self.model.beginResetModel()
        self.model.endResetModel()


class DataFrameTable(EditableTable):

    data = d_(Typed(pd.DataFrame))
    columns = d_(Typed(object))

    def _observe_columns(self, event):
        self._reset_model()

    @d_func
    def get_columns(self):
        if self.columns is not None:
            return self.columns
        if self.data is None:
            return []
        return self.data.columns

    def get_data(self, row_index, column_index):
        return str(self.data.iat[row_index, column_index])

    def set_data(self, row_index, column_index, value):
        value = self.coerce_to_type(column_index, value)
        self.data.iat[row_index, column_index] = value

    def remove_row(self, row_index):
        label = self.data.index[row_index]
        self.data.drop(label, inplace=True)
        self.data.index = range(len(self.data))

    def insert_row(self, row_index):
        values = self.get_default_row()
        self.data.loc[row_index + 0.5] = values
        self.data.sort_index(inplace=True)
        self.data.index = range(len(self.data))


class ListDictTable(EditableTable):

    def get_data(self, row_index, column_index):
        column = self.get_columns()[column_index]
        return self.data[row_index][column]

    def set_data(self, row_index, column_index, value):
        value = self.coerce_to_type(column_index, value)
        column = self.get_columns()[column_index]
        self.data[row_index][column] = value

    def get_default_row(self):
        values = super().get_default_row()
        keys = self.get_columns()
        return dict(zip(keys, values))

    def insert_row(self, row_index):
        values = self.get_default_row()
        self.data.insert(row_index+1, values)

    def remove_row(self, row_index):
        self.data.pop(row_index)
