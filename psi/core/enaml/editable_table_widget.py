#----------------------------------------------------------------------------
#  Adapted from BSD-licensed module used by Enthought, Inc.
#----------------------------------------------------------------------------
import pandas as pd

from atom.api import (Typed, set_default, observe, Event, Property,
                      Bool, Dict, Unicode, Atom, List, Value)
from enaml.core.declarative import d_, d_func
from enaml.widgets.api import RawWidget

from enaml.qt.QtCore import QAbstractTableModel, QModelIndex, Qt
from enaml.qt.QtWidgets import QTableView, QHeaderView, QAbstractItemView
from enaml.qt.QtGui import QColor

from .event_filter import EventFilter


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
        with self.interface.live_edit:
            r = index.row()
            c = index.column()
            try:
                self.interface.set_data(r, c, value)
            except:
                pass
            self.dataChanged.emit(index, index)
            return True

    def removeRows(self, row, count, index):
        with self.interface.live_edit:
            self.beginRemoveRows(index, row, row)
            self.interface.remove_row(row)
            self.endRemoveRows()
            return True

    def insertRows(self, row, count, index):
        with self.interface.live_edit:
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

    def sort(self, column_index, order):
        ascending = order == Qt.AscendingOrder
        self.interface.sort_rows(column_index, ascending)


class QEditableTableView(QTableView):

    def __init__(self, model, parent=None, **kwds):
        super().__init__(parent=parent, **kwds)
        self.model = model
        self.setModel(model)
        self._setup_hheader()
        self._setup_vheader()
        self.setVerticalScrollMode(QAbstractItemView.ScrollPerItem)
        self.setHorizontalScrollMode(QAbstractItemView.ScrollPerItem)
        self._set_default_column_widths()

    def _set_default_column_widths(self):
        widths = self.model.interface.get_default_column_widths()
        self.set_column_widths(widths)

    def _setup_vheader(self):
        header = self.verticalHeader()
        header.setSectionResizeMode(QHeaderView.Fixed)
        header.setDefaultSectionSize(20)
        header.setSectionsMovable(False)

    def _setup_hheader(self):
        header = self.horizontalHeader()
        header.setSectionsMovable(True)
        #header.setSortIndicatorShown(True)
        #header.sortIndicatorChanged.connect(self.model.sort)

    def remove_selected_rows(self):
        selection_model = self.selectionModel()
        rows = [index.row() for index in selection_model.selectedRows()]
        for row in sorted(rows, reverse=True):
            self.model.removeRow(row)

    def insert_row(self):
        selection_model = self.selectionModel()
        rows = [index.row() for index in selection_model.selectedRows()]
        rows.sort()
        if len(rows) == 0:
            self.model.insertRow(0)
        for row in sorted(rows, reverse=True):
            self.model.insertRow(row+1)

    def get_column_widths(self):
        widths = {}
        columns = self.model.interface.get_columns()
        header = self.horizontalHeader()
        for i, c in enumerate(columns):
            widths[c] = header.sectionSize(i)
        return widths

    def set_column_widths(self, widths):
        columns = self.model.interface.get_columns()
        for i, c in enumerate(columns):
            try:
                width = widths[c]
                self.setColumnWidth(i, width)
            except KeyError:
                pass


class LiveEdit:

    def __init__(self):
        self._editing = False

    def __enter__(self):
        self._editing = True

    def __exit__(self, type, value, traceback):
        self._editing = False

    def __bool__(self):
        return self._editing


class EditableTable(RawWidget):

    # Expand the table by default
    hug_width = set_default('weak')
    hug_height = set_default('weak')

    model = Typed(QEditableTableModel)
    view = Typed(QEditableTableView)
    event_filter = Typed(EventFilter)

    editable = d_(Bool(False))
    autoscroll = d_(Bool(False))

    # Can include label, compact_label, default value (for adding
    # rows), coerce function (for editing data).
    column_info = d_(Dict(Unicode(), Typed(object), {}))
    column_widths = Property()

    data = d_(Typed(object))
    update = d_(Bool())

    live_edit = Typed(LiveEdit, {})

    def get_column_attribute(self, column_name, attribute, default,
                             raise_error=False):

        column = self.column_info.get(column_name, {})
        try:
            return column[attribute]
        except (KeyError, TypeError):
            try:
                return getattr(column, attribute)
            except AttributeError:
                if raise_error:
                    raise
                else:
                    return default

    @d_func
    def get_cell_color(self, row_index, column_index):
        '''
        Parameters
        ----------
        row_index : int
            Row index (zero-based)
        column_index : int
            Column index (zero-based)

        Result
        ------
        color : SVG color name or hex color code
            Color to use for the background cell. Defaults to white. See
            http://www.december.com/html/spec/colorsvg.html for SVG color
            names.
        '''
        # Given the row and column
        # This must return one of the SVG color names (see
        return 'white'

    @d_func
    def get_row_label(self, row_index):
        '''
        Parameters
        ----------
        row_index : int
            Row index (zero-based)

        Result
        ------
        label : str
            Label to use for column header. Defaults to a 1-based row number.
        '''
        return str(row_index+1)

    @d_func
    def get_column_label(self, column_index):
        '''
        Parameters
        ----------
        column_index : int
            Column index (zero-based)

        Result
        ------
        label : str
            Label to use for row header. Defaults to the 'compact_label' key in
            'column_info'. If 'compact_label' is not found, checks for the
            'label' key.
        '''
        column = self.get_columns()[column_index]
        try:
            return self.get_column_attribute(column, 'compact_label', column,
                                             raise_error=True)
        except AttributeError:
            return self.get_column_attribute(column, 'label', column)

    @d_func
    def get_rows(self):
        if self.data is None:
            return []
        return range(len(self.data))

    @d_func
    def get_columns(self):
        '''
        Result
        ------
        column_labels : list of str
            List of column labels.
        '''
        raise NotImplementedError

    @d_func
    def get_data(self, row_index, column_index):
        '''
        Parameters
        ----------
        row_index : int
            Row index (zero-based)
        column_index : int
            Column index (zero-based)

        Result
        ------
        data : object
            Data to be shown in cell.
        '''
        raise NotImplementedError

    @d_func
    def set_data(self, row_index, column_index, value):
        '''
        Save value at specified row and column index to data

        Parameters
        ----------
        row_index : int
            Row index (zero-based)
        column_index : int
            Column index (zero-based)
        value : object
        '''
        raise NotImplementedError

    @d_func
    def remove_row(self, row_index):
        raise NotImplementedError

    @d_func
    def insert_row(self, row=None):
        raise NotImplementedError

    @d_func
    def sort_rows(self, column_index, ascending):
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
            self.event_filter = EventFilter(self.view)
            self.view.installEventFilter(self.event_filter)
        return self.view

    def _observe_data(self, event):
        # TODO: for lists does not reset if the values are equivalent. We then
        # lose a reference to the actual list.
        self._reset_model()

    def _observe_update(self, event):
        if self.update:
            self._reset_model()
            self.update = False

    def _reset_model(self, event=None):
        # Forces a reset of the model and view
        self.model.beginResetModel()
        self.model.endResetModel()
        if self.autoscroll and self.view:
            self.view.scrollToBottom()

    def _get_column_widths(self):
        return self.view.get_column_widths()

    def _set_column_widths(self, widths):
        self.view.set_column_widths(widths)
        self._reset_model()

    def get_default_column_widths(self):
        return {c: self.get_column_attribute(c, 'initial_width', 100) \
                for c in self.get_columns()}


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
        row_label = self.data.index[row_index]
        column_label = self.get_columns()[column_index]
        return str(self.data.at[row_label, column_label])

    def set_data(self, row_index, column_index, value):
        value = self.coerce_to_type(column_index, value)
        row_label = self.data.index[row_index]
        column_label = self.get_columns()[column_index]
        self.data.at[row_label, column_label] = value

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

    data = d_(List())
    columns = d_(List())

    def get_columns(self):
        if self.columns is not None:
            return self.columns
        if (self.data is not None) and (len(self.data) != 0):
            return list(self.data[0].keys())
        else:
            return []

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
