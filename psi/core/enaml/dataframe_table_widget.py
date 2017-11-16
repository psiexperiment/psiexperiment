#----------------------------------------------------------------------------
#  Adapted from BSD-licensed module used by Enthought, Inc.
#----------------------------------------------------------------------------

# TODO - make this faster for simple row append. Do not redraw the entire
# dataframe on each iteration. For now, it's fast enough.

import logging
log = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from atom.api import (Typed, set_default, observe, Value, Event, Property,
                      ContainerList)

from enaml.application import deferred_call
from enaml.core.declarative import d_, d_func
from enaml.widgets.api import RawWidget
from enaml.qt.QtCore import QAbstractTableModel, QModelIndex, Qt
from enaml.qt.QtWidgets import QTableView, QHeaderView, QAbstractItemView
from enaml.qt.QtGui import QFont, QColor


class QDataFrameTableModel(QAbstractTableModel):

    def __init__(self, data, columns, column_info, cell_color=None, **kw):
        self._data = data
        self._columns = columns
        self._column_info = column_info
        self._cell_color = cell_color
        super(QDataFrameTableModel, self).__init__(**kw)

    def headerData(self, section, orientation, role):
        if role == Qt.TextAlignmentRole:
            if orientation == Qt.Horizontal:
                return int(Qt.AlignHCenter | Qt.AlignVCenter)
            return int(Qt.AlignRight | Qt.AlignVCenter)
        elif role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                c = self._columns[section]
                return self._column_info.get(c, 'UNDEFINED')
            else:
                return str(section+1)

    def data(self, index, role=Qt.DisplayRole):
        # Do nothing if the dataframe is empty
        if not index.isValid() or not (0 <= index.row() < len(self._data)):
            return None

        if role == Qt.DisplayRole:
            r = index.row()
            c = self._columns[index.column()]
            v = self._data.at[r, c]
            return str(v)
        elif role == Qt.TextAlignmentRole:
            return int(Qt.AlignRight | Qt.AlignVCenter)
        elif role == Qt.BackgroundRole:
            if self._cell_color is not None:
                r = index.row()
                c = self._columns[index.column()]
                name = self._cell_color(r, c)
                color = QColor()
                color.setNamedColor(name)
                return color

    def columnCount(self, index=QModelIndex()):
        return len(self._columns)

    def rowCount(self, index=QModelIndex()):
        if self._data is not None:
            return len(self._data)
        else:
            return 0

    def _get_formatted_value(self, i, j):
        return str(self._data.loc[i, self._columns[j]])


class QDataFrameTableView(QTableView):

    def __init__(self, model, parent=None, **kwds):
        super(QDataFrameTableView, self).__init__(parent=parent, **kwds)
        self.model = model
        self.setModel(model)
        self._setup_scrolling()
        self._setup_headers()

    def _setup_scrolling(self):
        self.setVerticalScrollMode(QAbstractItemView.ScrollPerItem)
        self.setHorizontalScrollMode(QAbstractItemView.ScrollPerItem)

    def _setup_headers(self):
        self.vheader = QHeaderView(Qt.Vertical)
        self.setVerticalHeader(self.vheader)
        self.vheader.setSectionResizeMode(QHeaderView.Fixed)
        self.vheader.setDefaultSectionSize(20)
        self.hheader = self.horizontalHeader()

    def get_column_widths(self):
        widths = {}
        for i, c in enumerate(self.model._columns):
            widths[c] = self.hheader.sectionSize(i)
        return widths

    def set_column_widths(self, widths):
        for i, c in enumerate(self.model._columns):
            width = widths[c]
            self.setColumnWidth(i, width)


class DataframeTable(RawWidget):

    dataframe = d_(Typed(pd.DataFrame))
    columns = d_(ContainerList())
    column_info = d_(Typed(dict))
    column_widths = Property()

    @d_func
    def cell_color(self, row, column):
        # This must return one of the SVG color names (see
        # http://www.december.com/html/spec/colorsvg.html) or a hex color code.
        return 'white'

    # Expand the table by default
    hug_width = set_default('weak')
    hug_height = set_default('weak')

    def create_widget(self, parent):
        model = QDataFrameTableModel(self.dataframe, self.columns,
                                     self.column_info,
                                     cell_color=self.cell_color)
        return QDataFrameTableView(model, parent=parent)

    def _observe_dataframe(self, event):
        deferred_call(self._update_table)

    def _observe_columns(self, event):
        deferred_call(self._update_table)

    def _observe_column_info(self, event):
        deferred_call(self._update_table)

    def _update_table(self):
        if self.get_widget() is not None:
            table = self.get_widget()
            old_model = table.model
            new_model = QDataFrameTableModel(self.dataframe,
                                             self.columns,
                                             self.column_info,
                                             cell_color=self.cell_color)
            table.model = new_model
            table.setModel(new_model)
            table.scrollToBottom()
            deferred_call(table.update)

    def _get_column_widths(self):
        return self.get_widget().get_column_widths()

    def _set_column_widths(self, widths):
        self.get_widget().set_column_widths(widths)
        self._update_table()