#----------------------------------------------------------------------------
#  Adapted from BSD-licensed module used by Enthought, Inc.
#----------------------------------------------------------------------------
import logging
log = logging.getLogger(__name__)

import pandas as pd

from atom.api import (Typed, set_default, observe, Enum, Event, Property,
                      Bool, Dict, Str, Atom, List, Value)
from enaml.core.declarative import d_, d_func
from enaml.widgets.api import RawWidget

from enaml.qt.QtCore import QAbstractTableModel, QModelIndex, QRect, Qt
from enaml.qt.QtWidgets import QAbstractItemView, QHeaderView, QStyledItemDelegate, QTableView
from enaml.qt.QtGui import QBrush, QColor

from .event_filter import EventFilter


class QDelegate(QStyledItemDelegate):

    def __init__(self, model, **kw):
        self.model = model
        super().__init__(**kw)

    def paint(self, painter, option, index):
        self.initStyleOption(option, index)
        painter.save()
        left_width = option.rect.width() * self.model.cellFrac(index)
        right_width = option.rect.width() - left_width
        left_rect = QRect(option.rect.left(), option.rect.top(), left_width, option.rect.height())
        right_rect = QRect(option.rect.left(), option.rect.top(), right_width, option.rect.height())

        left_brush = QBrush(self.model.cellColor(index))
        painter.fillRect(left_rect, left_brush)
        painter.fillRect(right_rect, Qt.NoBrush)
        painter.restore()
        option.backgroundBrush = QBrush(Qt.NoBrush)
        super().paint(painter, option, index)


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
        flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        if self.interface.editable:
            flags = flags | Qt.ItemIsEditable
        return flags

    def cellColor(self, index):
        r = index.row()
        c = index.column()
        color_name = self.interface.get_cell_color(r, c)
        color = QColor()
        color.setNamedColor(color_name)
        return color

    def cellFrac(self, index):
        r = index.row()
        c = index.column()
        return self.interface.get_cell_frac(r, c)

    def data(self, index, role):
        if not index.isValid():
            return
        elif role in (Qt.DisplayRole, Qt.EditRole):
            r = index.row()
            c = index.column()
            return self.interface._get_data(r, c)

    def setData(self, index, value, role):
        with self.interface.live_edit:
            r = index.row()
            c = index.column()
            try:
                self.interface._set_data(r, c, value)
            except:
                raise
                pass
            self.dataChanged.emit(index, index)
            return True

    def removeRows(self, row, count, index):
        with self.interface.live_edit:
            self.beginRemoveRows(index, row, row)
            self.interface._remove_row(row)
            self.endRemoveRows()
            return True

    def insertRows(self, row, count, index):
        with self.interface.live_edit:
            self.beginInsertRows(index, row, row)
            self.interface._insert_row(row)
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
        self.delegate = QDelegate(model)
        self.setItemDelegate(self.delegate)

        self._setup_hheader()
        self._setup_vheader()
        self.setVerticalScrollMode(QAbstractItemView.ScrollPerItem)
        self.setHorizontalScrollMode(QAbstractItemView.ScrollPerItem)
        self._set_default_column_widths()

        select_mode = self.model.interface.select_mode
        select_behavior = self.model.interface.select_behavior
        if select_mode is None:
            self.setSelectionMode(self.NoSelection)
        else:
            flag_name = '{}Selection'.format(select_mode.capitalize())
            self.setSelectionMode(getattr(self, flag_name))
        flag_name = 'Select{}'.format(select_behavior.capitalize())
        self.setSelectionBehavior(getattr(self, flag_name))
        self.selectionModel().selectionChanged.connect(self._selection_changed)
        self.setShowGrid(self.model.interface.show_grid)

    def _selection_changed(self, selected, deselected):
        locations = []
        selection_model = self.selectionModel()
        for index in selection_model.selectedIndexes():
            locations.append((index.row(), index.column()))
        self.model.interface.selected_coords = locations
        self.model.interface.selection_changed = True

    def _set_default_column_widths(self):
        widths = self.model.interface.get_default_column_widths()
        self.set_column_widths(widths)

    def _setup_vheader(self):
        header = self.verticalHeader()
        header.setSectionResizeMode(QHeaderView.Fixed)
        header.setDefaultSectionSize(20)
        header.setSectionsMovable(False)
        if not self.model.interface.show_row_labels:
            header.setVisible(False)

    def _setup_hheader(self):
        header = self.horizontalHeader()
        header.setSectionsMovable(self.model.interface.columns_movable)
        if not self.model.interface.show_column_labels:
            header.setVisible(False)
        if self.model.interface.stretch_last_section:
            header.setStretchLastSection(True)
        resize_mode = self.model.interface.header_resize_mode
        if resize_mode == 'contents':
            resize_mode = 'ResizeToContents'
        else:
            resize_mode = resize_mode.capitalize()
        log.debug('Setting header resize mode to %s', resize_mode)
        header.setSectionResizeMode(getattr(header, resize_mode))

    def remove_selected_rows(self):
        selection_model = self.selectionModel()
        rows = [index.row() for index in selection_model.selectedRows()]
        for row in sorted(rows, reverse=True):
            self.model.removeRow(row)

    def get_selected_rows(self):
        return sorted(r for r, c in self.model.interface.selected_coords)

    def last_row_current(self):
        selected_row = self.currentIndex().row()
        return (selected_row + 1) == self.model.rowCount()

    def insert_row(self):
        rows = self.get_selected_rows()
        if len(rows) == 0:
            self.model.insertRow(0)
        for row in sorted(rows, reverse=True):
            self.model.insertRow(row)

    def insert_row_at_end(self):
        self.model.insertRow(self.model.rowCount())

    def get_column_config(self):
        log.debug('Geting column config')
        try:
            config = {}
            columns = self.model.interface.get_columns()
            header = self.horizontalHeader()
            for i, c in enumerate(columns):
                config[c] = {'width': self.columnWidth(i)}
                if self.model.interface.columns_movable:
                    config[c]['visual_index'] = header.visualIndex(i)
            return config
        except Exception as e:
            log.exception(e)

    def set_column_config(self, config):
        columns = self.model.interface.get_columns()
        for i, c in enumerate(columns):
            try:
                width = config[c]['width']
                self.setColumnWidth(i, width)
                log.debug('Set column width for %s to %d', c, width)
            except KeyError as e:
                log.debug('Unable to set column width for %s', c)

        if self.model.interface.columns_movable:
            header = self.horizontalHeader()
            visual_indices = []
            for i, c in enumerate(columns):
                try:
                    vi = config[c]['visual_index']
                    visual_indices.append((vi, i, c))
                except KeyError as e:
                    log.debug('Unable to find visual index for %s', c)

            # Since the current visual index of each column will change as we
            # rearrange them, we need to figure out which column should appear
            # first and put it there, then move to the next column.
            for vi, li, c in sorted(visual_indices):
                current_vi = header.visualIndex(li)
                header.moveSection(current_vi, vi)
                log.debug('Set visual index for %s to %d', c, vi)

    # CAN DEPRECATE THIS
    def get_column_widths(self):
        widths = {}
        columns = self.model.interface.get_columns()
        header = self.horizontalHeader()
        for i, c in enumerate(columns):
            widths[c] = header.sectionSize(i)
        return widths

    # CAN DEPRECATE THIS
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
    '''
    Defines a table that can optionally be edited by the end-user.

    Subclasses implement handling for different types of containers (e.g.,
    DataFrame, dictionary of lists, list of lists, etc.).
    '''

    #: Expand the table by default
    hug_width = set_default('weak')

    #: Expand the table by default
    hug_height = set_default('weak')

    model = Typed(QEditableTableModel)

    #: Instance of QEditableTableView
    view = Typed(QEditableTableView)
    event_filter = Typed(EventFilter)

    #: Can the user edit the data in the table?
    editable = d_(Bool(False))

    #: Each time the table is updated, should it scroll to the bottom (last
    #: row)?
    autoscroll = d_(Bool(False))

    #: Should column labels be shown?
    show_column_labels = d_(Bool(True))

    #: Should row labels be shown?
    show_row_labels = d_(Bool(True))

    #: Should the grid between cells be shown?
    show_grid = d_(Bool(True))

    #: Dictionary mapping column name to a dictionary of settings for that
    #: column. Valid keys for each setting include:
    #: * compact_label - Column label (preferred).
    #: * label - Column label (used if compact_label not provided).
    #: * default - Value used for adding rows.
    #: * coerce - Function to coerce text entered in column to correct value.
    #: * initial_width - Initial width to set column to.
    #: * to_string - Function used to generate string representation.

    column_info = d_(Dict(Str(), Typed(object), {}))

    #: Widths of columns in table
    column_widths = Property()

    #: Dictionary mapping column name to a dictionary of column properties:
    #: * visual_index: Visual position of column in table
    #: * width: Width of column in table
    column_config = Property()

    #: Can columns be rearranged by dragging labels in the header?
    columns_movable = d_(Bool(True))

    #: Table data. Table is updated every time the object changes. Atom cannot
    #: listen for changes to an object, so you need to be sur ethat
    data = d_(Value())
    update = d_(Bool())
    updated = d_(Event())

    # List of row, col tuples of selections
    selection_changed = d_(Event())
    selected_coords = d_(List(), [])

    live_edit = Typed(LiveEdit, {})

    select_behavior = d_(Enum('items', 'rows', 'columns'))
    select_mode = d_(Enum('single', 'contiguous', 'extended', 'multi', None))

    #: Strectch width of last column so it fills rest of table?
    stretch_last_section = d_(Bool(True))

    #: How can column headers be resized?
    header_resize_mode = d_(Enum('interactive', 'fixed', 'stretch',
                                 'contents'))

    def get_selected_row_coords(self):
        if len(self.selected_coords) == 0:
            return []
        rows, cols = zip(*self.selected_coords)
        return sorted(list(set(rows)))

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
    def get_cell_frac(self, row_index, column_index):
        '''
        Parameters
        ----------
        row_index : int
            Row index (zero-based)
        column_index : int
            Column index (zero-based)

        Result
        ------
        frac : float
            Fraction to shade background cell. Used for progress bars within
            cells. Defaults to 1 (i.e., shade full width of cell).
        '''
        return 1.0

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
        Return a list of the columns

        Returns
        -------
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

    def _get_data(self, row_index, column_index):
        try:
            value = self.get_data(row_index, column_index)
            column = self.get_columns()[column_index]
            formatter = self.column_info.get(column, {}).get('to_string', str)
            return formatter(value)
        except Exception as e:
            log.warning(e)
            return ''

    def _set_data(self, *args):
        self.set_data(*args)
        self.updated = True

    def _remove_row(self, *args):
        self.remove_row(*args)
        self.updated = True

    def _insert_row(self, *args):
        self.insert_row(*args)
        self.updated = True

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

    def _observe_column_info(self, event):
        self._reset_model()

    def _observe_update(self, event):
        if self.update:
            self._reset_model()
            self.update = False

    def _reset_model(self, event=None):
        # Forces a reset of the model and view. Check if model has been created
        # first. If not, do nothing.
        if self.model is None:
            return
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

    def _get_column_config(self):
        return self.view.get_column_config()

    def _set_column_config(self, config):
        self.view.set_column_config(config)
        self._reset_model()

    def get_visual_columns(self):
        if not self.columns_movable:
            return self.get_columns()
        config = self.column_config
        indices = [(cfg['visual_index'], c) for c, cfg in config.items()]
        indices.sort()
        return [i[1] for i in indices]

    def as_string(self):
        rows = self.get_rows()
        visual_cols = self.get_visual_columns()
        cols = self.get_columns()
        table_strings = []
        for r in range(len(rows)):
            row_data = []
            for v in visual_cols:
                c = cols.index(v)
                row_data.append(self.get_data(r, c))
            row_string = '\t'.join(str(d) for d in row_data)
            table_strings.append(row_string)
        return '\n'.join(table_strings)


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
        return self.data.at[row_label, column_label]

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

    #: List of dictionaries where list index maps to row and dictionary key
    #: maps to column.
    data = d_(List())

    #: List of column names. If not provided, defaults to dictionary keys
    #: provided by the first entry in `data`.
    columns = d_(List())

    def get_columns(self):
        if self.columns:
            return self.columns
        if (self.data is not None) and (len(self.data) != 0):
            return list(self.data[0].keys())
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

    def get_selected_rows(self):
        return [self.data[i] for i in self.get_selected_row_coords()]


class ListTable(EditableTable):

    data = d_(List())
    column_name = d_(Str())
    selected = d_(List())
    show_column_labels = True
    stretch_last_section = True

    def get_columns(self):
        return [self.column_name]

    def get_data(self, row_index, column_index):
        return self.data[row_index]

    def set_data(self, row_index, column_index, value):
        value = self.coerce_to_type(column_index, value)
        self.data[row_index] = value

    def get_default_row(self):
        values = super().get_default_row()
        return values[0]

    def insert_row(self, row_index):
        value = self.get_default_row()
        self.data.insert(row_index+1, value)

    def remove_row(self, row_index):
        self.data.pop(row_index)

    def _observe_selected_coords(self, event):
        self.selected = [self.data[r] for r, c in self.selected_coords]
