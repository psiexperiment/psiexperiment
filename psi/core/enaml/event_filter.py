from enaml.qt.QtCore import Qt, QObject, QEvent


class EventFilter(QObject):

    def __init__(self, widget, *args, **kwargs):
        self.widget = widget
        super().__init__(*args, **kwargs)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Delete:
                self.widget.remove_selected_rows()
                return True
            if event.key() in (Qt.Key_Plus, Qt.Key_Equal):
                if (event.modifiers() & Qt.ControlModifier):
                    self.widget.insert_row()
                    return True
            if event.key() == Qt.Key_Down:
                if self.widget.last_row_current():
                    self.widget.insert_row_at_end()
        return super().eventFilter(obj, event)
