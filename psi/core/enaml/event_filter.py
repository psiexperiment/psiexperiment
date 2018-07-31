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
            if event.key() == Qt.Key_Plus:
                if (event.modifiers() & Qt.ControlModifier):
                    self.widget.insert_row()
                    return True
        return super().eventFilter(obj, event)
