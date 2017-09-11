from atom.api import Unicode, List, Bool
from enaml.core.api import d_
from enaml.qt import QtWidgets, QtCore
from enaml.widgets.api import RawWidget


class QtLineCompleter(RawWidget):
    """Simple line editor supporting completion.
    """
    text = d_(Unicode())
    choices = d_(List())

    hug_width = 'ignore'

    #features = Feature.FocusEvents

    #: Flag avoiding circular updates.
    _no_update = Bool(False)

    # PySide requires weakrefs for using bound methods as slots.
    # PyQt doesn't, but executes unsafe code if not using weakrefs.
    __slots__ = '__weakref__'

    def create_widget(self, parent):
        #widget = QtWidgets.QLineEdit(parent)
        widget = QtWidgets.QComboBox(parent)
        widget.addItems(self.choices)
        #widget.setEditable(True)
        completer = QtWidgets.QCompleter(self.choices, parent)
        completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        completer.setFilterMode(QtCore.Qt.MatchContains)
        widget.setCompleter(completer)
        #widget.setText(self.text)
        self.proxy.widget = widget  # Anticipated so that selection works
        #widget.textEdited.connect(self.update_object)
        return widget

    def update_object(self):
        """ Handles the user entering input data in the edit control.
        """
        if (not self._no_update) and self.activated:
            value = self.get_widget().text()
            self._no_update = True
            self.text = value
            self._no_update = False

    def _post_setattr_text(self, old, new):
        """Updates the editor when the object changes externally to the editor.
        """
        if (not self._no_update) and self.get_widget():
            self._no_update = True
            self.get_widget().setText(new)
            self._no_update = False

    #def focus_gained(self):
    #    """Notify the completer the focus was lost.
    #    """
    #    self._completer.on_focus_gained()
