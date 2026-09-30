'''
Implement subclass of AbstractButton that passes keyboard modifiers that were
active when button was clicked.
'''

from atom.api import Event, ForwardTyped, Typed

from enaml.core.api import d_
from enaml.widgets.abstract_button import ProxyAbstractButton, AbstractButton

from enaml.qt.qt_abstract_button import QtAbstractButton
from enaml.qt.QtCore import Signal, Qt
from enaml.qt.QtWidgets import QPushButton
from enaml.qt.qt_factories import QT_FACTORIES


class QModifierPushButton(QPushButton):

    mouse_click = Signal(set)

    def mousePressEvent(self, mouse_event):
        modifiers = set()
        if mouse_event.modifiers() & Qt.SHIFT:
            modifiers.add('shift')
        if mouse_event.modifiers() & Qt.CTRL:
            modifiers.add('control')
        if mouse_event.modifiers() & Qt.ALT:
            modifiers.add('alt')
        self.mouse_click.emit(modifiers)


class ProxyModifierButton(ProxyAbstractButton):

    declaration = ForwardTyped(lambda: ModifierButton)


class QtModifierButton(ProxyModifierButton, QtAbstractButton):

    def keyPressEvent(self, event):
        super().keyPressEvent(event)

    def create_widget(self):
        self.widget = QModifierPushButton(self.parent_widget())
        self.widget.mouse_click.connect(self.on_click)

    def on_click(self, modifiers):
        self.declaration.clicked(modifiers)


class ModifierButton(AbstractButton):

    proxy = Typed(ProxyModifierButton)
    clicked = d_(Event(set), writable=False)


def modifier_button_factory():
    return QtModifierButton


QT_FACTORIES['ModifierButton'] = modifier_button_factory
