'''
Tests for :mod:`psi.launcher.widgets`.
'''
import enaml
import pytest

with enaml.imports():
    from enaml.widgets.api import Container, ObjectCombo, Window
    from psi.launcher.widgets import AddItem


@pytest.fixture(scope='module')
def qt_app():
    '''
    A single QtApplication for the tests below. Skips them (rather than
    failing) on a headless box where a QtApplication can't be created.
    '''
    try:
        from enaml.qt.qt_application import QtApplication
        app = QtApplication.instance() or QtApplication()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f'QtApplication unavailable: {exc}')
    return app


def _find(widget, type_name):
    '''Recursively collect descendant widgets whose type name matches.'''
    found = []
    for child in widget.children:
        if type(child).__name__ == type_name:
            found.append(child)
        found.extend(_find(child, type_name))
    return found


@pytest.fixture
def popup(qt_app):
    win = Window()
    container = Container(win)
    combo = ObjectCombo(container, items=['a'], selected='a')
    win.show()
    popup = AddItem(parent=container, combo=combo, label='Item')
    popup.show()
    yield popup
    popup.close()
    win.close()


def _type(popup, text):
    '''
    Type `text` into the popup's field the way a user would, so the
    per-keystroke `textEdited` signal fires (unlike setting the text
    programmatically).
    '''
    field, = _find(popup, 'RegexField')
    widget = field.proxy.widget
    widget.clear()
    widget.insert(text)
    return field


def _button(popup, text):
    return next(b for b in _find(popup, 'PushButton') if b.text == text)


def test_ok_disabled_until_text_entered(popup):
    ok = _button(popup, 'OK')
    assert not ok.enabled
    _type(popup, 'b')
    assert ok.enabled


def test_ok_adds_and_selects_item(popup):
    _type(popup, 'new-item')
    # The OK click path: field.text is still empty here (it only updates on
    # Enter), so this checks that the live field content is what's added.
    _button(popup, 'OK').clicked(False)
    assert popup.combo.items == ['a', 'new-item']
    assert popup.combo.selected == 'new-item'


def test_enter_adds_valid_item(popup):
    field = _type(popup, 'new-item')
    field.proxy.widget.returnPressed.emit()
    assert popup.combo.items == ['a', 'new-item']


def test_enter_ignores_invalid_item(popup):
    # The default regex only allows word characters and dashes, so a space
    # fails validation and Enter must not add anything.
    field = _type(popup, 'bad item')
    field.proxy.widget.returnPressed.emit()
    assert popup.combo.items == ['a']


def test_custom_regex(qt_app):
    win = Window()
    container = Container(win)
    combo = ObjectCombo(container, items=[])
    win.show()
    popup = AddItem(parent=container, combo=combo, regex=r'^[\w/ ]+$')
    popup.show()
    try:
        field = _type(popup, 'path/with space')
        field.proxy.widget.returnPressed.emit()
        assert combo.items == ['path/with space']
    finally:
        popup.close()
        win.close()


class _Settings:
    '''What ExperimentSequence reads from its `settings`.'''
    def __init__(self):
        from types import SimpleNamespace
        self.process_manager = SimpleNamespace(current_subprocess=None)


def _sequence_window(**kwargs):
    from launcher_helpers import FakeParadigm
    from psi.launcher.experiment import Experiment
    with enaml.imports():
        from psi.launcher.widgets import ExperimentSequence

    sequence = [Experiment(FakeParadigm(modes=['ipsi', 'contra'])).freeze('contra')]
    win = Window()
    box = ExperimentSequence(Container(win), mode='sequence', sequence=sequence,
                             settings=_Settings(), **kwargs)
    win.show()
    return win, box


def _labels(widget):
    return [w.text for w in _find(widget, 'Label')]


def test_sequence_row_label_defaults_to_mode(qt_app):
    win, box = _sequence_window()
    try:
        assert 'contra' in _labels(box)
    finally:
        win.close()


def test_sequence_row_hooks(qt_app):
    from enaml.widgets.api import Label

    def row_editor(experiment):
        return [Label(text=f'editing {experiment.mode}')]

    win, box = _sequence_window(row_label=lambda e: f'{e.mode} left ear',
                                build_row_editor=row_editor)
    try:
        assert 'contra left ear' in _labels(box)
        assert 'editing contra' not in _labels(box)
        box.edit_mode = True
        assert 'editing contra' in _labels(box)
        box.edit_mode = False
        assert 'contra left ear' in _labels(box)
    finally:
        win.close()
