'''
Tests for the dialog reporting how an experiment ended
(`ResultPopup` in `psi/controller/manifest.enaml`).

The dialog is built without showing it: activating the proxy creates the
Qt widgets (and runs the wrap/size logic bound to `activated`) without
putting a window on the screen.
'''
import enaml
import pytest
from enaml.qt.QtWidgets import QLabel, QPushButton

from psi import get_config, set_config

with enaml.imports():
    from psi.controller import manifest as controller_manifest
    from psi.controller.manifest import ResultPopup


@pytest.fixture
def opened(monkeypatch):
    '''
    Record what the dialog asks the desktop to open, rather than actually
    launching an editor or a file browser.
    '''
    paths = []

    def fake_open_url(url):
        paths.append(url.toLocalFile())
        return True

    monkeypatch.setattr(controller_manifest.QDesktopServices, 'openUrl',
                        staticmethod(fake_open_url))
    return paths


@pytest.fixture
def logfile(tmp_path):
    '''
    Stand in for the log file psi publishes when it configures logging.
    '''
    path = tmp_path / 'experiment.log'
    path.write_text('log contents')
    original = get_config('LOG_FILENAME', '')
    set_config('LOG_FILENAME', str(path))
    yield path
    set_config('LOG_FILENAME', original)


def build(app, **kwargs):
    view = ResultPopup(**kwargs)
    view.initialize()
    view.activate_proxy()
    return view


def buttons(view):
    return {b.text(): b for b in view.proxy.widget.findChildren(QPushButton)}


class TestLogFileButtons:

    def test_offers_to_open_the_log(self, app, logfile, opened):
        view = build(app, error_message='Something went wrong.')
        assert view.logfile == str(logfile)

        buttons(view)['Open log'].click()
        assert opened == [str(logfile).replace('\\', '/')]

    def test_offers_to_open_the_folder(self, app, logfile, opened):
        view = build(app, error_message='Something went wrong.')

        buttons(view)['Open log folder'].click()
        assert opened == [str(logfile.parent).replace('\\', '/')]

    def test_offered_without_a_traceback(self, app, logfile, opened):
        # A run that ends without an error has no traceback to show, but
        # its log is still worth opening.
        view = build(app, message='All done.')
        assert 'Open log' in buttons(view)

    def test_hidden_when_there_is_no_log_file(self, app):
        # Logging to the console only.
        original = get_config('LOG_FILENAME', '')
        set_config('LOG_FILENAME', '')
        try:
            view = build(app, error_message='Something went wrong.')
            assert 'Open log' not in buttons(view)
            assert 'Open log folder' not in buttons(view)
        finally:
            set_config('LOG_FILENAME', original)


class TestDialogBehavior:

    def test_modal_and_on_top(self, app, logfile):
        # An error must not be dismissable by a stray click, or end up
        # behind the experiment window.
        view = build(app, error_message='Something went wrong.')
        assert view.modality == 'application_modal'
        assert view.always_on_top

    def test_title_reflects_the_outcome(self, app, logfile):
        error = build(app, error_message='Something went wrong.')
        assert error.title == 'Experiment error'
        result = build(app, message='All done.')
        assert result.title == 'Experiment results'

    def test_message_wraps_to_the_dialog_width(self, app, logfile):
        # The message arrives as one long line per paragraph; the label
        # has to wrap it to the dialog width and be tall enough for the
        # result (enaml's Label does neither on its own).
        message = 'This message is long enough that it has to wrap. ' * 6
        view = build(app, error_message=message)
        label = [l for l in view.proxy.widget.findChildren(QLabel)
                 if l.text() == message][0]
        assert label.wordWrap()
        assert label.width() == view.text_width
        assert label.height() >= label.heightForWidth(view.text_width)


@pytest.fixture
def no_app_icon(app):
    '''
    Start from an application with no default window icon.

    The Qt application is shared by the whole test session, so without this
    an icon set by an earlier test would hide a workbench that never sets
    one. Must be requested before `workbench`, which sets it.
    '''
    from enaml.qt.QtGui import QIcon
    from enaml.qt.QtWidgets import QApplication
    qapp = QApplication.instance()
    original = qapp.windowIcon()
    qapp.setWindowIcon(QIcon())
    yield
    qapp.setWindowIcon(original)


class TestIcon:

    def test_parentless_dialog_gets_psi_icon(self, app, no_app_icon, workbench,
                                             logfile):
        # The dialog has no parent, so it is a top-level window with its own
        # taskbar button. Without an application-wide icon it showed Qt's
        # generic one, which Windows then used for the whole psi.psi taskbar
        # group -- the psi icon appeared in the main window's title bar but
        # not on the taskbar.
        view = build(app, error_message='Something went wrong.')
        assert view.parent is None
        assert not view.proxy.widget.windowIcon().isNull()
