from enaml.qt.qt_application import QtApplication


def main():
    import enaml
    with enaml.imports():
        from psi.application.cfts_launcher_view import LauncherView
    app = QtApplication()
    view = LauncherView()
    view.show()
    app.start()
