'''
Console setup for psi's command-line entry points.

Deliberately free of enaml, atom and Qt imports, so that a command which
needs no GUI (``psi-config``) does not load it just to configure the
console.
'''
import os


def disable_quick_edit():
    # From https://stackoverflow.com/questions/73486528/python-script-pausing-in-cmd
    import win32console as con

    # Missing constants in pywin
    ENABLE_EXTENDED_FLAGS = 0x0080
    ENABLE_QUICK_EDIT_MODE = 0x0040

    # Modify console mode to disable quick edit mode
    h = con.GetStdHandle(con.STD_INPUT_HANDLE)
    oldMode = h.GetConsoleMode()
    h.SetConsoleMode((oldMode | ENABLE_EXTENDED_FLAGS) &
            ~ENABLE_QUICK_EDIT_MODE)


def setup_windows_console():
    '''
    Disable Windows quick-edit mode so that clicking in the console does not
    accidentally pause the application. Called from the CLI entry points;
    importing this module has no side effects.
    '''
    if os.name == 'nt':
        try:
            disable_quick_edit()
        except Exception:
            pass
