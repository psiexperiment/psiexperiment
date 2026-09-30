'''
Taskbar identity for psi programs.

Deliberately free of enaml, atom and Qt imports so that a program can call
`set_app_id` from the very top of its `main`, before the GUI machinery is
loaded.
'''
import logging
log = logging.getLogger(__name__)

import os


def set_app_id(app_id):
    '''
    Give this process its own identity on the Windows taskbar.

    Windows groups taskbar buttons by AppUserModelID, and a Python GUI that
    never sets one inherits the interpreter's. Without this every psi program
    shares a single taskbar button showing Python's icon (or the console-script
    wrapper's), no matter what icon its windows carry.

    psivideo keeps its own copy of this (it does not depend on psi). Keep the
    two in sync.

    Parameters
    ----------
    app_id : string
        Dotted identifier, by convention `psi.<program>` (e.g., `psi.cftscal`,
        `psi.noise-exp`). `psi` itself claims `psi.psi`, leaving the launchers
        that spawn it free to claim their own so that a launcher and its
        experiments get separate taskbar buttons.

    Notes
    -----
    Call this from the program's entry point before the Qt application is
    created. Once a window exists Windows has already bound the process to the
    default ID and this has no effect.

    No-op off Windows, and fails soft: a mis-grouped taskbar button is cosmetic
    and shouldn't keep the program from starting.
    '''
    if os.name != 'nt':
        return
    import ctypes
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:
        log.warning('Unable to set the AppUserModelID to %r', app_id,
                    exc_info=True)
