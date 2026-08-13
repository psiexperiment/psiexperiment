import os


def get_hook_dirs():
    '''
    Entry point PyInstaller uses (via the `pyinstaller40` group) to discover
    hooks shipped by this package. Without this, freezing any app that
    imports psi silently drops non-.py package data (e.g. psi-logo.png)
    since PyInstaller's static analysis only follows imports.
    '''
    return [os.path.dirname(__file__)]
