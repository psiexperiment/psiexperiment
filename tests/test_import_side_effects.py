"""Verify that importing psi packages has no side effects.

These run in a subprocess so the assertions are about a *fresh* interpreter,
not one where the test session has already imported everything.
"""
import subprocess
import sys

import pytest


def _run(code):
    return subprocess.run([sys.executable, '-c', code], capture_output=True,
                          text=True, timeout=120)


@pytest.mark.slow
def test_import_psi_does_not_load_config():
    result = _run(
        'import psi\n'
        'assert psi._config is None, "config loaded at import"\n'
        'assert psi.get_config("HOSTNAME")\n'
        'assert psi._config is not None\n'
        'print("ok")\n'
    )
    assert result.returncode == 0, result.stderr
    assert 'ok' in result.stdout


@pytest.mark.slow
def test_import_application_does_not_install_excepthook():
    result = _run(
        'import sys\n'
        'original = sys.excepthook\n'
        'import psi.application\n'
        'assert sys.excepthook is original, "excepthook replaced at import"\n'
        'psi.application.install_exception_handler()\n'
        'assert sys.excepthook is psi.application.exception_handler\n'
        'print("ok")\n'
    )
    assert result.returncode == 0, result.stderr
    assert 'ok' in result.stdout
