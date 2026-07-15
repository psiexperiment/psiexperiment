"""Regression test for psi.application.workbench's dock style registration.

Run in a subprocess so the assertion is about a *fresh* interpreter, not one
where some other test in the session already imported
psi.experiment.dock_area_styles and populated the (module-level, global)
Enaml style registry.
"""
import subprocess
import sys

import pytest


def _run(code):
    return subprocess.run([sys.executable, '-c', code], capture_output=True,
                          text=True, timeout=120)


@pytest.mark.slow
def test_importing_workbench_registers_custom_dock_styles():
    result = _run(
        'import enaml\n'
        'with enaml.imports():\n'
        '    from enaml.stdlib.dock_area_styles import available_styles\n'
        '    import psi.application.workbench\n'
        'styles = set(available_styles())\n'
        'expected = {"complete", "error", "nosave"}\n'
        'missing = expected - styles\n'
        'assert not missing, f"custom dock styles not registered: {missing}"\n'
        'print("ok")\n'
    )
    assert result.returncode == 0, result.stderr
    assert 'ok' in result.stdout
