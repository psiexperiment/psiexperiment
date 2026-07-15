"""Smoke test: every `api.py` aggregator module must import cleanly.

These modules exist solely to re-export names from internal submodules. A
rename or removal in a submodule that forgets to update the aggregator's
import list breaks the module unconditionally (see the `tone_power_conv`
regression in psi.controller.calibration.api), and since nothing else
necessarily imports these aggregator modules directly, this is the only
thing that catches that class of bug.

Hardware-vendor api modules (soundcard, TDT) are excluded: they require
optional extras that aren't installed in the standard test environment.
"""
import importlib

import enaml
import pytest


API_MODULES = [
    'psi.context.api',
    'psi.controller.api',
    'psi.controller.calibration.api',
    'psi.core.api',
    'psi.core.enaml.api',
    'psi.data.api',
    'psi.data.sinks.api',
    'psi.experiment.api',
    'psi.token.api',
]


@pytest.mark.parametrize('module_name', API_MODULES)
def test_api_module_imports(module_name):
    with enaml.imports():
        importlib.import_module(module_name)
