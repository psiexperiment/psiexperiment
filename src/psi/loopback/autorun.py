'''
Run one paradigm unattended and save its data.

Started in a subprocess by `psi.loopback.runner.run_paradigm` (as `python -m
psi.loopback.autorun`), so each paradigm gets a fresh Qt application and psi
workbench. Mirrors `psi.application._main` except that the autorun manifest
(and any extra manifests the test asks for) are added to the paradigm's
plugins and preferences/layouts are never loaded, so a rig's saved settings
can't leak into a test.

The outcome is written to `--result` as JSON by the autorun manifest. If the
file is missing when this exits, the run died before psi could wrap up.
'''
import argparse
import importlib
import json
import logging
import os
from pathlib import Path
import sys
import traceback

import enaml
from enaml.application import timed_call


def load_manifest(path, attrs):
    module_name, class_name = path.rsplit('.', 1)
    with enaml.imports():
        module = importlib.import_module(module_name)
    return getattr(module, class_name)(**attrs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('paradigm', help='e.g. abr_io')
    parser.add_argument('pathname', help='Folder to save the data to')
    parser.add_argument('--io', required=True)
    parser.add_argument('--import', dest='modules', nargs='*', default=[],
                        help='Modules that register the paradigm')
    parser.add_argument('--plugins', nargs='*', default=[])
    parser.add_argument('--settings', required=True,
                        help='JSON file of settings, extra manifests and '
                        'duration (see `runner.run_paradigm`)')
    parser.add_argument('--result', required=True)
    parser.add_argument('--log', required=True)
    args = parser.parse_args()

    # Run as `python -m`, the working directory is on the path but only as
    # '', which enaml's importer doesn't search.
    sys.path.insert(0, os.getcwd())

    from psi.application import (configure_logging, ensure_application,
                                 install_exception_handler,
                                 install_qt_message_handler)
    from psi.experiment.api import paradigm_manager

    configure_logging('INFO', 'DEBUG', args.log)
    install_exception_handler()
    install_qt_message_handler()

    ensure_application()
    for module in args.modules:
        importlib.import_module(module)
    paradigm = paradigm_manager.get_paradigm(args.paradigm)
    paradigm.disable_all_plugins()
    for plugin in args.plugins:
        paradigm.enable_plugin(plugin)
    manifests = [p.manifest for p in paradigm.plugins if p.selected or p.required]

    with enaml.imports():
        from psi.loopback.autorun_manifest import AutorunManifest, write_result
    run = json.loads(Path(args.settings).read_text())
    manifests.append(AutorunManifest(settings=run['settings'],
                                     duration=run.get('duration'),
                                     result_path=args.result))
    for path, attrs in run.get('manifests', {}).items():
        manifests.append(load_manifest(path, attrs))

    from psi.application.workbench import PSIWorkbench
    workbench = PSIWorkbench()

    # psi's handler stops a running experiment, which ends in the wrapup the
    # autorun manifest records. An error before the experiment starts (e.g.,
    # a bad setting) only gets a dialog, so record it and quit here instead
    # of leaving the dialog up until the runner times out.
    psi_hook = sys.excepthook

    def hook(*exc_info):
        controller = workbench.get_plugin('psi.controller')
        if controller.experiment_state == 'initialized':
            write_result(args.result, message='', stop_reason='',
                         error_message=str(exc_info[1]),
                         traceback=''.join(traceback.format_exception(*exc_info)))
            ui = workbench.get_plugin('enaml.workbench.ui')
            timed_call(1000, ui.stop_application)
        psi_hook(*exc_info)

    sys.excepthook = hook
    workbench.register_core_plugins(args.io, manifests)

    # The launcher limits which events experiments send it as soon as they
    # connect. Without that filter the websocket plugin (if the paradigm has
    # one) tries to serialize every event, and fails on the ones that carry
    # epochs.
    from psi.launcher.process_manager import RELAYED_EVENTS
    try:
        websocket = workbench.get_plugin('psi.websocket')
    except ValueError:
        websocket = None
    if websocket is not None:
        websocket.set_event_filter('|'.join(RELAYED_EVENTS))
    workbench.start_workspace(paradigm.name, args.pathname, commands=[],
                              load_preferences=False, load_layout=False)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        logging.getLogger(__name__).exception('Autorun failed')
        raise
