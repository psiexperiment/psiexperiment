import logging
log = logging.getLogger(__name__)
import argparse
import importlib

from psi.application import (get_default_calibration, get_default_io,
                             launch_experiment, load_paradigm_descriptions)

from psi.experiment.api import paradigm_manager


def parse_args(parser):
    args = parser.parse_args()
    if args.calibration is None:
        try:
            args.calibration = get_default_calibration(args.io)
        except ValueError as e:
            log.warn(str(e))
    return args


def add_default_options(parser):
    import argparse

    class CalibrationAction(argparse.Action):
        def __call__(self, parser, namespace, value, option_string=None):
            path = Path(value)
            if not path.exists():
                path = namespace.io / path
                path = path.with_suffix('.json')
                if not path.exists():
                    raise ValueError('%s does not exist'.format(value))
            setattr(namespace, self.dest, value)

    try:
        default_io = get_default_io()
    except ValueError:
        default_io = None
    parser.add_argument('pathname', type=str, help='Filename', nargs='?')
    parser.add_argument('--io', type=str, default=default_io,
                        help='Hardware configuration')
    parser.add_argument('--calibration', type=str, help='Hardware calibration',
                        action=CalibrationAction)
    parser.add_argument('--debug', default=True, action='store_true',
                        help='Debug mode?')
    parser.add_argument('--debug-warning', default=False, action='store_true',
                        help='Show warnings?')
    parser.add_argument('--debug-level-console', type=str, default='DEBUG',
                        help='Logging level for console')
    parser.add_argument('--debug-level-file', type=str, default='INFO',
                        help='Logging level for file')
    parser.add_argument('--debug-exclude', type=str, nargs='*',
                        help='Names to exclude from debugging')
    parser.add_argument('--pdb', default=False, action='store_true',
                        help='Autolaunch PDB?')
    parser.add_argument('--no-preferences', default=False, action='store_true',
                        help="Don't load existing preference files")
    parser.add_argument('--no-layout', default=False, action='store_true',
                        help="Don't load existing layout files")
    parser.add_argument('-c', '--commands', nargs='+', default=[],
                        help='Commands to invoke')
    parser.add_argument('-p', '--preferences', type=str, nargs='?',
                        help='Preferences file')
    parser.add_argument('-l', '--layout', type=str, nargs='?',
                        help='Layout file')
    parser.add_argument('--profile', action='store_true', help='Profile app')


def main():
    class ControllerAction(argparse.Action):
        def __call__(self, parser, namespace, value, option_string=None):
            # Ensure that all plugins are disabled by default (this won't
            # affect required plugins from loading). By disabling plugins by
            # default, we can then specifically enable them one-by-one in using
            # the PluginAction.
            if '::' in value:
                description, value = value.split('::')
                importlib.import_module(description)

            paradigm = paradigm_manager.get_paradigm(value)
            paradigm.disable_all_plugins()
            setattr(namespace, 'experiment', paradigm.name)
            setattr(namespace, 'controller', paradigm)

    class PluginAction(argparse.Action):
        def __call__(self, parser, namespace, value, option_string=None):
            for plugin_name in value:
                namespace.controller.enable_plugin(plugin_name)

    load_paradigm_descriptions()
    parser = argparse.ArgumentParser(description='Run experiment')
    # TODO: Add extended help that explains how to locate paradigm descriptions
    # if needed using the :: syntax.
    parser.add_argument('experiment', type=str, help='Experiment to run',
                        action=ControllerAction)
    parser.add_argument('--plugins', type=str, nargs='*',
                        help='Plugins to load', action=PluginAction)

    add_default_options(parser)
    args = parse_args(parser)
    launch_experiment(args)
