import argparse

from psi.application import (add_default_options, launch_experiment,
                             load_paradigm_descriptions, parse_args)

from psi.experiment.api import paradigm_manager


def main():
    class ControllerAction(argparse.Action):
        def __call__(self, parser, namespace, value, option_string=None):
            # Ensure that all plugins are disabled by default (this won't
            # affect required plugins from loading). By disabling plugins by
            # default, we can then specifically enable them one-by-one in using
            # the PluginAction.
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
    parser.add_argument('experiment', type=str, help='Experiment to run',
                        action=ControllerAction)
    parser.add_argument('--plugins', type=str, nargs='*',
                        help='Plugins to load', action=PluginAction)

    add_default_options(parser)
    args = parse_args(parser)
    launch_experiment(args)
