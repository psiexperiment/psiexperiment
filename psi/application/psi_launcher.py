import argparse

from psi.application import (add_default_options, launch_experiment,
                             parse_args)
from psi.application.experiment_description import experiments


def main():
    class ControllerAction(argparse.Action):
        def __call__(self, parser, namespace, value, option_string=None):
            setattr(namespace, 'experiment', value)
            setattr(namespace, 'controller', experiments[value])
            for plugin in experiments[value].plugins:
                plugin.selected = False

    class PluginAction(argparse.Action):
        def __call__(self, parser, namespace, value, option_string=None):
            for plugin_name in value:
                namespace.controller.enable_plugin(plugin_name)


    parser = argparse.ArgumentParser(description='Run experiment')
    parser.add_argument('experiment', type=str, help='Experiment to run',
                        choices=experiments.keys(), action=ControllerAction)
    parser.add_argument('--plugins', type=str, nargs='*',
                        help='Plugins to load', action=PluginAction)

    add_default_options(parser)
    args = parse_args(parser)
    launch_experiment(args)
