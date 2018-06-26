import argparse

from psi.application import add_default_options, launch_experiment, experiments


def main():
    class ControllerAction(argparse.Action):
        def __call__(self, parser, namespace, value, option_string=None):
            setattr(namespace, 'experiment', value)
            setattr(namespace, 'controller', experiments[value])

    parser = argparse.ArgumentParser(description='Run experiment')
    parser.add_argument('experiment', type=str, help='Experiment to run',
                        choices=experiments.keys(), action=ControllerAction)
    add_default_options(parser)
    args = parser.parse_args()
    launch_experiment(args)
