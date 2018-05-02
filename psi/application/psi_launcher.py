import argparse

from psi.application import add_default_options, launch_experiment, experiments


def main():
    parser = argparse.ArgumentParser(description='Run experiment')
    parser.add_argument('experiment', type=str, help='Experiment to run',
                        choices=experiments.keys())
    add_default_options(parser)
    args = parser.parse_args()
    launch_experiment(args)
