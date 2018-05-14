import os.path

import numpy as np

from psi.data.io import abr


nofilter_template = 'ABR {:.1f}ms to {:.1f}ms {{}}.csv'

filter_template = 'ABR {:.1f}ms to {:.1f}ms ' \
    'with {:.0f}Hz to {:.0f}Hz filter ' \
    '{{}}.csv'


columns = ['frequency', 'level', 'polarity']


def process_files(filenames, offset, duration, filter_settings=None):
    for filename in filenames:
        try:
            process_file(filename, offset, duration, filter_settings)
            print('Processed {}'.format(filename))
        except Exception as e:
            raise
            print('Error processing {}'.format(filename))


def process_file(filename, offset, duration, filter_settings):
    fh = abr.load(filename)
    if filter_settings is None:
        epochs = fh.get_epochs(offset=offset, duration=duration,
                               columns=columns)
        file_template = nofilter_template.format(offset*1e3, (offset+duration)*1e3)
    else:
        lb = filter_settings['lb']
        ub = filter_settings['ub']
        file_template = filter_template.format(offset*1e3,
                                               (offset+duration)*1e3, lb, ub)
        epochs = fh.get_epochs_filtered(offset=offset, duration=duration,
                                        filter_lb=lb, filter_ub=ub,
                                        columns=columns)

    # Apply the reject
    reject_threshold = fh.trial_log.at[0, 'reject_threshold']
    m = np.abs(epochs) < reject_threshold
    m = m.all(axis=1)
    epochs = epochs.loc[m]

    epoch_reject_ratio = 1-m.groupby(columns[:-1]).mean()
    epoch_mean = epochs.groupby(columns).mean() \
        .groupby(columns[:-1]).mean()
    epoch_n = epochs.groupby(columns[:-1]).size()

    file_template = os.path.join(filename, file_template)
    raw_epoch_file = file_template.format('individual waveforms')
    mean_epoch_file = file_template.format('average waveforms')
    n_epoch_file = file_template.format('number of epochs')
    reject_ratio_file = file_template.format('reject ratio')

    # Write the data to CSV files
    epoch_reject_ratio.name = 'epoch_reject_ratio'
    epoch_reject_ratio.to_csv(reject_ratio_file, header=True)
    epoch_reject_ratio.name = 'epoch_n'
    epoch_n.to_csv(n_epoch_file, header=True)
    epoch_mean.columns.name = 'time'
    epoch_mean.T.to_csv(mean_epoch_file)
    epochs.columns.name = 'time'
    epochs.T.to_csv(raw_epoch_file)


def main():
    import argparse
    parser = argparse.ArgumentParser('Filter and summarize ABR data')

    parser.add_argument('filenames', type=str,
                        help='Filename', nargs='+')
    parser.add_argument('--offset', type=float,
                        help='Epoch offset',
                        default=-0.001)
    parser.add_argument('--duration', type=float,
                        help='Epoch duration',
                        default=0.01)
    parser.add_argument('--filter-lb', type=float,
                        help='Highpass filter cutoff',
                        default=None)
    parser.add_argument('--filter-ub', type=float,
                        help='Lowpass filter cutoff',
                        default=None)
    args = parser.parse_args()

    if args.filter_lb is not None or args.filter_ub is not None:
        filter_settings = {
            'lb': args.filter_lb,
            'ub': args.filter_ub,
        }
    else:
        filter_settings = None
    process_files(args.filenames, args.offset, args.duration, filter_settings)


if __name__ == '__main__':
    main()
