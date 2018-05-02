import os.path

from psi.data.io import abr


template = 'ABR {:.1f}ms to {:.1f}ms ' \
    'with {:.0f}Hz to {:.0f}Hz filter ' \
    '{{}}.csv'


def _main(args):
    columns = ['frequency', 'level', 'polarity']
    for filename in args.filenames:
        print('Processing {}'.format(filename))
        try:
            fh = abr.load(filename)
            epochs = fh.get_epochs_filtered(offset=args.offset,
                                            duration=args.duration,
                                            filter_lb=args.filter_lb,
                                            filter_ub=args.filter_ub,
                                            columns=columns)

            file_template = template.format(args.offset*1e3,
                                            (args.offset+args.duration)*1e3,
                                            args.filter_lb, args.filter_ub)
            file_template = os.path.join(filename, file_template)

            # Apply the reject
            reject_threshold = fh.trial_log.at[0, 'reject_threshold']
            m = epochs < reject_threshold
            m = m.all(axis=1)
            epochs = epochs.loc[m]

            epoch_reject_ratio = 1-m.groupby(columns[:-1]).mean()
            epoch_mean = epochs.groupby(columns).mean().groupby(columns[:-1]).mean()
            epoch_n = epochs.groupby(columns[:-1]).size()

            # Write the data to files
            epoch_reject_ratio.unstack().to_csv(file_template.format('reject ratio'))
            epoch_n.unstack().to_csv(file_template.format('number of epochs'))
            epoch_mean.T.to_csv(file_template.format('average waveform'))
        except:
            print('Error processing {}'.format(filename))


def main():
    import argparse
    parser = argparse.ArgumentParser('Summarize ABR data')

    parser.add_argument('filenames', type=str, help='Filename', nargs='+')
    parser.add_argument('--offset', type=float, help='Epoch offset',
                        default=-0.001)
    parser.add_argument('--duration', type=float, help='Epoch duration',
                        default=0.01)
    parser.add_argument('--filter_lb', type=float, help='Highpass filter cutoff',
                        default=300)
    parser.add_argument('--filter_ub', type=float, help='Lowpass filter cutoff',
                        default=3000)

    args = parser.parse_args()
    _main(args)


if __name__ == '__main__':
    main()
