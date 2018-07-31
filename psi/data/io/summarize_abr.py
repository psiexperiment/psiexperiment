import argparse
from glob import glob
import os.path

import numpy as np

from psi.data.io import abr


nofilter_template = 'ABR {:.1f}ms to {:.1f}ms {{}}.csv'

filter_template = 'ABR {:.1f}ms to {:.1f}ms ' \
    'with {:.0f}Hz to {:.0f}Hz filter ' \
    '{{}}.csv'


columns = ['frequency', 'level', 'polarity']


def process_folder(folder, filter_settings=None):
    glob_pattern = os.path.join(folder, '*abr')
    filenames = glob(glob_pattern)
    process_files(filenames, filter_settings=filter_settings)


def process_files(filenames, offset=-0.001, duration=0.01,
                  filter_settings=None, reprocess=False):
    for filename in filenames:
        try:
            processed = process_file(filename, offset, duration,
                                     filter_settings, reprocess)
            if processed:
                print(f'\nProcessed {filename}\n')
            else:
                print('.', end='', flush=True)
        except Exception as e:
            print(f'\nError processing {filename}\n{e}\n')
            raise


def _get_file_template(fh, offset, duration, filter_settings):
    template_args = [offset*1e3, (offset+duration)*1e3]
    if filter_settings == 'saved':
        settings = fh.trial_log.iloc[0]
        if not settings.get('digital_filter', True):
            file_template = nofilter_template.format(*template_args)
        else:
            lb = settings.get('digital_highpass', 300)
            ub = settings.get('digital_lowpass', 3000)
            template_args += [lb, ub]
            file_template = filter_template.format(*template_args)
    elif filter_settings is None:
        file_template = nofilter_template.format(*template_args)
    else:
        lb = filter_settings['lb']
        ub = filter_settings['ub']
        template_args += [lb, ub]
        file_template = filter_template.format(*template_args)
    return file_template


def _get_epochs(fh, offset, duration, filter_settings):
    kwargs = {'offset': offset, 'duration': duration, 'columns': columns}
    if filter_settings == 'saved':
        settings = fh.trial_log.iloc[0]
        if not settings.get('digital_filter', True):
            epochs = fh.get_epochs(**kwargs)
        else:
            lb = settings.get('digital_highpass', 300)
            ub = settings.get('digital_lowpass', 3000)
            kwargs.update({'filter_lb': lb, 'filter_ub': ub})
            epochs = fh.get_epochs_filtered(**kwargs)
    elif filter_settings is None:
        epochs = fh.get_epochs(**kwargs)
    else:
        lb = filter_settings['lb']
        ub = filter_settings['ub']
        kwargs.update({'filter_lb': lb, 'filter_ub': ub})
        epochs = fh.get_epochs_filtered(**kwargs)
    return epochs


def process_file(filename, offset, duration, filter_settings, reprocess=False):
    '''
    Extract ABR epochs, filter and save result to CSV files

    Parameters
    ----------
    filename : path
        Path to ABR experiment. If it's a set of ABR experiments, epochs across
        all experiments will be combined for the analysis.
    offset : sec
        The start of the epoch to extract, in seconds, relative to tone pip
        onset. Negative values can be used to extract a prestimulus baseline.
    duration: sec
        The duration of the epoch to extract, in seconds, relative to the
        offset. If offset is set to -0.001 sec and duration is set to 0.01 sec,
        then the epoch will be extracted from -0.001 to 0.009 sec re tone pip
        onset.
    filter_settings : {None, 'saved', dict}
        If None, no additional filtering is done. If 'saved', uses the digital
        filter settings that were saved in the ABR file. If a dictionary, must
        contain 'lb' (the lower bound of the passband in Hz) and 'ub' (the
        upper bound of the passband in Hz).
    reprocess : bool
        If True, reprocess the file even if it already has been processed for
        the specified filter settings.
    '''
    fh = abr.load(filename)

    # Generate the filenames
    file_template = _get_file_template(fh, offset, duration, filter_settings)
    file_template = os.path.join(filename, file_template)
    raw_epoch_file = file_template.format('individual waveforms')
    mean_epoch_file = file_template.format('average waveforms')
    n_epoch_file = file_template.format('number of epochs')
    reject_ratio_file = file_template.format('reject ratio')

    # Check to see if all of them exist before reprocessing
    if not reprocess and \
            (os.path.exists(raw_epoch_file) and \
             os.path.exists(mean_epoch_file) and \
             os.path.exists(n_epoch_file) and \
             os.path.exists(reject_ratio_file)):
        return False

    # Load the epochs
    epochs = _get_epochs(fh, offset, duration, filter_settings)

    # Apply the reject
    reject_threshold = fh.trial_log.at[0, 'reject_threshold']
    m = np.abs(epochs) < reject_threshold
    m = m.all(axis=1)
    epochs = epochs.loc[m]

    epoch_reject_ratio = 1-m.groupby(columns[:-1]).mean()
    epoch_mean = epochs.groupby(columns).mean() \
        .groupby(columns[:-1]).mean()
    epoch_n = epochs.groupby(columns[:-1]).size()

    # Write the data to CSV files
    epoch_reject_ratio.name = 'epoch_reject_ratio'
    epoch_reject_ratio.to_csv(reject_ratio_file, header=True)
    epoch_reject_ratio.name = 'epoch_n'
    epoch_n.to_csv(n_epoch_file, header=True)
    epoch_mean.columns.name = 'time'
    epoch_mean.T.to_csv(mean_epoch_file)
    epochs.columns.name = 'time'
    epochs.T.to_csv(raw_epoch_file)
    return True


def main_auto():
    parser = argparse.ArgumentParser('Filter and summarize ABR files in folder')
    parser.add_argument('folder', type=str, help='Folder containing ABR data')
    args = parser.parse_args()
    process_folder(args.folder, filter_settings='saved')


def main():
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
    parser.add_argument('--reprocess',
                        help='Redo existing results',
                        action='store_true')
    args = parser.parse_args()

    if args.filter_lb is not None or args.filter_ub is not None:
        filter_settings = {
            'lb': args.filter_lb,
            'ub': args.filter_ub,
        }
    else:
        filter_settings = None
    process_files(args.filenames, args.offset, args.duration, filter_settings,
                  args.reprocess)


if __name__ == '__main__':
    main()
