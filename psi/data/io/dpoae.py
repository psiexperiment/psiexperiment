import logging
log = logging.getLogger(__name__)

from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import Recording

# Max size of LRU cache
MAXSIZE = 1024


def isodp_th(l2, dp, nf, criterion):
    '''
    Computes iso-DP threshold for a level sweep at a single frequency

    Parameters
    ----------
    l2 : array-like
        Requested F2 levels
    dp : array-like
        Measured DPOAE levels
    nf : array-like
        Measured DPOAE noise floor
    criterion : float
        Threshold criterion (e.g., value that the input-output function must
        exceed)

    Returns
    -------
    threshold : float
        If no threshold is identified, NaN is returned
    '''
    # First, discard up to the first level where the DPOAE exceeds one standard
    # deviation from the noisne floor
    nf_crit = np.mean(nf) + np.std(nf)
    i = np.flatnonzero(dp < nf_crit)
    if len(i):
        dp = dp[i[-1]:]
        l2 = l2[i[-1]:]

    # Now, loop through every pair of points until we find the first pair that
    # brackets criterion (for non-Python programmers, this uses chained
    # comparision operators and is not a bug)
    for l_lb, l_ub, d_lb, d_ub in zip(l2[:-1], l2[1:], dp[:-1], dp[1:]):
        if d_lb < criterion <= d_ub:
            return np.interp(criterion, [d_lb, d_ub], [l_lb, l_ub])
    return np.nan


def isodp_th_criterions(df, criterions, debug=False):
    '''
    Helper function that takes dataframe containing a single frequency and
    calculates threshold for each criterion.
    '''
    if ':dB' in df.columns:
        # This is used for thresholding data already in EPL CFTS format
        l2 = df.loc[:, ':dB']
        dp = df.loc[:, '2f1-f2(dB)']
        nf = df.loc[:, '2f1-f2Nse(dB)']
    else:
        # This is used for thresholding data from the psi DPOAE IO.
        if debug:
            # Use a measurable signal to estimate threshold.
            l2 = df.loc[:, 'secondary_tone_level'].values
            dp = df.loc[:, 'f2_level'].values
            nf = df.loc[:, 'dpoae_noise_floor'].values
        else:
            l2 = df.loc[:, 'secondary_tone_level'].values
            dp = df.loc[:, 'dpoae_level'].values
            nf = df.loc[:, 'dpoae_noise_floor'].values

    th = [isodp_th(l2, dp, nf, c) for c in criterions]
    index = pd.Index(criterions, name='criterion')
    return pd.Series(th, index=index, name='threshold')


def plot_isodp_thresholds(thresholds, filename=None):
    colors = ['k', '#9a0a0a', '#3770a9', '#22b222', '#44280b', 'k', '#9a0a0a']
    markers = ['o'] * 5 + ['x'] * 2
    figure, axes = plt.subplots(1, 1, figsize=(6, 4))
    for c, m, (crit, th) in zip(colors, markers, thresholds.T.iterrows()):
        axes.semilogx(th.index, th.values, '-', marker=m, color=c, ms=7, label=f'{crit} dB SPL')
    axes.axis(xmin=1e3, xmax=100e3, ymin=0, ymax=90)
    axes.grid(True, which='major')
    axes.grid(True, which='minor', ls=':')
    axes.xaxis.set_ticks([1e3, 10e3, 100e3])
    axes.xaxis.set_ticklabels([1, 10, 100])
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    axes.legend(loc='lower left')

    ticks = np.hstack((np.arange(1e3, 10e3, 1e3), np.arange(10e3, 100e3, 10e3)))
    axes.xaxis.set_ticks(ticks, minor=True);
    axes.set_xlabel('f2 Frequency (kHz)')
    axes.set_ylabel('f2 Level (L2) (dB SPL)')

    if filename is not None:
        figure.savefig(filename)


def plot_io(dpoae, filename_template=None):
    for f2, f2_data in dpoae.groupby('f2_frequency'):
        figure, axes = plt.subplots(1, 1, figsize=(6, 6))
        axes.plot(f2_data['secondary_tone_level'], f2_data['f1_level'], 'ks')
        axes.plot(f2_data['secondary_tone_level'], f2_data['f2_level'], 'ks', mfc='w')
        axes.plot(f2_data['secondary_tone_level'], f2_data['dpoae_level'], 'o', color='#3770a9')
        axes.plot(f2_data['secondary_tone_level'], f2_data['dpoae_noise_floor'], 'x', color='#3770a9')
        axes.set_title(f'f2 = {f2*1e-3:.1f} kHz', fontweight='bold')
        axes.set_xlabel('f2 Level (dB SPL)')
        axes.set_ylabel('Level (dB SPL)')
        axes.grid()

        l2 = f2_data['secondary_tone_level'].unique()
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        axes.axis(xmin=l2.min()-5, xmax=l2.max()+5, ymin=-30, ymax=l2.max()+15)

        if filename_template is not None:
            filename = filename_template.format(frequency=f2*1e-3)
            figure.savefig(filename)


DP_HEADER = '''
# Version: 0.0.1
# Experiment: DPOAE I-O
{metadata}
{data}
'''
def write_epl_cfts_dpoae_file(dpoae, metadata, filename):
    columns = {
        'secondary_tone_level': 'dB',
        'secondary_tone_frequency': 'f1(Hz)',
        'primary_tone_frequency': 'f2(Hz)',
        'f1_level': 'f1(dB)',
        'f2_level': 'f2(dB)',
        'dpoae_level': '2f1-f2(dB)',
        'dpoae_noise_floor': '2f1-f2Nse(dB)',
    }
    for column in columns.keys():
        if column not in dpoae.columns:
            dpoae[column] = np.nan

    keep = list(columns.keys())
    dpoae_subset = dpoae[keep].astype('float').rename(columns=columns)
    csv_string = dpoae_subset.to_csv(float_format='%.2f', sep='\t',
                                     na_rep='NaN', index=False, header=True,
                                     line_terminator='\n')
    md_string = '\n'.join(f'# {k}: {v}' for k, v in metadata.items())
    text = DP_HEADER.strip().format(data=csv_string, metadata=md_string)
    Path(filename).write_text(text)


def write_epl_cfts_isodp_file(thresholds, metadata, filename):
    th = thresholds.copy()
    th.index.name = 'f2(Hz)'
    th = th.stack(dropna=False).rename('Th(dB)').reset_index()
    th_string = th.to_csv(float_format='%0.2f', sep='\t', na_rep='NaN',
                          header=True, index=False, line_terminator='\n')
    md_string = '\n'.join(f'# {k}: {v}' for k, v in metadata.items())
    text = DP_HEADER.strip().format(metadata=md_string, data=th_string)
    Path(filename).write_text(text)


def convert_epl(base_path):
    base_path = Path(base_path)
    fh = DPOAEFile(base_path)

    crit = [-5, 0, 5, 10, 15, 20, 25]
    dp_filename = base_path / 'DP'
    isodp_filename = base_path / 'IsoDP'
    isodp_fig_filename = base_path / 'Isoresponse_curves.png'
    dp_fig_filename = str(base_path / '{frequency:.2f} kHz.png')

    data = fh.dpoae_store.copy()
    col_extra = ['response_window', 'n_time', 'n_fft', 'max_dpoae_noise_floor']
    md = data.iloc[0][col_extra]

    thresholds = data.groupby('secondary_tone_frequency') \
        .apply(isodp_th_criterions, criterions=crit)

    plot_io(data, filename_template=dp_fig_filename)
    plot_isodp_thresholds(thresholds, filename=isodp_fig_filename)
    write_epl_cfts_dpoae_file(data, md, filename=dp_filename)
    write_epl_cfts_isodp_file(thresholds, md, isodp_filename)
    log.debug('Converted to EPL CFTS format')


def dpoae_renamer(x):
    if x in ('f1_level', 'f2_level', 'dpoae_level'):
        return f'meas_{x}'
    return x.replace('primary_tone', 'f1') \
        .replace('secondary_tone', 'f2')


class DPOAEFile(Recording):

    @property
    @lru_cache(maxsize=MAXSIZE)
    def results(self):
        data = self._load_bcolz_table('dpoae_store')
        return data.rename(columns=dpoae_renamer)


def load(filename):
    return DPOAEFile(filename)
