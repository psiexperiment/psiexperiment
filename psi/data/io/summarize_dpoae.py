import logging
log = logging.getLogger(__name__)

import numpy as np
import pandas as pd


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


def isodp_th_criterions(df, criterions=None, debug=False):
    '''
    Helper function that takes dataframe containing a single frequency and
    calculates threshold for each criterion.
    '''
    if criterions is None:
        criterions = [-5, 0, 5, 10, 15, 20, 25]

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
