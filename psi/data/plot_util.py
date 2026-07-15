'''
Pure-Python helpers for the plotting subsystem.

Nothing in this module may import pyqtgraph or Qt: it holds the numeric and
formatting logic (decimation, tick formatting, color cycles) so it can be
tested without a GUI.
'''
import importlib
import string

import numpy as np


def get_freq(fs, duration):
    n_time = int(round(fs * duration))
    return np.fft.rfftfreq(n_time, fs**-1)


def get_color_cycle(name, n):
    module_name, cmap_name = name.rsplit('.', 1)
    module = importlib.import_module(module_name)

    # This generates a LinearSegmentedColormap instance that interpolates to
    # the requested number of colors. We can then extract these colors by
    # calling the colormap with a mapping of 0 ... 1 where the number of
    # values in the array is the number of colors we need (spaced equally
    # along 0 ... 1).
    cmap = getattr(module, cmap_name).mpl_colormap.resampled(n)
    for i in np.linspace(0, 1, n):
        yield tuple(int(v * 255) for v in cmap(i))


def format_time(seconds, fmt='{H:02d}:{M:02d}:{S:02.0f}'):
    names = [i[1] for i in string.Formatter().parse(fmt)]
    values = {}
    if 'H' in names:
        values['H'] = hours = int(seconds) // (60 * 60)
        seconds -= (hours * 60 * 60)
    if 'M' in names:
        values['M'] = minutes = int(seconds) // 60
        seconds -= (minutes * 60)
    if 'S' in names:
        values['S'] = seconds
    return fmt.format(**values)


def format_log_ticks(values, scale, spacing):
    values = 10**np.array(values).astype(float)
    return ['{:.1f}'.format(v * 1e-3) for v in values]


def _reshape_for_decimate(data, downsample):
    # Determine the "fragment" size that we are unable to decimate.  A
    # downsampling factor of 5 means that we perform the operation in chunks
    # of 5 samples.  If we have only 13 samples of data, then we cannot
    # decimate the last 3 samples and will simply discard them.
    offset = data.shape[-1] % downsample
    if offset > 0:
        data = data[..., :-offset]
    shape = (len(data), -1, downsample) if data.ndim == 2 else (-1, downsample)
    return data.reshape(shape)


def decimate_mean(data, downsample):
    # If data is empty, return immediately. Regression note: this used to
    # return a two-element tuple (copy-paste from decimate_extremes), which
    # broke the caller's len()/isnan() handling on an empty buffer.
    if data.size == 0:
        return np.array([])
    data = _reshape_for_decimate(data, downsample).copy()
    return data.mean(axis=-1)


def decimate_extremes(data, downsample):
    # If data is empty, return immediately
    if data.size == 0:
        return np.array([]), np.array([])

    # Force a copy to be made, which speeds up min()/max().  Apparently
    # min/max make a copy of a reshaped array before performing the
    # operation, so we force it now so the copy only occurs once.
    data = _reshape_for_decimate(data, downsample).copy()
    return data.min(axis=-1), data.max(axis=-1)
