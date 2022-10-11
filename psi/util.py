import logging
log = logging.getLogger(__name__)

import ast
import inspect
import json
from pathlib import Path
import threading
import uuid

import numpy as np
import pandas as pd
from atom.api import Atom, Property

from psiaudio.calibration import BaseCalibration


def as_numeric(x):
    if not isinstance(x, (np.ndarray, pd.DataFrame, pd.Series)):
        x = np.asanyarray(x)
    return x


def rpc(plugin_name, method_name):
    '''Decorator to map an Enaml command handler to the plugin method'''
    def wrapper(event):
        plugin = event.workbench.get_plugin(plugin_name)
        f = getattr(plugin, method_name)
        argnames = inspect.getfullargspec(f).args
        parameters = {}
        for k, v in event.parameters.items():
            if k in argnames:
                parameters[k] = v
        return getattr(plugin, method_name)(**parameters)
    return wrapper


def get_tagged_members(obj, tag_name, tag_value=True, exclude_properties=False):
    result = {}
    if exclude_properties:
        match = lambda m: m.metadata and m.metadata.get(tag_name) == tag_value \
            and not isinstance(m, Property)
    else:
        match = lambda m: m.metadata and m.metadata.get(tag_name) == tag_value
    return {n: m for n, m in obj.members().items() if match(m)}


def get_tagged_values(obj, tag_name, tag_value=True, exclude_properties=False):
    members = get_tagged_members(obj, tag_name, tag_value, exclude_properties)
    return {n: getattr(obj, n) for n in members}


def declarative_to_dict(value, tag_name, tag_value=True, include_dunder=True,
                        seen_objects=None):
    if seen_objects is None:
        seen_objects = []

    args = (tag_name, tag_value, include_dunder, seen_objects)

    if isinstance(value, int) \
            or isinstance(value, float) \
            or isinstance(value, str) \
            or value is None:
        return value

    if isinstance(value, Path):
        return str(value)

    if id(value) in seen_objects:
        return f'__obj__::{id(value)}'
    else:
        seen_objects.append(id(value))

    if isinstance(value, Atom):
        if include_dunder:
            result = {
                '__type__': value.__class__.__name__,
                '__id__': id(value),
            }
        else:
            result = {}
        for name, member in value.members().items():
            if member.metadata and member.metadata.get(tag_name) == tag_value:
                v = getattr(value, name)
                result[name] = declarative_to_dict(v, *args)
        return result

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, list):
        return [declarative_to_dict(v, *args) for v in value]

    if isinstance(value, BaseCalibration):
        # Special case for the Calibration data since it's from psiaudio (and
        # we do not wish to introduce extra dependencies in psiaudio).
        attrs = [a for a in dir(value) if not \
                 (a.startswith('_') or callable(getattr(value, a)))]
        result = {a: declarative_to_dict(getattr(value, a), *args) \
                  for a in attrs}
        if include_dunder:
            result.update({
                '__type__': value.__class__.__name__,
                '__id__': id(value)
            })
        return result

    if hasattr(value, '__dict__') or hasattr(value, '__slots__'):
        if include_dunder:
            return {
                '__type__': value.__class__.__name__,
                '__id__': id(value),
            }
        else:
            return {}

    return value


def declarative_to_json(filename, value, tag_name, tag_value=True,
                        include_dunder=True, seen_objects=None):
    filename = Path(filename)
    info = declarative_to_dict(value, tag_name, tag_value,
                               include_dunder=include_dunder)
    filename.write_text(json.dumps(info, indent=2))


def dict_to_declarative(obj, info, skip_errors=False):
    for k, v in info.items():
        if k in ('type', '__type__', '__id__'):
            continue
        if isinstance(v, dict) and ('type' in v or '__type__' in v):
            dict_to_declarative(getattr(obj, k), v, skip_errors)
        else:
            try:
                setattr(obj, k, v)
            except Exception as e:
                if not skip_errors:
                    raise


def copy_declarative(old, exclude=None, **kw):
    attributes = get_tagged_values(old, 'metadata', exclude_properties=True)
    if exclude is not None:
        for k in exclude:
            attributes.pop(k, None)
    attributes.update(kw)
    new = old.__class__(**attributes)
    return new


class _FullNameGetter(ast.NodeVisitor):

    def __init__(self, *args, **kwargs):
        self.names = []
        super(_FullNameGetter, self).__init__(*args, **kwargs)

    def visit_Name(self, node):
        self.names.append(node.id)

    def visit_Attribute(self, node):
        names = []
        while isinstance(node, ast.Attribute):
            names.append(node.attr)
            node = node.value
        names.append(node.id)
        name = '.'.join(names[::-1])
        self.names.append(name)


def get_dependencies(expression):
    tree = ast.parse(expression)
    ng = _FullNameGetter()
    ng.visit(tree)
    return ng.names


class SignalBuffer:

    def __init__(self, fs, size, fill_value=np.nan, dtype=np.double,
                 n_channels=None):
        '''

        Parameters
        ----------
        fs : float
            Sampling rate of buffered signal
        size : float
            Duration of buffer in sec
        fill_value
        dtype
        '''
        log.debug('Creating signal buffer with fs=%f and size=%f', fs, size)
        self._lock = threading.RLock()
        self._buffer_fs = fs
        self._buffer_samples = int(np.ceil(fs*size))
        self._n_channels = n_channels
        self._size = size

        if n_channels is not None:
            shape = (self._n_channels, self._buffer_samples)
        else:
            shape = self._buffer_samples

        self._buffer = np.full(shape, fill_value, dtype=dtype)
        self._fill_value = fill_value
        self._samples = 0

        # This is an attribute that represents the current "start" of the
        # buffered data. We always "push" data into the array from the right,
        # so the newest samples will appear at the end of the array. In the
        # beginning, the "start" of the buffered data is at the very end of hte
        # array, but as we fill up the array, `_ilb` will eventually always be
        # 0.
        self._ilb = self._buffer_samples

    def resize(self, size):
        '''
        Resize buffer to hold the specified number of time samples

        Notes
        -----
        * This does not allow you to add/remove channels.
        * A request to decrease buffer size is ignored.
        '''
        # Don't shrink buffer size. Not worth the effort.
        with self._lock:
            old_samples = self._buffer_samples
            self._buffer = self.get_latest(-size, fill_value=self._fill_value)
            self._buffer_samples = self._buffer.shape[-1]

            new_samples = self._buffer_samples
            # This corrects the lower bound index to point to the oldest data
            # in the buffer.
            self._ilb = max(0, self._ilb + (new_samples - old_samples))

    def time_to_samples(self, t):
        '''
        Convert time to samples (re acquisition start)
        '''
        return round(t*self._buffer_fs)

    def time_to_index(self, t):
        '''
        Convert time to index in buffer. Note that the index may fall out of
        the buffered range.
        '''
        i = self.time_to_samples(t)
        return self.samples_to_index(i)

    def samples_to_index(self, i):
        # Convert index to the index in the buffer. Note that the index can
        # fall outside the buffered range.
        return i - self._samples + self._buffer_samples

    def get_range_filled(self, lb, ub, fill_value):
        # Index of requested range
        with self._lock:
            ilb = self.time_to_samples(lb)
            iub = self.time_to_samples(ub)
            # Index of buffered range
            slb = self.get_samples_lb()
            sub = self.get_samples_ub()
            lpadding = max(slb-ilb, 0)
            elb = max(slb, ilb)
            rpadding = max(iub-sub, 0)
            eub = min(sub, iub)
            data = self.get_range_samples(elb, eub)

            padding = (lpadding, rpadding)
            if data.ndim == 2:
                padding = ((0, 0), (lpadding, rpadding))
            else:
                padding = (lpadding, rpadding)

            return np.pad(data, padding, 'constant',
                         constant_values=fill_value)

    def get_range(self, lb=None, ub=None):
        with self._lock:
            if lb is None:
                lb = self.get_time_lb()
            if ub is None:
                ub = self.get_time_ub()
            ilb = None if lb is None else self.time_to_samples(lb)
            iub = None if ub is None else self.time_to_samples(ub)
            return self.get_range_samples(ilb, iub)

    def get_range_samples(self, lb=None, ub=None):
        with self._lock:
            if lb is None:
                lb = self.get_samples_lb()
            if ub is None:
                ub = self.get_samples_ub()
            ilb = self.samples_to_index(lb)
            iub = self.samples_to_index(ub)
            if ilb < self._ilb:
                raise IndexError
            elif iub > self._buffer_samples:
                raise IndexError
            return self._buffer[..., ilb:iub]

    def append_data(self, data):
        if self._n_channels is not None:
            if data.ndim != 2:
                raise ValueError('Appended data must be two-dimensional')
            if data.shape[0] != self._n_channels:
                raise ValueError(f'Appended data must have {self._n_channels} channels.')
        else:
            if data.ndim != 1:
                raise ValueError('Appended data must be one-dimensional')

        with self._lock:
            samples = data.shape[-1]
            if samples > self._buffer_samples:
                self._buffer[..., :] = data[..., -self._buffer_samples:]
                self._ilb = 0
            else:
                self._buffer[..., :-samples] = self._buffer[..., samples:]
                self._buffer[..., -samples:] = data
                self._ilb = max(0, self._ilb - samples)
            self._samples += samples

    def _invalidate(self, i):
        # This is only called by invalidate or invalidate_samples, which are
        # already wrapped inside a lock block.
        if i <= 0:
            self._buffer[:] = self._fill_value
            self._ilb = self._buffer_samples
        else:
            self._buffer[..., -i:] = self._buffer[..., :i]
            self._buffer[..., :-i] = np.nan
            self._ilb = self._ilb + self._buffer_samples - i

    def invalidate(self, t):
        with self._lock:
            self.invalidate_samples(self.time_to_samples(t))

    def invalidate_samples(self, i):
        with self._lock:
            if i >= self._samples:
                return
            bi = self.samples_to_index(i)
            self._invalidate(bi)
            di = self.get_samples_ub() - i
            self._samples -= di

    def get_latest(self, lb, ub=0, fill_value=None):
        '''
        Returns buffered data relative to the most recently-buffered sample

        Parameters
        ----------
        lb : float
            Time in seconds relative to the most recently-buffered sample.
            Usually will be a negative value.
        ub : float
            Time in seconds relative to the most recently-buffered sample.
            Usually will be 0 or a negative value.

        Examples
        --------
        Get the most recent 1 second of buffered data
        >>> buffer.get_latest(-1)

        Get the buffered data from -2 to -1 relative to current time.
        >>> buffer.get_latest(-2, -1)
        '''
        with self._lock:
            log.trace('Converting latest %f to %f to absolute time', lb, ub)
            lb = lb + self.get_time_ub()
            ub = ub + self.get_time_ub()
            log.trace('Absolute time is %f to %f', lb, ub)
            if fill_value is None:
                return self.get_range(lb, ub)
            else:
                return self.get_range_filled(lb, ub, fill_value)

    def get_time_lb(self):
        return self.get_samples_lb()/self._buffer_fs

    def get_time_ub(self):
        with self._lock:
            return self.get_samples_ub()/self._buffer_fs

    def get_samples_lb(self):
        with self._lock:
            return self._samples - self._buffer_samples + self._ilb

    def get_samples_ub(self):
        with self._lock:
            return self._samples


class ConfigurationException(Exception):
    pass


def psi_json_decoder_hook(obj):
    '''
    This adds support for loading legacy files generated using json-tricks
    '''
    if isinstance(obj, dict) and '__ndarray__' in obj:
        return np.asarray(obj['__ndarray__'], dtype=obj['dtype'])
    else:
        return obj


class PSIJsonEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, uuid.UUID):
            return str(obj)
        elif isinstance(obj, slice):
            return str(obj)
        else:
            return super().default(obj)
