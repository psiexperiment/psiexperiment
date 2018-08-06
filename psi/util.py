import ast
import inspect
import threading

import numpy as np
import pandas as pd
from atom.api import Property


def as_numeric(x):
    if not isinstance(x, (np.ndarray, pd.DataFrame, pd.Series)):
        x = np.asanyarray(x)
    return x


def rpc(plugin_name, method_name):
    '''Decorator to map an Enaml command handler to the plugin method'''
    def wrapper(event):
        plugin = event.workbench.get_plugin(plugin_name)
        f = getattr(plugin, method_name)
        argnames = inspect.getargspec(f).args
        parameters = {}
        for k, v in event.parameters.items():
            if k in argnames:
                parameters[k] = v
        return getattr(plugin, method_name)(**parameters)
    return wrapper


def get_tagged_values(obj, tag_name, tag_value=True, exclude_properties=False):
    result = {}
    if exclude_properties:
        match = lambda m: m.metadata and m.metadata.get(tag_name) == tag_value \
            and not isinstance(member, Property)
    else:
        match = lambda m: m.metadata and m.metadata.get(tag_name) == tag_value

    for name, member in obj.members().items():
        if match(member):
            value = getattr(obj, name)
            result[name] = value
    return result


def declarative_to_dict(obj, tag_name, tag_value=True):
    from atom.api import Atom
    import numpy as np
    result = {}
    for name, member in obj.members().items():
        if member.metadata and member.metadata.get(tag_name) == tag_value:
            value = getattr(obj, name)
            if isinstance(value, Atom):
                # Recurse into the class
                result[name] = declarative_to_dict(value, tag_name, tag_value)
            elif isinstance(value, np.ndarray):
                # Convert to a list
                result[name] = value.tolist()
            else:
                result[name] = value
    result['type'] = obj.__class__.__name__
    return result


def coroutine(func):
    '''Decorator to auto-start a coroutine.'''
    def start(*args, **kwargs):
        cr = func(*args, **kwargs)
        next(cr)
        return cr
    return start


def copy_declarative(old, **kw):
    attributes = get_tagged_values(old, 'metadata', exclude_properties=True)
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

    def __init__(self, fs, size, fill_value=np.nan, dtype=np.double):
        self._lock = threading.RLock()
        self._buffer_fs = fs
        self._buffer_size = size
        self._buffer_samples = round(fs*size)
        self._buffer = np.full(self._buffer_samples, fill_value, dtype=dtype)
        self._fill_value = fill_value
        self._samples = 0
        self._ilb = self._buffer_samples

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
            return np.pad(data, (lpadding, rpadding), 'constant',
                        constant_values=fill_value)

    def get_range(self, lb=None, ub=None, fill_value=None):
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
            return self._buffer[ilb:iub]

    def append_data(self, data):
        with self._lock:
            samples = data.shape[-1]
            if samples > self._buffer_samples:
                self._buffer[:] = data[-self._buffer_samples:]
                self._ilb = 0
            else:
                self._buffer[:-samples] = self._buffer[samples:]
                self._buffer[-samples:] = data
                self._ilb = max(0, self._ilb - samples)
            self._samples += samples

    def _invalidate(self, i):
        # This is only called by invalidate or invalidate_samples, which are
        # already wrapped inside a lock block.
        if i <= 0:
            self._buffer[:] = self._fill_value
            self._ilb = self._buffer_samples
        else:
            self._buffer[-i:] = self._buffer[:i]
            self._buffer[:-i] = np.nan
            self._ilb = self._ilb + self._buffer_samples - i

    def invalidate(self, t):
        with self._lock:
            bi = self.time_to_index(t)
            self._invalidate(bi)
            dt = self.get_time_ub() - t
            di = round(dt * self._buffer_fs)
            self._samples -= di

    def invalidate_samples(self, i):
        with self._lock:
            bi = self.samples_to_index(i)
            self._invalidate(bi)
            di = self.get_samples_ub() - i
            self._samples -= di

    def get_latest(self, lb, ub=0):
        with self._lock:
            lb = lb + self.get_time_ub()
            ub = ub + self.get_time_ub()
            return self.get_range(lb, ub)

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


def octave_space(lb, ub, step):
    '''
    >>> freq = octave_space(4, 32, 1)
    >>> print(freq)
    [ 4.  8. 16. 32.]
    '''
    lbi = round(np.log2(lb) / step) * step
    ubi = round(np.log2(ub) / step) * step
    x = np.arange(lbi, ubi+step, step)
    return 2**x


class ConfigurationException(Exception):
    pass
