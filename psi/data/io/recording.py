import logging
log = logging.getLogger(__name__)

import functools
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


class Recording:
    '''
    Wrapper around a recording created by psiexperiment

    Parameters
    ----------
    base_path : :obj:`str` or :obj:`pathlib.Path`
        Folder containing recordings

    Attributes
    ----------
    base_path : pathlib.Path
        Folder containing recordings
    carray_names : set
        List of Bcolz carrays in this recording
    ctable_names : set
        List of Bcolz ctables in this recording
    ttable_names : set
        List of CSV-formatted tables in this recording

    The `__getattr__` method is implemented to allow accessing arrays and
    tables by name. For example, if you have a ctable called `erp_metadata`:

        recording = Recording(base_path)
        erp_md = recording.erp_metadata

    When using this approach, all tables are loaded into memory and returned as
    instances of `pandas.DataFrame`. All arrays are returned as `Signal`
    instances. Signal instances do not load the data into memory until the data
    is requested.
    '''

    #: Mapping of names for CSV-formatted table to a list of columns that
    #: should be used as indices. For example:
    #:     {'tone_sens': ['channel_name', 'frequency']}
    #: This attribute is typically used by subclasses to automate handling of
    #: loading tables into DataFrames.
    _ttable_indices = {}

    #: This is the name of the table containing the settings that we may wish
    #: to extract.
    _setting_table = None

    def __init__(self, base_path, setting_table=None):
        self._setting_table = setting_table
        self.base_path = Path(base_path)
        if self.base_path.suffix == '.zip':
            self._store = ZipStore(self.base_path, self._ttable_indices)
        elif self.base_path.is_dir():
            self._store = DirStore(self.base_path, self._ttable_indices)
        else:
            raise ValueError(f'Unrecognized recording format at {base_path}')

    def get_setting(self, setting_name):
        '''
        Return value for setting

        Parameters
        ----------
        setting_name : string
            Setting to extract

        Returns
        -------
        object
            Value of setting

        Raises
        ------
        ValueError
            If the setting is not identical across all trials.
        KeyError
            If the setting does not exist.
        '''
        table = getattr(self, self._setting_table)
        values = np.unique(table[setting_name])
        if len(values) != 1:
            raise ValueError('{name} is not unique across all epochs.')
        return values[0]

    def get_setting_default(self, setting_name, default):
        '''
        Return value for setting

        Parameters
        ----------
        setting_name : string
            Setting to extract
        default : obj
            Value to return if setting doesn't exist.

        Returns
        -------
        object
            Value of setting

        Raises
        ------
        ValueError
            If the setting is not identical across all trials.
        '''
        try:
            return self.get_setting(setting_name)
        except KeyError:
            return default

    def __getattr__(self, attr):
        return self._store.__getattr__(attr)

    def __repr__(self):
        lines = [f'Recording at {self.base_path.name} with:']
        if self._store.zarr_names:
            lines.append(f'* Zarr arrays {self._store.zarr_names}')
        if self._store.carray_names:
            lines.append(f'* Bcolz carrays {self._store.carray_names}')
        if self._store.ctable_names:
            lines.append(f'* Bcolz ctables {self._store.ctable_names}')
        if self._store.ttable_names:
            lines.append(f'* CSV tables {self._store.ttable_names}')
        return '\n'.join(lines)


class BaseStore:

    def _refresh_names(self):
        '''
        Utility function to refresh list of signals and tables
        '''
        raise NotImplementedError

    def __getattr__(self, attr):
        if attr in self.zarr_names:
            return self._load_zarr_signal(attr)
        if attr in self.carray_names:
            return self._load_bcolz_signal(attr)
        if attr in self.ctable_names:
            return self._load_bcolz_table(attr)
        if attr in self.ttable_names:
            return self._load_text_table(attr)
        raise AttributeError(attr)

    @functools.lru_cache()
    def _load_bcolz_signal(self, name):
        from .bcolz_tools import BcolzSignal
        return BcolzSignal(self.base_path / name)

    @functools.lru_cache()
    def _load_zarr_signal(self, name):
        from .zarr_tools import ZarrSignal
        return ZarrSignal.from_path(self.base_path / name)

    @functools.lru_cache()
    def _load_bcolz_table(self, name):
        from .bcolz_tools import load_ctable_as_df
        return load_ctable_as_df(self.base_path / name)

    def _get_text_table_stream(self, name):
        raise NotImplementedError

    @functools.lru_cache()
    def _load_text_table(self, name):
        import pandas as pd
        index_col = self._ttable_indices.get(name, None)
        with self._get_text_table_stream(name) as stream:
            df = pd.read_csv(stream, index_col=index_col)

        drop = [c for c in df.columns if c.startswith('Unnamed:')]
        return df.drop(columns=drop)


class DirStore(BaseStore):

    def __init__(self, base_path, ttable_indices):
        self.base_path = Path(base_path)
        self._ttable_indices = ttable_indices
        self._refresh_names()

    def _get_text_table_stream(self, name):
        path = (self.base_path / name).with_suffix('.csv')
        return path.open()

    def _refresh_names(self):
        bp = self.base_path
        self.carray_names = {d.parent.stem for d in bp.glob('*/meta')}
        self.ctable_names = {d.parent.parent.stem for d in bp.glob('*/*/meta')}
        self.ttable_names = {d.stem for d in bp.glob('*.csv')}
        self.zarr_names = {d.stem for d in bp.glob('*.zarr')}


class ZipStore(BaseStore):

    def __init__(self, base_path, ttable_indices):
        self.base_path = Path(base_path)
        self._ttable_indices = ttable_indices
        self.zip_fh = zipfile.ZipFile(base_path)
        self._refresh_names()

    def _refresh_names(self):
        self.carray_names = set()
        self.ctable_names = set()
        self.ttable_names = set()
        self.zarr_names = set()
        for name in self.zip_fh.namelist():
            if name.endswith('.zarr/'):
                self.zarr_names.add(name.split('.', 1)[0])
            elif name.endswith('.csv'):
                self.ttable_names.add(name.split('.', 1)[0])
            elif name.endswith('meta'):
                raise ValueError('ZipRecording does not support bcolz')

    def _get_text_table_stream(self, name):
        return self.zip_fh.open(f'{name}.csv')

    @functools.lru_cache()
    def _load_zarr_signal(self, name):
        from .zarr_tools import ZarrSignal
        return ZarrSignal.from_zip(self.base_path, name)
