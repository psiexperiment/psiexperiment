import logging
log = logging.getLogger(__name__)

import os.path
import atexit
import tempfile
import shutil
import json

from atom.api import Unicode
from enaml.core.api import d_

from .base_store import BaseStore


class TextStore(BaseStore):
    '''
    Simple class for storing data in human-readable formats (CSV and text)
    '''
    name = d_(Unicode('text_store'))

    def create_table(self, name, dataframe):
        path = self.get_filename(name, '.csv')
        if path.exists():
            raise IOError('{} already exists'.format(path))
        dataframe.to_csv(path)
        return path

    def update_table(self, name, dataframe):
        path = self.get_filename(name, '.csv')
        dataframe.to_csv(path)
        return path

    def create_mapping(self, name, mapping):
        path = self.get_filename(name, '.json')
        if path.exists():
            raise IOError('{} already exists'.format(path))
        with open(path, 'w') as fh:
            json.dump(mapping, fh, sort_keys=True, indent=4)
        return path

    def update_mapping(self, name, mapping):
        path = self.get_filename(name, '.json')
        with open(path, 'w') as fh:
            json.dump(mapping, fh, sort_keys=True, indent=4)
        return path
