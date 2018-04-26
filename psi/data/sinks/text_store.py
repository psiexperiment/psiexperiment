import logging
log = logging.getLogger(__name__)

import os.path
import atexit
import tempfile
import shutil
import json

from .base_store import BaseStore


class TextStore(BaseStore):
    '''
    Simple class for storing data in human-readable formats (CSV and text)
    '''
    name = 'text_store'

    def create_table(self, name, dataframe):
        path = self.get_filename(name, '.csv')
        if path.exists():
            raise IOError('{} already exists'.format(path))
        dataframe.to_csv(path)

    def update_table(self, name, dataframe):
        path = self.get_filename(name, '.csv')
        dataframe.to_csv(path)

    def create_mapping(self, name, mapping):
        path = self.get_filename(name, '.json')
        if path.exists():
            raise IOError('{} already exists'.format(path))
        with open(path, 'w') as fh:
            json.dump(mapping, fh, sort_keys=True, indent=4)

    def update_mapping(self, name, mapping):
        path = self.get_filename(name, '.json')
        with open(path, 'w') as fh:
            json.dump(mapping, fh, sort_keys=True, indent=4)
