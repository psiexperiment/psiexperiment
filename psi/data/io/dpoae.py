import logging
log = logging.getLogger(__name__)

from functools import lru_cache

from . import Recording

# Max size of LRU cache
MAXSIZE = 1024


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
