import blosc
import functools
import json
import struct
import numpy as np

from . import Signal


BLOSCPACK_HEADER_LENGTH = 16
BLOSC_HEADER_LENGTH = 16


def decode_uint32(x):
    return struct.unpack('<I', x)[0]


def decode_blosc_header(buffer):
    """ Read and decode header from compressed Blosc buffer.

    Parameters
    ----------
    buffer_ : string of bytes
        the compressed buffer

    Returns
    -------
    settings : dict
        a dict containing the settings from Blosc

    Notes
    -----

    The Blosc 1.1.3 header is 16 bytes as follows:

    | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | A | B | C | D | E | F |
      ^   ^   ^   ^ |     nbytes    |   blocksize   |    ctbytes    |
      |   |   |   |
      |   |   |   +--typesize
      |   |   +------flags
      |   +----------versionlz
      +--------------version

    The first four are simply bytes, the last three are are each unsigned ints
    (uint32) each occupying 4 bytes. The header is always little-endian.
    'ctbytes' is the length of the buffer including header and nbytes is the
    length of the data when uncompressed.

    """
    return {
        'version': buffer[0],
        'versionlz': buffer[1],
        'flags': buffer[2],
        'typesize': buffer[3],
        'nbytes': decode_uint32(buffer[4:8]),
        'blocksize': decode_uint32(buffer[8:12]),
        'ctbytes': decode_uint32(buffer[12:16]),
   }


def clip_chunk(nchunk, chunklen, start, stop, step):
    """Get the limits of a certain chunk based on its length."""
    startb = start - nchunk * chunklen
    stopb = stop - nchunk * chunklen
    # Check limits
    if (startb >= chunklen) or (stopb <= 0):
        return startb, stopb, 0  # null size
    if startb < 0:
        startb = 0
    if stopb > chunklen:
        stopb = chunklen
    # step corrections
    if step > 1:
        # Just correcting startb is enough
        distance = (nchunk * chunklen + startb) - start
        if distance % step > 0:
            startb += (step - (distance % step))
            if startb > chunklen:
                return startb, stopb, 0  # null size
    # Compute size of the clipped block
    blen = get_len_of_range(startb, stopb, step)
    return startb, stopb, blen


def get_len_of_range(start, stop, step):
    """Get the length of a (start, stop, step) range."""
    if start < stop:
        return ((stop - start - 1) // step + 1)
    return 0


class LegacyBcolzArray:

    def __init__(self, rootdir):
        self.filename = rootdir
        self.attrs_filename = rootdir / '__attrs__'
        self.sizes_filename = rootdir / 'meta' / 'sizes'
        self.storage_filename = rootdir / 'meta' / 'storage'
        self.data_path = rootdir / 'data'
        self.attrs = json.loads(self.attrs_filename.read_text())
        self.sizes = json.loads(self.sizes_filename.read_text())
        self.storage = json.loads(self.storage_filename.read_text())
        self.length = self.sizes['shape'][-1]
        self.dtype = np.dtype(self.storage['dtype'])
        self.nbytes = self.sizes['nbytes']
        self.chunklen = self.storage['chunklen']

    def read_chunk(self, nchunk):
        chunk_path = self.data_path / f'__{nchunk}.blp'
        with chunk_path.open('rb') as fh:
            bloscpack_header = fh.read(BLOSCPACK_HEADER_LENGTH)
            blosc_header_raw = fh.read(BLOSC_HEADER_LENGTH)
            blosc_header = decode_blosc_header(blosc_header_raw)
            ctbytes = blosc_header['ctbytes']
            fh.seek(-BLOSC_HEADER_LENGTH, 1)
            b = fh.read(ctbytes)
        return np.frombuffer(blosc.decompress(b), dtype=self.dtype)

    def __getitem__(self, key):
        start, stop, step = key.indices(self.length)
        if start > stop:
            raise ValueError('Invalid slice')

        n = get_len_of_range(start, stop, step)
        arr = np.empty(shape=(n,), dtype=self.dtype)
        if n == 0:
            return arr

        if self.dtype.char == 'O':
            raise ValueError('Unsupported dtype')

        lb = start // self.chunklen
        ub = stop // self.chunklen + 1

        i = 0
        for n_chunk in range(lb, ub):
            startb, stopb, blen = clip_chunk(n_chunk, self.chunklen, start, stop, step)
            if blen == 0:
                continue
            arr[i:i + blen] = self.read_chunk(n_chunk)[startb:stopb:step]
            i += blen
        return arr


class LegacyBcolzSignal(Signal):

    def __init__(self, base_path):
        self.base_path = base_path

    @property
    @functools.lru_cache()
    def array(self):
        return LegacyBcolzArray(rootdir=self.base_path)

    @property
    def fs(self):
        return self.array.attrs['fs']

    @property
    def duration(self):
        return self.array.length/self.fs

    def __getitem__(self, slice):
        return self.array[slice]

    @property
    def shape(self):
        return (self.array.length,)
