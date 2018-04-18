import numpy as np


class RollingSignalBuffer:

    def __init__(self, fs, size):
        self._buffer_fs = fs
        self._buffer_size = size
        self._buffer_samples = int(fs*size)
        self._buffer = np.full(self._buffer_samples, np.nan)
        self._offset = -self._buffer_samples

    def append_data(self, data):
        samples = data.shape[-1]
        if samples > self._buffer_samples:
            self._buffer[:] = data[-self._buffer_samples:]
        else:
            self._buffer[:-samples] = self._buffer[samples:]
            self._buffer[-samples:] = data
        self._offset += samples

    def get_data(self, lb, ub):
        ilb = int(lb*self._buffer_fs) - self._offset
        iub = int(ub*self._buffer_fs) - self._offset
        if ilb < 0:
            ilb = 0
        return self._buffer[ilb:iub]


def wrap(offset, length, buffer_size):
    if (offset+length) > buffer_size:
        a_length = buffer_size-offset
        b_length = length-a_length
        return ((offset, a_length), (0, b_length))
    else:
        return ((offset, length), )


class PlotBuffer:

    def __init__(self, fs, span):
        self._fs = fs
        self._span = span
        self._samples = np.round(fs*span)
        self._buffer = np.full(self._samples, np.nan)
        self._t0 = 0

    def append(self, data):
        pass

class NoRollSignalBuffer:

    def __init__(self, fs, size):
        self._buffer_fs = fs
        self._buffer_size = size
        self._buffer_samples = np.round(fs*size)
        self._buffer = np.full(self._buffer_samples, np.nan)
        self._last_write = 0
        self._total_samples = 0

    def append_data(self, data):
        samples = data.shape[-1]
        if samples > self._buffer_samples:
            self._buffer[:] = data[-self._buffer_samples:]
            self._last_write = 0
        else:
            o = 0
            indices = wrap(self._last_write, samples, self._buffer_samples)
            for i, s in indices:
                self._buffer[i:i+s] = data[o:o+s]
                o += s
            self._last_write = i+s
        self._total_samples += samples

    def get_data(self, lb, ub):
        # Oldest sample in buffer (re start of acquisition)
        i0 = self._total_samples - self._buffer_samples

        # Lower index relative to oldest sample in buffer
        ilb = round(lb*self._buffer_fs)
        iub = round(ub*self._buffer_fs)

        # Total number of samples requested
        samples = iub-ilb
        data = np.full(samples, np.nan, dtype=np.float32)

        # Data offset
        if ilb > self._total_samples:
            return data
        if iub <= i0:
            return data

        if ilb < i0:
            o = i0-ilb
            i = i0
        else:
            o = 0
            i = ilb

        s = min(iub, self._total_samples) - i
        i = i % self._buffer_samples

        indices = wrap(i, s, self._buffer_samples)
        print(indices)
        for di, ds in indices:
            data[o:o+ds] = self._buffer[di:di+ds]
            o += ds
        return data


if __name__ == '__main__':
    fs, size = 100, 1
    nr = NoRollSignalBuffer(fs, size)
    r = RollingSignalBuffer(fs, size)

    #nr.append_data(np.arange(10))
    #nr.append_data(np.arange(5))
    #print(nr._buffer)

    t = np.arange(50)
    nr.append_data(t)
    r.append_data(t)
    print()
    print(nr.get_data(0, 0.1))
    print(r.get_data(0, 0.1))
    print()
    print(nr.get_data(0.25, 3))
    print(r.get_data(0.25, 3))

    t = np.arange(200)
    nr.append_data(t)
    r.append_data(t)
    print()
    print(nr.get_data(0, 0.1))
    print(r.get_data(0, 0.1))
    print()
    print(nr.get_data(1.5, 2))
    print(r.get_data(1.5, 2))
    print()
    print(nr.get_data(2.5, 3))
    print(r.get_data(2.5, 3))
