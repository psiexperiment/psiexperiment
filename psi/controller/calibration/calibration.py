import os.path
import importlib
from pathlib import Path

from scipy.interpolate import interp1d
from scipy import signal
import numpy as np
import pandas as pd

import logging
log = logging.getLogger(__name__)

from atom.api import Atom, Callable, Dict, Float, Property, Typed, Value
from enaml.core.api import Declarative, d_
from psi import SimpleState

from . import util

from psi import get_config

from psiaudio.calibration import FlatCalibration, InterpCalibration


class EPLCalibration(InterpCalibration):

    @classmethod
    def load_data(cls, filename):
        filename = Path(filename)
        calibration = pd.io.parsers.read_csv(filename, skiprows=14,
                                             delimiter='\t')
        freq = calibration['Freq(Hz)']
        spl = calibration['Mag(dB)']
        return {
            'attrs': {
                'source': filename
            },
            'frequency': freq,
            'spl': spl,
        }

    @classmethod
    def load(cls, filename, **kwargs):
        data = cls.load_data(filename)
        data.update(kwargs)
        return cls.from_spl(**data)


class CochlearCalibration(InterpCalibration):

    @classmethod
    def load_data(cls, filename):
        import tables
        with tables.open_file(filename, 'r') as fh:
            mic_freq = np.asarray(fh.get_node('/frequency').read())
            mic_sens = np.asarray(fh.get_node('/exp_mic_sens').read())
            return {
                'attrs': {
                    'source': filename
                },
                'frequency': mic_freq,
                'sensitivity': mic_sens,
            }

    @classmethod
    def load(cls, filename, **kwargs):
        data = cls.load_data(filename)
        data.update(kwargs)
        return cls(**kwargs)


class GolayCalibration(InterpCalibration):

    def __init__(self, frequency, sensitivity, fs=None, phase=None,
                 fixed_gain=0, **kwargs):
        super().__init__(frequency, sensitivity, fixed_gain, phase, **kwargs)
        # fs and phase are required for the IIR stuff
        if fs is not None:
            self.fs = fs

    @staticmethod
    def load_data(folder, n_bits=None, output_gain=None):
        from psi.data.io.calibration import CalibrationFile
        fh = CalibrationFile(folder)
        return fh._get_golay_data(n_bits, output_gain)

    @classmethod
    def load(cls, folder, n_bits=None, output_gain=None, **kwargs):
        from psi.data.io.calibration import CalibrationFile
        fh = CalibrationFile(folder)
        return fh.get_golay_calibration(n_bits, output_gain)

    def get_iir(self, fs, fl, fh, truncate=None):
        fs_ratio = self.fs/fs
        if int(fs_ratio) != fs_ratio:
            m = 'Calibration sampling rate, {}, must be an ' \
                'integer multiple of the requested sampling rate'
            raise ValueError(m.format(self.fs))

        n = (len(self.frequency)-1)/fs_ratio + 1
        if int(n) != n:
            m = 'Cannot achieve requested sampling rate ' \
                'TODO: explain why'
            raise ValueError(m)
        n = int(n)

        fc = (fl+fh)/2.0
        freq = self.frequency
        phase = self.phase
        sens = self.sensitivity - (self.get_sens(fc) + self.fixed_gain)
        sens[freq < fl] = 0
        sens[freq >= fh] = 0
        m, b = np.polyfit(freq[freq < fh], phase[freq < fh], 1)
        invphase = 2*np.pi*np.arange(len(freq))*m
        inv_csd = util.dbi(sens)*np.exp(invphase*1j)

        # Need to trim so that the data is resampled accordingly
        if fs_ratio != 1:
            inv_csd = inv_csd[:n]
        iir = np.fft.irfft(inv_csd)

        if truncate is not None:
            n = int(truncate*fs)
            iir = iir[:n]

        return iir


class ChirpCalibration(InterpCalibration):

    @staticmethod
    def load_data(folder, output_gain=None):
        folder = Path(folder)
        sensitivity = pd.io.parsers.read_csv(folder / 'chirp_summary.csv')
        if output_gain is None:
            output_gain = sensitivity['hw_ao_chirp_level'].max()

        m = sensitivity['hw_ao_chirp_level'] == output_gain
        mic_freq = sensitivity.loc[m, 'frequency'].values
        mic_sens = sensitivity.loc[m, 'sens'].values
        mic_phase = sensitivity.loc[m, 'phase'].values
        return {
            'attrs': {
                'source': folder,
                'output_gain': output_gain,
                'calibration_type': 'psi_chirp',
            },
            'frequency': mic_freq,
            'sensitivity': mic_sens,
        }

    @classmethod
    def load(cls, folder, n_bits=None, output_gain=None, **kwargs):
        data = cls.load_psi_golay(folder, n_bits, output_gain)
        data.update(kwargs)
        return cls(**data)


class CalibrationRegistry:

    def __init__(self):
        self.registry = {}

    def register(self, klass, label=None):
        calibration_type = klass.__name__
        calibration_path = f'{klass.__module__}.{calibration_type}'
        if label is None:
            label = calibration_type
        if calibration_type in self.registry:
            m = f'{label} already registered as {calibration_type}'
            raise ValueError(m)
        self.registry[calibration_path] = klass, label
        log.debug('Registered %s', calibration_path)

    def clear(self):
        self.registry.clear()

    def register_basic(self, clear=False, unity=True, fixed=True, golay=True,
                       chirp=True):
        if clear:
            self.clear()
        #if unity:
            #self.register(UnityCalibration, 'unity gain')
        if fixed:
            self.register(FlatCalibration, 'fixed sensitivity')
        if golay:
            self.register(GolayCalibration, 'Golay calibration')
        if chirp:
            self.register(ChirpCalibration, 'Chirp calibration')

    def get_classes(self):
        return [v[0] for v in self.registry.values()]

    def get_class(self, calibration_type):
        return self.registry[calibration_type][0]

    def get_labels(self):
        return [v[1] for v in self.registry.values()]

    def get_label(self, obj):
        name = f'{obj.__module__}.{obj.__name__}'
        return self.registry[name][1]

    def from_dict(self, calibration_type, **kw):
        if calibration_type not in self.registry:
            log.debug('Importing and registering calibration')
            # Older calibration formats may still have only the class name, not
            # the full module + class name.
            try:
                module_name, class_name = calibration_type.rsplit('.', 1)
            except ValueError:
                module_name = __name__
                class_name = calibration_type
            module = importlib.import_module(module_name)
            klass = getattr(module, class_name)
        else:
            klass = self.get_class(calibration_type)
        return klass(**kw)


calibration_registry = CalibrationRegistry()
calibration_registry.register_basic()
calibration_registry.register(EPLCalibration, 'EPL calibration')
calibration_registry.register(CochlearCalibration, 'Golay calibration (old Cochlear format)')


if __name__ == '__main__':
    import doctest
    doctest.testmod()
