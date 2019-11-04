import pandas as pd

from . import Recording


def _get_unique(df, level_name):
    choices = df.index.unique(level_name)
    if len(choices) != 1:
        choices = ', '.join(str(c) for c in choices)
        m = f'Must specify {level_name}. Options are {choices}.'
        raise ValueError(m)
    return choices[0]


def repair_golay_sens_name(fh):
    if 'golay_sens' in fh.ttable_names:
        return
    for sens_name in ('golay_sensitivity', 'golay_summary', 'sensitivity'):
        if sens_name in fh.ttable_names:
            break
    else:
        raise ValueError(f'{fh} does not contain Golay data')
    old_filename = (fh.base_path / sens_name).with_suffix('.csv')
    new_filename = (fh.base_path / 'golay_sens.csv')
    old_filename.rename(new_filename)
    fh._refresh_names()


def repair_chirp_sens_name(fh):
    if 'chirp_sens' in fh.ttable_names:
        return
    for sens_name in ('chirp_sensitivity', 'chirp_summary', 'sensitivity'):
        if sens_name in fh.ttable_names:
            break
    else:
        raise ValueError(f'{fh} does not contain chirp data')
    old_filename = (fh.base_path / sens_name).with_suffix('.csv')
    new_filename = (fh.base_path / 'chirp_sens.csv')
    old_filename.rename(new_filename)
    fh._refresh_names()


class CalibrationFile(Recording):

    _ttable_indices = {
        'tone_sens': ['channel_name', 'frequency'],
        'golay_sens': ['n_bits', 'output_gain', 'frequency'],
    }

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        if 'golay' in self.base_path.name.lower():
            repair_golay_sens_name(self)
        elif 'chirp' in self.base_path.name.lower():
            repair_chirp_sens_name(self)

    def get_tone_calibration(self, channel_name=None):
        '''
        Return PointCalibration as derived from tone data
        '''
        from psi.controller.calibration.api import PointCalibration
        if channel_name is None:
            channel_name = _get_unique(self.tone_sens, 'channel_name')
        sens = self.tone_sens.loc[channel_name]['sens']
        return PointCalibration(sens.index, sens.values)

    def _get_golay_data(self, n_bits=None, output_gain=None):
        for epoch_name in ('pt_epoch', 'epoch'):
            if epoch_name in self.carray_names:
                break
        else:
            m = f'Path {self.base_path} does not appear to contain Golay data'
            raise ValueError(m)

        sens_filename = self.base_path / 'golay_sens.csv'
        sensitivity = pd.io.parsers.read_csv(sens_filename)
        if n_bits is None:
            n_bits = sensitivity['n_bits'].max()
        if output_gain is None:
            m = sensitivity['n_bits'] == n_bits
            output_gain = sensitivity.loc[m, 'output_gain'].max()
        m_n_bits = sensitivity['n_bits'] == n_bits
        m_output_gain = sensitivity['output_gain'] == output_gain
        m = m_n_bits & m_output_gain
        mic_freq = sensitivity.loc[m, 'frequency'].values
        mic_sens = sensitivity.loc[m, 'sens'].values
        mic_phase = sensitivity.loc[m, 'phase'].values
        #source = 'psi_golay', self.base_folder, n_bits, output_gain
        epoch = getattr(self, epoch_name)
        return {
            'source': self.base_path,
            'frequency': mic_freq,
            'sensitivity': mic_sens,
            'phase': mic_phase,
            'fs': epoch.fs,
        }

    def get_golay_calibration(self, n_bits=None, output_gain=None):
        '''
        Return GolayCalibration as derived from golay data
        '''
        from psi.controller.calibration.api import GolayCalibration
        data = self._get_golay_data(n_bits, output_gain)
        return GolayCalibration(**data)
