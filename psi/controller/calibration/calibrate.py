import logging
log = logging.getLogger(__name__)

from atom.api import Bool, Dict, Float, Int, List, Str, Value
from enaml.application import deferred_call
from enaml.core.api import d_
import pandas as pd

from psi.core.enaml.api import PSIContribution

from psiaudio.calibration import FlatCalibration, InterpCalibration, PointCalibration
from .chirp import chirp_sens
from .tone import tone_sens


def merge_results(results, names=['ao_channel']):
    to_merge = {}
    for master_key, result in results.items():
        to_merge.setdefault('calibration', {})[master_key] = result
        for key, value in result.attrs.items():
            to_merge.setdefault(key, {})[master_key] = value

    merged = {}
    for key, value in to_merge.items():
        if key == 'fs':
            index = pd.MultiIndex.from_tuples(value.keys(), names=names)
            merged[key] = pd.DataFrame(value.values(), index=index)
        else:
            merged[key] = pd.concat(value.values(), keys=value.keys(), names=names)
    return merged


class BaseCalibrate(PSIContribution):

    outputs = d_(Dict())
    input_name = d_(Str())
    gain = d_(Float(-40))

    result = d_(Dict())
    show_widget = d_(Bool(True))

    def save_dataframe(self, core_plugin, dataframe, name):
        save_command = 'calibration_data.save_dataframe'
        parameters = {'name': name, 'dataframe': dataframe}
        deferred_call(core_plugin.invoke_command, save_command, parameters)

    def save(self, core_plugin):
        for key, value in self.result.items():
            name = self.name if key == 'calibration' else f'{self.name}_{key}'
            self.save_dataframe(core_plugin, value, name)

    def calibrate(self, controller, core):
        '''
        Run calibration

        Parameters
        ----------
        workbench : Enaml workbench
            Enaml workbench instance
        '''
        ai_channel = controller.get_input(self.input_name).channel

        results = {}
        for ao_channel, kwargs in self.get_config(controller, core).items():
            log.debug('Running calibration for %s', ao_channel.name)
            results[(ao_channel.name,)] = self.run_calibration(ao_channel, ai_channel, kwargs)
        self.result = merge_results(results)

    def get_config(self, controller, core):
        raise NotImplementedError

    def run_calibration(self, ao_channel, ai_channel, kwargs):
        raise NotImplementedError


class ToneCalibrate(BaseCalibrate):
    '''
    Calibrate the specified output using the specified input

    Useful for in-ear calibrations. The calibration will be saved.

    Attributes
    ----------
    outputs : dict
        Dictionary whose keys are outputs and values are a list of parameters
        needed to determine the calibration frequencies.
    input_name : string
        Channel to calibrate
    selector_name : string (default='default')
        Selector to use
    gain : float
        Gain to set on output channel
    max_thd : {None, float}
        Maximum total harmonic distortion (in percent) to allow. Anything above
        this raises a calibration error.
    min_snr : {None, float}
        Minimum test level (re. noise floor). If the test tone is too close to
        the noise floor, this raises a calibration error.
    duration : float
        Duration of test tone
    iti : float
        Intertrial interval between test tones
    trim : float
        Amount to trim off of start and end of test tone response before
       analysis.
    widget_name : {None, string}
        Name of widget containing results set to update (for viewing).
    attr_name : {None, string}
        Name of attribute on widget to set.
    store_name : {None, string}
        Name of store to write data to.
    '''

    duration = d_(Float(100e3))
    iti = d_(Float(0))
    trim = d_(Float(10e-3))
    max_thd = d_(Value(None))
    min_snr = d_(Value(None))

    def get_config(self, controller, core):
        # Generate a list of frequencies to calibrate for each channel
        ao = {}
        for output_name, parameter_names in self.outputs.items():
            output = controller.get_output(output_name)
            frequencies = set()
            for parameter in parameter_names:
                p = {'item_names': parameter}
                new = core.invoke_command('psi.context.unique_values', p)
                frequencies.update(new)
            ao[output.channel] = {'frequencies': list(frequencies)}
        return ao

    def run_calibration(self, ao_channel, ai_channel, kwargs):
        result = tone_sens(
            ao_channel.engine,
            gain=self.gain,
            ao_channel_name=ao_channel.name,
            ai_channel_names=[ai_channel.name],
            max_thd=self.max_thd,
            min_snr=self.min_snr,
            duration=self.duration,
            iti=self.iti,
            trim=self.trim,
            **kwargs
        )
        result = result.sort_index()
        sens = result.loc[ai_channel.name, 'sens']
        ao_channel.calibration = PointCalibration(sens.index, sens.values)
        return result


class ChirpCalibrate(BaseCalibrate):
    '''
    Calibrate the specified output using the specified input

    Useful for in-ear calibrations. The calibration will be saved.

    Parameters
    ----------
    outputs : list of string
        Output names to calibrate
    input_name : string
        Input to calibrate
    gain : float
        Gain to set on output channel
    duration : float
        Duration of chirp
    iti : float
        Intertrial interval between chirps
    repetitions : int
        Number of repetitions to average
    '''
    duration = d_(Float(20e-3))
    iti = d_(Float(1e-3))
    repetitions = d_(Int(64))

    def calibrate(self, controller, core):
        '''
        Run calibration

        Parameters
        ----------
        workbench : Enaml workbench
            Enaml workbench instance
        '''
        ao_channels = set(controller.get_output(o).channel for o in self.outputs)
        ai_input = controller.get_input(self.input_name)
        ai_channel = ai_input.channel

        results = {}
        for ao_channel in ao_channels:
            log.debug('Running chirp calibration for %s', ao_channel.name)
            result = chirp_sens(ao_channel.engine,
                                self.gain,
                                ao_channel_name=ao_channel.name,
                                ai_channel_names=[ai_channel.name],
                                duration=self.duration,
                                iti=self.iti,
                                repetitions=self.repetitions)

            results[ao_channel.name] = result
            calibration = InterpCalibration(result.index.get_level_values('frequency'),
                                            result['sens'])
            ao_channel.calibration = calibration
            log.info('%s: %s', ao_channel.name, ao_channel.calibration)

        results = pd.concat(results.values(), keys=results.keys(),
                            names=['ao_channel'])

        results['gain'] = self.gain
        results['duration'] = self.duration
        results['repetitions'] = self.repetitions
        self.result = results
