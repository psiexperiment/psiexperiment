import logging
log = logging.getLogger(__name__)

from atom.api import Bool, Dict, Float, Int, List, Str, Value
from enaml.application import deferred_call
from enaml.core.api import d_, d_func
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
            for v in value.values():
                v.attrs = {}
            merged[key] = pd.concat(value.values(), keys=value.keys(), names=names)
    return merged


class BaseCalibrate(PSIContribution):

    #: Dictionary whose keys are outputs and values are a list of parameters
    #: needed to determine the calibration frequencies.
    outputs = d_(Dict())

    #: Name of channel to calibrate.
    input_name = d_(Str())

    #: Gain to set on output channel.
    gain = d_(Float(-40))

    result = d_(Dict())

    #: Add a DockItem showing the results of the calibration?
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
    '''

    #: Duration of calibration tone
    duration = d_(Float(100e3))

    #: Interval between calibration tones
    iti = d_(Float(0))

    #: Amount to trim off of start and end of calibration tone response before
    #: analysis.
    trim = d_(Float(10e-3))

    #: Maximum total harmonic distortion (in percent) to allow. Anything above
    #: this raises a calibration error.
    max_thd = d_(Value(None))

    #: Minimum test level (re. noise floor). If the test tone is too close to
    #: the noise floor, this raises a calibration error.
    min_snr = d_(Value(None))

    @d_func
    def get_values(self, values):
        return {si for i in values for si in i}

    def get_config(self, controller, core):
        # Generate a list of frequencies to calibrate for each channel
        ao = {}
        for output_name, parameter_names in self.outputs.items():
            output = controller.get_output(output_name)
            ao_info = ao.setdefault(output.channel, {'frequencies': set()})
            p = {'item_names': parameter_names}
            new = core.invoke_command('psi.context.unique_values', p)
            ao_info['frequencies'].update(self.get_values(new))

        # At this point, ao_items is a dictionary whose keys are output
        # channels. The values are another dictionary consisting of 
        # {'frequencies': set(frequencies to test)}. We now need to convert the
        # set of frequencies to a list (we used set to remove duplicate
        # frequencies).
        return {k: {sk: list(sv)} for k, v in ao.items() for sk, sv in v.items()}

    def run_calibration(self, ao_channel, ai_channel, kwargs):
        result = tone_sens(
            ao_channel.engine,
            gains=self.gain,
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
    '''
    #: If not provided and show_widget is True, the viewbox name will
    #: automatically be generated and a plot added to the experiment GUI. If
    #: you want this calibration result to share a viewbox with another plot,
    #: provide the name of that viewbox. Be sure to set show_widget to False as
    #: well otherwise you'll get a useless widget.
    viewbox_name = d_(Str())

    #: Duration of chirp
    duration = d_(Float(20e-3))

    #: Intertrial interval between chirps
    iti = d_(Float(1e-3))

    #: Number of repetitions to average
    repetitions = d_(Int(64))

    def _default_viewbox_name(self):
        return self.name

    def get_config(self, controller, core):
        # As of now, we do not do anything with the value provided for each
        # output name in the dictionary. This is for future compatibility.
        ao = {}
        for output_name, _ in self.outputs.items():
            output = controller.get_output(output_name)
            ao[output.channel] = {}
        return ao

    def run_calibration(self, ao_channel, ai_channel, kwargs):
        result = chirp_sens(
            ao_channel.engine,
            self.gain,
            ao_channel_name=ao_channel.name,
            ai_channel_names=[ai_channel.name],
            duration=self.duration,
            iti=self.iti,
            repetitions=self.repetitions
        )
        result = result.sort_index()
        sens = result.loc[ai_channel.name, 'sens']

        # Ensure that calibration is set
        ao_channel.calibration = InterpCalibration(sens.index, sens.values)
        result['gain'] = self.gain
        result['duration'] = self.duration
        result['repetitions'] = self.repetitions
        return result
