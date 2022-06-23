import logging
log = logging.getLogger(__name__)

import importlib
import json
from pathlib import Path

import numpy as np
from psi.util import get_tagged_values, psi_json_decoder_hook, PSIJsonEncoder


def save_calibration(channels, filename):
    settings = {}
    for channel in channels:
        metadata = get_tagged_values(channel.calibration, 'metadata')
        metadata['calibration_type'] = channel.calibration.__class__.__name__
        settings[channel.name] = metadata

    filename.write_text(json.dumps(settings, fh, indent=4, cls=PSIJsonEncoder))


def fix_legacy(calibration_type, metadata):
    if calibration_type == 'GolayCalibration':
        calibration_type = 'psiaudio.calibration.InterpCalibration'
        attrs = metadata.setdefault('attrs', {})
        for key in ('source', 'fs'):
            if key in metadata:
                attrs[key] = metadata.pop(key)
    elif calibration_type == 'FlatCalibration':
        calibration_type = 'psiaudio.calibration.FlatCalibration'
    return calibration_type, metadata


def load_calibration_data(filename):
    settings = json.loads(Path(filename).read_text(), object_hook=psi_json_decoder_hook)
    calibrations = {}
    for name, metadata in settings.items():
        calibration_type = metadata.pop('calibration_type')
        calibration_type, metadata = fix_legacy(calibration_type, metadata)
        cal_module, cal_name = calibration_type.rsplit('.', 1)
        cal_class = getattr(importlib.import_module(cal_module), cal_name)
        calibrations[name] = cal_class(**metadata)
    return calibrations


def load_calibration(filename, channels):
    '''
    Load calibration configuration for hardware from json file
    '''
    calibrations = load_calibration_data(filename)
    for c in channels:
        if c.name in calibrations:
            c.calibration = calibrations[c.name]
