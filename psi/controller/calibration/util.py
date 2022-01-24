import logging
log = logging.getLogger(__name__)

import json
from pathlib import Path

import numpy as np
from psi.util import psi_json_decoder_hook, PSIJsonEncoder


def save_calibration(channels, filename):
    from psi.util import get_tagged_values
    from json import dump
    settings = {}
    for channel in channels:
        metadata = get_tagged_values(channel.calibration, 'metadata')
        metadata['calibration_type'] = channel.calibration.__class__.__name__
        settings[channel.name] = metadata

    with open(filename, 'w') as fh:
        dump(settings, fh, indent=4, cls=PSIJsonEncoder)


def load_calibration_data(filename):
    from psi.controller.calibration.api import calibration_registry
    settings = json.loads(Path(filename).read_text(),
                          object_hook=psi_json_decoder_hook)
    calibrations = {}
    for c_name, c_calibration in settings.items():
        # This is will deal with legacy calibration configs in which the source
        # was a top-level key rather than being stored as an attribute.
        if 'source' in c_calibration:
            attrs = c_calibration.setdefault('attrs', {})
            attrs['source'] = c_calibration.pop('source')
        calibrations[c_name] = calibration_registry.from_dict(**c_calibration)
    return calibrations


def load_calibration(filename, channels):
    '''
    Load calibration configuration for hardware from json file
    '''
    calibrations = load_calibration_data(filename)
    for c in channels:
        if c.name in calibrations:
            c.calibration = calibrations[c.name]
