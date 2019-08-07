from psi.controller.calibration import (EPLCalibration, FlatCalibration,
                                        GolayCalibration, UnityCalibration)
from psi.controller.channel import HardwareAIChannel

def save_calibration(channels):
    from psi.util import get_tagged_values
    from json_tricks import dumps
    settings = {}
    #for channel in channels:
        #metadata = get_tagged_values(channel.calibration, 'metadata')
        #metadata['calibration_type'] = channel.calibration.__class__.__name__
        #settings[channel.name] = metadata
    #print(dumps(settings, indent=4))
    print(dumps(channels[1].calibration))

filename = r'C:\psi-dev\calibration\20181107-0832 Brad 2W starship 2 377C10 primary long coupler pt_calibration_golay'
channels = [
    HardwareAIChannel(name='temperature',
                        label='Temperature',
                        calibration=UnityCalibration(),
                        ),
    HardwareAIChannel(name='reference_microphone',
                        label='Cal. microphone',
                        calibration=FlatCalibration.from_mv_pa(1.0)
                        ),
    HardwareAIChannel(name='pt_microphone',
                        label='PT microphone',
                        calibration=GolayCalibration.from_psi_golay(filename)
                        ),
]


save_calibration(channels)
