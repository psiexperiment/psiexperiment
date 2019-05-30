import logging
log = logging.getLogger(__name__)

from atom.api import List
from enaml.workbench.api import Plugin

from psi.core.enaml.api import load_manifests

from .calibrate import ChirpCalibrate, ToneCalibrate


CALIBRATION_POINT = 'psi.controller.calibration.channels'


class CalibrationPlugin(Plugin):

    _tone_calibrations = List()
    _chirp_calibrations = List()

    def start(self):
        log.debug('Starting calibration plugin')
        self._refresh_calibrations()
        self._bind_observers()

    def stop(self):
        self._unbind_observers()

    def _bind_observers(self):
        self.workbench.get_extension_point(CALIBRATION_POINT) \
            .observe('extensions', self._refresh_calibrations)

    def _unbind_observers(self):
        self.workbench.get_extension_point(CALIBRATION_POINT) \
            .unobserve('extensions', self._refresh_calibrations)

    def _refresh_calibrations(self, event=None):
        point = self.workbench.get_extension_point(CALIBRATION_POINT)
        tone_calibrations = []
        chirp_calibrations = []
        for extension in point.extensions:
            tone_calibrations.extend(extension.get_children(ToneCalibrate))
        for extension in point.extensions:
            chirp_calibrations.extend(extension.get_children(ChirpCalibrate))
        load_manifests(tone_calibrations, self.workbench)
        load_manifests(chirp_calibrations, self.workbench)
        self._tone_calibrations = tone_calibrations
        self._chirp_calibrations = chirp_calibrations
