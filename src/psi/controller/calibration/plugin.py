import logging
log = logging.getLogger(__name__)

from atom.api import List
from enaml.workbench.api import Plugin

from psi.core.enaml.api import load_manifests

from .calibrate import BaseCalibrate


CALIBRATION_POINT = 'psi.controller.calibration.channels'


class CalibrationPlugin(Plugin):

    _calibrators = List()

    def start(self):
        log.debug('Starting calibration plugin')
        self._refresh_calibrators()
        self._bind_observers()

    def stop(self):
        self._unbind_observers()

    def _bind_observers(self):
        self.workbench.get_extension_point(CALIBRATION_POINT) \
            .observe('extensions', self._refresh_calibrators)

    def _unbind_observers(self):
        self.workbench.get_extension_point(CALIBRATION_POINT) \
            .unobserve('extensions', self._refresh_calibrators)

    def _refresh_calibrators(self, event=None):
        point = self.workbench.get_extension_point(CALIBRATION_POINT)
        calibrators = []
        for extension in point.extensions:
            calibrators.extend(extension.get_children(BaseCalibrate))
        load_manifests(calibrators, self.workbench)
        self._calibrators = calibrators

    def will_calibrate(self, output_name):
        for calibrator in self._calibrators:
            if output_name in calibrator.outputs:
                return True
        return False
