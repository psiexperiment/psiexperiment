import logging
log = logging.getLogger(__name__)

from atom.api import List

import enaml
from enaml.workbench.ui.api import Workspace

with enaml.imports():
    from .view import ExperimentView


class ExperimentWorkspace(Workspace):

    toolbars = List()

    def _default_content(self):
        return ExperimentView()

    @property
    def dock_area(self):
        return self.content.find('dock_area')
