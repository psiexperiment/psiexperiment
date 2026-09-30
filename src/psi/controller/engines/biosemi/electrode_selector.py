import logging
log = logging.getLogger(__name__)

from atom.api import Atom, Bool, Int, List, Typed
import numpy as np
import pandas as pd

from psiaudio.util import diff_matrix
from . import electrode_coords as coords


class ElectrodeSelector(Atom):

    coords = Typed(pd.DataFrame)
    reference = List()
    selected = List()
    diff_matrix = Typed(np.ndarray)

    def toggle_reference(self, index):
        reference = self.reference.copy()
        if index in reference:
            reference.remove(index)
        else:
            reference.append(index)
        self.reference = reference

    def toggle_selected(self, index):
        selected = self.selected[:]
        if index in selected:
            selected.remove(index)
        else:
            selected.append(index)
        self.selected = selected

    def _observe_reference(self, event):
        self._update_diff_matrix()

    def _observe_coords(self, event):
        self._update_diff_matrix()

    def _update_diff_matrix(self):
        n_channels = len(self.coords)
        self.diff_matrix = diff_matrix(n_channels, self.reference)


class BiosemiElectrodeSelector(ElectrodeSelector):

    n_channels = Int(32)
    include_exg = Bool(True)

    def _observe_n_channels(self, event):
        self.coords = self._default_coords()

    def _observe_include_exg(self, event):
        self.coords = self._default_coords()

    def _default_coords(self):
        return coords.load_normalized_coords(self.n_channels, self.include_exg)
