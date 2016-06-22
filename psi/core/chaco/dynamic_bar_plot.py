import numpy as np
from chaco.api import (ArrayDataSource, LabelAxis, BarPlot,
                                 LinearMapper, DataRange1D)

from traits.api import (Instance, Any, Str, Int, Trait, DelegatesTo,
                                  Callable, Float)

class DynamicBarplotAxis(LabelAxis):
    
    source = Any
    label_trait = Str

    def _source_changed(self, old, new):
        trait = " " + self.label_trait
        if old is not None:
            old.on_trait_change(self._update_labels, trait, remove=True)
        if new is not None:
            new.on_trait_change(self._update_labels, trait)
        self._update_labels()

    def _update_labels(self):
        labels = getattr(self.source, self.label_trait)
        text = []
        for l in labels:
            if type(l) in (list, tuple):
                label = ', '.join([str(e) for e in l])
            else:
                label = str(l)
            text.append(label)

        self.labels = text
        self.positions = np.arange(len(labels))
        self.invalidate_and_redraw()

class DynamicBarPlot(BarPlot):

    source              = Any
    value_trait         = Trait(None, Str)
    index_trait         = Trait(None, Str)
    index_spacing       = Int(1)

    preprocess_values   = Trait(None, Callable)

    value               = Instance(ArrayDataSource, ())
    index               = Instance(ArrayDataSource, ())

    index_offset        = Float(0)

    def _get_indices(self):
        if self.index_trait is not None:
            return getattr(self.source, self.index_trait)
        else:
            values = self._get_values()
            indices = np.arange(0, len(values) * self.index_spacing,
                                self.index_spacing) 
            return indices+self.index_offset

    def _get_values(self):
        if self.source is not None:
            value = np.array(getattr(self.source, self.value_trait))
            if self.preprocess_values is not None:
                return self.preprocess_values(value)
            # Replaces all NaN values with zero
            return np.nan_to_num(value)
        return []

    def _source_changed(self, old, new):
        traits = []
        if self.index_trait is not None:
            traits.append(' ' + self.index_trait)
        if self.value_trait is not None:
            traits.append(' ' + self.value_trait)

        for trait in traits:
            if old is not None:
                old.on_trait_change(self.update_data, trait, remove=True)
            if new is not None:
                new.on_trait_change(self.update_data, trait)
        self.update_data()

    def update_data(self):
        if self.traits_inited():
            values = self._get_values()
            indices = self._get_indices()
            if self.value_trait == 'par_warn_count':
                print values, indices
            if len(indices) == len(values):
                self.index.set_data(indices)
                self.value.set_data(values)
                self._either_data_changed()
