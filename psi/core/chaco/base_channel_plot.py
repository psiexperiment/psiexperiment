from chaco.api import BaseXYPlot
from enable.api import black_color_trait, LineStyle
from traits.api import Event, Float, Instance

class BaseChannelPlot(BaseXYPlot):
    '''
    Not meant for use as a stand-alone plot.  Provides the base properties and
    methods shared by all subclasses.
    '''

    source = Instance('psi.data.hdf_store.data_source.DataChannel')

    fill_color = black_color_trait
    line_color = black_color_trait
    line_width = Float(1.0)
    line_style = LineStyle

    data_changed = Event

    def _source_changed(self, old, new):
        # We need to call _update_index_mapper when fs changes since this
        # method precomputes the index value based on the sampling frequency of
        # the channel.
        if old is not None:
            old.unobserve('changed', self._data_changed)
            old.unobserve('added', self._data_added)
            old.unobserve('fs', self._index_mapper_updated)
        if new is not None:
            new.observe('changed', self._data_changed)
            new.observe('added', self._data_added)
            new.observe('fs', self._index_mapper_updated)
