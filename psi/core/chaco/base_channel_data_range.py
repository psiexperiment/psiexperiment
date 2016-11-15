from traits.api import Float, List, Instance
from chaco.api import DataRange1D


class BaseChannelDataRange(DataRange1D):

    current_time = Float()

    sources = List(Instance('psi.data.hdf_store.data_source.DataSource'))

    def _set_current_time(self, time):
        if self.current_time < time:
            self.current_time = time
            self.refresh()

    def _data_added(self, event):
        lb, ub = event['value']
        self._set_current_time(ub)

    def _sources_changed(self, old, new):
        for source in old:
            source.unobserve('added', self._data_added)
            source.unobserve('current_time', self._update_current_time)
        for source in new:
            source.observe('added', self._data_added)
            source.observe('current_time', self._update_current_time)
        self.refresh()

    def _sources_items_changed(self, event):
        for source in event.removed:
            source.unobserve('added', self._data_added)
            source.unobserve('current_time', self._update_current_time)
        for source in event.added:
            source.observe('added', self._data_added)
            source.observe('current_time', self._update_current_time)
        self.refresh()
