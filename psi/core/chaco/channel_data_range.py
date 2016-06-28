from __future__ import division

from chaco.api import DataRange1D
from traits.api import Float, List, Instance, Enum, on_trait_change

import logging
log = logging.getLogger(__name__)

class ChannelDataRange(DataRange1D):

    sources = List(Instance('psi.data.hdf_store.data_source.DataSource'))
    span = Float(20)
    trig_delay = Float(5)
    trigger = Float(0)
    update_mode = Enum('auto', 'auto full', 'triggered')
    scroll_period = Float(20)
    current_time = Float()

    def _trigger_changed(self):
        self.refresh()

    def _span_changed(self):
        self.refresh()

    def _trig_delay_changed(self):
        self.refresh()
        
    def _update_mode_changed(self):
        self.refresh()

    def _update_current_time(self, event):
        if self.current_time != event['value']:
            self.current_time = event['value']
            self.refresh()

    def _data_added(self, event):
        self.refresh()

    def refresh(self, event=None):
        '''
        Keep this very simple.  The user cannot change low/high settings.  If
        they use this data range, the assumption is that they've decided they
        want tracking.
        '''
        span = self.span
        if self.update_mode == 'auto':
            # Update the bounds as soon as the data scrolls into the next span
            spans = self.current_time//span
            high_value = (spans+1)*span-self.trig_delay
            low_value = high_value-span
        elif self.update_mode == 'auto full':
            # Don't update the bounds until we have a full span of data to
            # display
            spans = self.current_time//span
            high_value = spans*span-self.trig_delay
            low_value = high_value-span
        elif self.update_mode == 'triggered':
            # We want the lower bound of the range to be referenced to the
            # trigger itself.
            low_value = self.trigger-self.trig_delay
            high_value = low_value+span

        # Important!  Don't update the values unless they are different.
        # Needlessly updating these values results in excessive screen redraws,
        # computations, etc., since other components may be "listening" to
        # ChannelDataRange for changes to its bounds.
        if (self._low_value != low_value) or (self._high_value != high_value):
            self._low_value = low_value
            self._high_value = high_value
            self.updated = (low_value, high_value)

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
