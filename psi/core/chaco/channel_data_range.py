from __future__ import division

from chaco.api import DataRange1D
from traits.api import Float, List, Instance, Enum, on_trait_change

import logging
log = logging.getLogger(__name__)

class ChannelDataRange(DataRange1D):

    sources         = List(Instance('cns.channel.Channel'))
    span            = Float(20)
    trig_delay      = Float(5)
    trigger         = Float(0)
    update_mode     = Enum('auto', 'auto full', 'triggered')
    scroll_period   = Float(20)

    def _trigger_changed(self):
        self.refresh()

    def _span_changed(self):
        self.refresh()

    def _trig_delay_changed(self):
        self.refresh()
        
    def _update_mode_changed(self):
        self.refresh()

    def get_max_time(self):
        bounds = [s.get_bounds()[1] for s in self.sources if s.get_size()>0]
        return 0 if len(bounds) == 0 else max(bounds)

    @on_trait_change('sources.added')
    def refresh(self):
        '''
        Keep this very simple.  The user cannot change low/high settings.  If
        they use this data range, the assumption is that they've decided they
        want tracking.
        '''
        log.debug('refreshing channel data range')
        span = self.span
        if self.update_mode == 'auto':
            # Update the bounds as soon as the data scrolls into the next span
            spans = self.get_max_time()//span
            high_value = (spans+1)*span-self.trig_delay
            low_value = high_value-span
        elif self.update_mode == 'auto full':
            # Don't update the bounds until we have a full span of data to
            # display
            spans = self.get_max_time()//span
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

    #def _sources_changed(self, old, new):
    #    for source in old:
    #        source.on_trait_change(self.refresh, 'added', remove=True)
    #    for source in new:
    #        source.on_trait_change(self.refresh, 'added')
    #    self.refresh()

    #def _sources_items_changed(self, event):
    #    for source in event.removed:
    #        source.on_trait_change(self.refresh, 'added', remove=True)
    #    for source in event.added:
    #        source.on_trait_change(self.refresh, 'added')
    #    self.refresh()
