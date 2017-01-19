from __future__ import division

import logging
log = logging.getLogger(__name__)

from traits.api import Float, List, Instance, Enum, on_trait_change

from .base_channel_data_range import BaseChannelDataRange


class ChannelDataRange(BaseChannelDataRange):

    span = Float(20)
    trig_delay = Float(5)
    trigger = Float(0)
    update_mode = Enum('auto', 'auto full', 'triggered')
    scroll_period = Float(20)

    def _trigger_changed(self):
        self.refresh()

    def _span_changed(self):
        self.refresh()

    def _trig_delay_changed(self):
        self.refresh()
        
    def _update_mode_changed(self):
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
