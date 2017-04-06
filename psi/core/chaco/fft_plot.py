from .base_channel_plot import BaseChannelPlot


class FFTChannelPlot(BaseChannelPlot):

    def _data_changed(self, event):
        print('data change')

    def _data_added(self, event):
        print('data added')
