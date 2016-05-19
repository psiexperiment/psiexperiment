from enaml.core.api import d_func
from enaml.widgets.api import DockItem


class PersistentDockItem(DockItem):

    @d_func
    def get_state(self):
        raise NotImplementedError

    @d_func
    def set_state(self):
        raise NotImplementedError
