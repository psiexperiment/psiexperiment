from atom.api import Dict


class BaseCallbackMixin:

    #: Dictionary of callbacks
    _callbacks = Dict()

    def complete(self):
        log.debug('Triggering "done" callbacks')
        for cb in self._callbacks.get('done', []):
            cb()


class ChannelSliceCallbackMixin(BaseCallbackMixin):

    def _get_channel_slice(self, task_name, channel_name):
        if channel_name is None:
            return Ellipsis
        # We want the channel slice to preserve dimensiality (i.e, we don't
        # want to drop the channel dimension from the PipelineData object).
        i = self._tasks[task_name]._properties['names'].index(channel_name)
        return [i]

    def register_done_callback(self, callback):
        self._callbacks.setdefault('done', []).append(callback)

    def register_ao_callback(self, callback, channel_name=None):
        s = self._get_channel_slice('hw_ao', channel_name)
        self._callbacks.setdefault('ao', []).append((channel_name, s, callback))

    def register_ai_callback(self, callback, channel_name=None):
        s = self._get_channel_slice('hw_ai', channel_name)
        self._callbacks.setdefault('ai', []).append((channel_name, s, callback))

    def register_ci_callback(self, callback, channel_name=None):
        s = self._get_channel_slice('hw_ci', channel_name)
        self._callbacks.setdefault('ci', []).append((channel_name, s, callback))

    def register_di_callback(self, callback, channel_name=None):
        s = self._get_channel_slice('hw_di', channel_name)
        self._callbacks.setdefault('di', []).append((channel_name, s, callback))

    def unregister_done_callback(self, callback):
        try:
            self._callbacks['done'].remove(callback)
        except KeyError:
            log.warning('Callback no longer exists.')

    def unregister_ao_callback(self, callback, channel_name):
        try:
            s = self._get_channel_slice('hw_ao', channel_name)
            self._callbacks['ao'].remove((channel_name, s, callback))
        except (KeyError, AttributeError):
            log.warning('Callback no longer exists.')

    def unregister_ai_callback(self, callback, channel_name):
        try:
            s = self._get_channel_slice('hw_ai', channel_name)
            self._callbacks['ai'].remove((channel_name, s, callback))
        except (KeyError, AttributeError):
            log.warning('Callback no longer exists.')

    def unregister_di_callback(self, callback, channel_name):
        s = self._get_channel_slice('hw_di', channel_name)
        self._callbacks['di'].remove((channel_name, s, callback))

