from ..engine import Engine


class NullEngine(Engine):

    def register_ao_callback(self, callback, channel_name):
        pass

    def register_ai_callback(self, callback, channel_name):
        pass

    def register_et_callback(self, callback, channel_name):
        pass

    def unregister_ao_callback(self, callback, channel_name):
        pass

    def unregister_ai_callback(self, callback, channel_name):
        pass

    def unregister_et_callback(self, callback, channel_name):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def get_ts(self):
        pass
