import inspect


def rpc(plugin_name, method_name):
    def wrapper(event):
        plugin = event.workbench.get_plugin(plugin_name)
        f = getattr(plugin, method_name)
        argnames = inspect.getargspec(f).args
        parameters = {}
        for k, v in event.parameters.items():
            if k in argnames:
                parameters[k] = v
        return getattr(plugin, method_name)(**parameters)
    return wrapper
