def rpc(plugin_name, method_name):
    def wrapper(event):
        plugin = event.workbench.get_plugin(plugin_name)
        return getattr(plugin, method_name)(**event.parameters)
    return wrapper

