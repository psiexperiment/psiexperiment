def find_extension(workbench, plugin_id, extension_id, extension_class):
    extension = None
    manifest = workbench.get_manifest(plugin_id)
    for extension in manifest.extensions:
        if extension.id == extension_id:
            break
    if extension is None:
        raise ValueError('Extension does not exist')
    for child in extension.children:
        if isinstance(child, extension_class):
            return child
    raise ValueError('Extension does not exist')
