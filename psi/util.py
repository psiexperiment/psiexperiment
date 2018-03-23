import inspect

from atom.api import Property


def rpc(plugin_name, method_name):
    '''Decorator to map an Enaml command handler to the plugin method'''
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


def get_tagged_values(obj, tag_name, tag_value=True, exclude_properties=False):
    result = {}
    if exclude_properties:
        match = lambda m: m.metadata and m.metadata.get(tag_name) == tag_value \
            and not isinstance(member, Property)
    else:
        match = lambda m: m.metadata and m.metadata.get(tag_name) == tag_value

    for name, member in obj.members().items():
        if match(member):
            value = getattr(obj, name)
            result[name] = value
    return result


def declarative_to_dict(obj, tag_name, tag_value=True):
    from atom.api import Atom
    import numpy as np
    result = {}
    for name, member in obj.members().items():
        if member.metadata and member.metadata.get(tag_name) == tag_value:
            value = getattr(obj, name)
            if isinstance(value, Atom):
                # Recurse into the class
                result[name] = declarative_to_dict(value, tag_name, tag_value)
            elif isinstance(value, np.ndarray):
                # Convert to a list
                result[name] = value.tolist()
            else:
                result[name] = value
    result['type'] = obj.__class__.__name__
    return result


def coroutine(func):
    '''Decorator to auto-start a coroutine.'''
    def start(*args, **kwargs):
        cr = func(*args, **kwargs)
        next(cr)
        return cr
    return start


def copy_declarative(old, **kw):
    attributes = get_tagged_values(old, 'metadata', exclude_properties=True)
    print(old, attributes.keys())
    attributes.update(kw)
    new = old.__class__(**attributes)
    return new
