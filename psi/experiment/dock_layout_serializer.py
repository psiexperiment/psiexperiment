'''
Conversion between enaml's Atom-based dock layout tree and plain Python
data (dict/list/str/int/bool/None) that can be round-tripped through YAML
or JSON instead of pickle.

``enaml.widgets.dock_area.DockArea.save_layout()`` returns a small, closed
tree of ``enaml.layout.dock_layout.LayoutNode`` subclasses (``ItemLayout``,
``TabLayout``, ``SplitLayout``/``HSplitLayout``/``VSplitLayout``,
``DockBarLayout``, ``AreaLayout``, ``DockLayout``). These are Atom objects,
not plain containers, so they can't be handed directly to ``yaml.dump``.
The node grammar is fixed and small, so a hand-written recursive
converter is straightforward -- see ``enaml/qt/docking/layout_saver.py``
for the authoritative list of node types and fields this mirrors.

``HSplitLayout``/``VSplitLayout`` are reconstructed as plain
``SplitLayout`` -- they are only convenience constructors that pin the
``orientation`` field, which is what every consumer actually inspects (no
code in enaml isinstance-checks for the H/V subclasses specifically).
'''
from enaml.layout.dock_layout import (
    AreaLayout, DockBarLayout, DockLayout, ItemLayout, SplitLayout, TabLayout,
)
from enaml.layout.geometry import Rect


def _rect_to_list(rect):
    return [rect.x, rect.y, rect.width, rect.height]


def _rect_from_list(seq):
    return Rect(*seq)


def dock_layout_node_to_dict(node):
    '''Recursively convert a LayoutNode tree into plain dict/list data.'''
    if isinstance(node, ItemLayout):
        return {
            'type': 'item',
            'name': node.name,
            'floating': node.floating,
            'geometry': _rect_to_list(node.geometry),
            'linked': node.linked,
            'maximized': node.maximized,
        }
    if isinstance(node, TabLayout):
        return {
            'type': 'tab',
            'tab_position': node.tab_position,
            'index': node.index,
            'maximized': node.maximized,
            'items': [dock_layout_node_to_dict(i) for i in node.items],
        }
    if isinstance(node, SplitLayout):
        return {
            'type': 'split',
            'orientation': node.orientation,
            'sizes': list(node.sizes),
            'items': [dock_layout_node_to_dict(i) for i in node.items],
        }
    if isinstance(node, DockBarLayout):
        return {
            'type': 'dockbar',
            'position': node.position,
            'items': [dock_layout_node_to_dict(i) for i in node.items],
        }
    if isinstance(node, AreaLayout):
        return {
            'type': 'area',
            'item': dock_layout_node_to_dict(node.item)
                if node.item is not None else None,
            'dock_bars': [dock_layout_node_to_dict(b) for b in node.dock_bars],
            'floating': node.floating,
            'geometry': _rect_to_list(node.geometry),
            'linked': node.linked,
            'maximized': node.maximized,
        }
    if isinstance(node, DockLayout):
        return {
            'type': 'dock',
            'items': [dock_layout_node_to_dict(i) for i in node.items],
        }
    raise TypeError(f'Unsupported dock layout node type: {type(node)!r}')


def dock_layout_node_from_dict(d):
    '''Reverse of dock_layout_node_to_dict.'''
    if d is None:
        return None
    kind = d['type']
    if kind == 'item':
        return ItemLayout(
            d['name'],
            floating=d['floating'],
            geometry=_rect_from_list(d['geometry']),
            linked=d['linked'],
            maximized=d['maximized'],
        )
    if kind == 'tab':
        items = [dock_layout_node_from_dict(i) for i in d['items']]
        return TabLayout(
            *items,
            tab_position=d['tab_position'],
            index=d['index'],
            maximized=d['maximized'],
        )
    if kind == 'split':
        items = [dock_layout_node_from_dict(i) for i in d['items']]
        return SplitLayout(
            *items,
            orientation=d['orientation'],
            sizes=list(d['sizes']),
        )
    if kind == 'dockbar':
        items = [dock_layout_node_from_dict(i) for i in d['items']]
        return DockBarLayout(*items, position=d['position'])
    if kind == 'area':
        item = dock_layout_node_from_dict(d['item'])
        return AreaLayout(
            item,
            dock_bars=[dock_layout_node_from_dict(b) for b in d['dock_bars']],
            floating=d['floating'],
            geometry=_rect_from_list(d['geometry']),
            linked=d['linked'],
            maximized=d['maximized'],
        )
    if kind == 'dock':
        items = [dock_layout_node_from_dict(i) for i in d['items']]
        return DockLayout(*items)
    raise ValueError(f'Unknown dock layout node type: {kind!r}')


def workspace_layout_to_dict(layout):
    '''Convert the dict returned by ExperimentPlugin.get_layout() into
    plain dict/list/scalar data suitable for yaml.dump/json.dump.'''
    return {
        'geometry': _rect_to_list(layout['geometry']),
        'toolbars': layout['toolbars'],
        'dock_layout': dock_layout_node_to_dict(layout['dock_layout']),
    }


def workspace_layout_from_dict(d):
    '''Reverse of workspace_layout_to_dict; returns a dict suitable for
    ExperimentPlugin.set_layout().'''
    return {
        'geometry': _rect_from_list(d['geometry']),
        'toolbars': d['toolbars'],
        'dock_layout': dock_layout_node_from_dict(d['dock_layout']),
    }
