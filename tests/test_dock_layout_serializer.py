"""Tests for psi.experiment.dock_layout_serializer."""
from enaml.layout.dock_layout import (
    AreaLayout, DockBarLayout, DockLayout, HSplitLayout, ItemLayout,
    SplitLayout, TabLayout, VSplitLayout,
)
from enaml.layout.geometry import Rect

from psi.experiment.dock_layout_serializer import (
    dock_layout_node_from_dict, dock_layout_node_to_dict,
    workspace_layout_from_dict, workspace_layout_to_dict,
)


def _roundtrip(node):
    d = dock_layout_node_to_dict(node)
    node2 = dock_layout_node_from_dict(d)
    assert dock_layout_node_to_dict(node2) == d
    return d


def test_item_layout_roundtrip():
    item = ItemLayout('microphone', floating=True, geometry=(1, 2, 3, 4),
                       linked=True, maximized=True)
    d = _roundtrip(item)
    assert d == {
        'type': 'item', 'name': 'microphone', 'floating': True,
        'geometry': [1, 2, 3, 4], 'linked': True, 'maximized': True,
    }


def test_item_layout_default_geometry_is_minus_one():
    # Regression: an un-floated item's default geometry is (-1, -1, -1, -1)
    # -- must survive the roundtrip rather than being coerced to zeros.
    item = ItemLayout('a')
    d = _roundtrip(item)
    assert d['geometry'] == [-1, -1, -1, -1]


def test_tab_layout_roundtrip():
    tab = TabLayout(ItemLayout('a'), ItemLayout('b'), tab_position='left',
                     index=1, maximized=True)
    _roundtrip(tab)


def test_split_layout_roundtrip():
    split = SplitLayout(ItemLayout('a'), ItemLayout('b'),
                         orientation='vertical', sizes=[10, 20])
    _roundtrip(split)


def test_hsplit_and_vsplit_reconstruct_as_split_with_matching_orientation():
    # HSplitLayout/VSplitLayout are convenience constructors, not distinct
    # node types on the wire -- orientation is what every consumer reads.
    hsplit = HSplitLayout(ItemLayout('a'), ItemLayout('b'))
    node = dock_layout_node_from_dict(dock_layout_node_to_dict(hsplit))
    assert type(node) is SplitLayout
    assert node.orientation == 'horizontal'

    vsplit = VSplitLayout(ItemLayout('a'), ItemLayout('b'))
    node = dock_layout_node_from_dict(dock_layout_node_to_dict(vsplit))
    assert node.orientation == 'vertical'


def test_dockbar_layout_roundtrip():
    bar = DockBarLayout(ItemLayout('a'), ItemLayout('b'), position='right')
    _roundtrip(bar)


def test_area_layout_roundtrip_with_nested_split_and_dockbar():
    split = HSplitLayout(ItemLayout('a'), TabLayout(ItemLayout('b'),
                                                      ItemLayout('c')))
    bar = DockBarLayout(ItemLayout('d'), position='bottom')
    area = AreaLayout(split, dock_bars=[bar], floating=True,
                       geometry=(5, 6, 7, 8), linked=True, maximized=True)
    _roundtrip(area)


def test_area_layout_with_no_item_roundtrips_to_none():
    area = AreaLayout()
    d = _roundtrip(area)
    assert d['item'] is None


def test_dock_layout_roundtrip_multiple_areas():
    area1 = AreaLayout(ItemLayout('a'))
    area2 = AreaLayout(ItemLayout('b'), floating=True, geometry=(1, 1, 1, 1))
    dock = DockLayout(area1, area2)
    _roundtrip(dock)


def test_unsupported_node_type_raises():
    class _NotALayoutNode:
        pass
    import pytest
    with pytest.raises(TypeError, match='Unsupported dock layout node'):
        dock_layout_node_to_dict(_NotALayoutNode())


def test_unknown_dict_type_raises():
    import pytest
    with pytest.raises(ValueError, match='Unknown dock layout node'):
        dock_layout_node_from_dict({'type': 'bogus'})


def test_workspace_layout_roundtrip():
    layout = {
        'geometry': Rect(0, 0, 800, 600),
        'toolbars': {
            'main': {'floating': False, 'orientation': 'horizontal',
                      'dock_area': 'top', 'x': 0, 'y': 0},
        },
        'dock_layout': DockLayout(AreaLayout(ItemLayout('a'))),
    }
    d = workspace_layout_to_dict(layout)
    # Must be plain data -- no Atom/Rect objects left, safe for yaml.dump.
    import yaml
    text = yaml.dump(d, default_flow_style=False)
    d2 = yaml.load(text, Loader=yaml.Loader)

    layout2 = workspace_layout_from_dict(d2)
    assert layout2['geometry'] == layout['geometry']
    assert layout2['toolbars'] == layout['toolbars']
    assert dock_layout_node_to_dict(layout2['dock_layout']) == \
        dock_layout_node_to_dict(layout['dock_layout'])
