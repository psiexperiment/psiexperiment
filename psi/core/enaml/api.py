import logging
log = logging.getLogger(__name__)

from .manifest import PSIManifest
from .contribution import PSIContribution
from .editable_table_widget import ListDictTable, DataFrameTable, EditableTable
from .list_view import ListView


def load_manifests(objects, workbench):
    '''
    Recursively load manifests for all PSIConbtribution subclasses in hierarchy
    '''
    for o in objects:
        if isinstance(o, PSIContribution):
            o.load_manifest(workbench)
            load_manifests(o.children, workbench)
        elif isinstance(o, list):
            load_manifests(o, workbench)
        elif hasattr(o, 'children'):
            load_manifests(o.children, workbench)
