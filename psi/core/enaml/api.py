from .manifest import PSIManifest
from .contribution import PSIContribution
from .editable_table_widget import ListDictTable, DataFrameTable, EditableTable


def load_manifests(objects, workbench):
    for o in objects:
        if isinstance(o, PSIContribution):
            o.load_manifest(workbench)
            load_manifests(o.children, workbench)

