from .dataframe_table_widget import DataframeTable
from .manifest import PSIManifest
from .contribution import PSIContribution


def load_manifests(objects, workbench):
    for o in objects:
        if isinstance(o, PSIContribution):
            o.load_manifest(workbench)
            load_manifests(o.children, workbench)

