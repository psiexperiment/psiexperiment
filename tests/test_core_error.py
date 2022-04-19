import pytest
from psi.core.enaml.api import PSIContribution


class PSITestContribution4(PSIContribution):
    pass


def test_find_manifest_error():
    with pytest.raises(ModuleNotFoundError):
        contribution = PSITestContribution4()
        contribution.find_manifest_class()
