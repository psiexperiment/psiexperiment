import pytest

from psi import application


@pytest.mark.skip(reason="Paradigm modules (e.g. cfts, abr) live in separate "
                         "packages now; they aren't auto-imported here so the "
                         "registry is empty. The expected count of 5 reflects "
                         "an older bundled set.")
def test_list_paradigm_descriptions():
    result = application.list_paradigm_descriptions()
    assert len(result) == 5
    for r in result:
        assert r.startswith('psi.')
