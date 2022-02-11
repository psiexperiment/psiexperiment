from psi import application


def test_list_paradigm_descriptions():
    result = application.list_paradigm_descriptions()
    assert len(result) == 4
    for r in result:
        assert r.startswith('psi.')
