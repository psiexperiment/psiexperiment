from psi.core.enaml.editable_table_widget import LiveEdit


def test_live_edit():
    live_edit = LiveEdit()
    assert not live_edit
    with live_edit:
        assert live_edit
    assert not live_edit
