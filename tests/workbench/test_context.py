import pytest


def test_eval(workbench):
    expected = [
        dict(repetitions=2, level=60, fc=32e3/2),
        dict(repetitions=10, level=60, fc=32e3/10),
        dict(repetitions=15, level=60, fc=32e3/15),
        dict(repetitions=20, level=60, fc=32e3/20),
        dict(repetitions=20, level=60, fc=32e3/20),
        dict(repetitions=2, level=60, fc=32e3/2),
        dict(repetitions=10, level=60, fc=32e3/10),
    ]
    context = workbench.get_plugin('psi.context')

    # Ensure that we loop properly through the selector sequence
    context.apply_changes()
    for e in expected:
        context.next_setting('default', save_prior=False)
        assert e == context.get_values()

    # Ensure that apply_changes restarts the selector sequence.
    context.apply_changes()
    for e in expected:
        context.next_setting('default', save_prior=False)
        assert e == context.get_values()

    # Ensure that changes to expressions after apply_changes does not affect the
    # result.
    context.apply_changes()
    context.context_items['fc'].expression = '1e3'
    for e in expected:
        context.next_setting('default', save_prior=False)
        assert e == context.get_values()

    # Now, the result should change.
    context.apply_changes()
    for e in expected:
        context.next_setting('default', save_prior=False)
        e['fc'] = 1e3
        assert e == context.get_values()


def test_unique_values(workbench):
    context = workbench.get_plugin('psi.context')
    result = context.unique_values('repetitions')
    expected = {2, 10, 15, 20}
    assert result == expected


def test_update(workbench):
    '''
    Tests whether the change detection algorithm works as intended.
    '''
    context = workbench.get_plugin('psi.context')
    context.apply_changes()

    assert context.changes_pending == False
    assert context.get_value('level') == 60

    context.next_setting('default', False)
    assert context.changes_pending == False
    assert context.get_value('level') == 60
    context.context_items['level'].expression = '32'
    assert context.changes_pending == True
    assert context.get_value('level') == 60

    context.next_setting('default', False)
    assert context.changes_pending == True
    assert context.get_value('level') == 60

    context.apply_changes()
    context.next_setting('default', False)
    assert context.changes_pending == False
    assert context.get_value('level') == 32

    context.context_items['level'].expression = '60'
    assert context.changes_pending == True
    context.revert_changes()
    assert context.changes_pending == False

    item = context.context_items['repetitions']
    context.selectors['default'].set_value(0, item, '5')
    assert context.changes_pending == True
    assert context.selectors['default'].get_value(0, item) == 5
