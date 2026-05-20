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
        context.next_setting('default')
        assert e == context.get_values()

    # Ensure that apply_changes restarts the selector sequence.
    context.apply_changes()
    for e in expected:
        context.next_setting('default')
        assert e == context.get_values()

    # Ensure that changes to expressions after apply_changes does not affect the
    # result.
    context.apply_changes()
    context.context_items['fc'].expression = '1e3'
    for e in expected:
        context.next_setting('default')
        assert e == context.get_values()

    # Now, the result should change.
    context.apply_changes()
    for e in expected:
        context.next_setting('default')
        e['fc'] = 1e3
        assert e == context.get_values()


def test_unique_values(workbench):
    context = workbench.get_plugin('psi.context')
    result = context.unique_values('repetitions')
    expected = {2, 10, 15, 20}
    assert result == expected


def test_multiple_unique_values(workbench):
    context = workbench.get_plugin('psi.context')
    result = context.unique_values(['repetitions', 'level'])
    expected = {(2, 60.0), (10, 60.0), (15, 60.0), (20, 60.0)}
    assert result == expected


def test_update(workbench):
    '''
    Tests whether the change detection algorithm works as intended.
    '''
    context = workbench.get_plugin('psi.context')
    context.apply_changes()

    assert context.changes_pending == False
    assert context.get_value('level') == 60

    context.next_setting('default')
    assert context.changes_pending == False
    assert context.get_value('level') == 60
    context.context_items['level'].expression = '32'
    assert context.changes_pending == True
    assert context.get_value('level') == 60

    context.next_setting('default')
    assert context.changes_pending == True
    assert context.get_value('level') == 60

    context.apply_changes()
    context.next_setting('default')
    assert context.changes_pending == False
    assert context.get_value('level') == 32

    context.context_items['level'].expression = '60'
    assert context.changes_pending == True
    context.revert_changes()
    assert context.changes_pending == False

    item = context.context_items['repetitions']
    context.selectors['default'].set_value(0, item, '5')
    assert context.changes_pending == True
    # set_value goes through item.coerce_to_type, which converts '5' -> 5.
    assert context.selectors['default'].get_value(0, item) == 5


def test_duplicate_context_groups(workbench, helpers):
    error = 'ContextGroup with the same name has already been registered'
    with pytest.raises(ValueError, match=error):
        workbench.register(helpers.DuplicateContextGroupManifest())


def test_duplicate_context_items(workbench, helpers):
    error = 'Context item .* already defined'
    with pytest.raises(ValueError, match=error):
        workbench.register(helpers.DuplicateContextItemManifest())


def test_register_unregister_context_items(workbench, helpers):
    plugin = workbench.get_plugin('psi.context')

    def _check_items(plugin, include, exclude):
        # ContextGroup exposes ContextItem / ContextSet children via `subgroups`.
        item_names = [i.name for i in plugin.context_groups['default'].subgroups]
        for item in include:
            assert item in item_names
        for item in exclude:
            assert item not in item_names

    option1 = helpers.ContextItemOption1()
    option2 = helpers.ContextItemOption2()
    workbench.register(option1)
    _check_items(plugin, ['repetitions_1'], ['repetitions_2'])
    workbench.unregister(option1.id)
    _check_items(plugin, [], ['repetitions_1', 'repetitions_2'])
    workbench.register(option2)
    _check_items(plugin, ['repetitions_2'], ['repetitions_1'])
    workbench.unregister(option2.id)
    _check_items(plugin, [], ['repetitions_1', 'repetitions_2'])


def test_n_values_matches_selector_length(workbench):
    context = workbench.get_plugin('psi.context')
    # The helper manifest's default selector has 5 settings for repetitions.
    assert context.n_values('default') == 5


def test_get_range_for_repetitions(workbench):
    context = workbench.get_plugin('psi.context')
    # Repetitions values are {20, 15, 10, 2, 20} -> min 2, max 20.
    assert context.get_range('repetitions') == (2, 20)


def test_get_context_info_contains_all_items(workbench):
    context = workbench.get_plugin('psi.context')
    info = context.get_context_info()
    assert set(info.keys()) == {'repetitions', 'level', 'fc'}
    assert info['repetitions']['rove'] is True
    # 'level' and 'fc' are not roved.
    assert info['level']['rove'] is False
    assert info['fc']['rove'] is False


def test_rove_and_unrove_item(workbench):
    context = workbench.get_plugin('psi.context')
    level_item = context.context_items['level']
    selector = context.selectors['default']

    # Initially level is not roved.
    assert level_item not in selector.context_items

    level_item.rove = True
    assert level_item in selector.context_items

    level_item.rove = False
    assert level_item not in selector.context_items


def test_get_item_info_returns_metadata(workbench):
    context = workbench.get_plugin('psi.context')
    info = context.get_item_info('repetitions')
    assert info['rove'] is True
    assert info['default'] == 80


def test_unique_values_string_returns_set_of_scalars(workbench):
    context = workbench.get_plugin('psi.context')
    # Passing a string returns flat values; passing a list returns tuples
    # (the existing test_unique_values vs test_multiple_unique_values covers
    # this distinction). Re-verify the boundary explicitly.
    s = context.unique_values('repetitions')
    assert all(not isinstance(v, tuple) for v in s)


def test_revert_changes_restores_state(workbench):
    context = workbench.get_plugin('psi.context')
    context.apply_changes()
    original_expr = context.context_items['level'].expression
    context.context_items['level'].expression = '99'
    assert context.changes_pending is True
    context.revert_changes()
    assert context.context_items['level'].expression == original_expr
    assert context.changes_pending is False
