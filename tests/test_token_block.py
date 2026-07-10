import pytest

from psi.context.api import Parameter
from psi.token.block import Block


def test_block_invalid_hide_names_parameter():
    block = Block(name='blk')
    Parameter(name='a', parent=block)
    block.hide = ['bogus']
    with pytest.raises(AttributeError, match='bogus'):
        block.initialize()


def test_block_invalid_value_key_names_key():
    # Regression: the error message was missing its f-prefix, so the
    # offending key was never interpolated into the message.
    block = Block(name='blk')
    Parameter(name='a', parent=block)
    block.values = {'bogus': 1}
    with pytest.raises(AttributeError, match='bogus'):
        block.initialize()


def test_block_valid_hide_and_values():
    block = Block(name='blk')
    p = Parameter(name='a', parent=block)
    block.hide = ['a']
    block.initialize()
    assert p.visible is False
