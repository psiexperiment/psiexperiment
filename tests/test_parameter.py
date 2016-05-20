import unittest

from psi.context.api import Parameter


class TestParameter(unittest.TestCase):

    def setUp(self):
        self.parameter1 = Parameter(name='foo')
        self.parameter2 = Parameter(name='foo')
        self.parameter3 = Parameter(name='biz')

    def test_equality(self):
        self.assertTrue(self.parameter1 == self.parameter2)
        self.assertFalse(self.parameter1 == self.parameter3)
