import unittest

import numpy as np

from psiexperiment.parameter import Parameter


class TestParameter(unittest.TestCase):

    def test_parameter_init(self):
        parameter = Parameter('foo', np.float32, expression='1*2')
        self.assertEqual(parameter.default_value, np.float32())
