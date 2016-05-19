import unittest
import numpy as np

from atom.api import Atom, Bool

from psiexperiment.context.expression import Expr, ExpressionNamespace, \
    _RecursiveAttrDict


class TestExpression(unittest.TestCase):

    def test_eval(self):
        context = dict(a=1, b=2, c=3)
        test_cases = [
            ('1+2', 3),
            ('b*c', 6),
            ('1+c', 4)
        ]
        for expr, expected in test_cases:
            actual = Expr(expr).evaluate(context)
            self.assertEqual(actual, expected)

    def test_symtable(self):
        expr = Expr('randint(10)')
        expr.evaluate(dict(randint=np.random.randint))
        expr = Expr('np.random.randint(x)')
        expr.evaluate(dict(x=5, np=np))


class TestExpressionNamespace(unittest.TestCase):

    EXPRESSIONS = {
        'd.x':  Expr('10'),
        'a':    Expr('10*5'),
        'b':    Expr('a+1'),
        'c':    Expr('b*a'),
        'e':    Expr('d.x*2'),
        'f':    Expr('d.x*data.channel.bar'),
        'g':    Expr('random.randint(b)'),
    }

    def test_evaluation(self):
        ns = ExpressionNamespace(self.EXPRESSIONS)
        self.assertEqual(ns.get_value('c'), 2550)
        self.assertEqual(ns.get_value('e'), 20)
        self.assertEqual(ns.get_value('d.x'), 10)
        self.assertEqual(ns.get_value('f', {'data.channel.bar': 1.5}), 15)

    def test_evaluation_override(self):
        ns = ExpressionNamespace(self.EXPRESSIONS)
        self.assertEqual(ns.get_value('c', {'a': 2}), 6)
        self.assertEqual(ns.get_value('a', {'a': 2}), 2)

    def test_cache(self):
        # We know for this particular seed that second and third call to the
        # generator will not return the same value.
        random = np.random.RandomState(seed=1)
        ns = ExpressionNamespace(self.EXPRESSIONS, {'random': random})
        initial = ns.get_value('g')
        self.assertEqual(initial, ns.get_value('g'))
        self.assertEqual(initial, ns.get_value('g'))
        ns.reset()
        self.assertNotEqual(initial, ns.get_value('g'))

    def test_extra_context(self):
        random = np.random.RandomState(seed=1)
        ns = ExpressionNamespace(self.EXPRESSIONS, {'random': random})
        ns.set_values({'data.channel.bar': 3.1})
        ns.set_value('z', 32)
        values = ns.get_values()
        self.assertTrue('z' in values)
        self.assertTrue('data.channel.bar' in values)
        self.assertEqual(values['z'], 32)
        self.assertEqual(values['f'], 31)


class TestRecursiveAttrDict(unittest.TestCase):

    INITIAL = {'bigbang': {'stars': {'planet': 'earth'}, 'center': 'sol'}}

    def test_init(self):
        life = _RecursiveAttrDict(self.INITIAL)
        self.assertEqual(life.bigbang.stars.planet, 'earth')
        self.assertEqual(life.bigbang.center, 'sol')
        self.assertEqual(life.bigbang, life['bigbang'])

    def test_getsetattr(self):
        life = _RecursiveAttrDict(self.INITIAL)
        life['universe.galaxy'] = 'milky way'
        self.assertEqual(life.universe.galaxy, 'milky way')
        life['universe.a.b.c.d.e'] = 4
        self.assertEqual(life.universe.a.b.c.d.e, 4)
        self.assertEqual(life.universe.a.b.c.d, {'e': 4})
        self.assertEqual(life['universe.a.b.c.d.e'], 4)

    def test_builtin_getsetattr(self):
        life = _RecursiveAttrDict(self.INITIAL)
        self.assertEqual(getattr(life.bigbang.stars, 'planet'), 'earth')
        setattr(life, 'planet.stars', 'sol')
        self.assertEqual(getattr(life, 'planet.stars'), 'sol')

    def test_contains(self):
        life = _RecursiveAttrDict(self.INITIAL)
        self.assertTrue('bigbang.stars.planet' in life)
        self.assertFalse('bigbang.stars.planet.earth' in life)
        self.assertFalse('bigbang.d.x' in life)


class ANT(Atom):

    observed = Bool()

    def __init__(self, *args, **kwargs):
        super(ANT, self).__init__(*args, **kwargs)

    def mark_observed(self, event):
        print event
        self.observed = True


class TestAtomNotification(unittest.TestCase):

    def setUp(self):
        expressions = {
            'a': Expr('2'),
            'b': Expr('a*10'),
        }
        self.ant = ANT()
        self.ns = ExpressionNamespace(expressions)
        self.ns.observe('_locals', self.ant.mark_observed)

    def test_get_value_notification(self):
        for v in ('a', 'b'):
            self.ant.observed = False
            self.ns.get_value(v)
            self.assertTrue(self.ant.observed)

    def test_set_value_notification(self):
        self.ant.observed = False
        self.ns.set_value('c', 5)
        self.assertTrue(self.ant.observed)

    def test_get_value_notification_no_change(self):
        self.ant.observed = False
        self.ns.get_value('b')
        self.assertTrue(self.ant.observed)

        # Should not trigger notification because 'a' was already computed when
        # getting 'b', so there was no change in value.
        self.ant.observed = False
        self.ns.get_value('a')
        self.assertFalse(self.ant.observed)
