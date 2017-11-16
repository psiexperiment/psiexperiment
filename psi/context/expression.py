import logging
log = logging.getLogger(__name__)

import ast

from atom.api import Atom, Typed


class _FullNameGetter(ast.NodeVisitor):

    def __init__(self, *args, **kwargs):
        self.names = []
        super(_FullNameGetter, self).__init__(*args, **kwargs)

    def visit_Name(self, node):
        self.names.append(node.id)

    def visit_Attribute(self, node):
        names = []
        while isinstance(node, ast.Attribute):
            names.append(node.attr)
            node = node.value
        names.append(node.id)
        name = '.'.join(names[::-1])
        self.names.append(name)


class Expr(object):

    def __init__(self, expression):
        if not isinstance(expression, str):
            raise ValueError('Expression must be a string')
        if not expression:
            raise ValueError('No value provided for expression')
        self._expression = expression
        self._code = compile(expression, 'dynamic', 'eval')
        self._dependencies = Expr.get_dependencies(expression)

    def evaluate(self, context):
        return eval(self._expression, context)

    @staticmethod
    def get_dependencies(expression):
        tree = ast.parse(expression)
        ng = _FullNameGetter()
        ng.visit(tree)
        return ng.names


class ExpressionNamespace(Atom):

    _locals = Typed(dict, {})
    _expressions = Typed(dict, {})
    _globals = Typed(dict, {})

    def __init__(self, expressions=None, globals=None):
        if globals is None:
            globals = {}
        if expressions is None:
            expressions = {}
        self._locals = {}
        self._globals = globals
        self._expressions = {k: Expr(str(v)) for k, v in expressions.items()}

    def update_expressions(self, expressions):
        self._expressions.update({k: Expr(str(v)) for k, v in expressions.items()})

    def update_symbols(self, symbols):
        self._globals.update(symbols)

    def reset(self, context_item_names=None):
        '''
        Clears the computed values (as well as any user-provided values) in
        preparation for the next cycle.
        '''
        self._locals = {}

    def get_value(self, name, context=None):
        if name not in self._locals:
            self._evaluate_value(name, context)
        return self._locals[name]

    def get_values(self, context=None):
        for name in self._expressions.keys():
            if name not in self._locals:
                self._evaluate_value(name, context)
        return dict(self._locals.copy())

    def set_value(self, name, value):
        _locals = self._locals.copy()
        _locals[name] = value
        self._locals = _locals

    def set_values(self, values):
        _locals = self._locals.copy()
        _locals.update(values)
        self._locals = _locals

    def _evaluate_value(self, name, context=None):
        if context is None:
            context = {}

        if name in context:
            self._locals[name] = context[name]
            return

        expr = self._expressions[name]
        c = self._globals.copy()
        c.update(self._locals)
        c.update(context)

        # Build a context dictionary containing the dependencies required for
        # evaluating the expression.
        for d in expr._dependencies:
            if d not in c and d in self._expressions:
                c[d] = self.get_value(d, c)
        
        # Note that in the past I was forcing a copy of self._locals to ensure
        # that the GUI was updated as needed; however, this proved to be a very
        # slow process since it triggered a cascade of GUI updates. 
        self._locals[name] = expr.evaluate(c)
