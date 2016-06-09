import ast

from atom.api import Atom, Typed


def _dict_to_expr(d):
    if d is None:
        return d
    e = {}
    for k, v in d.items():
        if not isinstance(v, Expr):
            e[k] = Expr(unicode(v))
        else:
            e[k] = v
    return e


class _RecursiveAttrDict(dict):
    '''
    Dictionary that allows nesting and attribute-like access
    '''
    def __init__(self, items=None):
        if items is not None:
            if isinstance(items, dict):
                for key, value in items.items():
                    self.__setitem__(key, value)
            else:
                raise TypeError('expected dict')

    def __getitem__(self, key):
        if '.' in key:
            initial, remainder = key.split('.', 1)
            return self.__getitem__(initial)[remainder]
        else:
            return dict.__getitem__(self, key)

    def __setitem__(self, key, value):
        if '.' in key:
            initial, remainder = key.split('.', 1)
            target = self.setdefault(initial, _RecursiveAttrDict())
            if not isinstance(target, _RecursiveAttrDict):
                raise KeyError('Cannot set "{}" in "{}" ({})'
                               .format(remainder, initial, repr(target)))
            target[remainder] = value
        else:
            if isinstance(value, dict) and not \
                    isinstance(value, _RecursiveAttrDict):
                value = _RecursiveAttrDict(value)
        dict.__setitem__(self, key, value)

    def __contains__(self, key):
        if '.' in key:
            initial, remainder = key.split('.', 1)
            if dict.__contains__(self, initial):
                target = dict.__getitem__(self, initial)
                if not isinstance(target, _RecursiveAttrDict):
                    return False
                return remainder in target
            return False
        else:
            return dict.__contains__(self, key)

    def setdefault(self, key, default):
        if key not in self:
            self[key] = default
        return self[key]

    def copy(self):
        other = dict.copy(self)
        return _RecursiveAttrDict(other)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).iteritems():
            self[k] = v

    __setattr__ = __setitem__
    __getattr__ = __getitem__


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
        if not isinstance(expression, basestring):
            raise ValueError('Expression must be a string')
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

    # TODO - We make a copy of _locals, update the copy, then assign the copy
    # back to the class attribute. This ensures that the Enaml GUI is updated
    # properly. When signals are implemented in Enaml (hopefully by Enaml 1.0),
    # we will be able to get rid of this hack.
    _locals = Typed(_RecursiveAttrDict)
    _expressions = Typed(_RecursiveAttrDict)
    _globals = Typed(_RecursiveAttrDict)

    def __init__(self, expressions=None, globals=None):
        self._expressions = _RecursiveAttrDict(_dict_to_expr(expressions))
        self._globals = _RecursiveAttrDict(globals)
        self._locals = _RecursiveAttrDict()

    def update_expressions(self, expressions):
        self._expressions.update(_dict_to_expr(expressions))

    def update_symbols(self, symbols):
        self._globals.update(symbols)

    def reset(self):
        '''
        Clears the computed values (as well as any user-provided values) in
        preparation for the next cycle.
        '''
        self._locals = _RecursiveAttrDict()

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

        # Force a copy of the `_locals` dictionary to ensure that the GUI
        # updates as needed.
        _locals = self._locals.copy()
        _locals[name] = expr.evaluate(c)
        self._locals = _locals
