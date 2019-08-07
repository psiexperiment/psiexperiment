from atom.api import Typed

from enaml.workbench.api import Plugin

from psi.token.block import ContinuousBlock, EpochBlock


TOKENS_POINT = 'psi.token.tokens'


class TokenPlugin(Plugin):

    _continuous_tokens = Typed(dict)
    _epoch_tokens = Typed(dict)

    def start(self):
        self._refresh_tokens()
        self._bind_observers()

    def stop(self):
        self._unbind_observers()

    def _refresh_tokens(self, event=None):
        etokens = {}
        ctokens = {}
        point = self.workbench.get_extension_point(TOKENS_POINT)
        for extension in point.extensions:
            if extension.factory:
                extension.factory()
            for token in extension.get_children(EpochBlock):
                etokens[token.name] = token
            for token in extension.get_children(ContinuousBlock):
                ctokens[token.name] = token
        self._epoch_tokens = etokens
        self._continuous_tokens = ctokens

    def _bind_observers(self):
        self.workbench.get_extension_point(TOKENS_POINT) \
            .observe('extensions', self._refresh_tokens)

    def _unbind_observers(self):
        self.workbench.get_extension_point(TOKENS_POINT) \
            .unobserve('extensions', self._refresh_tokens)

    def get_token(self, token_name):
        if token_name in self._epoch_tokens:
            return self._epoch_tokens[token_name]
        if token_name in self._continuous_tokens:
            return self._continuous_tokens[token_name]
        raise ValueError('%s token does not exist', token_name)
