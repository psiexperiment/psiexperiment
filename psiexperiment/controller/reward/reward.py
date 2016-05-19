from enaml.core.api import Declarative, d_func


class Reward(Declarative):

    @d_func
    def provide_reward(self):
        raise NotImplementedError
