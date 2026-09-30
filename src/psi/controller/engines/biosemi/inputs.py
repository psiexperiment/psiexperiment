import logging
log = logging.getLogger(__name__)

import numpy as np

from psiaudio.pipeline import coroutine

from psi.controller.api import EventInput

from pyactivetwo.client import decode_trigger


@coroutine
def decode(fs, target):
    t_prior = -1
    prior_ttl = np.array([0])
    while True:
        data = (yield) >> 8
        ttl = np.r_[prior_ttl, (data & 1).astype('i')]
        prior_ttl = ttl[-1:]

        rising = np.diff(ttl, axis=-1) == 1
        info_set = []
        for i in (np.flatnonzero(rising) + 1):
            md = decode_trigger(data[i])
            trigger = md['trigger'] >> 1
            md['is_target'] = bool(trigger & 1)
            md['is_response'] = bool((trigger >> 1) & 1)
            md['stim_index'] = int(trigger >> 2)
            info = {
                't0': float((t_prior + i) / fs),
                'key': md['stim_index'],
                'metadata': md,
            }
            info_set.append(info)
        if info_set:
            target(info_set)

        t_prior += data.shape[-1]


class Decode(EventInput):

    def configure_callback(self):
        cb = super().configure_callback()
        return decode(self.fs, cb).send
