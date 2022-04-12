import logging
log = logging.getLogger(__name__)

import enaml

with enaml.imports():
    from psi.paradigms.cfts.cfts_mixins import FreqLevelSelector
    from psi.context.api import Parameter


TEST_VALUES = [
    {'settings': dict(freq_lb=5.6, freq_ub=45.2, freq_step=0.5, level_lb=10, level_ub=80, level_step=5),
     'frequencies': {5656, 8000, 11313, 16000, 22627, 32000, 45254},
     'levels': {10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80},
     },
    {'settings': dict(freq_lb=8, freq_ub=32, freq_step=1, level_lb=10, level_ub=80, level_step=20),
     'frequencies': {8000, 16000, 32000},
     'levels': {10, 30, 50, 70},
     },
    {'settings': dict(freq_lb=5, freq_ub=16, freq_step=1, level_lb=20, level_ub=80, level_step=20),
     'frequencies': {4000, 8000, 16000},
     'levels': {20, 40, 60, 80},
     },
    {'settings': dict(freq_lb=5, freq_ub=16, freq_step=0.5, level_lb=20, level_ub=80, level_step=20),
     'frequencies': {5656, 8000, 11313, 16000},
     'levels': {20, 40, 60, 80},
     },
]


def make_selector(settings):
    level_param = Parameter(name='level')
    freq_param = Parameter(name='freq')
    s = FreqLevelSelector(level_name='level', freq_name='freq', **settings)
    s.append_item(level_param)
    s.append_item(freq_param)
    return s


def get_sequence(selector):
    sequence = []
    for setting in selector.get_iterator():
        sequence.append(tuple(setting.values()))
    return sequence


def test_freq_level_selector():
    for test in TEST_VALUES:
        selector = make_selector(test['settings'])
        sequence = get_sequence(selector)
        for s in sequence:
            log.warning('... %r', s)
        assert len(sequence) == len(set(sequence))
        assert set(s[0] for s in sequence) == test['frequencies']
        assert set(s[1] for s in sequence) == test['levels']
