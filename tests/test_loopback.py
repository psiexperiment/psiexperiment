'''
The parts of psi.loopback that don't need a sound card.
'''
import numpy as np
import pytest

from psi.loopback.continuity import (check_continuity, CyclicSequence,
                                     pilot_sequence, pseudorandom_sequence)
from psi.loopback.fit import fit_stimuli
from psi.loopback.runner import log_problems


FS = 96000


def test_seek():
    # check_continuity regenerates any span of the pilot, which relies on
    # PCG64.advance skipping exactly one draw per sample.
    full = pseudorandom_sequence(8192, 2, seed=1234)
    a = pseudorandom_sequence(1000, 2, seed=1234)
    b = pseudorandom_sequence(8192 - 1000, 2, seed=1234, start=1000)
    np.testing.assert_array_equal(full, np.concatenate([a, b], axis=-1))


def test_cyclic_fill_matches_reference():
    seq = CyclicSequence(1, FS, period=2 ** 16, amplitude=0.1, highpass=100,
                         lowpass=0.4 * FS)
    out = np.zeros((1, 4096), dtype='float32')
    for s0 in (0, 4096, 2 ** 16 - 2048, 3 * 2 ** 16 + 5):
        seq.fill(s0, out)
        np.testing.assert_array_equal(out, seq.reference(s0, 4096))
    assert np.abs(seq.buffer).max() == pytest.approx(0.1)


def record(seq, n, offset=0, gain=8.9, edit=None):
    '''
    What a loopback records of `seq` over `n` samples, with `edit` applied
    to the sample numbers (to drop or repeat some).
    '''
    i = np.arange(n) + offset
    if edit is not None:
        i = edit(i)
    return gain * seq.buffer[0, i % seq.period].astype('float64')


@pytest.fixture(scope='module')
def seq():
    return pilot_sequence(FS)


def test_clean(seq):
    x = record(seq, FS * 3, offset=-37)
    c = check_continuity(x, seq)
    assert c.offset == -37
    assert c.gain == pytest.approx(8.9, rel=1e-4)
    assert c.n_bad == 0
    assert c.slips == []
    assert c.describe() == 'none'


@pytest.mark.parametrize('jump', [8192, -8192, 1])
def test_slip(seq, jump):
    # A dropped (jump > 0) or repeated (jump < 0) buffer 1.5 s in.
    at = int(1.5 * FS) + 100
    x = record(seq, FS * 3, edit=lambda i: np.where(i >= at, i + jump, i))
    c = check_continuity(x, seq)
    assert 1 <= c.n_bad <= 2
    assert c.bad_times[0] == pytest.approx(at / FS, abs=1024 / FS)
    assert [s.samples for s in c.slips] == [jump]


def test_zeroed(seq):
    x = record(seq, FS * 3)
    x[FS:FS + 8192] = 0
    c = check_continuity(x, seq)
    # Every block touching the zeros, and no slip afterwards.
    assert c.n_bad == 9
    assert [s.samples for s in c.slips if s.samples is not None] == []


def test_missing_pilot(seq):
    with pytest.raises(ValueError):
        check_continuity(np.zeros(FS), seq)


def test_fit_stimuli():
    rng = np.random.default_rng(0)
    template = rng.normal(size=200)
    x = np.zeros(FS)
    onsets = [0.1, 0.4, 0.7]
    for t in onsets:
        i = int(t * FS) + 50
        x[i:i + 200] += 2 * template
    # A silent stimulus is reported but not fit.
    fit = fit_stimuli(x, FS, onsets + [0.9],
                      [template] * 3 + [np.zeros(200)])
    assert fit.latency == 50
    assert list(fit.epochs.lag[:3]) == [0, 0, 0]
    np.testing.assert_allclose(fit.epochs.gain[:3], 2)
    assert np.isnan(fit.epochs.gain[3])
    assert fit.unexplained_db < -100


def test_log_problems(tmp_path):
    log = tmp_path / 'psi.log'
    log.write_text(
        '2026-10-27 13:09:16,767 INFO      : MainThread  - a :: fine\n'
        '2026-10-27 13:09:16,768 WARNING   : MainThread  - a :: careful\n'
        '2026-10-27 13:09:16,769 ERROR     : MainThread  - b :: broke\n'
        'Traceback (most recent call last):\n'
        '  ValueError\n'
        '2026-10-27 13:09:16,770 ERROR     : MainThread  - c :: expected\n'
        '2026-10-27 13:09:16,771 DEBUG     : MainThread  - a :: after\n'
    )
    problems = log_problems(log, ignore=['expected'])
    assert len(problems) == 1
    assert problems[0].endswith('broke\nTraceback (most recent call last):'
                                '\n  ValueError')
    assert len(log_problems(log, level='WARNING')) == 3
