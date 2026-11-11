'''
The parts of psi.loopback that don't need a sound card.
'''
import numpy as np
import pytest

from psi.loopback.continuity import (check_continuity, CyclicSequence,
                                     pilot_sequence, pseudorandom_sequence)
from psi.loopback import rig as loopback_rig
from psi.loopback.fit import fit_stimuli
from psi.loopback.rig import (analyze_routing, measure_routing,
                              probe_noise, Rig, Route, Routing)
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


################################################################################
# Routing probe
################################################################################
PROBE_FS = 10000

#: in_0 hears a (x0.9) and b (x0.3) 37 samples late; in_1 hears b (x0.5) 50
#: samples late.
ROUTES = [('a', 'in_0', 0.9, 37), ('b', 'in_0', 0.3, 37),
          ('b', 'in_1', 0.5, 50)]


@pytest.fixture
def probe_rig():
    return Rig(io_manifest='', device='', outputs={'a': 0, 'b': 1},
               inputs={'in_0': 0, 'in_1': 1})


def probe_recording(rig, noise, start, routes=ROUTES):
    n = len(noise)
    recorded = np.zeros((start + n + PROBE_FS // 10, 2))
    outputs = list(rig.outputs)
    for output, input, gain, lag in routes:
        o = outputs.index(output)
        i = rig.inputs[input]
        recorded[start + lag:start + lag + n, i] += gain * noise[:, o]
    return recorded


def check_routes(routing, routes=ROUTES, tolerance_db=1e-6):
    assert len(routing.routes) == len(routes)
    for output, input, gain, lag in routes:
        r = routing.get(output, input)
        assert r is not None, (output, input)
        assert r.lag == lag
        assert abs(r.gain_db - 20 * np.log10(gain)) < tolerance_db


def test_analyze_routing(probe_rig):
    noise = probe_noise(probe_rig, PROBE_FS, 1)
    start = int(PROBE_FS * probe_rig.probe_lead_in)
    recorded = probe_recording(probe_rig, noise, start)
    routing = analyze_routing(probe_rig, noise, recorded, PROBE_FS, start)
    check_routes(routing)
    assert routing.get('a', 'in_1') is None
    assert all(r.gain_spread_db < 1e-6 for r in routing.routes)


@pytest.mark.parametrize('lost', ['zeroed', 'dropped'])
def test_analyze_routing_lost_noise(probe_rig, lost):
    '''
    A stretch of the noise that never came back (7%, as when a route once
    read 0.6 dB low) doesn't change the gain, but shows in its spread.
    '''
    noise = probe_noise(probe_rig, PROBE_FS, 1)
    start = int(PROBE_FS * probe_rig.probe_lead_in)
    recorded = probe_recording(probe_rig, noise, start)
    lb, n_lost = start + 2000, int(0.07 * PROBE_FS)
    if lost == 'zeroed':
        recorded[lb:lb + n_lost] = 0
        routes = ROUTES
    else:
        # The rest of the recording comes back that much earlier, so it is
        # the larger part that sets the delay.
        recorded = np.delete(recorded, np.s_[lb:lb + n_lost], axis=0)
        routes = [(o, i, g, lag - n_lost) for o, i, g, lag in ROUTES]
    routing = analyze_routing(probe_rig, noise, recorded, PROBE_FS, start)
    check_routes(routing, routes)
    assert routing.get('a', 'in_0').gain_spread_db > 1


def routing_with_gain(gain, lag=37):
    routing = Routing(fs=PROBE_FS, output_names=['a'], input_names=['in_0'],
                      threshold=0.05)
    routing.routes.append(Route('a', 'in_0', gain, lag))
    return routing


@pytest.mark.parametrize('other, agrees', [
    (routing_with_gain(0.9), True),
    (routing_with_gain(0.9 * 10 ** (0.04 / 20)), True),
    (routing_with_gain(0.9 * 10 ** (0.06 / 20)), False),
    (routing_with_gain(-0.9), False),
    (routing_with_gain(0.9, lag=38), False),
    (Routing(fs=PROBE_FS, output_names=['a'], input_names=['in_0'],
             threshold=0.05), False),
])
def test_routing_agrees(other, agrees):
    assert routing_with_gain(0.9).agrees(other, 0.05) == agrees


def test_measure_routing_repeats_until_two_agree(probe_rig, monkeypatch):
    probes = iter([routing_with_gain(0.84), routing_with_gain(0.9),
                   routing_with_gain(0.9), routing_with_gain(0.7)])
    monkeypatch.setattr(loopback_rig, 'probe_routing',
                        lambda *args, **kwargs: next(probes))
    routing = measure_routing(probe_rig)
    assert routing.get('a', 'in_0').gain == 0.9
    assert next(probes).get('a', 'in_0').gain == 0.7


def test_measure_routing_gives_up(probe_rig, monkeypatch):
    gains = iter([0.5, 0.6, 0.7, 0.8])
    monkeypatch.setattr(loopback_rig, 'probe_routing',
                        lambda *args, **kwargs: routing_with_gain(next(gains)))
    with pytest.raises(RuntimeError, match='did not agree'):
        measure_routing(probe_rig)
