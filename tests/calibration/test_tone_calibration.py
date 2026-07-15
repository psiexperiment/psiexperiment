"""Tests for the tone calibration math (tone_power / tone_spl / tone_sens).

The hardware acquisition step is replaced with a synthetic recording
containing tones of known amplitude, so these tests verify the analysis
chain: RMS extraction, SPL conversion via the input calibration, and the
normalized-SPL/sensitivity computation.
"""
import numpy as np
import pandas as pd
import pytest

from psiaudio.util import db

from psi.controller.calibration import tone
from psi.controller.calibration.acquire import AcquireResult


FS = 100e3
DURATION = 0.1
FREQUENCIES = [500, 1000]
GAIN = -40
AMPLITUDE = 0.1  # Peak amplitude of the synthetic "measured" tone.
RMS = AMPLITUDE / np.sqrt(2)


def _make_recording(ai_channel, repetitions=2):
    n = int(FS * DURATION)
    t = np.arange(n) / FS
    rows = []
    index_rows = []
    for f in FREQUENCIES:
        for rep in range(repetitions):
            rows.append(AMPLITUDE * np.sin(2 * np.pi * f * t))
            index_rows.append((GAIN, f, rep, 0.0))
    # Near-zero noise for the silence epochs used in the SNR calculation.
    rng = np.random.RandomState(0)
    for rep in range(repetitions):
        rows.append(rng.normal(scale=1e-6, size=n))
        index_rows.append((-400, 0, rep, 0.0))
    index = pd.MultiIndex.from_tuples(
        index_rows, names=['gain', 'frequency', 'repeat', 't0'])
    df = pd.DataFrame(np.vstack(rows), index=index,
                      columns=pd.Index(t, name='time'))
    result = AcquireResult()
    result[ai_channel] = df
    return result


@pytest.fixture
def recording(monkeypatch, ai_channel):
    recording = _make_recording(ai_channel)
    monkeypatch.setattr('psi.controller.calibration.acquire.acquire',
                        lambda *args, **kw: recording)
    return recording


def test_tone_power(engine, ai_channel, recording):
    result = tone.tone_power(engine, FREQUENCIES, 'ao', ['ai'])
    assert set(result.index) == {('ai', f) for f in FREQUENCIES}
    for f in FREQUENCIES:
        row = result.loc[('ai', f)]
        # The measured RMS must match the synthetic tone's RMS.
        assert row['rms'] == pytest.approx(RMS, rel=1e-3)
        # Pure tone against near-zero noise: enormous SNR, negligible THD.
        assert row['snr'] > 60
        assert row['thd'] == pytest.approx(0, abs=1e-3)


def test_tone_spl(engine, ai_channel, recording):
    result = tone.tone_spl(engine, FREQUENCIES, 'ao', ['ai'])
    for f in FREQUENCIES:
        row = result.loc[('ai', f)]
        # The ai_channel fixture uses FlatCalibration.as_attenuation()
        # (sensitivity 0), so SPL is just dB re 1 VRMS.
        assert row['spl'] == pytest.approx(db(RMS), rel=1e-3)


def test_tone_sens(engine, ai_channel, recording):
    result = tone.tone_sens(engine, FREQUENCIES, ao_channel_name='ao',
                            ai_channel_names=['ai'], gains=GAIN)
    for f in FREQUENCIES:
        row = result.loc[('ai', f)]
        spl = db(RMS)
        # Sensitivity is the SPL normalized to a 1 VRMS, 0 dB gain output:
        # norm_spl = spl - gain - db(vrms).
        expected = spl - GAIN - db(1)
        assert row['norm_spl'] == pytest.approx(expected, rel=1e-3)
        assert row['sens'] == pytest.approx(expected, rel=1e-3)
        assert row['vrms'] == 1
