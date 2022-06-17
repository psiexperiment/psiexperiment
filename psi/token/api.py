import enaml

from .block import Block, ContinuousBlock, EpochBlock

with enaml.imports():
    from .primitives import (
        BandlimitedNoise, BandlimitedNoiseFactory, Chirp, ChirpFactory,
        Cos2Envelope, Cos2EnvelopeFactory, Gate, GateFactory, SAMEnvelope,
        SAMEnvelopeFactory, Silence, SilenceFactory, SquareWave,
        SquareWaveFactory, Tone, ToneFactory)
