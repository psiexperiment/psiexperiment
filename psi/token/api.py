import enaml

with enaml.imports():
    from .primitives import (BandlimitedNoise, Chirp, Cos2Envelope, Gate,
                            SAMEnvelope, Silence, SquareWave, Tone)
