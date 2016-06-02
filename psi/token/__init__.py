import enaml

with enaml.imports():
    from .discrete.tone_manifest import ToneManifest
    from .continuous.bandlimited_noise_manifest import BandlimitedNoiseManifest
    from .continuous.silence_manifest import SilenceManifest
    from .continuous.tone_manifest import ToneManifest as CToneManifest


continuous_tokens = {
    'bandlimited noise': BandlimitedNoiseManifest,
    'silence': SilenceManifest,
    'continuous tone': CToneManifest,
}


discrete_tokens = {
    'tone': ToneManifest,
}


def get_token_manifest(token_name):
    try:
        return continuous_tokens[token_name]
    except KeyError:
        return discrete_tokens[token_name]
