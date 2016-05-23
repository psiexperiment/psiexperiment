import enaml

with enaml.imports():
    from .base_manifest import TokenManifest
    from .tone_manifest import ToneManifest
    from .bandlimited_noise_manifest import BandlimitedNoiseManifest


available_tokens = {
    'tone': ToneManifest,
    'bandlimited noise': BandlimitedNoiseManifest,
}
