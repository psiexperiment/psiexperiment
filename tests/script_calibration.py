import enaml

with enaml.imports():
    from psi.controller.calibration import tone
    from psi.application.io import pika


io_manifest = pika.IOManifest()
engine = io_manifest.find('NI_audio')

result = tone.tone_sens(engine, ao_channel_name='speaker_0',
                         ai_channel_names=['microphone_channel'],
                         frequencies=[500, 1e3, 5000], gain=-40, min_snr=None,
                         max_thd=None)
print(result)

result = tone.tone_sens(engine, ao_channel_name='speaker_1',
                         ai_channel_names=['microphone_channel'],
                         frequencies=[500, 1e3, 5000], gain=-40, min_snr=None,
                         max_thd=None)
print(result)
