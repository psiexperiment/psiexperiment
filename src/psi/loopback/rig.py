'''
The test rig: which IO manifest the paradigms run on, which sound card
channel is which, and a probe that measures how the sound card currently
routes outputs back to inputs.

Which output reaches which input is decided in the sound card's mixer (e.g.,
TotalMix), not in the IO manifest, so before any paradigm runs the probe
plays independent noise on every output and finds where each one comes back,
at what gain and with what delay. Tests then check that the routes they need
exist, and use the measured gain as the reference for level checks.
'''
from dataclasses import dataclass, field

import numpy as np
from scipy import signal


@dataclass
class Rig:
    #: IO manifest the paradigms run on, as passed to `psi --io`.
    io_manifest: str
    #: Sound card, as sounddevice names it.
    device: str
    #: Output and input names (as in the IO manifest) to channel numbers
    #: (0-based).
    outputs: dict
    inputs: dict
    #: Full-scale input level of the manifest's inputs (`Input.dBFS`). psi
    #: scales the samples it reads by this, while outputs are written
    #: unscaled, so a digital loopback at unity reads back this much louder
    #: than it was played.
    input_dbfs: float = 0
    #: The sound card engine's sampling rate and blocksize in the IO
    #: manifest.
    engine_fs: float = 96000
    engine_blocksize: int = 8192
    #: Probe signal level (peak, re full scale). Low enough to be harmless if
    #: something is plugged into an output.
    probe_amplitude: float = 0.01
    #: Gain below which an output is treated as not reaching an input.
    route_threshold: float = 0.05
    #: Longest round-trip delay through the sound card to allow for, in
    #: seconds.
    max_latency: float = 0.5


@dataclass
class Route:
    output: str
    input: str
    #: Linear gain from the digital value written to the value read.
    gain: float
    #: Delay through the sound card, in samples.
    lag: int

    @property
    def gain_db(self):
        return 20 * np.log10(abs(self.gain))


@dataclass
class Routing:
    fs: float
    output_names: list
    input_names: list
    #: Gain below which an output was treated as not reaching an input.
    threshold: float
    routes: list = field(default_factory=list)
    #: gain[input, output], including routes below threshold.
    gain: np.ndarray = None

    def get(self, output, input):
        for r in self.routes:
            if r.output == output and r.input == input:
                return r
        return None

    def sources(self, input):
        return [r.output for r in self.routes if r.input == input]

    def missing(self, required):
        '''
        Routes in `required` ({input: [outputs]}) that the mixer isn't set up
        for, as readable strings.
        '''
        return [f'{o} -> {i}' for i, outputs in required.items()
                for o in outputs if self.get(o, i) is None]


def measure_routing(rig, fs=96000, duration=1):
    '''
    Play independent noise on every output of `rig` and find which inputs it
    reaches.
    '''
    # Imported first: it enables sounddevice's ASIO support, which has to
    # happen before sounddevice is.
    import psi.controller.engines.soundcard.playrec  # noqa: F401
    import sounddevice as sd

    outputs, inputs = rig.outputs, rig.inputs
    n = int(fs * duration)
    n_ch = max(max(outputs.values()), max(inputs.values())) + 1
    rng = np.random.default_rng(0)
    noise = rng.uniform(-1, 1, (n, len(outputs))).astype('float32')
    noise *= rig.probe_amplitude
    # Keep recording after the noise ends so all of it comes back despite the
    # round-trip delay; otherwise the gain reads low by the fraction missed.
    played = np.zeros((n + int(fs * rig.max_latency), n_ch), 'float32')
    for i, ch in enumerate(outputs.values()):
        played[:n, ch] = noise[:, i]
    mapping = list(range(1, n_ch + 1))
    recorded = sd.playrec(played, fs, input_mapping=mapping,
                          output_mapping=mapping, device=rig.device,
                          blocking=True)

    routing = Routing(fs=fs, output_names=list(outputs),
                      input_names=list(inputs), threshold=rig.route_threshold)
    routing.gain = np.zeros((len(inputs), len(outputs)))
    out_names = list(outputs)
    for i, (in_name, in_ch) in enumerate(inputs.items()):
        x = recorded[:, in_ch]
        lags = {}
        for o in range(len(out_names)):
            c = signal.correlate(x, noise[:, o], 'full', 'fft')
            peak = np.argmax(np.abs(c))
            routing.gain[i, o] = c[peak] / np.sum(noise[:, o] ** 2)
            if abs(routing.gain[i, o]) > rig.route_threshold:
                lags[o] = peak - (n - 1)
        if not lags:
            continue
        # Where several outputs sum into one input, each one's correlation
        # also picks up the others by chance, so fit them jointly for the
        # gains.
        sources = np.zeros((len(x), len(lags)))
        for j, (o, lag) in enumerate(lags.items()):
            sources[lag:lag + n, j] = noise[:, o]
        gains = np.linalg.lstsq(sources, x, rcond=None)[0]
        for (o, lag), gain in zip(lags.items(), gains):
            routing.gain[i, o] = gain
            routing.routes.append(Route(out_names[o], in_name, gain, lag))
    return routing


def measure_latency(rig, output, input, fs=None, blocksize=None, duration=1):
    '''
    Samples from when psi writes a sample to `output` to when it reads it
    back on `input`, which is what the engine's `latency_samples` has to be.

    This goes through psi's own `PlayRec` rather than `sd.playrec`, since the
    delay depends on how the stream is opened and buffered (sampling rate,
    blocksize, and PlayRec dropping the first input block), and on the ASIO
    buffer size set in the sound card's settings.
    '''
    import time
    from psi.controller.engines.soundcard.playrec import PlayRec

    fs = rig.engine_fs if fs is None else fs
    blocksize = rig.engine_blocksize if blocksize is None else blocksize
    rng = np.random.default_rng(0)
    probe = rng.uniform(-1, 1, int(fs * 0.05)) * rig.probe_amplitude
    recorded = []

    def ao_cb(s0, samples, buffer):
        if s0 < len(probe):
            chunk = probe[s0:s0 + samples]
            buffer[0, :len(chunk)] = chunk

    def ai_cb(s0, samples, data):
        recorded.append(data[0])

    stream = PlayRec(fs, rig.device, [rig.inputs[input]],
                     [rig.outputs[output]], ai_cb, ao_cb, blocksize)
    stream.start()
    time.sleep(duration + rig.max_latency)
    stream.stop()

    x = np.concatenate(recorded)
    c = signal.correlate(x, probe, 'full', 'fft')
    peak = np.argmax(np.abs(c))
    gain = c[peak] / np.sum(probe ** 2)
    if abs(gain) < rig.route_threshold:
        raise ValueError(f'{output} does not reach {input}')
    return int(peak - (len(probe) - 1))


def measure_run_latency(rig, output, input):
    '''
    Latency at the engine's sampling rate and blocksize. The first stream
    after the sound card changes rate can come back thousands of samples off
    from the rest, so measure until two agree.
    '''
    previous = None
    for _ in range(4):
        latency = measure_latency(rig, output, input)
        if latency == previous:
            return latency
        previous = latency
    raise RuntimeError(f'Latency from {output} to {input} is not stable')
