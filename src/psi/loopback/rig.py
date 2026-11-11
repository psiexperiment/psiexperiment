'''
The test rig: which IO manifest the paradigms run on, which sound card
channel is which, and a probe that measures how the sound card currently
routes outputs back to inputs.

Which output reaches which input is decided in the sound card's mixer (e.g.,
TotalMix), not in the IO manifest, so before any paradigm runs the probe
plays independent noise on every output and finds where each one comes back,
at what gain and with what delay. Tests then check that the routes they need
exist, and use the measured gain as the reference for level checks.

Because the gain is the reference for every level check, the probe guards
against a stream that loses part of the noise (e.g., while it starts up),
which once read a route 0.6 dB low and failed every level check of the run
that followed: the noise starts after a lead-in of silence, each route's gain
is the median of its gains over short blocks, and the probe is repeated until
two measurements agree.
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
    #: Silence before the probe noise (s), so anything lost while the stream
    #: starts is silence rather than noise.
    probe_lead_in: float = 0.25
    #: Blocks (s) the gain of each route is measured in; the route's gain is
    #: the median over them.
    probe_block: float = 0.05
    #: How closely (dB) two probes must agree on every route's gain.
    probe_tolerance_db: float = 0.05
    #: Probes to run before giving up on two agreeing.
    probe_attempts: int = 4


@dataclass
class Route:
    output: str
    input: str
    #: Linear gain from the digital value written to the value read.
    gain: float
    #: Delay through the sound card, in samples.
    lag: int
    #: Largest difference (dB) of a block's gain from `gain`. Near 0 on a
    #: clean digital route; large if part of the noise went missing.
    gain_spread_db: float = 0.0

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

    def agrees(self, other, tolerance_db):
        '''
        True if `other` found the same routes, with the same delays and gains
        within `tolerance_db`.
        '''
        if {(r.output, r.input) for r in self.routes} != \
                {(r.output, r.input) for r in other.routes}:
            return False
        for r in self.routes:
            o = other.get(r.output, r.input)
            if o.lag != r.lag or np.sign(o.gain) != np.sign(r.gain) \
                    or abs(o.gain_db - r.gain_db) > tolerance_db:
                return False
        return True


def measure_routing(rig, fs=96000, duration=1):
    '''
    Play independent noise on every output of `rig` and find which inputs it
    reaches, probing until two measurements agree (see `Rig.probe_*`).
    '''
    previous = None
    for _ in range(rig.probe_attempts):
        routing = probe_routing(rig, fs, duration)
        if previous is not None \
                and routing.agrees(previous, rig.probe_tolerance_db):
            return routing
        previous = routing
    raise RuntimeError(f'{rig.probe_attempts} measurements of the routing did '
                       f'not agree to {rig.probe_tolerance_db} dB')


def probe_routing(rig, fs=96000, duration=1):
    '''
    Play the probe noise on every output of `rig` once, and analyze what came
    back (see `analyze_routing`).
    '''
    # Imported first: it enables sounddevice's ASIO support, which has to
    # happen before sounddevice is.
    import psi.controller.engines.soundcard.playrec  # noqa: F401
    import sounddevice as sd

    outputs, inputs = rig.outputs, rig.inputs
    noise = probe_noise(rig, fs, duration)
    n = len(noise)
    start = int(fs * rig.probe_lead_in)
    n_ch = max(max(outputs.values()), max(inputs.values())) + 1
    # Keep recording after the noise ends so all of it comes back despite the
    # round-trip delay.
    played = np.zeros((start + n + int(fs * rig.max_latency), n_ch),
                      'float32')
    for i, ch in enumerate(outputs.values()):
        played[start:start + n, ch] = noise[:, i]
    mapping = list(range(1, n_ch + 1))
    recorded = sd.playrec(played, fs, input_mapping=mapping,
                          output_mapping=mapping, device=rig.device,
                          blocking=True)
    return analyze_routing(rig, noise, recorded, fs, start)


def probe_noise(rig, fs, duration):
    '''
    Independent noise for each output of `rig`, (samples, outputs).
    '''
    rng = np.random.default_rng(0)
    noise = rng.uniform(-1, 1, (int(fs * duration), len(rig.outputs)))
    return (noise * rig.probe_amplitude).astype('float32')


def analyze_routing(rig, noise, recorded, fs, start=0):
    '''
    Find where the probe noise came back.

    Parameters
    ----------
    rig : Rig
    noise : array (samples, outputs)
        The noise each output of `rig` played.
    recorded : array (samples, channels)
        What every channel of the sound card recorded.
    start : int
        Sample the noise started at.
    '''
    inputs = rig.inputs
    n = len(noise)
    block = max(int(fs * rig.probe_block), 1)
    routing = Routing(fs=fs, output_names=list(rig.outputs),
                      input_names=list(inputs), threshold=rig.route_threshold)
    routing.gain = np.zeros((len(inputs), len(rig.outputs)))
    out_names = list(rig.outputs)
    for i, (in_name, in_ch) in enumerate(inputs.items()):
        x = np.asarray(recorded[:, in_ch], dtype=float)
        positions = {}
        for o in range(len(out_names)):
            c = signal.correlate(x, noise[:, o], 'full', 'fft')
            peak = np.argmax(np.abs(c))
            routing.gain[i, o] = c[peak] / np.sum(noise[:, o] ** 2)
            if abs(routing.gain[i, o]) > rig.route_threshold:
                positions[o] = peak - (n - 1)
        if not positions:
            continue
        # Where several outputs sum into one input, each one's correlation
        # also picks up the others by chance, so fit them jointly. The fit is
        # made block by block over where every source is playing, and each
        # gain is the median over blocks, so a stretch of the recording that
        # lost the noise (or shifted it) doesn't pull the gain down.
        sources = np.zeros((len(x), len(positions)))
        for j, (o, p) in enumerate(positions.items()):
            lb, ub = max(p, 0), min(p + n, len(x))
            sources[lb:ub, j] = noise[lb - p:ub - p, o]
        lb = max(max(positions.values()), 0)
        ub = min(min(positions.values()) + n, len(x))
        edges = range(lb, ub - block + 1, block)
        if len(edges):
            fits = np.array([
                np.linalg.lstsq(sources[e:e + block], x[e:e + block],
                                rcond=None)[0]
                for e in edges])
            gains = np.median(fits, axis=0)
        else:
            fits = gains = np.linalg.lstsq(sources, x, rcond=None)[0][None]
            gains = gains[0]
        with np.errstate(divide='ignore'):
            spread = np.max(np.abs(20 * np.log10(np.abs(fits / gains))),
                            axis=0)
        for j, (o, p) in enumerate(positions.items()):
            routing.gain[i, o] = gains[j]
            routing.routes.append(Route(out_names[o], in_name, gains[j],
                                        p - start, spread[j]))
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
