'''
Engine whose inputs are driven from the computer keyboard.

This lets a behavior paradigm run on any computer, with keys standing in for
the sensors (e.g., hold Space to nose-poke). The inputs are sampled streams,
like the hardware they stand in for, so everything downstream (thresholds,
edge detection, plots and stored data) runs exactly as it does on a rig.

Each `KeyboardAIChannel` reads back what the engine's `KeyboardAOChannel`
outputs play, as an IR sensor reads back its emitter. While the channel's key
is held (the beam is broken), the signal drops to `held_level`. Without an
active output, the inputs read 1 while the key is up.

A `LoopbackAIChannel` reads back the samples written to an output channel on
the clock engine (e.g., the speaker on a sound card), or to a
`KeyboardDOChannel` on this engine, as a digital loopback. Given several
outputs, it reads back their sum (e.g., both speakers of a starship, as its
microphone would hear them).

A `KeyboardDOChannel` is a hardware-timed digital output (e.g., a TTL sync
pulse scheduled by a `TimedTrigger`). Its samples are generated with the
inputs, so a loopback of it records the pulse where a rig would record it.

The engine generates samples to keep pace with `clock` (typically the sound
card playing the stimuli), and key presses are timestamped on that clock, so
edges line up with the stimuli. If `clock` is not in use (e.g., a paradigm
that only monitors a loopback plays nothing on the sound card), the engine
keeps its own time and becomes the master engine (see
`IOManager.configure_engines`). Software-timed digital outputs (e.g., a
pellet dispenser) are accepted and logged.

Example IO manifest::

    SoundcardEngine: sound:
        name = 'sound'
        master_clock = True
        ...

    KeyboardEngine:
        name = 'keyboard'
        clock = sound
        KeyboardAOChannel:
            name = 'ir_emitter'
        KeyboardAIChannel:
            name = 'np_contact'
            key = 'Space'
'''
import logging
log = logging.getLogger(__name__)

from collections import deque
import sys
from threading import Event, Thread, Timer
import time

from atom.api import Bool, Dict, Float, List, Property, Str, Typed, Value
from enaml.application import Application, deferred_call
from enaml.core.api import d_
import numpy as np

from psiaudio.pipeline import PipelineData

from ..channel import HardwareAIChannel, HardwareAOChannel, HardwareDOChannel
from ..engine import Engine
from .callback import BaseCallbackMixin


class KeyboardAIChannel(HardwareAIChannel):
    '''
    Input that reads back the engine's outputs, dropping to `held_level`
    while `key` is held.
    '''
    #: Qt name of the key, without the `Key_` prefix (e.g., 'Space', 'Left',
    #: 'J'). Several channels can share a key.
    key = d_(Str()).tag(metadata=True)

    #: Fraction of the signal read back while the key is held.
    held_level = d_(Float(0.1)).tag(metadata=True)

    fs = Property().tag(metadata=True)

    def _get_fs(self):
        return self.engine.fs


class KeyboardAOChannel(HardwareAOChannel):
    '''
    Output that is read back by the engine's `KeyboardAIChannel` inputs
    (e.g., an IR emitter).
    '''
    fs = Property().tag(metadata=True)

    def _get_fs(self):
        return self.engine.fs


class KeyboardDOChannel(HardwareDOChannel):
    '''
    Hardware-timed digital output (e.g., a TTL sync pulse). Its samples are
    generated at the engine's rate, so a `LoopbackAIChannel` can read it back.
    '''
    fs = Property().tag(metadata=True)

    def _get_fs(self):
        return self.engine.fs


class LoopbackAIChannel(HardwareAIChannel):
    '''
    Input that reads back the samples written to `source_name` (or the sum of
    those written to `source_names`), output channels on the engine's clock
    engine or on this engine.

    Outputs that are not in use contribute silence. If none is in use, the
    input reads silence, as a rig's input would.
    '''
    #: Name of the output channel to read back (e.g., 'speaker_1').
    source_name = d_(Str()).tag(metadata=True)

    #: Names of the output channels to read back the sum of (e.g., both
    #: speakers of a starship). Overrides `source_name`. The outputs must
    #: share a sampling rate.
    source_names = d_(List(Str())).tag(metadata=True)

    fs = Property().tag(metadata=True)

    def _get_fs(self):
        return self.source.fs

    def sync_start(self, channel):
        # Sample 0 read back is sample 0 written by the clock engine (or by
        # this engine), so the loopback is already in sync with outputs on
        # either.
        if channel.engine not in (self.engine, self.engine.clock):
            raise ValueError(f'{self.name} can only be synced with outputs '
                             f'on {self.engine.clock.name} or '
                             f'{self.engine.name}.')

    @property
    def source(self):
        '''
        The first output read back (sets the sampling rate).
        '''
        return self.sources[0]

    @property
    def sources(self):
        names = self.source_names or [self.source_name]
        engines = [self.engine.clock, self.engine]
        channels = {}
        for engine in engines:
            for channel in engine.get_channels(active=False):
                channels.setdefault(channel.name, channel)
        sources = []
        for name in names:
            if name not in channels:
                engine_names = ', '.join(sorted({e.name for e in engines}))
                raise AttributeError(f'{name} channel does not exist on '
                                     f'{engine_names}.')
            sources.append(channels[name])
        if len({s.fs for s in sources}) > 1:
            raise ValueError(f'Outputs read back by {self.name} must share a '
                             'sampling rate.')
        return sources


class KeyboardEngine(BaseCallbackMixin, Engine):

    #: Sampling rate of the keyboard inputs and outputs.
    fs = d_(Float(8000)).tag(metadata=True)

    #: Engine whose clock the inputs keep pace with. If None, the engine
    #: keeps its own clock.
    clock = d_(Value())

    #: Whether `get_ts` reads `clock` (set by `configure`): only if that
    #: engine is in use, since otherwise it is never started.
    _follow_clock = Bool(False)

    #: How often (in seconds) new input samples are generated.
    poll_interval = d_(Float(0.01)).tag(metadata=True)

    #: Current state of each software digital output, by channel name.
    sw_do_state = Typed(dict, ())

    #: Key presses and releases not yet in the inputs, as (timestamp, channel
    #: name, held).
    _events = Typed(deque, ())

    #: Whether each key is held, as of the last press or release.
    _pressed = Dict()

    #: Whether each key is held, as of the last sample generated.
    _held = Dict()

    #: Callbacks for each input channel, by channel name.
    _ai_callbacks = Dict()

    #: Samples generated so far (at `fs`).
    _samples = Value(0)

    #: For each loopback channel, the samples written to each of its sources
    #: in use not yet read back, as (offset, samples) by source name, and the
    #: samples read back so far.
    _loopback_queue = Dict()
    _loopback_samples = Dict()
    _monitors = List()

    _t0 = Value()
    _thread = Typed(Thread)
    _stop_requested = Typed(Event)
    _key_filter = Value()

    def _default_clock(self):
        return self

    ############################################################################
    # Engine interface
    ############################################################################
    def configure(self, active=True):
        log.info('Configuring keyboard engine %s', self.name)
        self._follow_clock = self.clock is not self and self.clock.in_use()
        if not self._follow_clock and self.clock is not self:
            log.info('Clock engine %s is not in use. Keeping own time.',
                     self.clock.name)
        self._samples = 0
        self._events = deque()
        self._held = {}
        self._pressed = {}
        for channel in self.key_channels(active):
            self._held[channel.name] = False
            self._pressed[channel.name] = False

        self._loopback_queue = {}
        self._loopback_samples = {}
        self._monitors = []
        for channel in self.loopback_channels(active):
            queues = self._loopback_queue[channel.name] = {}
            self._loopback_samples[channel.name] = 0
            # Outputs not in use are never written, so would hold up the
            # others. They contribute silence instead.
            for source in channel.sources:
                if not source.active:
                    continue
                queue = queues[source.name] = deque()
                monitor = self._make_monitor(queue)
                source.monitors.append(monitor)
                self._monitors.append((source, monitor))

        super().configure()
        if Application.instance() is not None:
            deferred_call(self._install_key_filter)
        else:
            log.warning('No application. Keys will not be read.')

    def start(self):
        self._t0 = time.perf_counter()
        self._stop_requested = Event()
        self._thread = Thread(target=self._run, daemon=True,
                              name=f'{self.name}-keyboard')
        self._thread.start()

    def stop(self):
        if self._stop_requested is None or self._stop_requested.is_set():
            return
        log.info('Stopping keyboard engine %s', self.name)
        # The thread exits once its current block is through the inputs. It
        # is not joined: an input's callback can wait on the thread stopping
        # the engine (e.g., a DPOAE's `dpoae_acquired` action waits on the
        # control dispatcher), so joining would deadlock. The sound card's
        # input thread is not joined either.
        self._stop_requested.set()
        self._t0 = None
        for source, monitor in self._monitors:
            source.monitors.remove(monitor)
        self._monitors = []
        if Application.instance() is not None:
            deferred_call(self._remove_key_filter)
        self.complete()

    def get_ts(self):
        if self._follow_clock:
            return self.clock.get_ts()
        if self._t0 is None:
            return 0.0
        return time.perf_counter() - self._t0

    def register_ai_callback(self, callback, channel_name=None):
        self._ai_callbacks.setdefault(channel_name, []).append(callback)

    def unregister_ai_callback(self, callback, channel_name=None):
        try:
            self._ai_callbacks[channel_name].remove(callback)
        except (KeyError, ValueError):
            log.warning('Callback no longer exists.')

    def register_done_callback(self, callback):
        self._callbacks.setdefault('done', []).append(callback)

    def set_sw_do(self, name, state):
        self.sw_do_state[name] = state
        log.info('Digital output %s set %s', name, 'high' if state else 'low')

    def fire_sw_do(self, name, duration=0.1):
        log.info('Digital output %s fired for %.3f s', name, duration)
        self.set_sw_do(name, 1)
        Timer(duration, lambda: self.set_sw_do(name, 0)).start()

    ############################################################################
    # Keys
    ############################################################################
    def key_channels(self, active=True):
        return [c for c in self.get_channels('analog', 'input', active=active)
                if isinstance(c, KeyboardAIChannel)]

    def loopback_channels(self, active=True):
        return [c for c in self.get_channels('analog', 'input', active=active)
                if isinstance(c, LoopbackAIChannel)]

    def set_key(self, channel_name, held, ts=None):
        '''
        Record that the key for `channel_name` was pressed (`held` is True)
        or released at `ts` (now if None).
        '''
        with self.lock:
            if channel_name not in self._pressed:
                return
            if self._pressed[channel_name] == held:
                return
            if ts is None:
                ts = self.get_ts()
            self._pressed[channel_name] = held
            self._events.append((ts, channel_name, held))
        log.debug('Key for %s %s at %f', channel_name,
                  'pressed' if held else 'released', ts)

    def release_keys(self):
        '''
        Release every key that is held (e.g., the window lost focus, so the
        release would never be seen).
        '''
        for name, held in list(self._pressed.items()):
            if held:
                self.set_key(name, False)

    def _install_key_filter(self):
        key_map = {}
        for channel in self.key_channels():
            key_map.setdefault(qt_key(channel.key), []).append(channel.name)
        self._key_filter = make_key_filter(self, key_map)
        from enaml.qt.QtWidgets import QApplication
        QApplication.instance().installEventFilter(self._key_filter)

    def _remove_key_filter(self):
        if self._key_filter is None:
            return
        from enaml.qt.QtWidgets import QApplication
        QApplication.instance().removeEventFilter(self._key_filter)
        self._key_filter = None

    ############################################################################
    # Sample generation
    ############################################################################
    def _make_monitor(self, queue):
        def monitor(offset, out):
            queue.append((offset, out.copy()))
        return monitor

    def _run(self):
        try:
            while not self._stop_requested.wait(self.poll_interval):
                self.generate()
        except Exception:
            deferred_call(sys.excepthook, *sys.exc_info())

    def generate(self, ts=None):
        '''
        Generate the input samples up to `ts` (now if None) and pass them to
        the inputs.
        '''
        if ts is None:
            ts = self.get_ts()
        if not np.isfinite(ts):
            return
        # Generates the digital outputs too, so must come before the
        # loopbacks of them.
        self._generate_key_samples(ts)
        self._generate_loopback_samples(ts)

    def _generate_key_samples(self, ts):
        s0 = self._samples
        n = int(np.floor(ts * self.fs)) - s0
        if n <= 0:
            return
        channels = self.key_channels()

        # Generating the samples of the hardware-timed digital outputs passes
        # them to their monitors (i.e., loopbacks).
        for channel in self.get_channels('digital', 'output', 'hardware'):
            channel.get_samples(s0, n, np.zeros(n))

        emitter = np.zeros(n)
        outputs = self.get_channels('analog', 'output')
        for channel in outputs:
            channel.get_samples(s0, n, emitter)
        if not outputs:
            emitter[:] = 1

        # Find where each key changed state within this block.
        changes = {c.name: [] for c in channels}
        with self.lock:
            while self._events:
                event_ts, name, held = self._events[0]
                i = int(round(event_ts * self.fs)) - s0
                if i >= n:
                    # Belongs to the next block.
                    break
                self._events.popleft()
                if name in changes:
                    changes[name].append((max(i, 0), held))

        for channel in channels:
            level = np.empty(n)
            held = self._held[channel.name]
            i = 0
            for j, new_held in changes[channel.name]:
                level[i:j] = channel.held_level if held else 1
                i, held = j, new_held
            level[i:] = channel.held_level if held else 1
            self._held[channel.name] = held
            data = PipelineData((emitter * level)[np.newaxis], fs=self.fs,
                                s0=s0)
            for cb in self._ai_callbacks.get(channel.name, []):
                cb(data)

        self._samples = s0 + n

    def _generate_loopback_samples(self, ts):
        for channel in self.loopback_channels():
            queues = self._loopback_queue[channel.name]
            s0 = self._loopback_samples[channel.name]
            # Read back what has been played, but only as far as every source
            # in use has been written.
            s1 = int(np.floor(ts * channel.fs))
            for queue in queues.values():
                s1 = min(s1, queued_end(queue, s0))
            if s1 <= s0:
                continue
            samples = np.zeros(s1 - s0)
            for queue in queues.values():
                samples += read_queue(queue, s0, s1)
            data = PipelineData(samples[np.newaxis], fs=channel.fs, s0=s0)
            self._loopback_samples[channel.name] = s1
            for cb in self._ai_callbacks.get(channel.name, []):
                cb(data)


def queued_end(queue, s0):
    '''
    The sample after the last one written to `queue` (`s0` if nothing is
    queued).
    '''
    try:
        offset, samples = queue[-1]
    except IndexError:
        return s0
    return max(s0, offset + samples.shape[-1])


def read_queue(queue, s0, s1):
    '''
    Samples `s0` up to `s1` from `queue` of (offset, samples), dropping blocks
    once read. Samples nothing was written for are 0.
    '''
    result = np.zeros(s1 - s0)
    while queue:
        offset, samples = queue[0]
        end = offset + samples.shape[-1]
        if offset >= s1:
            break
        lb, ub = max(offset, s0), min(end, s1)
        if ub > lb:
            result[lb - s0:ub - s0] = samples[lb - offset:ub - offset]
        if end > s1:
            break
        queue.popleft()
    return result


################################################################################
# Qt
################################################################################
def qt_key(name):
    '''
    The Qt key code for a key name such as 'Space' or 'J'.
    '''
    from enaml.qt.QtCore import Qt
    attr = f'Key_{name}'
    try:
        return getattr(Qt.Key, attr)
    except AttributeError:
        raise ValueError(f'Unknown key "{name}"')


def make_key_filter(engine, key_map):
    '''
    An application-wide event filter that passes the keys in `key_map` (Qt
    key code to a channel name, or a list of them) to `engine`. A key mapped
    to several channels drives them all (e.g., one sensor that a rig makes
    available under two names).

    Keys are read whichever widget has focus, except a widget the user types
    into (e.g., a parameter field). Key repeats while a key is held are
    ignored, and every key is released when the application loses focus
    (otherwise a key released in another window would stay held).
    '''
    from enaml.qt.QtCore import QEvent, QObject
    from enaml.qt.QtWidgets import (QAbstractSpinBox, QApplication, QComboBox,
                                    QLineEdit, QPlainTextEdit, QTextEdit)

    text_widgets = (QAbstractSpinBox, QLineEdit, QPlainTextEdit, QTextEdit)

    def typing():
        widget = QApplication.focusWidget()
        if isinstance(widget, text_widgets):
            return True
        return isinstance(widget, QComboBox) and widget.isEditable()

    class KeyFilter(QObject):

        def eventFilter(self, obj, event):
            kind = event.type()
            if kind in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
                names = key_map.get(event.key())
                if names is None or typing():
                    return False
                if isinstance(names, str):
                    names = [names]
                if not event.isAutoRepeat():
                    for name in names:
                        engine.set_key(name, kind == QEvent.Type.KeyPress)
                return True
            if kind == QEvent.Type.ApplicationDeactivate:
                engine.release_keys()
            return False

    return KeyFilter()
