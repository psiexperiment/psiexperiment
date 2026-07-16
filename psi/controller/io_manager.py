import logging
log = logging.getLogger(__name__)

import threading

from atom.api import Atom, Bool, Typed, Value
from enaml.workbench.api import Extension

from .channel import Channel, OutputMixin, InputMixin
from .engine import Engine
from .output import BaseOutput, Synchronized
from .input import Input


def get_obj(o, klass):
    objects = []
    if isinstance(o, klass):
        objects.append(o)
    for child in o.children:
        objects.extend(get_obj(child, klass))
    return objects


general_error = '''
More than one {obj_type} named "{name}"

To fix this, please review the IO manifest (i.e., hardware configuration) you
selected and verify that all {obj_type}s have unique names.
'''


class ErrorDict(dict):

    def __init__(self, obj_type, *args, **kwargs):
        self.obj_type = obj_type
        super().__init__(*args, **kwargs)


    def __setitem__(self, key, value):
        if key in self:
            mesg = general_error.format(obj_type=self.obj_type, name=key)
            raise ValueError(mesg)
        return super().__setitem__(key, value)


def find_engines(point):
    master_engine = None
    engines = ErrorDict('engine')
    for extension in point.extensions:
        for e in extension.get_children(Engine):
            # ErrorDict raises a descriptive error on duplicate names.
            engines[e.name] = e
            if e.master_clock:
                if master_engine is not None:
                    m = 'Only one engine can be defined as the master'
                    raise ValueError(m)
                master_engine = e
    engines = dict(sorted(engines.items(), key=lambda e: e[1].weight))

    # The last engine is the master by default
    if master_engine is None:
        master_engine = list(engines.values())[-1]

    return engines, master_engine


def find_channels(engines):
    channels = ErrorDict('channel')
    for e in engines.values():
        for c in e.get_channels(active=False):
            channels[c.reference] = c
    return channels


def find_outputs(channels, point):
    outputs = ErrorDict('output')
    supporting = ErrorDict('synchronized output')

    # Find all the outputs already connected to a channel
    for c in channels.values():
        if isinstance(c, OutputMixin):
            # Channel is an output. Now check the children. ErrorDict raises
            # a descriptive error on duplicate names.
            for o in c.children:
                for oi in get_obj(o, BaseOutput):
                    outputs[oi.name] = oi

    # Find unconnected outputs and inputs (these are allowed so that we can
    # split processing hierarchies across multiple manifests).
    for extension in point.extensions:
        for o in extension.get_children(BaseOutput):
            outputs[o.name] = o
        for s in extension.get_children(Synchronized):
            supporting[s.name] = s
            for o in s.outputs:
                outputs[o.name] = o

    return outputs, supporting


def find_inputs(channels, point):
    inputs = ErrorDict('input')

    # Find all the outputs already connected to a channel
    for c in channels.values():
        if isinstance(c, InputMixin):
            for i in c.children:
                for ci in get_obj(i, Input):
                    inputs[ci.name] = ci

    for extension in point.extensions:
        for i in extension.get_children(Input):
            # Recurse through input tree. Currently we assume that
            # inputs can be nested/hierarchial while outputs are not.
            for ci in get_obj(i, Input):
                inputs[ci.name] = ci

    return inputs


class IOManager(Atom):
    '''
    Owns the hardware configuration declared through the
    ``psi.controller.io`` extension point — engines, channels, outputs and
    inputs — including the wiring between them and the engine lifecycle.

    This is pure hardware bookkeeping with no knowledge of the experiment
    action system. Per the threading contract (docs/threading.md), methods
    here may hold the engine-state lock but never invoke experiment actions;
    the ControllerPlugin wrappers fire those after these methods return.
    '''

    #: Available engines, indexed by name and sorted by weight. The last one
    #: is the default master (responsible for the experiment clock).
    engines = Typed(dict, {})

    #: Available channels, indexed by reference.
    channels = Typed(dict, {})

    #: Available outputs, indexed by name.
    outputs = Typed(dict, {})

    #: Supporting classes for outputs (right now only Synchronized).
    supporting = Typed(dict, {})

    #: Available inputs, indexed by name.
    inputs = Typed(dict, {})

    #: Engine responsible for the experiment clock. May be reassigned during
    #: setup (e.g., NIDAQ start-trigger configuration in
    #: psi.controller.engines.nidaq).
    master_engine = Typed(Engine)

    #: Are engines running?
    engines_running = Bool(False)

    # Guards engine state transitions.
    _lock = Value()

    def _default__lock(self):
        return threading.Lock()

    def refresh(self, point, workbench):
        '''
        Rebuild the registries from the extension point and load the manifest
        of every discovered object.
        '''
        # TODO: Allow disabling of devices.
        log.debug('Loading IO')
        self.engines, self.master_engine = find_engines(point)
        self.channels = find_channels(self.engines)
        self.outputs, self.supporting = find_outputs(self.channels, point)
        self.inputs = find_inputs(self.channels, point)

        for c in self.channels.values():
            c.load_manifest(workbench)

        for s in self.supporting.values():
            s.load_manifest(workbench)

        for e in self.engines.values():
            e.load_manifest(workbench)

        for o in self.outputs.values():
            o.load_manifest(workbench)

        for i in self.inputs.values():
            i.load_manifest(workbench)

    def connect_outputs(self):
        to_init = list(self.outputs.values())
        n_init = len(to_init)

        while True:
            for o in to_init[:]:
                # This ensures that all blocks that need to be linked to channels
                # are connected first.
                if o.target is None and o.target_name:
                    if self.connect_output(o.name, o.target_name):
                        to_init.remove(o)
                elif o.target is None and not isinstance(o.parent, Extension):
                    o.parent.add_output(o)
                    to_init.remove(o)
                elif o.target is None:
                    log.warning('Unconnected output %s', o.name)
                    to_init.remove(o)
                else:
                    to_init.remove(o)
            if len(to_init) == 0:
                break
            elif n_init == len(to_init):
                raise ValueError(f'Unable to configure outputs {", ".join(o.name for o in to_init)}')
            else:
                n_init = len(to_init)

    def connect_inputs(self):
        for i in self.inputs.values():
            # First, make sure the input is connected to a source
            if i.source is None and i.source_name:
                self.connect_input(i.name, i.source_name)
            elif i.source is None and not isinstance(i.parent, Extension):
                i.parent.add_input(i)
            elif i.source is None:
                log.warning('Unconnected input %s', i.name)

    def connect_output(self, output_name, target_name):
        # Link up outputs with channels if needed.
        if target_name in self.channels:
            target = self.channels[target_name]
        elif target_name in self.outputs:
            target = self.outputs[target_name]
        else:
            valid_targets = list(self.channels) + list(self.outputs)
            valid_targets = ', '.join(valid_targets)
            m = "Unknown target {} specified for output {}. Valid targets are {}" \
                .format(target_name, output_name, valid_targets)
            raise ValueError(m)

        if not isinstance(target, Channel) and target.target is None:
            return False

        o = self.outputs[output_name]
        target.add_output(o)
        m = 'Connected output %s to target %s'
        log.debug(m, output_name, target_name)
        return True

    def connect_input(self, input_name, source_name):
        if source_name in self.inputs:
            source = self.inputs[source_name]
        elif source_name in self.channels:
            source = self.channels[source_name]
        else:
            m = "Unknown source {}. Cannot configure {}.".format(source_name, input_name)
            raise ValueError(m)

        i = self.inputs[input_name]
        source.add_input(i)
        m = 'Connected input %s to source %s'
        log.debug(m, input_name, source_name)

    def configure_engines(self, done_callback_factory=None):
        '''
        Configure every engine that has active channels.

        Parameters
        ----------
        done_callback_factory : callable, optional
            Called with each configured engine; must return the callback to
            register as that engine's done callback.
        '''
        with self._lock:
            for engine in self.engines.values():
                # Check to see if engine is being used
                if engine.get_channels():
                    engine.configure()
                    if done_callback_factory is not None:
                        engine.register_done_callback(
                            done_callback_factory(engine))

    def start_engines(self):
        with self._lock:
            if self.engines_running:
                raise ValueError('Engines already running')
            log.debug('Starting engines')
            for engine in self.engines.values():
                # Check to see if engine is being used
                if engine.get_channels():
                    if engine is not self.master_engine:
                        engine.start()
            self.master_engine.start()
            self.engines_running = True

    def stop_engines(self):
        with self._lock:
            if not self.engines_running:
                raise ValueError('Engines not running')
            for engine in self.engines.values():
                if engine.get_channels():
                    log.info('Stopping engine %r', engine)
                    engine.stop()
            self.engines_running = False

    def reset_engines(self):
        with self._lock:
            for engine in self.engines.values():
                engine.reset()

    def get_output(self, output_name):
        try:
            return self.outputs[output_name]
        except KeyError as e:
            outputs = ', '.join(self.outputs.keys())
            m = f'No such output "{output_name}". Valid outputs are {outputs}. ' \
                'Did you accidentally specify a channel name instead?'
            raise ValueError(m) from e

    def get_input(self, input_name):
        try:
            return self.inputs[input_name]
        except KeyError as e:
            valid_inputs = ', '.join(self.inputs)
            raise KeyError(f'{input_name}: valid inputs are {valid_inputs}') from e

    def set_input_attr(self, input_name, attr_name, value):
        setattr(self.inputs[input_name], attr_name, value)

    def get_channel(self, channel_name):
        try:
            return self.channels[channel_name]
        except KeyError as e:
            channels = ', '.join(self.channels.keys())
            m = f'No such channel "{channel_name}". Valid channels are {channels}. ' \
                'Did you accidentally specify an output name instead?'
            raise ValueError(m) from e

    def get_channels(self, mode=None, direction=None, timing=None,
                     active=True):
        '''
        Return channels matching criteria across all engines

        Parameters
        ----------
        mode : {None, 'analog', 'digital'
            Type of channel
        direction : {None, 'input, 'output'}
            Direction
        timing : {None, 'hardware', 'software'}
            Hardware or software-timed channel. Hardware-timed channels have a
            sampling frequency greater than 0.
        active : bool
            If True, return only channels that have configured inputs or
            outputs.
        '''
        channels = []
        for engine in self.engines.values():
            ec = engine.get_channels(mode, direction, timing, active)
            channels.extend(ec)
        return channels

    def get_ts(self):
        return self.master_engine.get_ts()
