import logging
log = logging.getLogger(__name__)

import builtins
import difflib
from functools import partial

from atom.api import Enum, Bool, Typed, Property
from enaml.application import deferred_call
from enaml.workbench.plugin import Plugin

from .dispatcher import ControlDispatcher
from .io_manager import IOManager

from psi.core.exceptions import ActionError
from psi.core.experiment_action import (EventLogger, ExperimentAction,
                                ExperimentActionBase, ExperimentCallback,
                                ExperimentEvent, ExperimentState)


IO_POINT = 'psi.controller.io'
ACTION_POINT = 'psi.controller.actions'
WRAPUP_POINT = 'psi.controller.wrapup'


def invoke_action(core, action, event_name, timestamp, kw, skip_errors=False):
    try:
        if kw is None: kw = {}
        log.debug('Invoking action %s', action)
        return action.invoke(core, timestamp=timestamp, event=event_name, **kw)
    except Exception as e:
        m = f'When invoking {action} in response to the {event_name} ' \
            f'event, the following error was received:\n\n{e}.'
        log.error(m)
        if not skip_errors:
            raise ActionError(m, action=action, event=event_name) from e


class ControllerPlugin(Plugin):

    # Tracks the state of the controller.
    experiment_state = Enum('initialized', 'running', 'paused', 'stopped')

    # Provides direct access to plugins rather than going through the core
    # command system. Right now the context plugin is so fundamentally important
    # to the controller that it would be cumbersome to use the core command
    # system.
    core = Typed(Plugin)
    context = Typed(Plugin)
    data = Typed(Plugin)

    # Hardware configuration (engines, channels, outputs, inputs) and engine
    # lifecycle. See psi.controller.io_manager.IOManager. The get_channel/
    # get_output/get_input/get_channels/get_ts methods below delegate here.
    io = Typed(IOManager, ())

    # We should not respond to changes during the course of a trial. These
    # flags indicate changes or requests from the user are pending and should
    # be processed when the opportunity arises (e.g., at the end of the trial).
    _apply_requested = Bool(False)
    _pause_requested = Bool(False)
    _resume_requested = Bool(False)

    # List of events and actions that can be associated with the event
    _events = Typed(dict, {})
    _states = Typed(dict, {})
    _event_loggers = Typed(list, [])

    # Owned exclusively by the control dispatcher thread. Never read or
    # mutate from other threads; route through invoke_actions instead.
    _action_context = Typed(dict, {})

    # Single-owner thread for the experiment control plane. All action
    # matching/invocation and delayed events execute here, serialized. See
    # docs/threading.md.
    _dispatcher = Typed(ControlDispatcher, ())

    # Plugin actions are automatically registered when the manifests are
    # loaded. In contrast, registered actions are registered by setup code
    # (e.g., when one doesn't know in advance which output/input the user wants
    # to record from).
    _plugin_actions = Typed(list, {})
    _registered_actions = Typed(list, {})
    _actions = Property()

    def _get__actions(self):
        return self._registered_actions + self._plugin_actions

    def start(self):
        log.debug('Starting controller plugin %s', self.__class__.__name__)
        self._refresh_io()
        self._refresh_actions()
        self._bind_observers()
        self.core = self.workbench.get_plugin('enaml.workbench.core')
        self.context = self.workbench.get_plugin('psi.context')
        self.data = self.workbench.get_plugin('psi.data')

    def stop(self):
        self._unbind_observers()
        self._dispatcher.stop()

    def _wrapup(self, **kwargs):
        workbench = self.workbench
        point = workbench.get_extension_point(WRAPUP_POINT)
        for extension in point.extensions:
            if extension.factory is None:
                msg = "extension '%s' does not declare a factory"
                raise ValueError(msg % extension.qualified_id)
            cb = extension.factory(workbench)
            cb(**kwargs)

    def _bind_observers(self):
        self.workbench.get_extension_point(IO_POINT) \
            .observe('extensions', self._refresh_io)
        self.workbench.get_extension_point(ACTION_POINT) \
            .observe('extensions', self._refresh_actions)

    def _unbind_observers(self):
        self.workbench.get_extension_point(IO_POINT) \
            .unobserve('extensions',self._refresh_io)
        self.workbench.get_extension_point(ACTION_POINT) \
            .unobserve('extensions', self._refresh_actions)

    def _refresh_io(self, event=None):
        point = self.workbench.get_extension_point(IO_POINT)
        self.io.refresh(point, self.workbench)

    def connect_output(self, output_name, target_name):
        return self.io.connect_output(output_name, target_name)

    def connect_input(self, input_name, source_name):
        return self.io.connect_input(input_name, source_name)

    def _refresh_actions(self, event=None):
        actions = []
        event_loggers = []
        events = {}
        states = {}

        point = self.workbench.get_extension_point(ACTION_POINT)
        for extension in point.extensions:
            log.debug('Scanning extension %s', extension.id)
            found_states = extension.get_children(ExperimentState)
            found_events = extension.get_children(ExperimentEvent)
            found_actions = extension.get_children(ExperimentActionBase)
            found_loggers = extension.get_children(EventLogger)

            for state in found_states:
                if state.name in states:
                    m = '{} state already exists'.format(state.name)
                    raise ValueError(m)
                states[state.name] = state
                found_events.extend(state._generate_events())

            for event in found_events:
                if event.name in events:
                    m = '{} event already exists'.format(event.name)
                    raise ValueError(m)
                events[event.name] = event

            for action in found_actions:
                log.debug('... Found action %s', action)

            actions.extend(found_actions)
            event_loggers.extend(found_loggers)

        context = {}
        for state_name in states:
            context[state_name + '_active'] = False
        for event_name in events:
            context[event_name] = False

        actions.sort(key=lambda a: a.weight)
        self._states = states
        self._events = events
        self._plugin_actions = actions
        self._action_context = context
        self._event_loggers = event_loggers

    def register_action(self, event, command, kwargs=None):
        if kwargs is None:
            kwargs = {}
        if isinstance(command, str):
            action = ExperimentAction(event=event, command=command,
                                      kwargs=kwargs)
        else:
            action = ExperimentCallback(event=event, callback=command,
                                        kwargs=kwargs)
        self._registered_actions.append(action)

    def validate_action_dependencies(self):
        '''
        Verify that every action's event expression references only known
        events and state flags.

        Because action-event matching is expression-based, a typo (e.g.,
        `target_ended` instead of `target_end`) does not raise; the action
        simply never fires. This check converts that silent failure into a
        hard error at experiment start, before any data is acquired. Actions
        bound to events that are generated dynamically at runtime can opt
        out with `allow_unregistered = True`.
        '''
        # The action-evaluation context contains every registered event name
        # plus the <state>_active flags, which is exactly the universe of
        # names an event expression may reference. Builtins are permitted
        # since expressions are evaluated with eval.
        known = set(self._action_context)
        allowed = known | set(dir(builtins))
        errors = []
        for action in self._actions:
            if getattr(action, 'allow_unregistered', False):
                continue
            for name in sorted(set(action.dependencies) - allowed):
                close = difflib.get_close_matches(name, sorted(known), n=3)
                hint = ' Did you mean: {}?'.format(', '.join(close)) \
                    if close else ''
                errors.append(f'{action} references unknown event or state '
                              f'"{name}".{hint}')
        if errors:
            mesg = 'Invalid experiment actions:\n' + \
                '\n'.join(f' - {e}' for e in errors)
            raise ActionError(mesg)

    def finalize_io(self):
        log.info('Finalizing IO')
        self.io.connect_outputs()
        self.io.connect_inputs()
        self.invoke_actions('io_configured')

    # Note: the IOManager methods do the locked hardware work; actions are
    # always invoked here, *after* those methods return (and the lock is
    # released). Actions may invoke commands (e.g., reset_engines) that
    # acquire the lock again; invoking them under it would deadlock. See
    # docs/threading.md.

    def configure_engines(self):
        def done_callback(engine):
            return partial(self.invoke_actions, '{}_end'.format(engine.name))
        self.io.configure_engines(done_callback)
        self.invoke_actions('engines_configured')

    def start_engines(self):
        self.io.start_engines()
        self.invoke_actions('engines_started')

    def stop_engines(self):
        # Cancel pending control-plane timers first so delayed events do not
        # fire against stopped engines.
        self.stop_all_timers()
        self.io.stop_engines()
        self.invoke_actions('engines_stopped')

    def reset_engines(self):
        self.io.reset_engines()

    def get_output(self, output_name):
        return self.io.get_output(output_name)

    def get_input(self, input_name):
        return self.io.get_input(input_name)

    def set_input_attr(self, input_name, attr_name, value):
        self.io.set_input_attr(input_name, attr_name, value)

    def get_channel(self, channel_name):
        return self.io.get_channel(channel_name)

    def get_channels(self, mode=None, direction=None, timing=None,
                     active=True):
        '''
        Return channels matching criteria across all engines

        See `psi.controller.io_manager.IOManager.get_channels`.
        '''
        return self.io.get_channels(mode, direction, timing, active)

    def invoke_actions(self, event_name, timestamp=None, delayed=False,
                       cancel_existing=True, kw=None, skip_errors=False,
                       wait=True):
        '''
        Invoke all actions bound to `event_name`.

        The actions execute on the control dispatcher thread, serialized
        with all other control-plane work. May be called from any thread.

        Parameters
        ----------
        wait : bool
            If True (default), block until the actions complete and return
            their results (exceptions propagate to the caller). If False,
            enqueue and return immediately (exceptions are logged). Use
            False from data-plane callbacks that must not block on control
            work.
        '''
        if cancel_existing:
            self._dispatcher.cancel(event_name)
        if delayed:
            delay = timestamp-self.get_ts()
            if delay > 0:
                self._dispatcher.call_later(
                    event_name, delay, self._invoke_actions, event_name,
                    timestamp, kw, skip_errors)
                return
        if not wait:
            self._dispatcher.submit(self._invoke_actions, event_name,
                                    timestamp, kw, skip_errors)
            return
        return self._dispatcher.submit_sync(self._invoke_actions, event_name,
                                            timestamp, kw, skip_errors)

    def event_used(self, event_name):
        '''
        Returns True if the experiment event is bound to an experiment action.

        This is typically used internally as a performance-optimization so we
        don't configure callbacks for events that are unused. For example, we
        can attach actions to the <input_name>_acquired event. However, this
        event typically occurs several times a second for each input. This
        would result in unecessary calls to `invoke_actions`.

        Parameters
        ----------
        event_name : str
            Name of event

        Returns
        -------
        used : bool
            True if event is bound to an action, False otherwise.
        '''
        for action in self._actions:
            if event_name in action.dependencies:
                return True
        return False

    def _log_event(self, event_name, timestamp, kw):
        data = {
            'event': event_name,
            'timestamp': timestamp,
            'info': kw,
        }
        for logger in self._event_loggers:
            logger._invoke(self.core, data)

    def _invoke_actions(self, event_name, timestamp=None, kw=None, skip_errors=False):
        log.debug('Invoking actions for {}'.format(event_name))
        deferred_call(self._log_event, event_name, timestamp, kw)

        # If this is a stateful event, update the associated state.
        if event_name.endswith('_start'):
            key = event_name[:-6]
            self._action_context[key + '_active'] = True
        elif event_name.endswith('_end'):
            key = event_name[:-4]
            self._action_context[key + '_active'] = False

        # Make a copy of the context and set the event to True. We don't want
        # to set the state on the main context since it may affect recursive
        # notifications.
        context = self._action_context.copy()
        context[event_name] = True

        # We ignore any missing variables in the ExperimentAction system when
        # the experiment has not been fully initialized since these missing
        # variables may not exist until the experiment is fully initialized.
        # Further, if skip_errors is True, this usually indicates that we
        # absolutely want actions to run through to the end if possible.
        ignore_missing = skip_errors or \
            (self.experiment_state == 'initialized')
        results = []
        for action in self._actions:
            log.trace(f'... checking {action}')
            if action.match(context, ignore_missing):
                result = invoke_action(self.core, action, event_name,
                                       timestamp, kw, skip_errors)
                results.append(result)
        return results

    def request_apply(self):
        if not self.apply_changes():
            log.debug('Apply requested')
            deferred_call(lambda: setattr(self, '_apply_requested', True))

    def request_pause(self):
        if not self.pause_experiment():
            log.debug('Pause requested')
            deferred_call(lambda: setattr(self, '_pause_requested', True))

    def request_resume(self):
        if not self.resume_experiment():
            log.debug('Resume requested')
            deferred_call(lambda: setattr(self, '_resume_requested', True))

    # -- Subclass contract ---------------------------------------------------
    # The base controller knows how to start and stop an experiment; what
    # happens between trials is paradigm-specific. Subclasses hook in by
    # overriding the template methods below. `apply_changes`,
    # `pause_experiment` and `resume_experiment` share a request protocol:
    # return True if the change took effect immediately, or something falsy
    # to defer it, in which case the corresponding `request_*` method latches
    # a flag (`_apply_requested`, `_pause_requested`, `_resume_requested`)
    # that the subclass must consume — act on, then clear — at its next safe
    # point (typically between trials). See docs/threading.md ("Request
    # flags").

    def apply_changes(self):
        '''
        Apply pending context changes now.

        Returns
        -------
        handled : bool
            True if the changes were applied. Return something falsy to
            defer; `request_apply` then latches `_apply_requested` and the
            subclass must apply (and clear the flag) between trials.

        The base implementation applies immediately, which is only safe when
        no trial is in progress; subclasses running trials should defer while
        a trial is active.
        '''
        self.context.apply_changes()
        return True

    def pause_experiment(self):
        '''
        Attempt to pause the experiment now.

        Returns
        -------
        handled : bool
            True if the pause took effect immediately. Return something
            falsy to defer; `request_pause` then latches `_pause_requested`
            and the subclass must pause (clear the flag, set
            `experiment_state` to 'paused') at its next safe point.
        '''
        raise NotImplementedError(
            f'{type(self).__name__} does not support pausing. Controllers '
            'that support pause must override pause_experiment (and '
            'resume_experiment); see the method docstring for the expected '
            'return-value protocol.'
        )

    def resume_experiment(self):
        '''
        Attempt to resume a paused experiment now.

        Same return-value protocol as `pause_experiment` (deferred requests
        latch `_resume_requested`).
        '''
        raise NotImplementedError(
            f'{type(self).__name__} does not support resuming. Controllers '
            'that support pause/resume must override resume_experiment; see '
            'the method docstring for the expected return-value protocol.'
        )

    def end_trial(self):
        '''
        End the current trial and advance to the next.

        Invoked by the ``psi.controller.next_trial`` command (e.g., from a
        toolbar button or hotkey). The base controller has no concept of a
        trial; subclasses that expose next-trial UI must override this.
        '''
        raise NotImplementedError(
            f'{type(self).__name__} does not implement end_trial, which is '
            'required by the psi.controller.next_trial command. Override '
            'end_trial in your controller plugin to end the current trial '
            'and start the next one.'
        )

    def start_experiment(self):
        deferred_call(self._start_experiment)

    def _start_experiment(self):
        try:
            self.validate_action_dependencies()
            self.invoke_actions('experiment_initialize')
            self.invoke_actions('experiment_prepare')
            self.invoke_actions('experiment_start')
            self.experiment_state = 'running'
        except Exception:
            log.error('An error occured when attempting to start experiment. Stopping.')
            self.stop_experiment(True)
            raise

    def stop_experiment(self, skip_errors=False, kw=None):
        if self.experiment_state not in ('running', 'paused'):
            log.debug('Nothing to do since experiment is not running. Returning.')
            return []
        # Set this flag before invoking actions since some of the actions may
        # trigger a circular loop that results in `stop_experiment` getting
        # called again.
        self.experiment_state = 'stopped'
        if kw is None:
            kw = {}
        return self.invoke_actions('experiment_end', self.get_ts(), skip_errors=skip_errors, kw=kw)

    def get_ts(self):
        return self.io.get_ts()

    def start_timer(self, name, duration, callback, cancel_existing=True):
        # The callback runs on the control dispatcher thread.
        self._dispatcher.call_later(name, duration, callback,
                                    cancel_existing=cancel_existing)

    def stop_timer(self, name):
        self._dispatcher.cancel(name)

    def stop_all_timers(self):
        self._dispatcher.cancel_all()
