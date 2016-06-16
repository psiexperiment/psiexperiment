import enum
import threading
from functools import partial

import logging
log = logging.getLogger(__name__)

from atom.api import Int, Typed, Unicode, observe
from enaml.core.api import d_func
import numpy as np

from .base_plugin import BaseController


from enaml.qt.QtCore import QTimer, QMutex


ACTION_POINT = 'psi.controller.action'


class TrialScore(enum.Enum):

    hit = 'HIT'
    miss = 'MISS'
    correct_reject = 'CR'
    false_alarm = 'FA'


class TrialState(enum.Enum):
    '''
    Defines the possible states that the experiment can be in. We use an Enum to
    minimize problems that arise from typos by the programmer (e.g., they may
    accidentally set the state to "waiting_for_nose_poke_start" rather than
    "waiting_for_np_start").

    This is specific to appetitive reinforcement paradigms.
    '''
    waiting_for_resume = 'waiting for resume'
    waiting_for_np_start = 'waiting for nose-poke start'
    waiting_for_np_duration = 'waiting for nose-poke duration'
    waiting_for_hold_period = 'waiting for hold period'
    waiting_for_response = 'waiting for response'
    waiting_for_reward = 'waiting for reward retrieval'
    waiting_for_to = 'waiting for timeout'
    waiting_for_iti = 'waiting for intertrial interval'


class Event(enum.Enum):
    '''
    Defines the possible events that may occur during the course of the
    experiment.

    This is specific to appetitive reinforcement paradigms.
    '''
    np_start = 'initiated nose poke'
    np_end = 'withdrew from nose poke'
    np_duration_elapsed = 'nose poke duration met'
    hold_duration_elapsed = 'hold period over'
    response_duration_elapsed = 'response timed out'
    reward_start = 'reward contact'
    reward_end = 'withdrew from reward'
    to_duration_elapsed = 'timeout over'
    iti_duration_elapsed = 'ITI over'
    trial_start = 'trial start'


class Mutex(object):
    '''
    Wrapper around the QMutex object for use with the Python context manager

    Example
    -------
    mutex = Mutex()
    with mutex:
        # Mutex is automatically locked when entering this block. Do some work
        # safely. When the block is complete, the mutex is unlocked.
    '''

    def __init__(self):
        self.mutex = QMutex()

    def __enter__(self):
        self.mutex.lock()

    def __exit__(self, type, value, traceback):
        self.mutex.unlock()


class AppetitivePlugin(BaseController):
    '''
    Plugin for controlling appetitive experiments that are based on a reward.
    Eventually this may become generic enough that it can be used with aversive
    experiments as well (it may already be sufficiently generic).
    '''

    # Current trial
    trial = Int(0)

    # Current number of consecutive nogos 
    consecutive_nogo = Int(0)

    # Used by the trial sequence selector to randomly select between go/nogo.
    rng = Typed(np.random.RandomState)
    trial_type = Unicode()
    trial_info = Typed(dict, ())
    trial_state = Typed(TrialState)

    timer = Typed(QTimer)
    mutex = Typed(Mutex, ())

    samples = Typed(list, ())

    event_map = {
        ('rising', 'nose_poke'): Event.np_start,
        ('falling', 'nose_poke'): Event.np_end,
        ('rising', 'reward_contact'): Event.reward_start,
        ('falling', 'reward_contact'): Event.reward_end,
    }

    score_map = {
        ('nogo', 'reward'): TrialScore.false_alarm,
        ('nogo', 'poke'): TrialScore.correct_reject,
        ('nogo', 'no response'): TrialScore.correct_reject,
        ('go', 'reward'): TrialScore.hit,
        ('go', 'poke'): TrialScore.miss,
        ('go', 'no response'): TrialScore.miss,
    }

    def next_selector(self):
        '''
        Determine next trial type
        '''
        try:
            max_nogo = self.context.get_value('max_nogo')
            go_probability = self.context.get_value('go_probability')
            score = self.context.get_value('score')
        except KeyError:
            self.trial_type = 'go_remind'
            return 'remind'

        if self._remind_requested:
            self.trial_type = 'go_remind'
            return 'remind'
        elif self.consecutive_nogo >= max_nogo:
            self.trial_type = 'go_forced'
            return 'go'
        elif score == TrialScore.false_alarm:
            self.trial_type = 'nogo_repeat'
            return 'nogo'
        else:
            if self.rng.uniform() <= go_probability:
                self.trial_type = 'go'
                return 'go'
            else:
                self.trial_type = 'nogo'
                return 'nogo'

    def start_experiment(self):
        try:
            self.trial += 1
            self.context.apply_changes()
            self.start_engines()
            self.rng = np.random.RandomState()
            self.context.next_setting(self.next_selector(), save_prior=False)
            self.core.invoke_command('psi.data.prepare')
            self.experiment_state = 'running'
            self.trial_state = TrialState.waiting_for_np_start
            self.invoke_actions('experiment_start')
        except Exception as e:
            # TODO - provide user interface to notify of errors. How to
            # recover?
            raise

    def start_trial(self):
        self.invoke_actions('trial_start')
        # TODO - the hold duration will include the update delay. Do we need
        # super-precise tracking of hold period or can it vary by a couple 10s
        # to 100s of msec?
        self.trial_state = TrialState.waiting_for_hold_period
        self.start_timer('hold_duration', Event.hold_duration_elapsed)

    def end_trial(self, response):
        log.debug('Animal responded by {}, ending trial'.format(response))
        self.stop_timer()

        trial_type = self.trial_type.split('_', 1)[0]
        score = self.score_map[trial_type, response]
        self.consecutive_nogo = self.consecutive_nogo + 1 \
            if trial_type == 'nogo' else 0

        response_ts = self.trial_info.get('response_ts', np.nan)
        target_start = self.trial_info.get('target_start', np.nan)
        np_end = self.trial_info.get('np_end', np.nan)
        np_start = self.trial_info.get('np_start', np.nan)
        self.trial_info.update({
            'response': response,
            'trial_type': self.trial_type,
            'score': score.value,
            'correct': score in (TrialScore.correct_reject, TrialScore.hit),
            'response_time': response_ts-target_start,
            'reaction_time': np_end-np_start,
        })
        if score == TrialScore.false_alarm:
            self.invoke_actions('timeout_start')
            self.trial_state = TrialState.waiting_for_to
            self.start_timer('to_duration', Event.to_duration_elapsed)
        else:
            if score == TrialScore.hit:
                self.invoke_actions('deliver_reward')
            self.trial_state = TrialState.waiting_for_iti
            self.start_timer('iti_duration', Event.iti_duration_elapsed)

        self.context.set_values(self.trial_info)
        self.core.invoke_command('psi.data.process_trial')

    def request_resume(self):
        super(AppetitivePlugin, self).request_resume()
        self.trial_state = TrialState.waiting_for_np_start

    def ao_callback(self, name, offset, samples):
        # TODO: This assumes there is only one output. Need to figure out how
        # to handle two.
        
        waveforms = []
        output = self._outputs[name]
        waveform = output.get_waveform(offset, samples)
        output.engine.append_hw_ao(waveform)

    def ai_callback(self, name, data):
        log.trace('Acquired {} samples from {}'.format(data.shape, name))
        self.samples.append(data)

    def di_callback(self, name, data):
        log.trace('Acquired {} samples from {}'.format(data.shape, name))
        self.samples.append(data)

    def et_callback(self, *args):
        log.debug('Detected {} on {} at {}'.format(change, name, event_time))
        event = self.event_map[change, line]
        self.handle_event(event, event_time)

    def stop_experiment(self):
        self.stop_engines()
        self.experiment_state = 'stopped'

    def pause_experiment(self):
        with self.mutex:
            if self.trial_state != TrialState.waiting_for_np_start:
                return False
            self._pause_experiment()
            return True

    def _pause_experiment(self):
        self.experiment_state = 'paused'
        self._pause_requested = False

    def apply_changes(self):
        with self.mutex:
            if self.trial_state != TrialState.waiting_for_np_start:
                return False
            self._apply_changes()
            return True

    def _apply_changes(self):
        self.context.apply_changes()
        self._apply_requested = False
        log.debug('applied changes')

    def handle_event(self, event, timestamp=None):
        # Ensure that we don't attempt to process several events at the same
        # time. This essentially queues the events such that the next event
        # doesn't get processed until `_handle_event` finishes processing the
        # current one.
        with self.mutex:
            # Only events generated by NI-DAQmx callbacks will have a timestamp.
            # Since we want all timing information to be in units of the analog
            # output sample clock, we will capture the value of the sample clock
            # if a timestamp is not provided. Since there will be some delay
            # between the time the event occurs and the time we read the analog
            # clock, the timestamp won't be super-accurate. However, it's not
            # super-important since these events are not reference points around
            # which we would do a perievent analysis. Important reference points
            # would include nose-poke initiation and withdraw, reward contact,
            # sound onset, lights on, lights off. These reference points will
            # be tracked via NI-DAQmx or can be calculated (i.e., we know
            # exactly when the target onset occurs because we precisely specify
            # the location of the target in the analog output buffer).
            try:
                if timestamp is None:
                    timestamp = self.get_ts()
                log.debug('{} at {}'.format(event, timestamp))
                params = {'event': event.value, 'timestamp': timestamp}
                self.core.invoke_command('psi.data.process_event', params)
                self._handle_event(event, timestamp)
            except Exception as e:
                log.exception(e)
                raise

    def _handle_event(self, event, timestamp):
        '''
        Give the current experiment state, process the appropriate response for
        the event that occured. Depending on the experiment state, a particular
        event may not be processed.
        '''
        # HOWTO: If training, then set np duration and hold duration to 0. This
        # means that the trial will start immediately after the intertrial
        # interval is over. We then go into the response period.
        self.invoke_actions(event.name)

        if self.experiment_state == 'paused':
            # If the experiment is paused, don't do anything.
            return

        if self.trial_state == TrialState.waiting_for_np_start:
            if event == Event.np_start:
                # Animal has nose-poked in an attempt to initiate a trial.
                self.trial_state = TrialState.waiting_for_np_duration
                self.start_timer('np_duration', Event.np_duration_elapsed)
                # If the animal does not maintain the nose-poke long enough,
                # this value will get overwritten with the next nose-poke. In
                # general, set np_end to NaN (sometimes the animal never
                # withdraws).
                self.trial_info['np_start'] = timestamp
                self.trial_info['np_end'] = np.nan

        elif self.trial_state == TrialState.waiting_for_np_duration:
            if event == Event.np_end:
                # Animal has withdrawn from nose-poke too early. Cancel the
                # timer so that it does not fire a 'event_np_duration_elapsed'.
                log.debug('Animal withdrew too early')
                self.stop_timer()
                self.trial_state = TrialState.waiting_for_np_start
            elif event == Event.np_duration_elapsed:
                log.debug('Animal initiated trial')
                self.start_trial()

        elif self.trial_state == TrialState.waiting_for_hold_period:
            # All animal-initiated events (poke/reward) are ignored during this
            # period but we may choose to record the time of nose-poke withdraw
            # if it occurs.
            if event == Event.np_end:
                # Record the time of nose-poke withdrawal if it is the first
                # time since initiating a trial.
                log.debug('Animal withdrew during hold period')
                if np.isnan(self.trial_info['np_end']):
                    log.debug('Recording np_end')
                    self.trial_info['np_end'] = timestamp
            elif event == Event.hold_duration_elapsed:
                log.debug('Animal maintained poke through hold period')
                self.trial_state = TrialState.waiting_for_response
                self.start_timer('response_duration',
                                 Event.response_duration_elapsed)

        elif self.trial_state == TrialState.waiting_for_response:
            # If the animal happened to initiate a nose-poke during the hold
            # period above and is still maintaining the nose-poke, they have to
            # manually withdraw and re-poke for us to process the event.
            if event == Event.np_end:
                # Record the time of nose-poke withdrawal if it is the first
                # time since initiating a trial.
                log.debug('Animal withdrew during response period')
                if np.isnan(self.trial_info['np_end']):
                    log.debug('Recording np_end')
                    self.trial_info['np_end'] = timestamp
            elif event == Event.np_start:
                log.debug('Animal repoked')
                self.trial_info['response_ts'] = timestamp
                self.end_trial(response='poke')
            elif event == Event.reward_start:
                log.debug('Animal went to reward')
                self.trial_info['response_ts'] = timestamp
                self.end_trial(response='reward')
            elif event == Event.response_duration_elapsed:
                log.debug('Animal provided no response')
                self.trial_info['response_ts'] = np.nan
                self.end_trial(response='no response')

        elif self.trial_state == TrialState.waiting_for_to:
            if event == Event.to_duration_elapsed:
                # Turn the light back on
                self.invoke_actions('timeout_end')
                self.trial_state = TrialState.waiting_for_iti
                self.start_timer('iti_duration',
                                 Event.iti_duration_elapsed)
            elif event in (Event.reward_start, Event.np_start):
                log.debug('Resetting timeout duration')
                self.stop_timer()
                self.start_timer('to_duration', Event.to_duration_elapsed)

        elif self.trial_state == TrialState.waiting_for_iti:
            if event == Event.iti_duration_elapsed:
                log.debug('Setting up for next trial')
                # Apply pending changes that way any parameters (such as
                # repeat_FA or go_probability) are reflected in determining the
                # next trial type.
                if self._apply_requested:
                    self._apply_changes()
                selector = self.next_selector()
                self.context.next_setting(selector, save_prior=True)
                self.trial += 1

                if self._pause_requested:
                    self._pause_experiment()
                    self.trial_state = TrialState.waiting_for_resume
                else:
                    self.trial_state = TrialState.waiting_for_np_start

    def stop_timer(self):
        self.timer.timeout.disconnect()
        self.timer.stop()

    def start_timer(self, variable, event):
        # Even if the duration is 0, we should still create a timer because this
        # allows the `_handle_event` code to finish processing the event. The
        # timer should execute as soon as `_handle_event` finishes processing.
        # Since the Enaml application is Qt-based, we need to use the QTimer to
        # avoid issues with a Qt-based multithreading application.
        log.debug('timer for {} set to {}'.format(event, variable))
        duration = self.context.get_value(variable)
        receiver = partial(self.handle_event, event)

        if self.timer is not None:
            self.timer.deleteLater()

        self.timer = QTimer()
        self.timer.timeout.connect(receiver)    # set up new callback
        self.timer.setSingleShot(True)          # call only once
        self.timer.start(duration*1e3)

    @observe('trial_state')
    def _log_trial_state(self, event):
        log.debug('trial state set to {}'.format(event['value']))

    @observe('experiment_state')
    def _log_experiment_state(self, event):
        log.debug('experiment state set to {}'.format(event['value']))
