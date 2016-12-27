import enum
import threading
from functools import partial

import logging
log = logging.getLogger(__name__)

from atom.api import Int, Typed, Unicode, observe
from enaml.application import deferred_call
from enaml.core.api import d_func
from enaml.qt.QtCore import QTimer
import numpy as np

from .base_plugin import BasePlugin


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

    hold_start = 'hold period started'
    hold_end = 'hold period over'

    response_start = 'response period started'
    response_end = 'response timed out'

    reward_start = 'reward contact'
    reward_end = 'withdrew from reward'

    to_start = 'timeout started'
    to_end = 'timeout over'

    iti_start = 'ITI started'
    iti_end = 'ITI over'

    trial_start = 'trial start'
    trial_end = 'trial end'

    target_start = 'target start'
    target_end = 'target end'


class AppetitivePlugin(BasePlugin):
    '''
    Plugin for controlling appetitive experiments that are based on a reward.
    Eventually this may become generic enough that it can be used with aversive
    experiments as well (it may already be sufficiently generic).
    '''
    lock = Typed(object)

    # Current trial
    trial = Int(0)

    # Current number of consecutive nogos
    consecutive_nogo = Int(0)

    # Used by the trial sequence selector to randomly select between go/nogo.
    rng = Typed(np.random.RandomState)
    trial_type = Unicode()
    trial_info = Typed(dict, ())
    trial_state = Typed(TrialState)
    #trial_epoch_running = Bool(False)

    timer = Typed(QTimer)

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
        self.lock = threading.RLock()
        try:
            self.trial += 1
            self.rng = np.random.RandomState()
            self.context.next_setting(self.next_selector(), save_prior=False)
            self.experiment_state = 'running'
            self.trial_state = TrialState.waiting_for_np_start
            self.context.get_values()
            self.invoke_actions('experiment_start', self.get_ts())
            self.invoke_actions('trial_prepare', self.get_ts())
        except Exception as e:
            # TODO - provide user interface to notify of errors. How to
            # recover?
            raise

    def start_trial(self):
        ts = self.get_ts()
        self.invoke_actions(Event.trial_start.name, ts)
        self.invoke_actions(Event.hold_start.name, ts)
        self.trial_state = TrialState.waiting_for_hold_period
        self.start_timer('hold_duration', Event.hold_end)

    def end_trial(self, response):
        log.debug('Animal responded by {}, ending trial'.format(response))
        self.stop_timer()

        ts = self.get_ts()
        #self.invoke_actions(Event.response_end.name, ts)

        trial_type = self.trial_type.split('_', 1)[0]
        score = self.score_map[trial_type, response]
        self.consecutive_nogo = self.consecutive_nogo + 1 \
            if trial_type == 'nogo' else 0

        response_ts = self.trial_info.setdefault('response_ts', np.nan)
        target_start = self.trial_info.setdefault('target_start', np.nan)
        np_end = self.trial_info.setdefault('np_end', np.nan)
        np_start = self.trial_info.setdefault('np_start', np.nan)

        self.trial_info.update({
            'response': response,
            'trial_type': self.trial_type,
            'score': score.value,
            'correct': score in (TrialScore.correct_reject, TrialScore.hit),
            'response_time': response_ts-target_start,
            'reaction_time': np_end-np_start,
        })
        if score == TrialScore.false_alarm:
            self.trial_state = TrialState.waiting_for_to
            self.invoke_actions(Event.to_start.name, ts)
            self.start_timer('to_duration', Event.to_end)
        else:
            if score == TrialScore.hit:
                if not self.context.get_value('training_mode'):
                    self.invoke_actions('deliver_reward', ts)
            self.trial_state = TrialState.waiting_for_iti
            self.invoke_actions(Event.iti_start.name, ts)
            self.start_timer('iti_duration', Event.iti_end)

        self.context.set_values(self.trial_info)
        parameters ={'results': self.context.get_values()}
        self.core.invoke_command('psi.data.process_trial', parameters)

        log.debug('Setting up for next trial')
        # Apply pending changes that way any parameters (such as repeat_FA or
        # go_probability) are reflected in determining the next trial type.
        if self._apply_requested:
            self._apply_changes()
        selector = self.next_selector()
        self.context.next_setting(selector, save_prior=True)
        self.trial += 1
        self.trial_info = {}

    def request_resume(self):
        super(AppetitivePlugin, self).request_resume()
        self.trial_state = TrialState.waiting_for_np_start

    def ao_callback(self, name):
        log.debug('Updating output {}'.format(name))
        self._outputs[name].update()

    def ai_callback(self, name, data):
        parameters = {'name': name, 'data': data}
        self.core.invoke_command('psi.data.process_ai', parameters)

    def di_callback(self, name, data):
        pass

    def et_callback(self, name, edge, event_time):
        if edge == 'processed':
            parameters = {'name': 'event_log', 'timestamp': event_time}
            self.core.invoke_command('psi.data.set_current_time', parameters)
        else:
            log.debug('Detected {} on {} at {}'.format(edge, name, event_time))
            event = self.event_map[edge, name]
            self.handle_event(event, event_time)

    def pause_experiment(self):
        if self.trial_state == TrialState.waiting_for_np_start:
            deferred_call(self._pause_experiment)

    def _pause_experiment(self):
        self.experiment_state = 'paused'
        self._pause_requested = False

    def apply_changes(self):
        if self.trial_state == TrialState.waiting_for_np_start:
            deferred_call(self._apply_changes)

    def _apply_changes(self):
        self.context.apply_changes()
        self._apply_requested = False
        log.debug('applied changes')

    def handle_event(self, event, timestamp=None):
        # Ensure that we don't attempt to process several events at the same
        # time. This essentially queues the events such that the next event
        # doesn't get processed until `_handle_event` finishes processing the
        # current one.

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
            log.trace('Emitting handle_event signal')
            deferred_call(self._handle_event, event, timestamp)
        except Exception as e:
            log.exception(e)
            raise

    def _handle_event(self, event, timestamp):
        '''
        Give the current experiment state, process the appropriate response for
        the event that occured. Depending on the experiment state, a particular
        event may not be processed.
        '''
        log.debug('Recieved handle_event signal')
        self.invoke_actions(event.name, timestamp)

        if self.experiment_state == 'paused':
            # If the experiment is paused, don't do anything.
            return

        if self.trial_state == TrialState.waiting_for_np_start:
            if event == Event.np_start:
                # Animal has nose-poked in an attempt to initiate a trial.
                self.trial_state = TrialState.waiting_for_np_duration
                self.start_timer('np_duration', Event.np_duration_elapsed)
                # If the animal does not maintain the nose-poke long enough,
                # this value will be deleted.
                self.trial_info['np_start'] = timestamp

        elif self.trial_state == TrialState.waiting_for_np_duration:
            if event == Event.np_end:
                # Animal has withdrawn from nose-poke too early. Cancel the
                # timer so that it does not fire a 'event_np_duration_elapsed'.
                log.debug('Animal withdrew too early')
                self.stop_timer()
                self.trial_state = TrialState.waiting_for_np_start
                del self.trial_info['np_start']
            elif event == Event.np_duration_elapsed:
                log.debug('Animal initiated trial')
                self.start_trial()

                # We want to deliver the reward immediately when in training
                # mode so the food is already in the hopper. Not sure how
                # *critical* this is?
                if self.context.get_value('training_mode'):
                    if self.trial_type.startswith('go'):
                        self.invoke_actions('deliver_reward', self.get_ts())

        elif self.trial_state == TrialState.waiting_for_hold_period:
            # All animal-initiated events (poke/reward) are ignored during this
            # period but we may choose to record the time of nose-poke withdraw
            # if it occurs.
            if event == Event.np_end:
                # Record the time of nose-poke withdrawal if it is the first
                # time since initiating a trial.
                log.debug('Animal withdrew during hold period')
                if 'np_end' not in self.trial_info:
                    log.debug('Recording np_end')
                    self.trial_info['np_end'] = timestamp
            elif event == Event.hold_end:
                log.debug('Animal maintained poke through hold period')
                self.trial_state = TrialState.waiting_for_response
                self.invoke_actions(Event.response_start.name, self.get_ts())
                self.start_timer('response_duration', Event.response_end)

        elif self.trial_state == TrialState.waiting_for_response:
            # If the animal happened to initiate a nose-poke during the hold
            # period above and is still maintaining the nose-poke, they have to
            # manually withdraw and re-poke for us to process the event.
            if event == Event.np_end:
                # Record the time of nose-poke withdrawal if it is the first
                # time since initiating a trial.
                log.debug('Animal withdrew during response period')
                if 'np_end' not in self.trial_info:
                    log.debug('Recording np_end')
                    self.trial_info['np_end'] = timestamp
            elif event == Event.np_start:
                log.debug('Animal repoked')
                self.trial_info['response_ts'] = timestamp
                self.invoke_actions(Event.response_end.name, timestamp)
                self.end_trial(response='poke')
                # At this point, trial_info should have been cleared by the
                # `end_trial` function so that we can prepare for the next
                # trial. Save the start of the nose-poke.
                self.trial_info['np_start'] = timestamp
            elif event == Event.reward_start:
                log.debug('Animal went to reward')
                self.invoke_actions(Event.response_end.name, timestamp)
                self.trial_info['response_ts'] = timestamp
                self.end_trial(response='reward')
            elif event == Event.response_end:
                log.debug('Animal provided no response')
                self.trial_info['response_ts'] = np.nan
                self.end_trial(response='no response')

        elif self.trial_state == TrialState.waiting_for_to:
            if event == Event.to_end:
                # Turn the light back on
                self.invoke_actions('timeout_end', self.get_ts())
                self.trial_state = TrialState.waiting_for_iti
                self.invoke_actions(Event.iti_start.name, self.get_ts())
                self.start_timer('iti_duration', Event.iti_end)
            elif event in (Event.reward_start, Event.np_start):
                # Animal repoked. Reset timeout duration.
                log.debug('Resetting timeout duration')
                self.stop_timer()
                self.start_timer('to_duration', Event.to_end)

        elif self.trial_state == TrialState.waiting_for_iti:
            if event == Event.iti_end:
                if self._pause_requested:
                    self.pause_experiment()
                    self.trial_state = TrialState.waiting_for_resume
                elif 'np_start' in self.trial_info:
                    # The animal had initiated a nose-poke during the ITI.
                    # Allow this to contribute towards the start of the next
                    # trial by calculating how much is pending in the nose-poke
                    # duration.
                    self.trial_state = TrialState.waiting_for_np_duration
                    current_poke_duration = self.get_ts()-self.trial_info['np_start']
                    poke_duration = self.context.get_value('np_duration')
                    remaining_poke_duration = poke_duration-current_poke_duration
                    delta = max(0, remaining_poke_duration)
                    self.start_timer(delta, Event.np_duration_elapsed)
                else:
                    # Call get_values to seed the context before it's actually
                    # needed. Should we also prepare the token?
                    self.trial_state = TrialState.waiting_for_np_start
                    self.context.get_values()
                    self.invoke_actions('trial_prepare', self.get_ts())
            elif event == Event.np_end and 'np_start' in self.trial_info:
                del self.trial_info['np_start']

    def start_timer(self, duration, event):
        deferred_call(self._start_timer, duration, event)

    def stop_timer(self):
        deferred_call(self._stop_timer)

    def _stop_timer(self):
        if self.timer is not None:
            self.timer.timeout.disconnect()
            self.timer.stop()

    def _start_timer(self, duration, event):
        # The duration can be specified as a string naming the context variable
        # to extract.
        if isinstance(duration, basestring):
            duration = self.context.get_value(duration)
        log.debug('Timer for {} with duration {}'.format(event, duration))
        receiver = partial(self.handle_event, event)

        if duration == 0:
            deferred_call(receiver)
        else:
            self.timer = QTimer()
            self.timer.timeout.connect(receiver)    # set up new callback
            self.timer.setSingleShot(True)          # call only once
            self.timer.start(duration*1e3)
