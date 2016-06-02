import enum
import threading

import logging
log = logging.getLogger(__name__)

from atom.api import Int, Typed, Unicode
import numpy as np

from .base_plugin import BaseController


class TrialState(enum.Enum):
    '''
    Defines the possible states that the experiment can be in. We use an Enum to
    minimize problems that arise from typos by the programmer (e.g., they may
    accidentally set the state to "waiting_for_nose_poke_start" rather than
    "waiting_for_np_start").

    This is specific to appetitive reinforcement paradigms.
    '''
    waiting_for_np_start = 'waiting for nose-poke start'
    waiting_for_np_duration = 'waiting for nose-poke duration'
    waiting_for_hold_period = 'waiting for hold period'
    waiting_for_response = 'waiting for response'
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
    spout_start = 'spout contact'
    spout_end = 'withdrew from spout'
    to_duration_elapsed = 'timeout over'
    iti_duration_elapsed = 'ITI over'
    trial_start = 'trial start'


class AppetitivePlugin(BaseController):

    trial = Int(0)
    consecutive_nogo = Int(0)
    rng = Typed(np.random.RandomState)
    trial_type = Unicode()
    trial_info = Typed(dict, {})
    trial_state = Typed(TrialState)
    _lock = threading.Lock()
    timer = Typed(object)

    event_map = {
        ('rising', 'np'): Event.np_start,
        ('falling', 'np'): Event.np_end,
        ('rising', 'spout'): Event.spout_start,
        ('falling', 'spout'): Event.spout_end,
    }

    def next_selector(self):
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
        elif score == 'FA':
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
            self.context.apply_changes()
            self.configure_engines()
            self.rng = np.random.RandomState()
            self.context.next_setting(self.next_selector(), save_prior=False)
            self.core.invoke_command('psi.data.prepare')
            self.state = 'running'
            self.trial_state = TrialState.waiting_for_np_start
        except Exception as e:
            # TODO - provide user interface
            raise

    def start_trial(self):
        epoch_output = self._channel_outputs['epoch']['speaker']
        continuous_output = self._channel_outputs['continuous']['speaker']
        target = epoch_output.get_waveform()
        samples = target.shape[-1]
        ts = self.get_ts()+0.25
        fs = continuous_output.channel.engine.ao_fs
        offset = int(fs*ts)
        masker = continuous_output.get_waveform(offset, samples)[0]
        waveform = target+masker
        epoch_output.channel.engine.write_hw_ao(waveform, offset=offset)

        # TODO - the hold duration will include the update delay. Do we need
        # super-precise tracking of hold period or can it vary by a couple 10s
        # to 100s of msec?
        self.trial_state = TrialState.waiting_for_hold_period
        self.start_timer('hold_duration', Event.hold_duration_elapsed)
        self.trial_info['target_start'] = ts
        self.trial_info['target_end'] = ts+(samples/fs)

    def end_trial(self, response):
        if self.trial_type in ('nogo', 'nogo_repeat'):
            self.consecutive_nogo += 1
            if response == 'spout':
                score = 'FA'
            elif response == 'poke':
                score = 'CR'
        else:
            self.consecutive_nogo = 0
            if response == 'spout':
                score = ' HIT'
            elif response == 'poke':
                score = 'MISS'

        self.trial_info.update({
            'response': response,
            'trial_type': self.trial_type,
            'score': score,
            'correct': score in ('CR', 'HIT'),
        })
        self.context.set_values(self.trial_info)
        self.core.invoke_command('psi.data.process_trial')

        # Apply pending changes that way any parameters (such as repeat_FA or
        # go_probability) are reflected in determining the next trial type.
        if self._apply_requested:
            self.context.apply_changes()
            self._apply_requested = False
        selector = self.next_selector()
        self.context.next_setting(selector, save_prior=True)

        if self._pause_requested:
            self.state = 'paused'
            self._pause_requested = False
        else:
            self.start_trial()

    def ao_callback(self, engine_name, channel_names, offset, samples):
        waveforms = []
        for channel_name in channel_names:
            output = self._channel_outputs['continuous'][channel_name]
            waveforms.append(output.get_waveform(offset, samples)[0])
        waveforms = np.r_[waveforms]
        engine = self._engines[engine_name]
        engine.write_hw_ao(waveforms)

    def ai_callback(self, engine_name, samples):
        print 'acquired', samples.shape

    def et_callback(self, engine_name, change, line, event_time):
        event = self.event_map[edge, line]
        self.handle_event(event, event_time)

    def handle_event(self, event, timestamp=None):
        # Ensure that we don't attempt to process several events at the same
        # time. This essentially queues the events such that the next event
        # doesn't get processed until `_handle_event` finishes processing the
        # current one.
        with self._lock:
            # Only events generated by NI-DAQmx callbacks will have a timestamp.
            # Since we want all timing information to be in units of the analog
            # output sample clock, we will capture the value of the sample clock
            # if a timestamp is not provided. Since there will be some delay
            # between the time the event occurs and the time we read the analog
            # clock, the timestamp won't be super-accurate. However, it's not
            # super-important since these events are not reference points around
            # which we would do a perievent analysis. Important reference points
            # would include nose-poke initiation and withdraw, spout contact,
            # sound onset, lights on, lights off. These reference points will be
            # tracked via NI-DAQmx or can be calculated (i.e., we know exactly
            # when the target onset occurs because we precisely specify the
            # location of the target in the analog output buffer).
            if timestamp is None:
                timestamp = self.get_ts()
            print event, timestamp
            self._handle_event(event, timestamp)

    def _handle_event(self, event, timestamp):
        '''
        Give the current experiment state, process the appropriate response for
        the event that occured. Depending on the experiment state, a particular
        event may not be processed.
        '''
        # TODO: log event
        if self.trial_state == TrialState.waiting_for_np_start:
            if event == Event.np_start:
                # Animal has nose-poked in an attempt to initiate a trial.
                self.trial_state = TrialState.waiting_for_np_duration
                self.start_timer('np_duration', Event.np_duration_elapsed)
                # If the animal does not maintain the nose-poke long enough,
                # this value will get overwritten with the next nose-poke.
                self.trial_info['np_start'] = timestamp

        elif self.trial_state == TrialState.waiting_for_np_duration:
            if event == Event.np_end:
                # Animal has withdrawn from nose-poke too early. Cancel the
                # timer so that it does not fire a 'event_np_duration_elapsed'.
                log.debug('Animal withdrew too early')
                self.timer.cancel()
                self.trial_state = TrialState.waiting_for_np_start
            elif event == Event.np_duration_elapsed:
                self.start_trial()

        elif self.trial_state == TrialState.waiting_for_hold_period:
            # All animal-initiated events (poke/spout) are ignored during this
            # period but we may choose to record the time of nose-poke withdraw
            # if it occurs.
            if event == Event.np_end:
                # Record the time of nose-poke withdrawal if it is the first
                # time since initiating a trial.
                log.debug('Animal withdrew during hold period')
                if 'np_end' not in self.trial_info:
                    log.debug('Recording np_end')
                    self.trial_info['np_end'] = timestamp
            elif event == Event.hold_duration_elapsed:
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
                if 'np_end' not in self.trial_info:
                    self.trial_info['np_end'] = timestamp
            elif event == Event.np_start:
                self.trial_info['response_ts'] = timestamp
                self.stop_trial(response='nose poke')
            elif event == Event.spout_start:
                self.trial_info['response_ts'] = timestamp
                self.stop_trial(response='spout contact')
            elif event == Event.response_duration_elapsed:
                self.trial_info['response_ts'] = timestamp
                self.stop_trial(response='no response')

        elif self.trial_state == TrialState.waiting_for_to:
            if event == Event.to_duration_elapsed:
                # Turn the light back on
                #self.engine.set_sw_do('light', 1)
                self.start_timer('iti_duration',
                                 Event.iti_duration_elapsed)
                self.trial_state = TrialState.waiting_for_iti
            elif event in (Event.spout_start, Event.np_start):
                self.timer.cancel()
                self.start_timer('to_duration', Event.to_duration_elapsed)

        elif self.trial_state == TrialState.waiting_for_iti:
            if event == Event.iti_duration_elapsed:
                self.trial_state = TrialState.waiting_for_np_start

    def start_timer(self, variable, event):
        # Even if the duration is 0, we should still create a timer because this
        # allows the `_handle_event` code to finish processing the event. The
        # timer will execute as soon as `_handle_event` finishes processing.
        duration = self.context.get_value(variable)
        self.timer = threading.Timer(duration, self.handle_event, [event])
        self.timer.start()
