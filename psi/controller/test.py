
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


class Controller(
        PositiveCMRControllerMixin,
        AbstractExperimentController,
        CLControllerMixin,
        #PumpControllerMixin,
        ):
    '''
    Controls experiment logic (i.e. communicates with the TDT hardware,
    responds to input by the user, etc.).
    '''
    random_generator = Any
    random_seed = Int
    remind_requested = Bool

    # Track the current state of the experiment. How the controller responds to
    # events will depend on the state.
    trial_state = Instance(TrialState, TrialState.waiting_for_np_start)

    def _get_status(self):
        return self.trial_state.value

    preload_samples = 200000*5
    update_delay = 100000

    _lock = threading.Lock()
    engine = Instance('daqengine.ni.Engine')

    fs = 100e3

    def start_trial(self):
        # Get the current position in the analog output buffer, and add a cetain
        # update_delay (to give us time to generate and upload the new signal).
        ts = self.get_ts()
        offset = int(round(ts*self.fs)) + self.update_delay
        log.debug('Inserting target at %d', offset)
        # TODO - should be able to calculate a precise duration.
        duration = self.engine.ao_write_space_available(offset)/10
        log.debug('Overwriting %d samples in buffer', duration)

        masker_sf = self.get_current_value('masker_sf')
        target_sf = self.get_current_value('target_sf')

        # Generate combined signal
        signal = self.get_masker(offset, duration)*masker_sf
        target = self.get_target()*target_sf

        signal[:target.shape[-1]] += target
        self.engine.write_hw_ao(signal, offset)
        self._masker_offset = offset + signal.shape[-1]

        # TODO - the hold duration will include the update delay. Do we need
        # super-precise tracking of hold period or can it vary by a couple 10s
        # to 100s of msec?
        self.trial_state = TrialState.waiting_for_hold_period
        self.start_timer('hold_duration', Event.hold_duration_elapsed)
        self.trial_info['target_start'] = ts
        self.trial_info['target_end'] = ts+duration/self.fs

    def stop_trial(self, response):
        trial_type = self.get_current_value('ttype')
        if response != 'no response':
            self.trial_info['response_time'] = \
                self.trial_info['response_ts']-self.trial_info['target_start']
        else:
            self.trial_info['response_time'] = np.nan

        self.trial_info['reaction_time'] = \
            self.trial_info.get('np_end', np.nan)-self.trial_info['np_start']

        if trial_type in ('GO', 'GO_REMIND'):
            score = 'HIT' if response == 'spout contact' else 'MISS'
        elif trial_type in ('NOGO', 'NOGO_REPEAT'):
            score = 'FA' if response == ' spout contact' else 'CR'

        if score == 'FA':
            # Turn the light off
            #self.engine.set_sw_do('light', 0)
            self.start_timer('to_duration', Event.to_duration_elapsed)
            self.trial_state = TrialState.waiting_for_to
        else:
            if score == 'HIT':
                #self.engine.fire_sw_do('pump', 0.2)
                pass
            self.start_timer('iti_duration', Event.iti_duration_elapsed)
            self.trial_state = TrialState.waiting_for_iti

        print(self.trial_info)
        self.log_trial(score=score, response=response, ttype=trial_type,
                       **self.trial_info)
        self.trigger_next()

    event_map = {
        ('rising', 'np'): Event.np_start,
        ('falling', 'np'): Event.np_end,
        ('rising', 'spout'): Event.spout_start,
        ('falling', 'spout'): Event.spout_end,
    }

    def et_fired(self, edge, line, timestamp):
        # The timestamp is the number of analog output samples that have been
        # generated at the time the event occured. Convert to time in seconds
        # since experiment start.
        timestamp /= self.fs
        log.debug('detected {} edge on {} at {}'.format(edge, line, timestamp))
        event = self.event_map[edge, line]
        self.handle_event(event, timestamp)

    def handle_event(self, event, timestamp=None):
        # Ensure that we don't attempt to process several events at the same
        # time. This essentially queues the events such that the next event
        # doesn't get processed until `_handle_event` finishes processing the
        # current one.
        with self._lock:
            # Only events generated by NI-DAQmx callbacks will have a timestamp.
            # Since we want all timing information to be in units of the analog
            # output sample clock, we will capture the value of the sample
            # clock if a timestamp is not provided. Since there will be some
            # delay between the time the event occurs and the time we read the
            # analog clock, the timestamp won't be super-accurate. However, it's
            # not super-important since these events are not reference points
            # around which we would do a perievent analysis. Important reference
            # points would include nose-poke initiation and withdraw, spout
            # contact, sound onset, lights on, lights off. These reference
            # points will be tracked via NI-DAQmx or can be calculated (i.e., we
            # know exactly when the target onset occurs because we precisely
            # specify the location of the target in the analog output buffer).
            if timestamp is None:
                timestamp = self.get_ts()
            self._handle_event(event, timestamp)

    def _handle_event(self, event, timestamp):
        '''
        Give the current experiment state, process the appropriate response for
        the event that occured. Depending on the experiment state, a particular
        event may not be processed.
        '''
        self.model.data.log_event(timestamp, event.value)

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
        duration = self.get_value(variable)
        self.timer = threading.Timer(duration, self.handle_event, [event])
        self.timer.start()

    def get_masker(self, masker_offset, masker_duration):
        '''
        Get the next `duration` samples of the masker starting at `offset`. If
        reading past the end of the array, loop around to the beginning.
        '''
        masker_size = self.masker.shape[-1]
        offset = masker_offset % masker_size
        duration = masker_duration
        result = []
        while True:
            if (offset+duration) < masker_size:
                subset = self.masker[offset:offset+duration]
                duration = 0
            else:
                subset = self.masker[offset:]
                offset = 0
                duration = duration-subset.shape[-1]
            result.append(subset)
            if duration == 0:
                break
        return np.concatenate(result, axis=-1)

    def get_target(self):
        return self.target

    def get_ts(self):
        return self.engine.ao_sample_clock()/self.fs
