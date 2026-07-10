'''
Command handlers for output manifests.

These functions implement the behavior behind the workbench commands
declared in :mod:`psi.controller.output_manifest`. They are kept in a plain
Python module so they can be imported and tested without the Enaml import
machinery.
'''
import logging
log = logging.getLogger(__name__)

from collections import ChainMap, Counter

import numpy as np

from enaml.application import deferred_call

from .token_context import get_parameters, initialize_factory


def set_token(event, output, output_type):
    if not event.parameters['token']:
        log.warning('No token provided for %s.', output.name)
        return

    output.token = event.parameters['token']

    # TODO: This is a hack. Maybe?
    context_plugin = event.workbench.get_plugin('psi.context')
    context_plugin._refresh_items()


def prepare_output(event, output):
    '''
    Set up the factory in preparation for producing the signal. This allows
    the factory to cache some potentially expensive computations in advance
    rather than just before we actually want the signal played.
    '''
    log.debug('Preparing output %s', output.name)
    core = event.workbench.get_plugin('enaml.workbench.core')
    parameters = {'context_names': get_parameters(output, output.token)}
    md = core.invoke_command('psi.context.get_values', parameters)

    context = md.copy()
    context['fs'] = output.fs
    context['calibration'] = output.calibration

    if output.token is not None:
        factory = initialize_factory(output, output.token, context)
        output.source = factory

    output.source_md = md
    return md


def prepare_output_queue(event, output):
    log.info('Setting up queue for {}'.format(output.name))

    selector_name = event.parameters.get('selector_name', 'default')
    iti_key = event.parameters.get('selector_name', f'{output.name}_iti_duration')
    averages_key = event.parameters.get('selector_name', f'{output.name}_averages')

    controller = event.workbench.get_plugin('psi.controller')
    context = event.workbench.get_plugin('psi.context')
    action_name = output.name + '_end'

    # Link a callback that's invoked in the main thread (to avoid race
    # conditions)
    def cb(event):
        log.info('Output queue complete. Invoking %s in MainThread',
                 action_name)
        deferred_call(controller.invoke_actions, action_name)
    output.observe('complete', cb)

    for setting in context.iter_settings(selector_name, 1):
        iti_duration = setting[iti_key]
        averages = setting[averages_key]
        output.add_setting(setting, averages=averages,
                           iti_duration=iti_duration)

    controller.invoke_actions(f'{output.name}_queue_ready')


def prepare_synchronized(synchronized, event):
    log.debug('Preparing synchronized %s', synchronized.name)
    for output in synchronized.outputs:
        prepare_output(event, output)


def start_synchronized(synchronized, event):
    try:
        _start_synchronized(synchronized, event)
    except Exception as e:
        log.exception(e)
        controller = event.workbench.get_plugin('psi.controller')
        controller.invoke_actions('{}_failure'.format(synchronized.name))


def _start_synchronized(synchronized, event):
    log.debug('Starting synchronized %s', synchronized.name)
    controller = event.workbench.get_plugin('psi.controller')

    # First, lock all engines involved to avoid race conditions
    for engine in synchronized.engines:
        engine.lock.acquire()
    log.debug('#> Acquired lock for all engines')

    ts = event.parameters['timestamp']
    start = event.parameters.get('start', ts)
    delay = event.parameters.get('delay', 0)
    clear_delay = event.parameters.get('clear_delay', delay)

    # Is this a hack? If no timestamp is defined, then assume that the start
    # is 0 (e.g., in the case of experiment_prepare).
    if start is None:
        start = controller.get_ts()
        log.debug('No starting time provided. Setting start to %f.', start)
    log.debug('Timestamp %r, start %r, delay %r', ts, start, delay)

    if clear_delay > delay:
        raise ValueError('Delay for clearing buffer cannot be greater than delay')

    # For each engine involved in this synchronized output, store a list of
    # the offsets and channel names that need to be updated.
    settings = {e: {'offsets': [], 'names': []} for e in synchronized.engines}
    notify_end = False
    durations = {}

    # Activate the outputs
    for output in synchronized.outputs:
        offset = round((start+delay)*output.channel.fs)
        offset_delay = round((start+clear_delay)*output.channel.fs)

        if output.active:
            output.deactivate(offset_delay)
            log.debug('Output %s was active. Deactivating first.', output.name)
            notify_end = True

        if not output.is_ready():
            log.debug('Output %s is not ready. Preparing.', output.name)
            prepare_output(event, output)

        durations[output.name] = output.get_duration()

        log.debug('Starting output %s', output.name)
        output.activate(offset)

        setting = settings[output.engine]
        setting['offsets'].append(offset_delay)
        setting['names'].append(output.channel.name)

    # Tell the engines to update the waveform
    for engine, setting in settings.items():
        log.debug('%s update %r', engine.name, setting)
        engine.update_hw_ao_multiple(**setting)

    # Finally, release the lock.
    for engine in synchronized.engines:
        engine.lock.release()

    if notify_end:
        controller.invoke_actions('{}_end'.format(synchronized.name),
                                  start+clear_delay)

    md = dict(ChainMap(*[o.source_md for o in synchronized.outputs]))
    duration = max(durations.values())
    kw = {
        't0': start+delay,
        'duration': duration,
        'key': None,
        'decrement': None,
        'metadata': md
    }
    controller.invoke_actions('{}_start'.format(synchronized.name),
                              start+delay, kw=kw)
    if duration is not np.inf:
        controller.invoke_actions('{}_end'.format(synchronized.name),
                                  start+delay+duration, delayed=True)


def clear_synchronized(synchronized, event):
    end = event.parameters['timestamp']
    delay = event.parameters.get('delay', 0)

    # For each engine involved in this synchronized output, store a list of
    # the offsets and channel names that need to be updated.
    settings = {e: {'offsets': [], 'names': []} for e in synchronized.engines}

    # First, lock all engines involved to avoid race conditions
    log.debug('Locking all engines')
    for engine in synchronized.engines:
        engine.lock.acquire()

    # Activate the outputs
    for output in synchronized.outputs:
        offset = round((end+delay)*output.channel.fs)

        if output.active:
            output.deactivate(offset)
            log.debug('Deactivating output %s', output.name)

        setting = settings[output.engine]
        setting['offsets'].append(offset)
        setting['names'].append(output.channel.name)

    # Tell the engines to update the waveform
    for engine, setting in settings.items():
        engine.update_hw_ao_multiple(**setting)

    # Finally, release the lock.
    for engine in synchronized.engines:
        engine.lock.release()

    controller = event.workbench.get_plugin('psi.controller')
    controller.invoke_actions('{}_end'.format(synchronized.name), end+delay)


def start_output(event, output):
    '''
    clear_delay : The time at which to stop the output if it's currently
    active.
    '''
    ts = event.parameters['timestamp']
    start = event.parameters.get('start', ts)
    delay = event.parameters.get('delay', 0)
    clear_delay = event.parameters.get('clear_delay', delay)

    if clear_delay > delay:
        raise ValueError('Delay for clearing buffer cannot be greater than delay')

    if isinstance(delay, str):
        context = event.workbench.get_plugin('psi.context')
        delay = context.get_value(delay)

    # Is this a hack? If no timestamp is defined, then assume that the start
    # is 0 (e.g., in the case of experiment_prepare).
    if start is None:
        start = 0

    offset = round((start+delay)*output.channel.fs)
    offset_delay = round((start+clear_delay)*output.channel.fs)
    notify_end = False

    try:
        output.engine.lock.acquire()
        # Important to lock the engine when activating the output to prevent
        # race conditions
        log.debug('Starting output %s at %d', output.name, offset)
        if output.active:
            output.deactivate(offset_delay)
            log.debug('Output %s was active. Deactivating first.', output.name)
            notify_end = True

        if not output.is_ready():
            log.debug('Output %s is not ready', output.name)
            prepare_output(event, output)

        # This needs to be done before we update the engine since it gets set
        # to None once all samples have been generated.
        duration = output.get_duration()

        output.activate(offset_delay)
        output.engine.update_hw_ao(output.channel.name, offset)
        output.engine.lock.release()
    except SystemError as e:
        # There wasn't enough time to write the data to the buffer. Need to
        # reset the generator and deactivate the output. The invoking code
        # can then decide what to do about the problem.
        output.deactivate(offset)
        output.engine.lock.release()
        log.exception(e)
        log.debug('Invalidated buffer at %d', offset)
    else:
        filter_delay = output.channel.filter_delay
        log.debug('Compensating %s start event for filter delay %f',
                  output.name, filter_delay)
        controller = event.workbench.get_plugin('psi.controller')

        kw = {'duration': duration, 't0_end': start+delay+duration}
        controller.invoke_actions('{}_start'.format(output.name),
                                  start+delay+filter_delay, kw=kw)
        if duration is not np.inf:
            controller.invoke_actions('{}_end'.format(output.name),
                                      start+delay+duration, delayed=True)

        if notify_end:
            controller.invoke_actions('{}_end'.format(output.name),
                                      start+delay+duration)


def clear_output(event, output):
    end = event.parameters['timestamp']
    delay = event.parameters.get('delay', 0)
    with output.engine.lock:
        log.debug('Clearing output {}'.format(output.name))
        # First, deactivate the output if it's still running
        if not output.active:
            return

        log.debug('Deactivating output {}'.format(output.name))
        output.deactivate()
        # Now, update the channel once the output has been deactivated. This
        # will overwrite any data that contains a fragment of the output
        # waveform.
        offset = round((end+delay)*output.fs)
        output.engine.update_hw_ao(output.channel.name, offset)

    controller = event.workbench.get_plugin('psi.controller')
    controller.invoke_actions('{}_end'.format(output.name), end+delay)


def decrement_key(event, output):
    with output.engine.lock:
        keys = [e.metadata['key'] for e in event.parameters['data']]
        counts = Counter(keys)
        complete_keys = []
        for key, count in counts.items():
            try:
                if output.queue.decrement_key(key, count):
                    complete_keys.append(key)
            except KeyError:
                # This can happen if we are playing epochs so rapidly we
                # gather a few extra before the output queue knows to stop.
                complete_keys.append(key)
    if complete_keys:
        log.debug('Queued keys complete: %r', complete_keys)
        controller = event.workbench.get_plugin('psi.controller')
        controller.invoke_actions(f'{output.name}_keys_complete', keys)


def output_pause(event, output):
    delay = event.parameters.get('delay', 1)
    controller = event.workbench.get_plugin('psi.controller')
    with output.engine.lock:
        time = output.engine.get_ts() + delay
        log.debug('Pausing output %s at %f', output.name, time)
        output.pause(time)
        controller.invoke_actions(output.name + '_paused', time)


def output_resume(event, output):
    delay = event.parameters.get('delay', 1)
    controller = event.workbench.get_plugin('psi.controller')
    with output.engine.lock:
        time = output.engine.get_ts() + delay
        log.debug('Resuming output %s at %f', output.name, time)
        output.resume(time)
        controller.invoke_actions(output.name + '_resumed', time)


def get_tokens(workbench, ttype):
    try:
        plugin = workbench.get_plugin('psi.token')
        if ttype == 'epoch':
            return plugin._epoch_tokens
        elif ttype == 'continuous':
            return plugin._continuous_tokens
    except ValueError:
        return {}


def get_pause_text(contribution, is_paused):
    base = 'Resume' if is_paused else 'Pause'
    return '{} {}'.format(base, contribution.name)


def subscribe_to_queue(event, output):
    controller = event.workbench.get_plugin('psi.controller')
    event_added = f'{output.name}_added'
    event_removed = f'{output.name}_removed'

    def stim_added(info):
        info = info.copy()
        t0 = info.pop('t0')
        controller.invoke_actions(event_added, t0, kw=info)

    def stim_removed(info):
        info = info.copy()
        t0 = info.pop('t0')
        controller.invoke_actions(event_removed, t0, kw=info)

    output.connect(stim_added, 'added')
    output.connect(stim_removed, 'removed')


def toggle_off(event, output):
    output.set_low()


def toggle_on(event, output):
    output.set_high()


def toggle(event, output):
    if event.parameters['state']:
        toggle_on(event, output)
    else:
        toggle_off(event, output)
