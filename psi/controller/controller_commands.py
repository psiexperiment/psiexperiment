'''
Command handlers for the controller manifest.

These functions implement the behavior behind the workbench commands
declared in :mod:`psi.controller.manifest`. They are kept in a plain Python
module so they can be imported and tested without the Enaml import
machinery.
'''
import logging
log = logging.getLogger(__name__)

from ..util import get_tagged_values, log_with_header


def accumulate_actions(plugin):
    accumulator = {}
    for action in plugin._actions:
        a = accumulator.setdefault(action.event, [])
        info = '{} (weight={})'.format(action, action.weight)
        a.append(info)
    lines = []
    for event, actions in accumulator.items():
        lines.append(event)
        for action in actions:
            lines.append('  ' + action)
    return lines


def log_actions(event):
    plugin = event.workbench.get_plugin('psi.controller')
    log_with_header('Configured actions', accumulate_actions(plugin))


def log_events(event):
    plugin = event.workbench.get_plugin('psi.controller')
    events = {}
    max_base = 0
    for e in sorted(plugin._events):
        base, extra = e.split('_', 1)
        max_base = max(len(base), max_base)
        events.setdefault(base, []).append(extra)

    lines = []
    for event_base, event_list in events.items():
        event_list = ','.join(event_list)
        event_base = event_base.ljust(max_base)
        lines.append(f" * {event_base}_{{{event_list}}}")
    log_with_header('Available events', lines)


def accumulate_io_tree(plugin):
    accumulator = ['Output']
    for channel in plugin.get_channels(direction='output'):
        _accumulate_io_tree(accumulator, channel, 'outputs', prefix='')
    accumulator.append('Input')
    for channel in plugin.get_channels(direction='input'):
        _accumulate_io_tree(accumulator, channel, 'inputs', prefix='')
    return accumulator


def _accumulate_io_tree(accumulator, obj, direction, prefix=''):
    md = get_tagged_values(obj, 'metadata')
    exclude = {'engine', 'outputs', 'inputs', 'channel', 'source_name',
               'calibration', 'source', 'force_active', 'unit', 'dtype'}
    md_string = ', '.join(f'{k}={v}' for k, v in md.items() if k not in exclude)
    accumulator.append(f'{prefix} * {obj._default_name()} ({md_string})')
    for i in getattr(obj, direction, []):
        _accumulate_io_tree(accumulator, i, direction, f'{prefix} |')


def log_io(event):
    plugin = event.workbench.get_plugin('psi.controller')
    log_with_header('IO configuration', accumulate_io_tree(plugin))


def log_commands(event):
    plugin = event.workbench.get_plugin('enaml.workbench.core')
    commands = [f' * {s}' for s in sorted(plugin._commands.keys())]
    log_with_header('Available commands', commands)


def start_experiment(event):
    controller = event.workbench.get_plugin('psi.controller')
    controller.start_experiment()


def stop_experiment(event):
    # Todo: this does not capture *everything* possible (e.g., stopping
    # experiment leads to a cascade of actions).
    error_message = event.parameters.get('error_message', '')
    stop_reason = event.parameters.get('stop_reason', '')
    skip_errors = event.parameters.get('skip_errors', True)
    controller = event.workbench.get_plugin('psi.controller')
    kw = {
        'error_message': error_message,
        'stop_reason': stop_reason,
    }
    results = controller.stop_experiment(skip_errors=skip_errors, kw=kw)
    if results is not None:
        # Deduplicate the messages returned by the experiment_end actions.
        # Note: this previously called controller.stop_experiment a second
        # time; since the experiment was already stopped, that returned an
        # empty list and the wrapup message was always lost.
        messages = {r.strip() for r in results if isinstance(r, str)}
        mesg = '\n'.join(sorted(messages))
        controller._wrapup(message=mesg, stop_reason=stop_reason,
                           error_message=error_message)


def invoke_actions(event):
    controller = event.workbench.get_plugin('psi.controller')
    event_name = event.parameters['event_name']
    timestamp = event.parameters['timestamp']
    controller.invoke_actions(event_name, timestamp)


def get_hw_ao_choices(workbench):
    plugin = workbench.get_plugin('psi.controller')
    channels = plugin.get_channels('analog', 'output', 'hardware', False)
    return {f'{c.label}': f'"{c.reference}"' for c in channels}


def get_hw_ai_choices(workbench):
    plugin = workbench.get_plugin('psi.controller')
    channels = plugin.get_channels('analog', 'input', 'hardware', False)
    return {f'{c}': f'"{c.reference}"' for c in channels}
