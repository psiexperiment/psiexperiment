'''
Command handlers for the experiment manifest.

These functions implement layout and preference persistence plus assorted
window-management commands declared in :mod:`psi.experiment.manifest`. They
are kept in a plain Python module so they can be imported and tested without
the Enaml import machinery.
'''
import logging
log = logging.getLogger(__name__)

import os
import pickle
from pathlib import Path

import yaml

from enaml.application import deferred_call
from enaml.widgets.api import FileDialogEx

from .. import get_config
from .util import LAYOUT_WILDCARD, PREFERENCES_WILDCARD


def get_default_path(which):
    root = get_config('{}_ROOT'.format(which.upper()))
    experiment = get_config('EXPERIMENT')
    default_path = os.path.join(root, experiment)
    if not os.path.exists(default_path):
        os.makedirs(default_path)
    return default_path


def get_default_filename(which):
    default_path = get_default_path(which)
    return os.path.join(default_path, 'default.{}'.format(which))


def save_layout(event):
    filename = FileDialogEx.get_save_file_name(
        name_filters=[LAYOUT_WILDCARD],
        current_path=get_default_path('layout')
    )
    if filename:
        _save_layout(event, filename)


def _save_layout(event, filename):
    if not filename.endswith('.layout'):
        filename += '.layout'
    plugin = event.workbench.get_plugin('psi.experiment')
    layout = plugin.get_layout()
    with open(filename, 'wb') as fh:
        pickle.dump(layout, fh)


def load_layout(event):
    filename = event.parameters.get('filename', None)
    if filename is None:
        filename = FileDialogEx.get_open_file_name(
            name_filters=[LAYOUT_WILDCARD],
            current_path=get_default_path('layout')
        )
    if filename:
        _load_layout(event, filename)


def _load_layout(event, filename):
    plugin = event.workbench.get_plugin('psi.experiment')
    with open(filename, 'rb') as fh:
        layout = pickle.load(fh)
        plugin.set_layout(layout)


def set_default_layout(event):
    filename = get_default_filename('layout')
    _save_layout(event, filename)


def get_default_layout(event):
    try:
        filename = get_default_filename('layout')
        _load_layout(event, filename)
    except IOError:
        pass


def save_preferences(event):
    filename = event.parameters.get('filename', None)
    if filename is None:
        filename = FileDialogEx.get_save_file_name(
            name_filters=[PREFERENCES_WILDCARD],
            current_path=get_default_path('preferences')
        )
    if filename:
        _save_preferences(event, filename)


def _save_preferences(event, filename):
    filename = Path(filename).with_suffix('.preferences')
    plugin = event.workbench.get_plugin('psi.experiment')
    preferences = plugin.get_preferences()
    with open(filename, 'w') as fh:
        yaml.dump(preferences, fh, default_flow_style=False)


def load_preferences(event):
    filename = event.parameters.get('filename', None)
    if filename is None:
        filename = FileDialogEx.get_open_file_name(
            name_filters=[PREFERENCES_WILDCARD],
            current_path=get_default_path('preferences')
        )
    if filename:
        _load_preferences(event, filename)


def _load_preferences(event, filename):
    log.debug('Loading preferences from {}'.format(filename))
    with open(filename, 'r') as fh:
        preferences = yaml.load(fh, Loader=yaml.Loader)
    plugin = event.workbench.get_plugin('psi.experiment')
    plugin.set_preferences(preferences)


def set_default_preferences(event):
    filename = get_default_filename('preferences')
    _save_preferences(event, filename)


def get_default_preferences(event):
    try:
        filename = get_default_filename('preferences')
        _load_preferences(event, filename)
    except IOError:
        pass


def minimize_window(event):
    ui = event.workbench.get_plugin('enaml.workbench.ui')
    deferred_call(ui.window.proxy.widget.showMinimized)


def hide_window(event):
    ui = event.workbench.get_plugin('enaml.workbench.ui')
    deferred_call(ui.window.proxy.widget.hide)


def show_window(event):
    ui = event.workbench.get_plugin('enaml.workbench.ui')
    deferred_call(ui.window.proxy.widget.showNormal)


def set_dock_style(event):
    ui = event.workbench.get_plugin('enaml.workbench.ui')
    ui.workspace.dock_area.style = event.parameters['style_name']


def update_dock_style(event):
    ui = event.workbench.get_plugin('enaml.workbench.ui')
    style = 'complete' if event.parameters.get('stop_reason', '') == '' else 'error'
    ui.workspace.dock_area.style = style


def write_metadata(event):
    store = event.workbench.get_plugin('psi.data').find_sink('metadata')
    experiment = event.workbench.get_plugin('psi.experiment')
    store.save_mapping('metadata', experiment.metadata_to_dict())
