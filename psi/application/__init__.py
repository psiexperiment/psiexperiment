import logging.config
log = logging.getLogger(__name__)

from contextlib import contextmanager
import datetime as dt
from glob import glob
import importlib
import os
import os.path
from pathlib import Path
import pdb
import re
import sys
import traceback
import warnings

import enaml
from enaml.application import deferred_call
with enaml.imports():
    from enaml.stdlib.message_box import critical

from psi import get_config, set_runtime
from psi.util import wrap_text
from psi.core.enaml.api import load_manifest, load_manifest_from_file


def disable_quick_edit():
    # From https://stackoverflow.com/questions/73486528/python-script-pausing-in-cmd
    import win32console as con
    import signal

    # Missing constants in pywin
    ENABLE_EXTENDED_FLAGS = 0x0080
    ENABLE_QUICK_EDIT_MODE = 0x0040

    # Modify console mode to disable quick edit mode
    h = con.GetStdHandle(con.STD_INPUT_HANDLE)
    oldMode = h.GetConsoleMode()
    h.SetConsoleMode((oldMode | ENABLE_EXTENDED_FLAGS) &
            ~ENABLE_QUICK_EDIT_MODE)


def setup_windows_console():
    '''
    Disable Windows quick-edit mode so that clicking in the console does not
    accidentally pause the application. Called from the CLI entry points;
    importing this module has no side effects.
    '''
    if os.name == 'nt':
        try:
            disable_quick_edit()
        except Exception:
            pass


def set_app_id(app_id):
    '''
    Give this process its own identity on the Windows taskbar.

    Windows groups taskbar buttons by AppUserModelID, and a Python GUI that
    never sets one inherits the interpreter's. Without this every psi program
    shares a single taskbar button showing Python's icon (or the console-script
    wrapper's), no matter what icon its windows carry.

    Deliberately duplicates `psiapp.util.set_app_id`, which is the one the
    launchers (cftscal, noise-exp, cfts) call. psiapp is built on psi rather
    than the other way around, so importing it here would point the dependency
    backwards for the sake of eight lines. Keep the two in sync.

    Parameters
    ----------
    app_id : string
        Dotted identifier, by convention `psi.<program>`. `psi` itself claims
        `psi.psi`, leaving the launchers that spawn it free to claim their own
        so that a launcher and its experiments get separate taskbar buttons.

    Notes
    -----
    Call this from the CLI entry point before the Qt application is created.
    Once a window exists Windows has already bound the process to the default
    ID and this has no effect.

    No-op off Windows, and fails soft: a mis-grouped taskbar button is cosmetic
    and shouldn't keep the program from starting.
    '''
    if os.name != 'nt':
        return
    import ctypes
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:
        log.warning('Unable to set the AppUserModelID to %r', app_id,
                    exc_info=True)


mesg_template = '''
A critical exception has occurred. While we do our best to prevent these
issues, they sometimes happen. We are now attempting to shut down the program
gracefully so acquired data can be saved. Please notify the developers.

{}

{}
'''


class ExceptionHandler:
    '''
    This provides a custom exception handler. Since, during the course of
    custom exception handling, new exceptions may be raised, we temporarily
    restore the default exception handling on __enter__ and then restore the
    custom exception handler on __exit__. By using the context manager, we make
    sure that __exit__ always gets called properly.
    '''

    def __init__(self):
        self.workbench = None
        self.logfile = None
        self.stopping = False

    def __enter__(self):
        sys.excepthook = sys.__excepthook__

    def __exit__(self, exc_type, exc_value, exc_tb):
        sys.excepthook = self

    def format_exception(self, args):
        if self.logfile is not None:
            log_mesg = f'The log file has been saved to {self.logfile}'
        else:
            log_mesg = 'Unfortunately, no log file was saved.'

        err_mesg = f'The error message is:\n{args[1]}'
        if args[1].__cause__ is not None:
            err_mesg = f'{err_mesg}\n\nThe above error was caused by ' \
                        f'the following error:\n{args[1].__cause__}'

        # Collapse each paragraph onto a single line (keeping the blank
        # lines that separate them) and leave it at that. This message is
        # only ever shown in the GUI, which wraps it to the width of the
        # window it is displayed in; hard-wrapping it here as well (via
        # wrap_text, which is meant for console output) would make it wrap
        # at 70 columns no matter how wide that window is.
        mesg = mesg_template.format(args[1], log_mesg)
        mesg = re.sub(r'(?<!\n)\n(?!\n)', ' ', mesg)
        mesg = re.sub(r' +', ' ', mesg)
        return mesg.strip()

    def __call__(self, *args):
        with self:
            log.exception("Uncaught exception", exc_info=args)
            mesg = self.format_exception(args)
            tb_text = ''.join(traceback.format_exception(*args))

            if self.workbench is not None:
                core = self.workbench.get_plugin('enaml.workbench.core')
                parameters = {'stop_reason': 'error', 'skip_errors': True,
                              'error_message': mesg, 'traceback': tb_text}
                if not self.stopping:
                    try:
                        self.stopping = True
                        log.info('Invoking stop command')
                        core.invoke_command('psi.set_dock_style', {'style_name': 'error'})
                        core.invoke_command('psi.controller.stop', parameters)
                    except Exception as e:
                        log.exception(e)
                        window = self.workbench.get_plugin('enaml.workbench.ui').window
                        deferred_call(critical, window, 'Oops :(', mesg)
            sys.excepthook(*args)


exception_handler = ExceptionHandler()


def install_exception_handler():
    '''
    Install psiexperiment's exception handler as sys.excepthook so that
    uncaught exceptions attempt a graceful experiment shutdown (saving
    acquired data) before the process dies.

    Called automatically by launch_experiment and the CLI entry points.
    Programs that embed psiexperiment without going through those paths and
    want this behavior must call it explicitly; importing psi.application
    no longer installs the hook as a side effect.
    '''
    sys.excepthook = exception_handler


def install_qt_message_handler():
    '''
    Install a Qt message handler that treats Qt-level warnings/errors (e.g.
    "QObject::setParent: Cannot set parent, new parent is in a different
    thread") as hard failures, with a full Python stack trace attached.

    Qt's own warnings are emitted from C++ with no Python context, so by
    default they just print to stderr and execution continues -- easy to
    scroll past even though many of them (cross-thread widget/graphics-item
    access chief among them) indicate a real, silently-corrupting bug rather
    than something safe to ignore. This is what caught the DPOAE IO
    black-canvas-on-resize bug (a missing `deferred_call`). `QtDebugMsg`/
    `QtInfoMsg` are logged only; `QtWarningMsg` and above are additionally
    routed through psiexperiment's own exception handler (see
    install_exception_handler) so they get the same graceful-shutdown
    treatment as any other uncaught exception.

    Called automatically by launch_experiment.
    '''
    import traceback
    from enaml.qt.QtCore import QtMsgType, qInstallMessageHandler

    def _handler(msg_type, context, message):
        stack = ''.join(traceback.format_stack(limit=20))
        log.error('[QT MESSAGE] %s\n%s', message, stack)
        if msg_type in (QtMsgType.QtDebugMsg, QtMsgType.QtInfoMsg):
            return
        try:
            raise RuntimeError(f'Qt error: {message}')
        except RuntimeError:
            sys.excepthook(*sys.exc_info())

    qInstallMessageHandler(_handler)


def configure_logging(level_console=None, level_file=None, filename=None,
                      debug_exclude=None):

    logging.captureWarnings(True)
    log = logging.getLogger()

    if level_file is None and level_console is None:
        return
    elif level_file is None:
        log.setLevel(level_console.upper())
    elif level_console is None:
        log.setLevel(level_file.upper())
    else:
        level_console = getattr(logging, level_console.upper())
        level_file = getattr(logging, level_file.upper())
        min_level = min(level_console, level_file)
        log.setLevel(min_level)

    fmt = '{asctime:s} {levelname:10s}: {threadName:11s} - {name:40s}:: {message}'

    formatter = logging.Formatter(fmt, style='{')
    if level_console is not None:
        try:
            level_styles = {
                'trace': dict(color='cyan'),
                'debug': dict(color='green'),
                'info': dict(color='white'),
                'warning': dict(color='yellow'),
                'error': dict(color='magenta'),
                'critical': dict(color='red'),
            }
            import coloredlogs
            import humanfriendly
            humanfriendly.terminal.enable_ansi_support()
            formatter = coloredlogs.ColoredFormatter(fmt, style='{',
                                                     level_styles=level_styles)
        except ImportError:
            formatter = logging.Formatter(fmt, style='{')

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(level_console)
        log.addHandler(stream_handler)

    if filename is not None and level_file is not None:
        formatter = logging.Formatter(fmt, style='{')
        file_handler = logging.FileHandler(filename, 'w', 'UTF-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level_file)
        log.addHandler(file_handler)
        exception_handler.logfile = filename
        # Publish the logfile location so lower-level plugins (e.g., the
        # Logger sink) can find it without importing psi.application.
        set_runtime('LOG_FILENAME', filename)

    if debug_exclude is not None:
        for name in debug_exclude:
            logging.getLogger(name).setLevel('CRITICAL')
    tdt_logger = logging.getLogger('tdt')
    tdt_logger.setLevel('INFO')
    websockets_logger = logging.getLogger('websockets')
    websockets_logger.setLevel('INFO')
    install_exception_handler()


def warn_with_traceback(message, category, filename, lineno, file=None,
                        line=None):
    log = file if hasattr(file,'write') else sys.stderr
    m = warnings.formatwarning(message, category, filename, lineno, line)
    log.write(m)


def _main(args):
    set_runtime('EXPERIMENT', args.experiment)

    if args.debug:
        # Show debugging information. This includes full tracebacks for
        # warnings.
        dt_string = dt.datetime.now().strftime('%Y-%m-%d %H%M')
        filename = '{} {}'.format(dt_string, args.experiment)
        log_root = Path(get_config('PSI_LOG_ROOT'))
        log_root.mkdir(parents=True, exist_ok=True)
        log_file = os.path.join(log_root, filename)
        configure_logging(args.debug_level_console,
                          args.debug_level_file,
                          log_file,
                          args.debug_exclude)

        log.debug('Logging configured')
        log.info('Logging information captured in %s', log_file)
        log.info('Python executable: %s', sys.executable)
        if args.debug_warning:
            warnings.showwarning = warn_with_traceback

    from psi.application.workbench import PSIWorkbench

    workbench = PSIWorkbench()
    plugins = [p.manifest for p in args.controller.plugins \
               if p.selected or p.required]
    workbench.register_core_plugins(args.io, plugins)

    if args.pathname is None:
        log.warning('All data will be destroyed at end of experiment')

    exception_handler.workbench = workbench
    workbench.start_workspace(args.experiment,
                              args.pathname,
                              commands=args.commands,
                              load_preferences=not args.no_preferences,
                              load_layout=not args.no_layout,
                              preferences_file=args.preferences,
                              layout_file=args.layout
                              )


# Re-exported for backwards compatibility; the implementation lives in
# psi.experiment.util to respect the package layering.
from psi.experiment.util import list_preferences  # noqa: E402,F401


def list_io():
    result = []
    result.extend(Path(get_config('PSI_IO_ROOT')).glob('*.enaml'))
    result.extend(get_config('PSI_STANDARD_IO'))
    return result


def launch_experiment(args):
    install_exception_handler()
    install_qt_message_handler()
    setup_windows_console()
    set_app_id('psi.psi')
    set_runtime('ARGS', args)
    set_runtime('PROFILE', args.profile)
    if args.profile:
        import cProfile, pstats
        pr = cProfile.Profile()
        pr.enable()

    try:
        _main(args)
    except Exception as e:
        if args.pdb:
            type, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
        else:
            log.exception(e)
            critical(None, 'Error starting experiment', str(e))

    if args.profile:
        pr.disable()
        path = get_config('PSI_LOG_ROOT') / 'main_thread.pstat'
        pr.dump_stats(path)
        stat_files = [str(p) for p in path.parent.glob('*.pstat')]
        merged_stats = pstats.Stats(*stat_files)
        merged_stats.dump_stats(path.parent / 'merged.pstat')


def get_default_io(method='hostname'):
    '''
    Attempt to figure out the default IO configuration file

    Parameters
    ----------
    method : {'hostname'}
        If 'hostname', returns the IO config matching the full hostname.
    '''
    mesg = f'''
    {{}}

    Please create an IO config file. This file should go in
    {get_config('PSI_IO_ROOT')}. The location of the IO config files can be set via
    the `PSI_IO_ROOT` environment variable.
    '''
    available_io = list_io()
    log.debug('Found the following IO files: %r', available_io)
    if method == 'hostname':
        hostname = get_config('PSI_HOSTNAME').lower()
        for io in available_io:
            if hostname in str(io):
                return io
        else:
            err = f'No IO named {hostname}.enaml found for the system.'
            raise ValueError(wrap_text(mesg.format(err)))
    else:
        raise ValueError('Unsupported method')


class IOManifestError(ValueError):
    '''
    Raised when the hardware IO configuration cannot be loaded.

    Subclasses `ValueError` rather than `Exception` on purpose. The errors it
    replaces are overwhelmingly `ValueError` (that is what `sounddevice`
    raises for an unknown device), and callers already probe for absent
    hardware by catching `ValueError` around a manifest load -- see cftscal's
    `list_inputs`/`list_outputs`/`list_connections`, whose `raise_error=False`
    path lets its plugin manifest decide which plugins a machine can offer.
    Narrowing the base class would turn those graceful degradations into
    crashes.

    The IO manifest is the only part of the startup sequence that is specific
    to an individual rig, so a failure here is nearly always a configuration
    problem (hardware that is no longer connected, a device that has been
    renamed, a typo in the manifest) rather than a bug. The exception raised by
    the underlying library is usually useless on its own for tracking that down
    -- `sounddevice`, for example, raises a bare ``ValueError: No input/output
    device matching 'FrontMic'`` that never mentions which file named
    `FrontMic` -- so the message built by `format_io_manifest_error` names the
    manifest that was loaded, how it was selected, and what else is available.
    '''


def _exception_involves(exc, module_name, _seen=None):
    '''
    True if any frame in `exc`'s traceback (or that of a chained exception)
    belongs to the top-level package `module_name`.
    '''
    if _seen is None:
        _seen = set()
    if exc is None or id(exc) in _seen:
        return False
    _seen.add(id(exc))
    tb = exc.__traceback__
    while tb is not None:
        name = tb.tb_frame.f_globals.get('__name__', '')
        if name.split('.')[0] == module_name:
            return True
        tb = tb.tb_next
    return _exception_involves(exc.__cause__, module_name, _seen) \
        or _exception_involves(exc.__context__, module_name, _seen)


def list_sound_devices():
    '''
    Describe the sound devices PortAudio can currently see.

    Only inspects `sounddevice` if it has already been imported, so that
    generating an error message never has the side effect of initializing
    PortAudio (which is slow and, on some systems, noisy).

    Returns
    -------
    devices : list of string
        One human-readable entry per device. Empty if `sounddevice` is not
        loaded or the query fails.
    '''
    sd = sys.modules.get('sounddevice')
    if sd is None:
        return []
    try:
        hostapis = sd.query_hostapis()
        return [
            '{!r} ({}, {} in, {} out)'.format(
                d['name'],
                hostapis[d['hostapi']]['name'],
                d['max_input_channels'],
                d['max_output_channels'],
            ) for d in sd.query_devices()
        ]
    except Exception as e:
        log.debug('Could not query sound devices: %r', e)
        return []


def _resolve_io_manifest_reference(io_manifest):
    '''
    Split an IO manifest reference into the pieces worth reporting.

    Returns
    -------
    source : string
        The file or module that declares the manifest.
    klass : string
        Name of the manifest class within it.
    is_file : bool
        True for a `.enaml` file the user maintains, False for a manifest
        provided by an installed package. The distinction matters for the
        advice we give: "edit this file" is only correct for the former.
    '''
    io_path, _, io_class = str(io_manifest).partition('::')
    if io_path.endswith('.enaml'):
        return io_path, io_class or 'IOManifest', True
    module, _, klass = io_path.rpartition('.')
    # Report the module's file when it has already been imported -- the dotted
    # path alone is not something most users can turn into a location on disk,
    # and we can't ask importlib without paying for the import.
    mod = sys.modules.get(module)
    source = getattr(mod, '__file__', None) or module
    return source, klass, False


def _describe_sound_device_env():
    '''
    Describe the `PSI_SOUND_DEVICE_*` overrides, when they are in play.

    `AutoSoundCardEngine` takes its device and sampling rate from these
    environment variables rather than from anything written in an IO manifest,
    so when they are set they -- not the manifest -- are what has to change.
    They are set by the launching application (cftscal, and the tools built on
    it such as noise-exp, set them from their saved sound card selection), so
    a user looking at the manifest alone has no way to discover them.

    Returns
    -------
    sections : list of string
        Empty if the variables are not set.
    '''
    env = {k: os.environ[k] for k
           in ('PSI_SOUND_DEVICE_NAME', 'PSI_SOUND_DEVICE_FS')
           if k in os.environ}
    if not env:
        return []

    width = max(len(k) for k in env) + 1
    listing = '\n'.join(f'    {k + ":":<{width}} {v!r}' for k, v in env.items())
    sections = [
        wrap_text('''
            The sound device is being set by environment variables rather than
            by the IO configuration itself. These take priority, so this is
            the device that has to be connected:
            ''') + '\n\n' + listing
    ]

    advice = '''
        These are set by the application that launched this one. If that was
        cftscal (or a tool built on it, such as noise-exp), the value comes
        from the sound card selected in its hardware settings -- change the
        selection there to a device that is currently connected.
        '''
    try:
        from psi import get_config_file
        config_file = get_config_file()
        if config_file.exists():
            advice = advice.rstrip() + (
                f' The saved selection is in {config_file}, under'
                ' CFTSCAL_DEVICE_NAME.')
    except Exception as e:
        log.debug('Could not locate the configuration file: %r', e)
    sections.append(wrap_text(advice))
    return sections


def format_io_manifest_error(io_manifest, exc):
    '''
    Build the message for `IOManifestError`.

    Parameters
    ----------
    io_manifest : {str, Path}
        IO manifest reference that was being loaded, i.e., the value handed to
        `load_io_manifest` *after* the default has been resolved (so that the
        message names an actual file rather than `None`).
    exc : Exception
        Exception raised while loading or instantiating the manifest.

    Returns
    -------
    message : string
    '''
    source, klass, is_file = _resolve_io_manifest_reference(io_manifest)
    # get_config coerces a path setting to Path regardless of whether the
    # value came from the config file, the environment or the default, so
    # the mixed separators Windows routinely produces are already
    # normalized into something the user can paste.
    io_root = get_config('PSI_IO_ROOT')
    hostname = get_config('PSI_HOSTNAME')

    sections = [wrap_text('''
        Unable to load the hardware IO configuration. The IO configuration
        (also known as the IO manifest) describes the hardware attached to
        this particular system, so this is usually a configuration problem
        (e.g., hardware that is no longer connected or that has been renamed)
        rather than a problem with the experiment itself.
        ''')]

    detail = [
        ('IO configuration', source),
        ('Manifest class', klass),
        ('Error', f'{type(exc).__name__}: {exc}'),
    ]
    width = max(len(label) for label, _ in detail) + 1
    sections.append('\n'.join(f'    {label + ":":<{width}} {value}'
                              for label, value in detail))

    if is_file:
        sections.append(wrap_text('''
            Open the IO configuration listed above and check the hardware it
            declares (device names, channel numbers, sampling rates). Either
            connect the hardware it expects or edit the file so it matches the
            hardware that is currently attached to this system.
            '''))
        sections.append(wrap_text(f'''
            If that is not the IO configuration you expected, it is either the
            one passed via the `--io` command-line option or the one
            auto-detected by matching this computer's hostname ({hostname!r})
            against the IO configurations in {io_root}. Set the `PSI_IO_ROOT`
            environment variable to search a different folder.
            '''))
        try:
            available = [str(p) for p in list_io()]
        except Exception as e:
            log.debug('Could not list IO configurations: %r', e)
            available = []
        if available:
            sections.append('IO configurations available on this system:\n'
                            + '\n'.join(f'    {p}' for p in available))
    else:
        sections.append(wrap_text('''
            This IO configuration is provided by an installed package rather
            than by a file you maintain, so it is not the thing to edit. It
            was chosen by whatever launched this program -- either the `--io`
            command-line option or the hardware settings of the application
            that started it.
            '''))

    sections.extend(_describe_sound_device_env())

    if _exception_involves(exc, 'sounddevice'):
        devices = list_sound_devices()
        if devices:
            sections.append(
                wrap_text('''
                    The error came from the sound card driver. The device the
                    IO configuration asks for must exactly match one of the
                    sound devices currently visible to this computer:
                    ''')
                + '\n' + '\n'.join(f'    {d}' for d in devices)
            )
        else:
            sections.append(wrap_text('''
                The error came from the sound card driver, and no sound
                devices are currently visible to this computer. Check that the
                sound card is connected and powered on. To list the devices
                yourself, run `python -m sounddevice`.
                '''))

    return '\n\n'.join(sections)


@contextmanager
def io_manifest_errors(io_manifest):
    '''
    Re-raise anything that goes wrong as an `IOManifestError`

    Wraps the whole of loading *and* registering the IO manifest, not just the
    import: the manifest is Enaml, so the expressions that actually touch the
    hardware (`sd.query_devices(device_name)` and friends) are evaluated when
    the manifest is instantiated and its extensions are resolved, not when the
    file is imported.

    Parameters
    ----------
    io_manifest : {str, Path}
        IO manifest reference being loaded, with the default already resolved.
    '''
    try:
        yield
    except IOManifestError:
        raise
    except Exception as exc:
        raise IOManifestError(format_io_manifest_error(io_manifest, exc)) \
            from exc


def load_io_manifest(io_manifest=None):
    '''
    Load the IOManifest from the specified file or module

    Parameters
    ----------
    io_manifest : {str, Path, None}
        If a path (ending in `.enaml`), load the `IOManifest` from the file
        specified by the path. If a module (e.g., `psilbhb.io.badger`) load the
        `IOManifest` from the module. If None, loads the manifest returned by
        `get_default_io`.

    Returns
    -------
    io_manifest : IOManifest
        IOManifest class.

    Raises
    ------
    IOManifestError
        If the manifest cannot be imported. Note that the manifest is not
        instantiated here, so hardware that the manifest declares is not
        touched until the returned class is called -- wrap that call in
        `io_manifest_errors` so those failures are reported the same way.
    '''
    if io_manifest is None:
        io_manifest = get_default_io()
    # Coerce Path (or any path-like) to str for the checks below.
    io_manifest = str(io_manifest)
    # Split off an optional '::ClassName' suffix *before* checking for
    # '.enaml' -- the suffix is what makes this a file-based reference in
    # the first place, so checking the un-split string would never match
    # once a class name is appended (e.g. 'foo.enaml::IOManifest' does not
    # itself end in '.enaml').
    io_path, sep, io_class = io_manifest.partition('::')
    with io_manifest_errors(io_manifest):
        if io_path.endswith('.enaml'):
            klass = load_manifest_from_file(io_path, io_class or 'IOManifest')
        else:
            klass = load_manifest(io_manifest)
    return klass


def initialize_io_manifest(io_manifest=None):
    '''
    Load *and* instantiate the IOManifest

    Prefer this over `load_io_manifest(...)()`. The IO manifest is Enaml, so
    the expressions that actually reach for the hardware (e.g., the
    `sd.query_devices(device_name)` behind `AutoSoundCardEngine`) run when the
    manifest is instantiated, not when it is imported. Splitting the two leaves
    the interesting failure outside `load_io_manifest`, where it surfaces as
    whatever the driver raised (e.g., a bare "No input/output device matching
    'ASIO Fireface USB, ASIO'") with no indication of which configuration
    named that device.

    Parameters
    ----------
    io_manifest : {str, Path, None}
        See `load_io_manifest`.

    Returns
    -------
    io_manifest : IOManifest
        IOManifest instance.

    Raises
    ------
    IOManifestError
        If the manifest cannot be imported or instantiated.
    '''
    if io_manifest is None:
        # Resolve up front so the error message can name a real file.
        io_manifest = get_default_io()
    with io_manifest_errors(io_manifest):
        return load_io_manifest(io_manifest)()


def load_paradigm_descriptions():
    '''
    Loads paradigm descriptions
    '''
    from psi.experiment.api import ParadigmDescription

    default = list_paradigm_descriptions()
    descriptions = get_config('PSI_PARADIGM_DESCRIPTIONS')
    for description in descriptions:
        importlib.import_module(description)


def list_io_templates():
    io_template_path = Path(__file__).parent.parent / 'templates' / 'io'
    return list(io_template_path.glob('*.enaml'))


def list_paradigm_descriptions():
    '''
    List default paradigms descriptions provided by psiexperiment

    Returns
    -------
    modules : list of strings
        List of strings identifying the module path for the description
    '''
    paradigm_path = Path(__file__).parent.parent / 'paradigms' / 'descriptions'
    result = []
    for filename in paradigm_path.glob('*.py'):
        s = str(filename.with_suffix(''))
        i = s.rfind('psi')
        module = s[i:].replace('/', '.').replace('\\', '.')
        result.append(module)
    return result


def _render_setting(value, verbose=False):
    '''
    One line for a setting's value.

    Paths print as the path rather than as WindowsPath('...'), and
    strings without quotes: this listing is read to check a location
    or a device name, and those should be pasteable. Containers are
    summarized, because a nested table (cftscal's per-plugin state
    runs to dozens of keys) otherwise buries every other setting on
    the screen. `verbose` prints them in full.
    '''
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        return value if value else '(empty)'
    if isinstance(value, bool) or not isinstance(value, (list, tuple, dict)):
        return str(value)

    if verbose:
        return repr(value)
    if not value:
        return '(none)'
    if isinstance(value, dict):
        keys = ', '.join(sorted(value))
        n = len(value)
        return f'({n} {"entry" if n == 1 else "entries"}: {keys})'
    if all(isinstance(v, (str, int, float, Path)) for v in value):
        return ', '.join(str(v) for v in value)
    n = len(value)
    return f'({n} {"item" if n == 1 else "items"})'


def _register_downstream_settings():
    '''
    Import the packages that register settings of their own.

    Registration is an import side effect, and `psi-config` imports only
    psi. Without this, every CFTSCAL_ or NOISE_EXP_ key in the
    configuration file is reported as belonging to no setting at all --
    exactly backwards, since those are the settings a user is most likely
    to be checking.

    PSI_PARADIGM_DESCRIPTIONS already names the modules this installation
    uses, and importing one imports its package, so there is no separate
    registry to keep in step. A package installed but named nowhere in
    the configuration is still invisible; declaring an entry point would
    fix that, at the cost of reinstalling every package.

    Failures are logged rather than raised: `psi-config show` is what
    somebody runs when something is already broken, so it has to survive
    a package that will not import.
    '''
    loaded = []
    try:
        descriptions = get_config('PSI_PARADIGM_DESCRIPTIONS')
    except Exception as e:
        log.warning('Could not read PSI_PARADIGM_DESCRIPTIONS: %s', e)
        return loaded
    for description in descriptions:
        try:
            importlib.import_module(description)
            loaded.append(description)
        except Exception as e:
            log.warning('Could not import %s, so any settings it registers '
                        'will be reported as unknown: %s', description, e)
    return loaded


def _group_label(names):
    '''
    The prefix a group of settings shares.

    Taken from the names rather than from a hard-coded list of
    packages, since any package can register its own. Segments are
    only consumed while every name agrees and never down to a
    name's last segment, so NOISE_EXP_* is labelled NOISE_EXP
    while a lone CFTS_ROOT is labelled CFTS rather than
    CFTS_ROOT.
    '''
    parts = [n.split('_') for n in names]
    common = []
    for i in range(min(len(p) - 1 for p in parts)):
        segment = {p[i] for p in parts}
        if len(segment) != 1:
            break
        common.append(segment.pop())
    return '_'.join(common) or names[0].split('_')[0]


def config():
    import argparse
    import psi

    setup_windows_console()

    # Identify all the possible hardware configurations. Thoe prefixed by an
    # underscore are not available for running directly as they need to be
    # modified before use. All hardware configurations can be used as a
    # template for a skeleton that's copied to the IO_ROOT folder.
    io_template_paths = list_io_templates()
    io_skeleton_choices = [p.stem.strip('_') for p in io_template_paths]

    paradigms = list_paradigm_descriptions()
    paradigm_choices = {p.rsplit('.', 1)[1]: p for p in paradigms}

    def show_config(args):
        loaded = _register_downstream_settings()
        config_file = psi.get_config_file()
        exists = 'exists' if config_file.exists() else 'does not exist'
        print(f'Configuration file: {config_file} ({exists})')
        if loaded:
            print(f'Also loaded: {", ".join(sorted(loaded))}')

        settings = psi.get_all_config()
        if not settings:
            print('\nNo settings are known.')
            return

        # A key in the file that no package registered is read by nothing.
        # Listing it beside the real settings implies it does something.
        known = {n: v for n, v in settings.items() if n in psi.config._defaults}
        unknown = {n: v for n, v in settings.items() if n not in known}

        # The source is the whole point of this listing -- "why is this not
        # what I put in the file?" -- but most settings are defaults, and
        # labelling every one of them just crowds out the few that matter.
        labels = {'config file': 'file', 'environment': 'env', 'default': ''}
        width = max(len(n) for n in settings)

        groups = {}
        for name, value in known.items():
            groups.setdefault(name.split('_', 1)[0], []).append((name, value))

        rendered = {n: _render_setting(v, args.verbose)
                    for n, v in known.items()}
        # Wide enough for most values, but not so wide that one long entry
        # pushes the source column off the screen for everything else.
        vwidth = min(max((len(v) for v in rendered.values()), default=0), 44)

        for key in sorted(groups):
            entries = sorted(groups[key])
            print(f'\n{_group_label([n for n, _ in entries])}')
            for name, _ in entries:
                source = labels[psi.config_source(name)]
                print(f'  {name:<{width}}  {rendered[name]:<{vwidth}}  '
                      f'{source}'.rstrip())

        if unknown:
            print('\nIn the configuration file but not registered by any '
                  'package loaded here')
            for name, value in sorted(unknown.items()):
                print(f'  {name:<{width}}  '
                      f'{_render_setting(value, args.verbose)}')
            print('  (left over from an older version, misspelled, or owned '
                  'by a package this')
            print('   installation does not load -- see '
                  'PSI_PARADIGM_DESCRIPTIONS)')

        print('\nA blank source means the package default.')
        if not args.verbose:
            print('Run with --verbose to print container values in full.')

    def set_config_value(args):
        # Same reason as show: without the owning package loaded, a
        # downstream setting has no registered default, so the value would
        # not be coerced to its type and config_source could not report
        # that the environment is shadowing the write.
        _register_downstream_settings()
        psi.save_config({args.setting: args.value})
        source = psi.config_source(args.setting)
        value = psi.get_config(args.setting)
        if isinstance(value, Path):
            value = str(value)
        print(f'{args.setting} = {value}')
        if source == 'environment':
            # Writing succeeded but changed nothing the application will
            # see. Saying so here avoids a long hunt later.
            print(f'WARNING: {args.setting} is also set in the environment, '
                  'which takes precedence. The value written to the '
                  'configuration file will not take effect until the '
                  'environment variable is cleared.')

    def migrate_config(args):
        from psi.config_migrate import migrate
        migrate(args.source, dry_run=args.dry_run)

    def create_config(args):
        base_directory = args.base_directory.rstrip('\\')
        if args.paradigm_description is None:
            paradigms = None
        else:
            paradigms = [paradigm_choices.get(p, p) \
                         for p in args.paradigm_description]

        psi.create_config(base_directory=base_directory, standard_io=args.io,
                          paradigm_descriptions=paradigms)
        if args.base_directory:
            psi.create_config_dirs()

    def create_folders(args):
        psi.create_config_dirs()

    def create_io(args):
        i = io_skeleton_choices.index(args.template)
        template = io_template_paths[i].name
        psi.create_io_manifest(template)

    parser = argparse.ArgumentParser(
        'psi-config',
        description='Configure psiexperiment'
    )
    subparsers = parser.add_subparsers(
        dest='cmd',
        description='Available actions',
    )
    subparsers.required = True

    show = subparsers.add_parser(
        'show',
        description='Show the config file location, every setting, its '
                    'resolved value and which layer supplied it.',
    )
    show.set_defaults(func=show_config)
    show.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print container values in full instead of summarizing them.',
    )

    set_parser = subparsers.add_parser(
        'set',
        description='Write a setting to the configuration file.',
    )
    set_parser.set_defaults(func=set_config_value)
    set_parser.add_argument('setting', type=str, help='Name of the setting.')
    set_parser.add_argument('value', type=str, help='Value to write.')

    migrate = subparsers.add_parser(
        'migrate',
        description='Convert a pre-rework config.py (and any cftscal '
                    'workspace.json beside it) into config.toml.',
    )
    migrate.set_defaults(func=migrate_config)
    migrate.add_argument(
        'source',
        type=Path,
        help='Path to the legacy config.py to convert.',
    )
    migrate.add_argument(
        '--dry-run',
        action='store_true',
        help='Print what would be written without writing it.',
    )

    create = subparsers.add_parser('create')
    create.set_defaults(func=create_config)
    create.add_argument(
        '--base-directory',
        type=str,
        help='Root directory to store data and settings for psiexperiment.'
    )
    create.add_argument(
        '--io',
        nargs='*',
        type=str,
        help='Default hardware configurations.',
    )
    create.add_argument(
        '--paradigm-description',
        nargs='*',
        type=str,
        help=f'''Default paradigm descriptions. Can either specify a
        fully-qualified module path (e.g., psilbhb.paradigms.lbhb) or the names
        of one of the built-in paradigms. Available built-in paradigms include
        {', '.join(paradigm_choices.keys())}.'
        '''
    )

    make = subparsers.add_parser(
        'create-folders',
        description='Create folders defined in the config file.',
    )
    make.set_defaults(func=create_folders)

    io = subparsers.add_parser(
        'create-io',
        description='Creates a hardware configuration skeleton that you can edit'
    )
    io.set_defaults(func=create_io)
    io.add_argument(
        'template',
        type=str,
        choices=io_skeleton_choices,
        help='Template to use for hardware configuration skeleton.',
    )

    args = parser.parse_args()
    args.func(args)
