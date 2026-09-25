'''
Tests for the diagnostics attached to IO manifest failures.

The IO manifest is the rig-specific part of the startup sequence, so when it
fails the underlying exception is usually a bare driver error ("No
input/output device matching 'FrontMic'") that never names the file that
declared the device. `psi.application.io_manifest_errors` re-raises these as
`IOManifestError` with enough context to find and fix the configuration.
'''

import pytest

from psi import application
from psi.application import IOManifestError, io_manifest_errors


@pytest.fixture
def io_root(tmp_path, monkeypatch):
    '''
    Point PSI_IO_ROOT at a folder containing a single, plausible IO config.
    '''
    root = tmp_path / 'io'
    root.mkdir()
    (root / 'rig1.enaml').write_text('')
    # Every setting the code under test reads has to be present: these
    # settings now have registered defaults, so nothing passes a fallback
    # to get_config any more and a missing key would resolve to None.
    config = {
        'PSI_IO_ROOT': root,
        'PSI_HOSTNAME': 'rig1',
        'PSI_STANDARD_IO': [],
    }
    monkeypatch.setattr(application, 'get_config',
                        lambda key, *args: config.get(key, *args))
    return root


def test_error_names_the_manifest_and_how_to_find_it(io_root):
    manifest = str(io_root / 'rig1.enaml')
    with pytest.raises(IOManifestError) as excinfo:
        with io_manifest_errors(manifest):
            raise ValueError("No input/output device matching 'FrontMic'")

    mesg = str(excinfo.value)
    # The file to edit, the error that was raised, where IO configs live and
    # how to point psi somewhere else.
    assert manifest in mesg
    assert "ValueError: No input/output device matching 'FrontMic'" in mesg
    assert str(io_root) in mesg
    assert 'PSI_IO_ROOT' in mesg
    assert '--io' in mesg
    # The original exception is preserved for anyone reading the traceback.
    assert isinstance(excinfo.value.__cause__, ValueError)


def test_error_reports_manifest_class(io_root):
    manifest = f"{io_root / 'rig1.enaml'}::CustomManifest"
    with pytest.raises(IOManifestError) as excinfo:
        with io_manifest_errors(manifest):
            raise RuntimeError('boom')
    mesg = str(excinfo.value)
    assert 'CustomManifest' in mesg
    # The '::ClassName' suffix is reported separately rather than mangling
    # the path the user is told to open.
    assert f"IO configuration: {io_root / 'rig1.enaml'}\n" in mesg


def test_available_io_configurations_are_listed(io_root):
    (io_root / 'rig2.enaml').write_text('')
    with pytest.raises(IOManifestError) as excinfo:
        with io_manifest_errors(str(io_root / 'rig1.enaml')):
            raise RuntimeError('boom')
    mesg = str(excinfo.value)
    assert str(io_root / 'rig2.enaml') in mesg


def test_sounddevice_errors_list_available_devices(io_root, monkeypatch):
    devices = [
        {'name': 'ASIO Lynx Hilo USB', 'hostapi': 0,
         'max_input_channels': 8, 'max_output_channels': 8},
    ]

    class FakeSoundDevice:
        @staticmethod
        def query_devices():
            return devices

        @staticmethod
        def query_hostapis():
            return [{'name': 'ASIO'}]

    monkeypatch.setitem(__import__('sys').modules, 'sounddevice',
                        FakeSoundDevice)

    def raise_from_sounddevice():
        # The frame this raises from has to look like it lives in
        # `sounddevice`, since that is how the device list is triggered.
        exec(compile("raise ValueError(\"No input/output device matching "
                     "'FrontMic'\")", 'sounddevice.py', 'exec'),
             {'__name__': 'sounddevice'})

    with pytest.raises(IOManifestError) as excinfo:
        with io_manifest_errors(str(io_root / 'rig1.enaml')):
            raise_from_sounddevice()

    mesg = str(excinfo.value)
    assert 'ASIO Lynx Hilo USB' in mesg
    assert 'ASIO, 8 in, 8 out' in mesg
    assert 'sound card driver' in mesg


def test_non_sounddevice_errors_do_not_list_devices(io_root):
    with pytest.raises(IOManifestError) as excinfo:
        with io_manifest_errors(str(io_root / 'rig1.enaml')):
            raise RuntimeError('boom')
    assert 'sound card driver' not in str(excinfo.value)


def test_iomanifesterror_is_not_rewrapped(io_root):
    original = IOManifestError('already formatted')
    with pytest.raises(IOManifestError) as excinfo:
        with io_manifest_errors(str(io_root / 'rig1.enaml')):
            raise original
    assert excinfo.value is original


def test_load_io_manifest_wraps_import_errors(io_root, monkeypatch):
    def boom(path, name):
        raise ImportError('no such module')
    monkeypatch.setattr(application, 'load_manifest_from_file', boom)
    with pytest.raises(IOManifestError) as excinfo:
        application.load_io_manifest(str(io_root / 'rig1.enaml'))
    assert 'ImportError: no such module' in str(excinfo.value)


def test_list_sound_devices_is_empty_when_not_imported(monkeypatch):
    # Generating the message must never import (and thus initialize)
    # PortAudio as a side effect.
    monkeypatch.delitem(__import__('sys').modules, 'sounddevice',
                        raising=False)
    assert application.list_sound_devices() == []


def test_module_manifest_is_not_reported_as_editable(io_root):
    # cftscal hands psi a manifest that lives inside psiexperiment itself.
    # Telling the user to edit that file (or listing the .enaml files in
    # IO_ROOT, which have nothing to do with it) would send them to the
    # wrong place.
    manifest = 'psi.controller.engines.soundcard.standard_io.AutoSoundCardManifest'
    with pytest.raises(IOManifestError) as excinfo:
        with io_manifest_errors(manifest):
            raise RuntimeError('boom')
    mesg = str(excinfo.value)
    assert 'AutoSoundCardManifest' in mesg
    assert 'provided by an installed package' in mesg
    assert 'Open the IO configuration listed above' not in mesg
    assert 'PSI_IO_ROOT' not in mesg


def test_sound_device_env_overrides_are_reported(io_root, monkeypatch):
    # AutoSoundCardEngine reads the device from the environment, not from the
    # manifest, so the manifest alone never reveals where 'Fireface' came
    # from.
    monkeypatch.setenv('PSI_SOUND_DEVICE_NAME', 'ASIO Fireface USB, ASIO')
    monkeypatch.setenv('PSI_SOUND_DEVICE_FS', '96000')
    manifest = 'psi.controller.engines.soundcard.standard_io.AutoSoundCardManifest'
    with pytest.raises(IOManifestError) as excinfo:
        with io_manifest_errors(manifest):
            raise ValueError("No input/output device matching "
                             "'ASIO Fireface USB, ASIO'")
    mesg = str(excinfo.value)
    assert "PSI_SOUND_DEVICE_NAME: 'ASIO Fireface USB, ASIO'" in mesg
    assert "PSI_SOUND_DEVICE_FS:   '96000'" in mesg
    assert 'hardware settings' in mesg


def test_sound_device_env_omitted_when_unset(io_root, monkeypatch):
    monkeypatch.delenv('PSI_SOUND_DEVICE_NAME', raising=False)
    monkeypatch.delenv('PSI_SOUND_DEVICE_FS', raising=False)
    with pytest.raises(IOManifestError) as excinfo:
        with io_manifest_errors(str(io_root / 'rig1.enaml')):
            raise RuntimeError('boom')
    assert 'PSI_SOUND_DEVICE_NAME' not in str(excinfo.value)


def test_initialize_io_manifest_wraps_instantiation(io_root, monkeypatch):
    # The regression this whole module exists for: the hardware is touched
    # when the manifest is instantiated, not when it is imported.
    class Manifest:
        def __init__(self):
            raise ValueError("No input/output device matching 'Fireface'")

    monkeypatch.setattr(application, 'load_manifest_from_file',
                        lambda path, name: Manifest)
    with pytest.raises(IOManifestError) as excinfo:
        application.initialize_io_manifest(str(io_root / 'rig1.enaml'))
    assert "No input/output device matching 'Fireface'" in str(excinfo.value)


def test_initialize_io_manifest_resolves_default(io_root, monkeypatch):
    default = str(io_root / 'rig1.enaml')
    monkeypatch.setattr(application, 'get_default_io', lambda: default)

    class Manifest:
        def __init__(self):
            raise RuntimeError('boom')

    monkeypatch.setattr(application, 'load_manifest_from_file',
                        lambda path, name: Manifest)
    with pytest.raises(IOManifestError) as excinfo:
        application.initialize_io_manifest(None)
    # The message names the resolved file rather than 'None'.
    assert default in str(excinfo.value)


def test_iomanifesterror_is_a_valueerror():
    # cftscal probes for absent hardware with `except ValueError` around a
    # manifest load (raise_error=False); narrowing the base class would turn
    # those graceful degradations into crashes.
    assert issubclass(IOManifestError, ValueError)
