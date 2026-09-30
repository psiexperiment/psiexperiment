'''
Tests for the configuration system.

The behaviour pinned down here is the behaviour that was wrong before the
rework: the environment must beat the config file, a bare install with no
config file must still run, and a value must arrive as the type its
default implies no matter which layer supplied it.
'''
from pathlib import Path

import numpy as np
import pytest

from psi import get_config
from psi import config as psi_config
from psi import runtime as psi_runtime


def test_get_config_default_returned_for_missing_setting():
    default = 'fallback'
    assert get_config('__no_such_setting__', default) == default


def test_get_config_none_is_valid_default():
    assert get_config('__no_such_setting__', None) is None


def test_get_config_array_default_identity():
    # Regression: the sentinel check used `!=` instead of `is not`, so any
    # default whose __eq__ returns a non-scalar (e.g. numpy arrays) raised
    # "truth value is ambiguous" inside get_config itself.
    default = np.array([1.0, 2.0])
    result = get_config('__no_such_setting__', default)
    assert result is default


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    '''
    An empty config file that psi will actually load.
    '''
    path = tmp_path / 'config.toml'
    path.write_text('', encoding='utf-8')
    monkeypatch.setenv('PSI_CONFIG_FILE', str(path))
    psi_config.reload_config()
    yield path
    psi_config.reload_config()


@pytest.fixture
def no_config_file(tmp_path, monkeypatch):
    '''
    A config file location that does not exist.
    '''
    path = tmp_path / 'missing.toml'
    monkeypatch.setenv('PSI_CONFIG_FILE', str(path))
    psi_config.reload_config()
    yield path
    psi_config.reload_config()


def write(path, text):
    path.write_text(text, encoding='utf-8')
    psi_config.reload_config()


def test_no_config_file_still_resolves(no_config_file):
    '''
    Every registered setting must have a usable built-in default: cftscal
    is end-user installable with no configuration step, so a missing
    config file is a supported state rather than an error.
    '''
    for name in list(psi_config._defaults):
        psi_config.get_config(name)


def test_environment_overrides_config_file(config_file, monkeypatch):
    '''
    The whole point of the rework. The old implementation had this
    backwards: config.py silently beat the environment.
    '''
    write(config_file, 'PSI_DATA_ROOT = "C:/from_file"\n')
    assert get_config('PSI_DATA_ROOT') == Path('C:/from_file')
    assert psi_config.config_source('PSI_DATA_ROOT') == 'config file'

    monkeypatch.setenv('PSI_DATA_ROOT', 'C:/from_env')
    assert get_config('PSI_DATA_ROOT') == Path('C:/from_env')
    assert psi_config.config_source('PSI_DATA_ROOT') == 'environment'


def test_config_file_overrides_default(config_file):
    write(config_file, 'PSI_DATA_ROOT = "C:/from_file"\n')
    assert get_config('PSI_DATA_ROOT') == Path('C:/from_file')


def test_default_when_nothing_set(no_config_file):
    assert psi_config.config_source('PSI_DATA_ROOT') == 'default'
    assert get_config('PSI_DATA_ROOT') == \
        get_config('PSI_BASE_DIRECTORY') / 'data'


def test_derived_roots_follow_base_directory(config_file):
    '''
    Setting the base directory must move every derived root with it.
    This is why defaults are callables: resolving them at import would
    freeze the built-in base directory into the derived values.
    '''
    write(config_file, 'PSI_BASE_DIRECTORY = "C:/rig"\n')
    assert get_config('PSI_DATA_ROOT') == Path('C:/rig/data')
    assert get_config('PSI_SETTINGS_ROOT') == Path('C:/rig/settings')
    assert get_config('PSI_IO_ROOT') == Path('C:/rig/io')


def test_logs_do_not_follow_the_base_directory(config_file):
    '''
    Logs and profiling output are written continuously while an
    experiment runs, and a rig points its base directory at the share
    where the data belongs. Deriving the log root from it would send
    those writes over the network.
    '''
    write(config_file, 'PSI_BASE_DIRECTORY = "//server/share/psi"\n')
    assert Path('//server/share/psi') not in get_config('PSI_LOG_ROOT').parents


def test_log_root_is_still_overridable(config_file):
    write(config_file, 'PSI_LOG_ROOT = "D:/logs"\n')
    assert get_config('PSI_LOG_ROOT') == Path('D:/logs')


def test_derived_root_can_be_overridden_individually(config_file):
    write(config_file, 'PSI_BASE_DIRECTORY = "C:/rig"\n'
                       'PSI_DATA_ROOT = "D:/bulk"\n')
    assert get_config('PSI_DATA_ROOT') == Path('D:/bulk')
    assert get_config('PSI_SETTINGS_ROOT') == Path('C:/rig/settings')


@pytest.mark.parametrize('raw,expected', [
    ('1', True), ('true', True), ('TRUE', True), ('yes', True), ('on', True),
    ('0', False), ('false', False), ('no', False), ('off', False),
])
def test_bool_coercion(raw, expected):
    '''
    bool('false') is True, which would make every spelling of "off" mean
    "on". Coercion must not go through the bool constructor.
    '''
    assert psi_config._coerce(raw, True) is expected


def test_bool_coercion_rejects_nonsense():
    with pytest.raises(ValueError):
        psi_config._coerce('maybe', True)


def test_numeric_and_path_coercion():
    assert psi_config._coerce('48000', 0) == 48000
    assert psi_config._coerce('96000.5', 0.0) == 96000.5
    assert psi_config._coerce('C:/x', Path('.')) == Path('C:/x')


def test_list_coercion_from_environment():
    assert psi_config._coerce('a, b ,c', []) == ['a', 'b', 'c']


def test_list_from_toml_is_untouched(config_file):
    # A TOML array arrives as a list, unlike an environment variable,
    # which arrives as a string and has to be split.
    write(config_file, 'PSI_AD_HOC_LIST = ["a", "b"]\n')
    assert get_config('PSI_AD_HOC_LIST') == ['a', 'b']


def test_unknown_setting_raises(no_config_file):
    with pytest.raises(KeyError):
        get_config('PSI_NOT_A_SETTING')


def test_save_config_preserves_comments(config_file):
    write(config_file, '# a comment worth keeping\nPSI_DATA_ROOT = "C:/a"\n')
    psi_config.save_config({'PSI_LOG_ROOT': Path('C:/logs')})
    assert '# a comment worth keeping' in \
        config_file.read_text(encoding='utf-8')
    assert get_config('PSI_LOG_ROOT') == Path('C:/logs')
    assert get_config('PSI_DATA_ROOT') == Path('C:/a')


def test_save_config_creates_missing_file(no_config_file):
    psi_config.save_config({'PSI_DATA_ROOT': Path('C:/new')})
    assert no_config_file.exists()
    assert get_config('PSI_DATA_ROOT') == Path('C:/new')


def test_save_config_removes_with_none(config_file):
    write(config_file, 'PSI_DATA_ROOT = "C:/a"\n')
    psi_config.save_config({'PSI_DATA_ROOT': None})
    assert psi_config.config_source('PSI_DATA_ROOT') == 'default'


def test_save_config_leaves_no_temp_file(config_file):
    psi_config.save_config({'PSI_DATA_ROOT': Path('C:/a')})
    assert list(config_file.parent.glob('*.tmp')) == []


def test_register_defaults_rejects_non_callable():
    with pytest.raises(ValueError):
        psi_config.register_defaults({'PSI_X': 'not callable'})


def test_register_defaults_rejects_conflict():
    psi_config.register_defaults({'PSI_TEST_ONLY': lambda: 1})
    try:
        with pytest.raises(ValueError):
            psi_config.register_defaults({'PSI_TEST_ONLY': lambda: 2})
    finally:
        psi_config._defaults.pop('PSI_TEST_ONLY', None)


class TestRuntime:
    '''
    Runtime values must have no environment or config-file path. Someone
    who sets PSI_LOG_FILENAME should find it does nothing, rather than
    half-working.
    '''

    def setup_method(self):
        psi_runtime.clear_runtime()

    def teardown_method(self):
        psi_runtime.clear_runtime()

    def test_roundtrip(self):
        psi_runtime.set_runtime('PROFILE', True)
        assert psi_runtime.get_runtime('PROFILE') is True

    def test_unset_raises(self):
        with pytest.raises(KeyError):
            psi_runtime.get_runtime('PROFILE')

    def test_unset_with_default(self):
        assert psi_runtime.get_runtime('LOG_FILENAME', '') == ''

    def test_not_reachable_through_environment(self, monkeypatch):
        monkeypatch.setenv('LOG_FILENAME', 'C:/sneaky.log')
        monkeypatch.setenv('PSI_LOG_FILENAME', 'C:/sneaky.log')
        assert psi_runtime.get_runtime('LOG_FILENAME', None) is None

    @pytest.mark.parametrize('name', ['LOG_FILENAME', 'PROFILE'])
    def test_runtime_names_are_not_settings(self, name):
        assert name not in psi_config._defaults
        assert f'PSI_{name}' not in psi_config._defaults


def test_no_call_site_shadows_a_registered_default():
    '''
    Passing `default=` for a setting that already has a registered
    default is how one setting ends up with two different defaults in two
    modules. LOG_FILENAME really did default to '' in one module and None
    in another before the rework.
    '''
    import ast

    root = Path(__file__).parent.parent / 'src' / 'psi'
    offenders = []
    for path in root.rglob('*.py'):
        try:
            tree = ast.parse(path.read_text(encoding='utf-8'))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, 'id', None) or \
                getattr(node.func, 'attr', None)
            if name != 'get_config' or len(node.args) < 2:
                continue
            first = node.args[0]
            if isinstance(first, ast.Constant) and \
                    first.value in psi_config._defaults:
                offenders.append(f'{path}:{node.lineno} {first.value}')
    assert offenders == []


def test_config_file_with_bom(config_file):
    '''
    A configuration file saved with a byte order mark still loads.

    Notepad, PowerShell's Out-File and VS Code's "UTF-8 with BOM" all
    produce one, and handing tomllib a binary handle made it reject the
    file as "Invalid statement (at line 1, column 1)" -- naming neither
    the BOM nor the file. Anything catching that exception then fell back
    to defaults, which looks exactly like a machine nobody configured.
    '''
    config_file.write_text('PSI_DATA_ROOT = "C:/from_file"\n',
                           encoding='utf-8-sig')
    psi_config.reload_config()
    assert get_config('PSI_DATA_ROOT') == Path('C:/from_file')


def test_save_config_preserves_content_of_a_bom_file(config_file):
    config_file.write_text('# a comment\nPSI_DATA_ROOT = "C:/a"\n',
                           encoding='utf-8-sig')
    psi_config.reload_config()
    psi_config.save_config({'PSI_LOG_ROOT': Path('C:/logs')})
    assert get_config('PSI_DATA_ROOT') == Path('C:/a')
    assert get_config('PSI_LOG_ROOT') == Path('C:/logs')
    assert '# a comment' in config_file.read_text(encoding='utf-8-sig')


def test_empty_environment_variable_is_treated_as_unset(config_file,
                                                        monkeypatch):
    '''
    A launcher building `env[name] = str(field)` from a field nobody
    filled in would otherwise win over the file and the default -- and
    for a path setting, Path('') is the working directory.
    '''
    write(config_file, 'PSI_DATA_ROOT = "C:/from_file"\n')
    monkeypatch.setenv('PSI_DATA_ROOT', '')
    assert get_config('PSI_DATA_ROOT') == Path('C:/from_file')
    assert psi_config.config_source('PSI_DATA_ROOT') == 'config file'


def test_a_default_that_raises_does_not_hide_a_configured_value(config_file):
    '''
    The default is only needed to say what type to coerce to when a value
    was supplied. One that cannot be computed must not make an explicitly
    configured setting unreadable.
    '''
    def boom():
        raise RuntimeError('cannot compute')

    psi_config._defaults['PSI_BOOM_TEST'] = boom
    try:
        write(config_file, 'PSI_BOOM_TEST = "set by hand"\n')
        assert get_config('PSI_BOOM_TEST') == 'set by hand'

        # With nothing configured, the default is the only source of a
        # value, so its failure is the caller's problem after all.
        write(config_file, '')
        with pytest.raises(RuntimeError):
            get_config('PSI_BOOM_TEST')
    finally:
        psi_config._defaults.pop('PSI_BOOM_TEST', None)


def test_empty_path_is_refused():
    with pytest.raises(ValueError, match='empty value cannot be a path'):
        psi_config._coerce('', Path('.'))


def test_unquoted_path_in_toml_is_refused(config_file):
    write(config_file, 'PSI_DATA_ROOT = 5\n')
    with pytest.raises(ValueError, match='Expected a path'):
        get_config('PSI_DATA_ROOT')


def test_nested_none_is_dropped_on_save(config_file):
    psi_config.save_config({'PSI_TABLE_TEST': {'a': 1, 'b': None}})
    assert psi_config.load_config()['PSI_TABLE_TEST'] == {'a': 1}


def test_none_inside_a_list_is_refused(config_file):
    with pytest.raises(ValueError, match='element 1 is None'):
        psi_config.save_config({'PSI_LIST_TEST': ['a', None]})
