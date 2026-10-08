"""Remove code and metadata together while preserving serving dependencies."""
from pathlib import Path
import tempfile
from strip_runtime_installers import strip
with tempfile.TemporaryDirectory() as temporary:
    prefixes=[Path(temporary)/'venv',Path(temporary)/'python']
    for prefix in prefixes:
        package=prefix/'lib/python3.12/site-packages'
        for name in ['pip/_vendor','pip-26.2.1.dist-info','setuptools','torch','pipx']:
            folder=package/name;folder.mkdir(parents=True);(folder/'keep.py').write_text('payload')
        ensure=prefix/'lib/python3.12/ensurepip/_bundled';ensure.mkdir(parents=True);(ensure/'pip.whl').write_text('bundled installer')
        binaries=prefix/'bin';binaries.mkdir()
        for name in ['pip','pip3','pip3.12','python']: (binaries/name).write_text('executable')
    removed=strip(prefixes,'3.12')
    assert removed
    for prefix in prefixes:
        package=prefix/'lib/python3.12/site-packages'
        assert not (package/'pip').exists() and not (package/'pip-26.2.1.dist-info').exists()
        assert not (prefix/'lib/python3.12/ensurepip').exists()
        assert all(not (prefix/'bin'/name).exists() for name in ['pip','pip3','pip3.12'])
        assert all((package/name/'keep.py').exists() for name in ['torch','setuptools','pipx'])
        assert (prefix/'bin/python').exists()
print('PASS: both prefixes, full installer/vendor/metadata/console removal, preserved runtime packages')
