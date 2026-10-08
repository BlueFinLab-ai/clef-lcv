"""Remove complete installers from final images after all dependencies exist."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys


def strip(prefixes, version):
    removed=[]
    for prefix in set(map(Path, prefixes)):
        for folder in (prefix/'lib').glob('python*'):
            paths=[folder/'ensurepip',folder/'site-packages'/'pip']
            paths.extend((folder/'site-packages').glob('pip-*.dist-info'))
            for path in paths:
                if path.is_dir():shutil.rmtree(path);removed.append(str(path))
        for name in ['pip','pip3',f'pip{version}']:
            path=prefix/'bin'/name
            if path.exists() or path.is_symlink():path.unlink();removed.append(str(path))
    return removed


if __name__ == '__main__':
    # Fail before any deletion when invoked accidentally in a developer env.
    if sys.prefix!='/opt/venv' or sys.base_prefix not in {'/usr/local','/opt/python'}:
        raise RuntimeError('Installer removal is restricted to the Clef runtime image')
    removed=strip([sys.prefix,sys.base_prefix],f'{sys.version_info.major}.{sys.version_info.minor}')
    assert importlib.util.find_spec('pip') is None,'pip still importable'
    assert importlib.util.find_spec('ensurepip') is None,'ensurepip still importable'
    print(json.dumps({'removed_installers':removed,'passed':True}))
