"""CPU checks: ABI/version binding, path containment and artifact tamper rejection."""
import hashlib,json,sys,sysconfig,platform
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from clef_service.prebuilt_launchers import load_manifest,install_prebuilt_launchers


def rejects(fn):
    try:fn()
    except (RuntimeError,ValueError):return
    raise AssertionError('Expected rejection')

with TemporaryDirectory() as folder:
    root=Path(folder);binary=root/'helper.so';binary.write_bytes(b'fixture, not executable')
    manifest={'format_version':1,'python_ext_suffix':sysconfig.get_config_var('EXT_SUFFIX'),
              'platform':platform.machine(),'triton_version':'pinned','torch_version':'pinned','fla_version':'pinned',
              'modules':[{'key':'hash','name':'helper','file':'helper.so','sha256':hashlib.sha256(binary.read_bytes()).hexdigest()}]}
    def write(): (root/'manifest.json').write_text(json.dumps(manifest))
    write()
    with patch('clef_service.prebuilt_launchers.metadata.version',return_value='pinned'):
        assert load_manifest(root)[1][('hash','helper')]==str(binary.resolve())
        binary.write_bytes(b'tampered');rejects(lambda:load_manifest(root));binary.write_bytes(b'fixture, not executable')
        manifest['modules'][0]['file']='../outside.so';write();rejects(lambda:load_manifest(root))
        manifest['modules'][0]['file']='helper.so';manifest['python_ext_suffix']='wrong ABI';write()
        rejects(lambda:load_manifest(root))
        manifest['python_ext_suffix']=sysconfig.get_config_var('EXT_SUFFIX');write()
    with patch('clef_service.prebuilt_launchers.metadata.version',return_value='different'):
        rejects(lambda:load_manifest(root))
    rejects(lambda:install_prebuilt_launchers('rocm',root))
print('PASS: artifact ABI/version binding, tamper detection, path guard, HIP separation')
