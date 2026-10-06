"""Use version-bound Triton host helpers without a runtime C compiler.

GPU compilation by Triton remains enabled; this replaces only host C builds.
The payload contains upstream-generated Python/CUDA driver extension modules.
"""
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sysconfig


def load_manifest(directory):
    root = Path(directory).resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('format_version') != 1:
        raise RuntimeError('Unsupported prebuilt Triton host-helper format')
    if manifest['python_ext_suffix'] != sysconfig.get_config_var('EXT_SUFFIX') or manifest['platform'] != platform.machine():
        raise RuntimeError('Prebuilt Triton host helpers require the matching Python ABI and CPU platform')
    for package, field in [('triton', 'triton_version'), ('torch', 'torch_version'), ('fla-core', 'fla_version')]:
        if metadata.version(package) != manifest[field]:
            raise RuntimeError(f'Prebuilt Triton host helpers require {package}={manifest[field]}')
    modules = {}
    for record in manifest['modules']:
        path = (root / record['file']).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise RuntimeError('Invalid prebuilt host-helper path')
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise RuntimeError(f'Prebuilt host-helper checksum mismatch: {path.name}')
        key = (record['key'], record['name'])
        if key in modules:
            raise RuntimeError('Duplicate prebuilt host-helper key')
        modules[key] = str(path)
    return manifest, modules


def install_prebuilt_launchers(runtime_backend, directory=None):
    directory = directory if directory is not None else os.environ.get('CLEF_PREBUILT_LAUNCHERS_DIR')
    if not directory:
        return {'enabled': False}
    if runtime_backend != 'cuda':
        raise ValueError('Prebuilt CUDA host helpers cannot be used in a HIP image')
    manifest, modules = load_manifest(directory)
    from triton import knobs
    from triton.runtime.build import platform_key

    def prebuilt(name, src, srcdir, library_dirs, include_dirs, libraries):
        source = Path(src).read_text()
        key = hashlib.sha256((source + platform_key()).encode()).hexdigest()
        try:
            return modules[(key, name)]
        except KeyError:
            raise RuntimeError(
                f'No prebuilt Triton host helper for {name} ({key}); '
                'regenerate the helper pack in the builder image. Runtime C compilation is disabled.'
            ) from None

    knobs.build.impl = prebuilt
    info = {'enabled': True, 'modules': len(modules), 'triton_version': manifest['triton_version'],
            'runtime_host_compilation': False, 'gpu_jit_compilation': True}
    os.environ['CLEF_PREBUILT_LAUNCHERS_INFO'] = json.dumps(info)
    return info
