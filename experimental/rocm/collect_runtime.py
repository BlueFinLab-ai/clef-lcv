"""Copy serving runtime payload, excluding SDK executables and build tools."""
import json,os,shutil,sys
from pathlib import Path
root=Path('/runtime-root'); core=Path('/opt/rocm/core-10.0')
def copy(p):
 dest=root/p.relative_to('/')
 dest.parent.mkdir(parents=True,exist_ok=True)
 if p.is_symlink():
  if not dest.exists() and not dest.is_symlink():dest.symlink_to(os.readlink(p))
  target=p.resolve()
  if target.is_file() and target.is_relative_to(core):copy(target)
 elif p.is_dir():shutil.copytree(p,dest,symlinks=True,dirs_exist_ok=True)
 else:shutil.copy2(p,dest)
# Core libraries include libamd_comgr/LLVM used by in-process HIP runtime compilation.
# Keep code objects, BLAS solution indexes, MIOpen caches and required headers.
for p in core.glob('lib/*.so*'):copy(p)
for name in ['lib/rocblas','lib/hipblaslt','lib/rocfft','lib/rocsparse','lib/host-math/lib',
             'lib/rocm_sysdeps/lib','lib/llvm/lib/clang','lib/llvm/amdgcn','share/doc','include','.kpack','share/miopen','.info']:
 p=core/name
 if p.exists():copy(p)
for p in (core/'lib/llvm/lib').glob('*.so*'):copy(p)
# Preserve core/alias layout and the community SGEMM preload shim.
for name,target in {'core':'core-10.0','core-10':'core-10.0','lib':'core-10.0/lib',
                    'include':'core-10.0/include','share':'core-10.0/share',
                    'llvm':'core-10.0/lib/llvm','amdgcn':'core-10.0/lib/llvm/amdgcn'}.items():
    dest=root/'opt/rocm'/name
    if dest.is_symlink():dest.unlink()
    dest.symlink_to(target)
for p in [Path('/opt/rocm/migraphx-version.txt')]:
 if p.exists():copy(p)
base=Path(sys.executable).resolve().parent.parent
shutil.copytree(base,root/'opt/python',symlinks=True)
shutil.copytree('/opt/venv',root/'opt/venv',symlinks=True)
for name in ['python','python3','python3.12']:
 p=root/'opt/venv/bin'/name
 if p.exists() or p.is_symlink():p.unlink()
 p.symlink_to('/opt/python/bin/python3.12')
p=root/'opt/venv/pyvenv.cfg';text=p.read_text().replace(str(base),'/opt/python');p.write_text(text)
# Capture removal of optional builder packages. Their files are outside the
# runtime environment after pip uninstallation in this isolated collector stage.
for folder in ['opt/python/include','opt/venv/include']:
 shutil.rmtree(root/folder,ignore_errors=True)
(root/'runtime-payload.json').write_text(json.dumps({'gpu':'gfx803','compiler_executables':False,
 'runtime_compilation_libraries_retained':['HIPRTC','COMGR','LLVM shared libraries'],
 'base_python':str(base)},indent=2))
print('Collected runtime payload')
