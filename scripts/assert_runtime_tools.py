"""Verify runtime images exclude general compiler/build executables.

This does not claim removal of GPU JIT libraries or Triton's PTX assembler.
"""
import json
from pathlib import Path
import shutil

names=['gcc','g++','cc','c++','clang','clang++','cmake','make','ninja','nvcc','hipcc','git','rustc','pip','pip3']
found={name:shutil.which(name) for name in names if shutil.which(name)}
for pattern in ['/usr/bin/gcc-[0-9]*','/usr/bin/g++-[0-9]*','/usr/bin/clang*',
                '/opt/rocm/lib/llvm/bin/clang*','/opt/rocm/bin/hipcc']:
    # Paths listed here are deliberate compiler locations, not runtime .so files.
    parent,glob=pattern.rsplit('/',1)
    for p in Path(parent).glob(glob):
        if p.is_file():found[str(p)]=str(p)
assert not found,found
print(json.dumps({'general_compilers_and_build_tools':found,
                  'gpu_runtime_compilation_may_remain':True,'passed':True}))
