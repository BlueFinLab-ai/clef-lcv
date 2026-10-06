"""Capture upstream-generated Triton host helper binaries in a CUDA builder.

Use --module clef_service --output-dir /artifacts -- serve ... then exercise the
API to capture serving signatures, or --script scripts/kernel_smoke.py for
operator regressions. Requires GCC and the pinned CUDA runtime in the builder.
"""
import argparse
import hashlib,json,os,platform,shutil,sys,sysconfig,runpy
from pathlib import Path
import importlib.metadata as m
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir',type=Path,required=True)
mode=parser.add_mutually_exclusive_group(required=True)
mode.add_argument('--module')
mode.add_argument('--script')
parser.add_argument('arguments',nargs=argparse.REMAINDER)
a=parser.parse_args()
os.environ.pop('CLEF_PREBUILT_LAUNCHERS_DIR',None)
if not shutil.which('gcc'):parser.error('Run in the pinned CUDA builder with GCC')
from triton.backends.nvidia import driver
from triton.runtime.build import platform_key
from triton.runtime.cache import get_cache_manager
root=a.output_dir;(root/'sources').mkdir(parents=True,exist_ok=True)
manifest_path=root/'manifest.json'
manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {'format_version':1,
 'python_ext_suffix':sysconfig.get_config_var('EXT_SUFFIX'),'platform':platform.machine(),
 'triton_version':m.version('triton'),'torch_version':m.version('torch'),'fla_version':m.version('fla-core'),
 'scope':'Host driver/launcher modules, independent of GPU SM; GPU Triton JIT retained.','modules':[]}
original=driver.compile_module_from_src

def capture(src,name,*args,**kwargs):
 key=hashlib.sha256((src+platform_key()).encode()).hexdigest()
 source_path=root/'sources'/(key+'.c');source_path.write_text(src)
 module=original(src,name,*args,**kwargs)
 file=Path(get_cache_manager(key).get_file(name+sysconfig.get_config_var('EXT_SUFFIX')))
 relative='modules/'+key+'/'+file.name
 dest=root/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,dest)
 record={'key':key,'name':name,'file':relative,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
         'source':'sources/'+key+'.c','source_sha256':hashlib.sha256(src.encode()).hexdigest()}
 manifest['modules']=[r for r in manifest['modules'] if (r['key'],r['name'])!=(key,name)]+[record]
 manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
 return module

driver.compile_module_from_src=capture
arguments=a.arguments[1:] if a.arguments[:1]==['--'] else a.arguments
sys.argv=[a.module or a.script,*arguments]
if a.module:runpy.run_module(a.module,run_name='__main__')
else:runpy.run_path(a.script,run_name='__main__')
