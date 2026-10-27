"""Use the production planner to verify exact-prefix safety for unrelated text rows."""
import ast,sys,unittest
from pathlib import Path
from types import SimpleNamespace as NS
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
source=Path(__file__).resolve().parents[1]/'runtime/optimized_inference.py'
tree=ast.parse(source.read_text())
cls=next(c for c in tree.body if isinstance(c,ast.ClassDef) and any(isinstance(n,ast.FunctionDef) and n.name=='plan_text_batch' for n in c.body))
node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='plan_text_batch')
namespace={}
exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
class Engine:
    max_entries=32
    def __init__(self,gpu=None,host=None):
        self.entries=gpu or {};self.host=NS(enabled=True);self.bank={**self.entries,**(host or {})}
    def _budget(self):return 1024
    def _lookup(self,key):return self.bank[key]
    def _match(self,model,tokens,boundary,*args):
        matches=[(len(v['ids']),k) for k,v in self.bank.items() if len(v['ids'])<=boundary and tuple(tokens[:len(v['ids'])])==v['ids']]
        return (max(matches)[1],0) if matches else (None,0)
    plan=namespace['plan_text_batch']
rows=[NS(input_ids=(1,2,3,10,11,90)),NS(input_ids=(1,2,3,20,21,90))]
class CrossContextCachePlan(unittest.TestCase):
    def test_only_exact_common_ancestry_is_reused(self):
        e=Engine(gpu={'root':{'ids':(1,2,3)},'unrelated':{'ids':(1,9,8)}})
        p=e.plan(None,rows,[5,5],'text')
        self.assertEqual((p['key'],p['common'],p['reason']),('root',3,None))
    def test_uncached_unrelated_rows_need_no_common_checkpoint(self):
        p=Engine().plan(None,rows,[5,5],'text',False)
        self.assertFalse(p['enabled']);self.assertIsNone(p['reason']);self.assertIsNone(p['key'])
    def test_better_individual_hit_still_forces_individual_execution(self):
        for tier in ['gpu','host']:
            e=Engine(gpu={'root':{'ids':(1,2,3)}})
            extra={'stateA':{'ids':(1,2,3,10,11)}}
            e.bank.update(extra)
            if tier=='gpu':e.entries.update(extra)
            self.assertEqual(e.plan(None,rows,[5,5],'text')['reason'],'better_individual_prefix')
    def test_host_common_prefix_keeps_existing_restore_path(self):
        e=Engine(host={'root':{'ids':(1,2,3)}})
        self.assertEqual(e.plan(None,rows,[5,5],'text')['reason'],'host_prefix_requires_single_request')
if __name__=='__main__':unittest.main()
