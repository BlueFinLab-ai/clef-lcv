"""Scheduler compatibility excludes content; native cache planning validates ancestry."""
import ast,sys,unittest
from pathlib import Path
from types import SimpleNamespace as NS
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from cpu_preparation import PreparedDecision
source=Path(__file__).resolve().parents[1]/'clef_service/app.py'
node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='text_batch_key')
namespace={'PreparedDecision':PreparedDecision}
exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
key=namespace['text_batch_key']

def request(**kw):
    return NS(**{'model':'clef-flash-nf4','prefix_cache':True,'input_cache':True,
                 'images':[],'context':'policy A',**kw})
class TextBatchKeys(unittest.TestCase):
    def test_different_contexts_and_questions_can_share_a_backbone_batch(self):
        self.assertEqual(key(request()),key(request(context={'document':'B'},questions={'other':'different'})))
        self.assertEqual(key(request(prefix_cache=False)),key(request(prefix_cache=False,context=None)))
    def test_model_and_cache_controls_remain_barriers(self):
        first=key(request())
        for row in [request(model='full'),request(prefix_cache=False),request(input_cache=False)]:
            self.assertNotEqual(first,key(row))
    def test_image_remains_a_fifo_barrier(self):
        self.assertIsNone(key(request(images=['image'])))
    def test_prepared_decision_preserves_compatibility(self):
        row=request()
        prepared=PreparedDecision(row,{},None,0,(),{},None)
        self.assertEqual(key(row),key(prepared))
if __name__=='__main__':unittest.main()
