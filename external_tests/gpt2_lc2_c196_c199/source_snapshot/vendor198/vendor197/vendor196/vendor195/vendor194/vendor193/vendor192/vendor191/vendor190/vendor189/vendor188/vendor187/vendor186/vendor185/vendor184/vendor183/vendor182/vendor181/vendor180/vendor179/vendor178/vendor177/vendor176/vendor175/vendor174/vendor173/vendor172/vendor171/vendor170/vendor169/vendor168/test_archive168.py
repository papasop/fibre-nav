"""Full thirteen-original synthetic chain: empty C167 cannot authorize a real probe."""
import os,copy,contextlib,io,tempfile,unittest,shutil
from core168 import *
from fixture168 import pack
from inputs168 import load
from worker168 import run
class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cache=Path(os.environ['C168_ELEVEN_SYNTHETIC_CACHE']);cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);cls.paths={k:cache/(k+'.zip') for k in prior.prior.INPUT_KEYS};cc166=clean_json((cache/'contract.json').read_bytes())
  from worker166 import run as produce166
  from worker167 import run as produce167
  with contextlib.redirect_stdout(io.StringIO()):s=produce166(cls.paths,cls.root/'C166',cc166)
  need(s['complete'],'Synthetic C166 complete: '+str(s['errors']));shutil.copytree(ROOT/'vendor167/vendor166',cls.root/'C166/source',ignore=shutil.ignore_patterns('__pycache__'));cls.paths['C166']=pack(cls.root/'C166',cls.root/'C166.zip');cc167=prior.contract();cc167.update(production=False,parent=cc166,pins={k:dict(path=str(p),sha256=sha(p)) for k,p in cls.paths.items()})
  with contextlib.redirect_stdout(io.StringIO()):s=produce167(cls.paths,cls.root/'C167',cc167)
  need(s['complete'],'Synthetic C167 complete: '+str(s['errors']));compare(s['active_response_loss_rows'],0,'Historical synthetic fixture has no active response loss');shutil.copytree(ROOT/'vendor167',cls.root/'C167/source',ignore=shutil.ignore_patterns('__pycache__'));cls.paths['C167']=pack(cls.root/'C167',cls.root/'C167.zip');cls.c=contract();cls.c.update(production=False,parent=cc167,pins={k:dict(path=str(p),sha256=sha(p)) for k,p in cls.paths.items()})
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup()
 def test_all13_original_chain_reaches_C167_and_blocks_empty_certificate_before_model(self):
  with contextlib.redirect_stdout(io.StringIO()):s=run(self.paths,self.root/'C168',self.c,lambda *a:(_ for _ in ()).throw(AssertionError('Model must not load for empty certificate')))
  self.assertFalse(s['complete']);self.assertEqual(s['new_model_forwards'],0);self.assertEqual(s['new_backward_calls'],0);self.assertTrue(any('bind167' in e and 'linear_program_calls_attempted' in e for e in s['errors']),s['errors']);self.assertTrue((self.root/'C168/binding/C166_reconstruction/C165_arithmetic/ALL672_ACTUAL_FP64_POINT_AUDIT_ROWS.json').is_file());self.assertFalse(s['LC2_closed'])
if __name__=='__main__':unittest.main(verbosity=2)
