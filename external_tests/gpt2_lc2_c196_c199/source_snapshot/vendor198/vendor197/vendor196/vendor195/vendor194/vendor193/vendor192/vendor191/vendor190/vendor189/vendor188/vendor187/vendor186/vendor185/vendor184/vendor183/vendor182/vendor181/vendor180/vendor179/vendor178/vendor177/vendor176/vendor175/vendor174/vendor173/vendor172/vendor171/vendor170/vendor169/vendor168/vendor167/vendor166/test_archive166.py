"""Full eleven-original synthetic archive chain; no parent context injection."""
import contextlib,io,tempfile,unittest,os,shutil
from core166 import *
from fixture166 import make165
from worker166 import run
class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);cls.prior=None;cache=Path(os.environ['C166_TEST_CACHE']) if os.environ.get('C166_TEST_CACHE') else None
  if cache is not None and (cache/'contract.json').is_file():
   cls.c=clean_json((cache/'contract.json').read_bytes());cls.paths={k:cache/(k+'.zip') for k in INPUT_KEYS}
  else:
   import test_archive165 as prior
   cls.prior=prior.Checks;cls.prior.setUpClass();cls.paths,cls.c,cls.ctx=make165(cls.prior.ctx,cls.prior.paths,cls.prior.c['parent'],cls.root/'C165',True)
   if cache is not None:
    cache.mkdir(parents=True,exist_ok=True)
    for k,p in cls.paths.items():shutil.copy2(p,cache/(k+'.zip'))
    dump(cache/'contract.json',cls.c)
 @classmethod
 def tearDownClass(cls):
  cls.tmp.cleanup()
  if cls.prior is not None:cls.prior.tearDownClass()
 def test_all11_originals_complete_worker_no_model_no_optimization(self):
  with contextlib.redirect_stdout(io.StringIO()):s=run(self.paths,self.root/'complete',self.c)
  self.assertTrue(s['complete'],s['errors']);self.assertTrue(s['all_eleven_originals_independently_bound']);self.assertEqual(s['actual_FP64_point_arrays_rehashed'],672);self.assertEqual(s['native_endpoint_records_audited'],1176);self.assertEqual(s['new_direction_optimization_iterations'],0);self.assertFalse('torch' in sys.modules);self.assertFalse(s['LC2_closed'])
 def test_production_pins_block_synthetic_original_before_parent_load(self):
  with contextlib.redirect_stdout(io.StringIO()):s=run(self.paths,self.root/'wrong_production',contract())
  self.assertFalse(s['complete']);self.assertEqual(s['new_model_forwards'],0)
if __name__=='__main__':unittest.main(verbosity=2)
