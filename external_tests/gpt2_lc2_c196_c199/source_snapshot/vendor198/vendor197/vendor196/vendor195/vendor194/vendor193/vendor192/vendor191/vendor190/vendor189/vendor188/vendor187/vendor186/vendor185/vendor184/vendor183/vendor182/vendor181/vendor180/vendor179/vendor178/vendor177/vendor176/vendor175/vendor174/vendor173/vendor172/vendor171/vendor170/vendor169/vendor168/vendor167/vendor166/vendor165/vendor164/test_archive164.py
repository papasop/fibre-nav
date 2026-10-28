"""Full nine-original synthetic binding, independent of model libraries."""
import contextlib,io,shutil,tempfile,unittest
from unittest.mock import patch
from core164 import *
from fixture164 import make163
from worker164 import run
class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  import test_archive163 as prior
  import core159,core160,worker159,worker160
  from synthetic_history164 import numpy_fixture_freeze
  cls.prior=prior.Checks
  with patch.object(worker159,'freeze',numpy_fixture_freeze(core159)),patch.object(worker160,'freeze',numpy_fixture_freeze(core160)):cls.prior.setUpClass()
  cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);pc=cls.prior.c['parent']
  cls.paths,cls.c,cls.ctx=make163(cls.prior.ctx,cls.prior.paths,pc,cls.root/'C163')
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup();cls.prior.tearDownClass()
 def test_all9_original_archives_full_worker_no_injected_binding(self):
  with contextlib.redirect_stdout(io.StringIO()):s=run(self.paths,self.root/'nine_archive_result',self.c)
  self.assertTrue(s['complete'],s['errors']);self.assertTrue(s['all_original_parent_archives_independently_bound']);self.assertTrue(s['Torch_FP64_forward_point_bytes_independently_rehashed']);self.assertEqual(s['actual_Torch_FP64_points_rehashed'],756);self.assertFalse(s['LC2_closed']);self.assertFalse('torch' in sys.modules)
 def test_production_pin_rejects_fixture_before_loader(self):
  with contextlib.redirect_stdout(io.StringIO()):s=run(self.paths,self.root/'wrongproduction',contract())
  self.assertFalse(s['complete']);self.assertFalse(s['Torch_FP64_forward_point_bytes_independently_rehashed']);self.assertEqual(s['new_training_steps'],0)
if __name__=='__main__':unittest.main(verbosity=2)
