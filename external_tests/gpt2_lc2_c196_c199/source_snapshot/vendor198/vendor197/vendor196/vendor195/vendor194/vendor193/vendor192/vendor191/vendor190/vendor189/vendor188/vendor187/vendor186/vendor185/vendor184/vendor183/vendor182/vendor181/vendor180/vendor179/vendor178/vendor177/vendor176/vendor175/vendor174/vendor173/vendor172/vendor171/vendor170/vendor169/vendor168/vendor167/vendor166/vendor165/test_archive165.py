"""Full ten synthetic originals; real readonly loader and rebound-tampering controls."""
import copy,contextlib,io,shutil,tempfile,unittest
from core165 import *
from inputs165 import load,check164
from fixture164 import changed,json_change,pack
class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  import test_archive164 as old
  import worker164
  cls.prior=old.Checks;cls.prior.setUpClass();cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);paths=dict(cls.prior.paths);pc=cls.prior.c;result=cls.root/'C164_result'
  with contextlib.redirect_stdout(io.StringIO()):s=worker164.run(paths,result,pc)
  need(s['complete'],'Synthetic C164 formal loader: '+str(s['errors']));shutil.copytree(ROOT/'vendor164',result/'source',ignore=shutil.ignore_patterns('__pycache__'));p=cls.root/'C164.zip';pack(result,p);paths['C164']=p;cls.paths=paths;cls.c=contract();cls.c.update(production=False,parent=pc,pins={k:dict(path=str(v),sha256=sha(v)) for k,v in paths.items()})
  with contextlib.redirect_stdout(io.StringIO()):cls.ctx=load(paths,cls.root/'ten_bound',cls.c)
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup();cls.prior.tearDownClass()
 def test_all10_originals_bound_before_model_and_exact_C163_final_anchor(self):
  self.assertEqual(self.ctx['C164_receipt']['actual_FP64_point_arrays_rehashed'],756);np.testing.assert_array_equal(self.ctx['initial_head'],self.ctx['C163_audit']['heads'][8]);self.assertEqual(len(self.ctx['parent_final_FP64_records']),84);self.assertFalse('torch' in sys.modules)
 def test_rebound_C164_summary_change_blocks(self):
  p=changed(self.paths['C164'],self.root/'forged164.zip','SUMMARY.json',json_change(lambda a:a['diagnostics'].__setitem__('recorded_smooth_gradient_uphill_updates',999)),True);out=self.root/'ten_bound';d=clean_json((out/'diagnostics/DIAGNOSTIC_SUMMARY.json').read_bytes())
  with self.assertRaises(ValueError):check164(p,self.ctx['C163_audit'],d,out,self.c)
 def test_rebound_C164_source_change_blocks(self):
  p=changed(self.paths['C164'],self.root/'forged_source164.zip','source/core164.py',lambda b:b+b'\n# changed\n',True);out=self.root/'ten_bound';d=clean_json((out/'diagnostics/DIAGNOSTIC_SUMMARY.json').read_bytes())
  with self.assertRaises(ValueError):check164(p,self.ctx['C163_audit'],d,out,self.c)
if __name__=='__main__':unittest.main(verbosity=2)
