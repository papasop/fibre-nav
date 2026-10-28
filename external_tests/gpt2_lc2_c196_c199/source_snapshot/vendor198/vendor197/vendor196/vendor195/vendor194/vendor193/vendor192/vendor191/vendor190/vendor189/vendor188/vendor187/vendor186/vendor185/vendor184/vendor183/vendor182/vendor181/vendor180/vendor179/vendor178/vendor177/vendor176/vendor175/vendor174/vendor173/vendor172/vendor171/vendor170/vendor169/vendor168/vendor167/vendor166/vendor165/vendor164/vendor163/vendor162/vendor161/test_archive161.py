"""Complete six-original synthetic archive integration, no scientific model claims."""
import contextlib,copy,io,shutil,unittest
from core161 import *
import test_archive160 as pa
from inputs160 import load as load160
from inputs161 import load,audit160
from worker160 import run as run160
from fixture160 import Analytic
from fixture156 import pack
class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.parent=pa.Checks;cls.parent.setUpClass();cls.root=cls.parent.root;cls.paths=dict(cls.parent.paths);cls.pc=cls.parent.c
  with contextlib.redirect_stdout(io.StringIO()):cls.ctx=load160(cls.paths,cls.root/'context160',cls.pc)
  out=cls.root/'c160'
  with contextlib.redirect_stdout(io.StringIO()):s=run160(cls.paths,out,cls.pc,Analytic,lambda *a:copy.deepcopy(cls.ctx))
  need(s['complete'],'Synthetic C160 complete: '+str(s['errors']));shutil.copytree(ROOT/'vendor160',out/'source',ignore=shutil.ignore_patterns('__pycache__'));dump(out/'ORIGINAL_INPUTS.json',{k:dict(path=str(p),sha256=sha(p)) for k,p in cls.paths.items()});p=cls.root/'C160.zip';pack(out,p);cls.paths['C160']=p;cls.c=contract();cls.c.update(production=False,parent=cls.pc,pins={k:dict(path=str(p),sha256=sha(p)) for k,p in cls.paths.items()})
 @classmethod
 def tearDownClass(cls):cls.parent.tearDownClass()
 def test_all_six_originals_full84_C160_records_and_gradient_audit(self):
  with contextlib.redirect_stdout(io.StringIO()):ctx=load(self.paths,self.root/'six161',self.c)
  self.assertTrue(ctx['completed160']['all84_states_and1008_records_reconstructed']);self.assertEqual(ctx['completed160']['initial_native']['n_evaluated'],84);self.assertFalse(ctx['completed160']['archived_gradients_independently_recomputed'])
 def test_rebound_C160_gradient_aggregate_change_blocks(self):
  out=self.root/'forged160';shutil.copytree(self.root/'c160',out);p=out/'SUMMARY.json';s=clean_json(p.read_bytes());s['aggregate_gradient_dot_original_delta']+=1.;dump(p,s);z=self.root/'forged160.zip';pack(out,z)
  with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(ValueError):audit160(z,self.ctx,self.root/'forged160_check',self.c)
 def test_rebound_C160_final_coefficient_tensor_change_blocks(self):
  out=self.root/'forged_head160';shutil.copytree(self.root/'c160',out);p=out/'SEVEN_COEFFICIENT_TENSORS.npz'
  with np.load(p,allow_pickle=False) as z:ar={n:z[n] for n in z.files}
  ar['zero'][-1,-1,-1]+=.001;np.savez_compressed(p,**ar);z=self.root/'forged_head160.zip';pack(out,z)
  with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(ValueError):audit160(z,self.ctx,self.root/'forged_head160_check',self.c)
if __name__=='__main__':unittest.main(verbosity=2)
