"""Actual tiny CPU Torch runtime/restore probes; GPT-2 and ancestors injected."""
import copy,contextlib,io,tempfile,unittest
from unittest.mock import patch
import torch
from core172 import *
from runtime172 import FinalHeadProbes
from worker172 import run
class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  import torch_smoke170
  cls.parent=torch_smoke170.Checks;cls.parent.setUpClass();cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);cls.ctx=copy.deepcopy(cls.parent.ctx);p=cls.ctx['initial_head'].copy();cls.ctx.update(C170_final_head=p,C170_final_FP64=cls.ctx['C168_FP64_anchor_records'],C170_final_native=cls.ctx['C168_native_anchor_records'],C171_receipt=dict(decision=prior.DECISION,synthetic_upstream_injected=True));cls.chain=cls.parent.chain;cls.native_factory=staticmethod(cls.parent.native_factory)
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup();cls.parent.tearDownClass()
 def factory(self,*a):return FinalHeadProbes(*a,native_factory=self.native_factory)
 def inputs(self,root):
  root=Path(root);root.mkdir();paths={}
  for k in INPUT_KEYS:q=root/(k+'.zip');q.write_bytes(b'Synthetic upstream context injection only');paths[k]=q
  c=contract();c.update(production=False,pins={k:dict(path=str(q),sha256=sha(q)) for k,q in paths.items()});return paths,c
 def test_full_real_tiny_Torch_single_LP_six_probes_and_counts(self):
  with tempfile.TemporaryDirectory() as t:
   t=Path(t);paths,c=self.inputs(t/'inputs')
   with contextlib.redirect_stdout(io.StringIO()):s=run(paths,t/'out',c,self.factory,lambda *a:copy.deepcopy(self.ctx))
   self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['new_model_forwards'],3699);self.assertEqual(s['functional_FP64_forwards'],1346);self.assertEqual(s['new_backward_calls'],168);self.assertEqual(s['new_linear_program_calls'],1);self.assertEqual(s['temporary_probe_heads'],6);self.assertEqual(s['records_completed'],1848);self.assertEqual(s['actual_FP64_point_tensors_saved'],672);self.assertEqual(s['new_training_steps'],0);self.assertFalse(s['candidate_selected']);self.assertFalse(s['LC2_closed'])
 def runtime(self):
  root=self.root/self._testMethodName;root.mkdir();p=self.ctx['C170_final_head'].copy();np.savez_compressed(root/'FROZEN_FULL84_CHAIN_CONSTANTS.npz',**self.chain);pt,nt,_=seal_head(self.chain,p,root/'anchor','anchor',True)
  with contextlib.redirect_stdout(io.StringIO()):rt=self.factory(root/'model_runtime',copy.deepcopy(self.ctx),self.chain,pt,nt,dict(zero=p),dict(model_forwards=0,functional_FP64_forwards=0,backward_calls=0,backward_attempts=0))
  rt.preseal(p,pt,root/'anchor','anchor');rt.activate('anchor',p);return rt,root,p,pt,nt
 def test_actual_saved_point_tamper_blocks_before_forward(self):
  rt,root,p,pt,nt=self.runtime();f=root/'anchor/ACTUAL_TORCH_FP64_POINTS.npz';f.write_bytes(f.read_bytes()+b'changed');before=rt.calls['model_forwards']
  with self.assertRaises(ValueError):rt.coefficient_record(IDS[0],p,pt[(IDS[0],'zero')],True)
  self.assertEqual(rt.calls['model_forwards'],before)
 def test_probe_AD_request_rejected_before_backward(self):
  rt,root,p,pt,nt=self.runtime();rt.batch['folder']=root/'plus_s1';before=rt.calls['backward_calls']
  with self.assertRaises(ValueError):rt.coefficient_record(IDS[0],p,pt[(IDS[0],'zero')],True)
  self.assertEqual(rt.calls['backward_calls'],before)
 def test_functional_failure_restores_tied_weights_and_chart(self):
  rt,root,p,pt,nt=self.runtime()
  with patch.object(rt.rt.model,'forward',side_effect=RuntimeError('injected functional failure')):
   with self.assertRaises(RuntimeError):rt.coefficient_record(IDS[0],p,pt[(IDS[0],'zero')],True)
  rt.verify(True);self.assertIs(rt.rt.model.embed.weight,rt.rt.model.proj.weight);np.testing.assert_array_equal(rt.rt.params[0].numpy(),0.)
 def test_native_failure_restores_tied_weights_and_chart(self):
  rt,root,p,pt,nt=self.runtime();point=nt[(IDS[0],'zero')]
  with patch.object(rt.rt.model,'forward',side_effect=RuntimeError('injected native failure')):
   with self.assertRaises(RuntimeError):rt.native_candidate(IDS[0],point,safe.tensor_hash(point))
  rt.verify(True);self.assertIs(rt.rt.model.embed.weight,rt.rt.model.proj.weight);np.testing.assert_array_equal(rt.rt.params[0].numpy(),0.)
 def test_frozen_reference_mutation_blocks(self):
  rt,root,p,pt,nt=self.runtime();rt.objective_response.add_(1.)
  with self.assertRaises(ValueError):rt.verify()
if __name__=='__main__':unittest.main(verbosity=2)
