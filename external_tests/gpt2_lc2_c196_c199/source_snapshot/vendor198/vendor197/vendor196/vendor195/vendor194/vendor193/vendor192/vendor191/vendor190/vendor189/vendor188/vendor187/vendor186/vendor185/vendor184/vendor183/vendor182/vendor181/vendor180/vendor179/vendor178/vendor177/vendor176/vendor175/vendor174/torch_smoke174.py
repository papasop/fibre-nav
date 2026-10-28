"""Actual CPU tiny tied-model coordinate AD, LP, six probes and restoration."""
import copy,contextlib,io,tempfile,unittest
from unittest.mock import patch
import torch
from core174 import *
from runtime174 import CoordinateProbes
from worker174 import run
class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  import torch_smoke172
  cls.parent=torch_smoke172.Checks;cls.parent.setUpClass();cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);cls.ctx=copy.deepcopy(cls.parent.ctx);cls.ctx.update(C173_receipt=dict(decision=prior.DECISION,synthetic_upstream_injected=True),C172_anchor_FP64=cls.ctx['C170_final_FP64'],C172_anchor_native=cls.ctx['C170_final_native']);cls.chain=cls.parent.chain;cls.native_factory=staticmethod(cls.parent.native_factory)
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup();cls.parent.tearDownClass()
 def factory(self,*a):return CoordinateProbes(*a,native_factory=self.native_factory)
 def inputs(self,root):
  root=Path(root);root.mkdir();paths={}
  for k in INPUT_KEYS:q=root/(k+'.zip');q.write_bytes(b'Synthetic upstream context injection only');paths[k]=q
  c=contract();c.update(production=False,pins={k:dict(path=str(q),sha256=sha(q)) for k,q in paths.items()});return paths,c
 def runtime(self):
  root=self.root/self._testMethodName;root.mkdir();p=self.ctx['C170_final_head'].copy();np.savez_compressed(root/'FROZEN_FULL84_CHAIN_CONSTANTS.npz',**self.chain);pt,nt,_=seal_head(self.chain,p,root/'anchor','anchor',True);calls=dict(model_forwards=0,functional_FP64_forwards=0,backward_calls=0,backward_attempts=0,response_coordinate_backward_calls=0,response_Jacobian_blocks_completed=0)
  with contextlib.redirect_stdout(io.StringIO()):rt=self.factory(root/'model_runtime',copy.deepcopy(self.ctx),self.chain,pt,nt,dict(zero=p),calls)
  rt.preseal(p,pt,root/'anchor','anchor');rt.activate('anchor',p);return rt,root,p,pt,nt
 def test_complete_actual_Torch_all_coordinates_one_LP_six_probes(self):
  with tempfile.TemporaryDirectory() as t:
   t=Path(t);paths,c=self.inputs(t/'inputs')
   with contextlib.redirect_stdout(io.StringIO()):s=run(paths,t/'out',c,self.factory,lambda *a:copy.deepcopy(self.ctx))
   self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['new_model_forwards'],3699);self.assertEqual(s['new_backward_calls'],672);self.assertEqual(s['new_Jacobians'],84);self.assertEqual(s['new_response_coordinate_Jacobian_rows'],504);self.assertEqual(s['new_model_parameter_Jacobians'],0);self.assertEqual(s['new_linear_program_calls'],1);self.assertEqual(s['temporary_probe_heads'],6);self.assertFalse(s['LC2_closed']);self.assertFalse(s['candidate_selected']);self.assertEqual(s['new_training_steps'],0)
 def test_coordinate_AD_matches_same_function_central_differences(self):
  rt,root,p,pt,nt=self.runtime();sid=IDS[0];row=rt.coefficient_record(sid,p,pt[(sid,'zero')],True);rng=np.random.default_rng(174);d=rng.normal(size=p.shape);d/=np.linalg.norm(d);h=1e-5;obs=[]
  for sign in (-1,1):
   point,e=rt.map_head(torch.tensor(p+sign*h*d,dtype=torch.float64),sid);rt.authorized_point=point_hash(point.numpy())
   try:o,kl=rt.full_forward(point,False);obs.append(o.detach().numpy()[:6])
   finally:rt.authorized_point=None
  J=np.asarray(row['response_coordinate_gradients']);ad=np.sum(J*d,axis=(1,2,3));fd=(obs[1]-obs[0])/(2*h);np.testing.assert_allclose(fd,ad,atol=2e-10,rtol=2e-6);self.assertTrue(loss_chain_check(row,p.shape)['within_declared_chain_tolerance']);self.assertEqual(rt.calls['backward_calls'],8);self.assertEqual(rt.calls['response_coordinate_backward_calls'],6);rt.verify(True)
 def test_partial_coordinate_backward_exception_restores_model_and_counts(self):
  rt,root,p,pt,nt=self.runtime();orig=torch.autograd.grad;counter=[0]
  def grad(*a,**kw):
   counter[0]+=1
   if counter[0]==3:raise RuntimeError('injected partial coordinate AD')
   return orig(*a,**kw)
  with patch.object(torch.autograd,'grad',grad):
   with self.assertRaises(RuntimeError):rt.coefficient_record(IDS[0],p,pt[(IDS[0],'zero')],True)
  self.assertEqual(rt.calls['backward_attempts'],3);self.assertEqual(rt.calls['backward_calls'],2);self.assertEqual(rt.calls['response_coordinate_backward_calls'],2);self.assertEqual(rt.calls['response_Jacobian_blocks_completed'],0);self.assertIsNone(rt.authorized_point);rt.verify(True);self.assertIs(rt.rt.model.embed.weight,rt.rt.model.proj.weight)
 def test_probe_AD_is_rejected_before_any_backward(self):
  rt,root,p,pt,nt=self.runtime();rt.batch['folder']=root/'plus_s1'
  with self.assertRaises(ValueError):rt.coefficient_record(IDS[0],p,pt[(IDS[0],'zero')],True)
  self.assertEqual(rt.calls['backward_calls'],0)
 def test_saved_actual_point_tamper_blocks_before_forward(self):
  rt,root,p,pt,nt=self.runtime();f=root/'anchor/ACTUAL_TORCH_FP64_POINTS.npz';f.write_bytes(f.read_bytes()+b'changed');before=rt.calls['model_forwards']
  with self.assertRaises(ValueError):rt.coefficient_record(IDS[0],p,pt[(IDS[0],'zero')],True)
  self.assertEqual(rt.calls['model_forwards'],before)
 def test_native_exception_restores_physical_chart_and_tied_weights(self):
  rt,root,p,pt,nt=self.runtime();point=nt[(IDS[0],'zero')]
  with patch.object(rt.rt.model,'forward',side_effect=RuntimeError('injected native failure')):
   with self.assertRaises(RuntimeError):rt.native_candidate(IDS[0],point,safe.tensor_hash(point))
  rt.verify(True);self.assertIs(rt.rt.model.embed.weight,rt.rt.model.proj.weight);np.testing.assert_array_equal(rt.rt.params[0].numpy(),0.)
 def test_frozen_reference_mutation_blocks(self):
  rt,root,p,pt,nt=self.runtime();rt.objective_response.add_(1.)
  with self.assertRaises(ValueError):rt.verify()
if __name__=='__main__':unittest.main(verbosity=2)
