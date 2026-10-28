"""Actual CPU tiny tied-model AD, all eight LP updates, seals and restoration."""
import copy,contextlib,io,tempfile,unittest
from unittest.mock import patch
import torch
from core194 import *
from runtime194 import CoordinateTraining
from worker194 import run
class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  import torch_smoke192
  cls.parent=torch_smoke192.Checks;cls.parent.setUpClass();cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);cls.ctx=copy.deepcopy(cls.parent.ctx);cls.chain=cls.parent.chain;cls.native_factory=staticmethod(cls.parent.native_factory)
  from worker192 import run as earlier_run
  from runtime192 import CoordinateTraining as EarlierRuntime
  from fixture194 import adopt
  priorctx=copy.deepcopy(cls.ctx);priorctx['C191_receipt']=dict(decision=previous_training.prior.DECISION,synthetic_upstream_injected=True)
  paths={}
  for k in previous_training.INPUT_KEYS:
   f=cls.root/(k+'.zip');f.write_bytes(b'Synthetic previous Torch run upstream context');paths[k]=f
  c=previous_training.contract();c.update(production=False,pins={k:dict(path=str(f),sha256=sha(f)) for k,f in paths.items()})
  def factory(*a):return EarlierRuntime(*a,native_factory=cls.native_factory)
  with contextlib.redirect_stdout(io.StringIO()):result=earlier_run(paths,cls.root/'actual_previous_C192',c,factory,lambda *a:copy.deepcopy(priorctx))
  need(result['complete'],'Actual tiny Torch previous C192: '+str(result['errors']));adopt(cls.ctx,cls.root/'actual_previous_C192');need(np.any(cls.ctx['C192_final_head']!=cls.ctx['C180_final_head']),'Nonzero actual Torch continuation anchor')
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup();cls.parent.tearDownClass()
 def factory(self,*a):return CoordinateTraining(*a,native_factory=self.native_factory)
 def inputs(self,root):
  root=Path(root);root.mkdir();paths={}
  for k in INPUT_KEYS:q=root/(k+'.zip');q.write_bytes(b'Synthetic upstream context injection only');paths[k]=q
  c=contract();c.update(production=False,pins={k:dict(path=str(q),sha256=sha(q)) for k,q in paths.items()});return paths,c
 def runtime(self):
  root=self.root/self._testMethodName;root.mkdir();p=self.ctx['C192_final_head'].copy();np.savez_compressed(root/'FROZEN_FULL84_CHAIN_CONSTANTS.npz',**self.chain);folder=root/'step_01_before';pt,nt,_=seal_head(self.chain,p,folder,'training_before_local_step1',True);calls=dict(model_forwards=0,functional_FP64_forwards=0,backward_calls=0,backward_attempts=0,response_coordinate_backward_calls=0,response_Jacobian_blocks_completed=0)
  with contextlib.redirect_stdout(io.StringIO()):rt=self.factory(root/'model_runtime',copy.deepcopy(self.ctx),self.chain,pt,nt,dict(zero=p),calls)
  rt.preseal(p,pt,folder,'training_before_local_step1');rt.activate('step_01_before',p);return rt,root,p,pt,nt
 def test_complete_actual_CPU_Torch_fixed8_LPs_AD_updates_and_repeats(self):
  with tempfile.TemporaryDirectory() as t:
   t=Path(t);paths,c=self.inputs(t/'inputs')
   with contextlib.redirect_stdout(io.StringIO()):s=run(paths,t/'out',c,self.factory,lambda *a:copy.deepcopy(self.ctx))
   self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['new_training_steps'],8);self.assertEqual(s['cumulative_C176_C178_C180_C182_C184_C186_C188_C190_C192_C194_training_steps'],80);self.assertEqual(s['new_model_forwards'],2355);self.assertEqual(s['functional_FP64_forwards'],1682);self.assertEqual(s['new_backward_calls'],5376);self.assertEqual(s['new_Jacobians'],672);self.assertEqual(s['new_response_coordinate_Jacobian_rows'],4032);self.assertEqual(s['new_model_parameter_Jacobians'],0);self.assertEqual(s['new_linear_program_calls'],8);self.assertEqual(s['actual_FP64_point_tensors_saved'],840);self.assertTrue(s['FP64_final_repeat_consistent']);self.assertFalse(s['LC2_closed']);self.assertFalse(s['candidate_selected']);self.assertFalse(s['best_checkpoint_selected']);self.assertEqual(s['endpoint_precision_margins']['rows'],336);self.assertEqual(s['endpoint_precision_margins']['additional_model_forwards'],0)
   with np.load(t/'out/INITIAL_C192_FIXED_FINAL_HEAD.npz') as z:np.testing.assert_array_equal(z['P'],self.ctx['C192_final_head']);self.assertFalse(np.array_equal(z['P'],self.ctx['C178_final_head']))
   with np.load(t/'out/UPDATE_STEP_08.npz') as z:q=z['after'].copy()
   with np.load(t/'out/FIXED_FINAL_HEAD_STEP8.npz') as z:np.testing.assert_array_equal(q,z['P'])
 def test_coordinate_AD_matches_same_function_directional_difference(self):
  rt,root,p,pt,nt=self.runtime();sid=IDS[0];row=rt.coefficient_record(sid,p,pt[(sid,'zero')],True);rng=np.random.default_rng(194);d=rng.normal(size=p.shape);d/=np.linalg.norm(d);h=1e-5;obs=[]
  for sign in (-1,1):
   point,e=rt.map_head(torch.tensor(p+sign*h*d,dtype=torch.float64),sid);rt.authorized_point=point_hash(point.numpy())
   try:o,kl=rt.full_forward(point,False);obs.append(o.detach().numpy()[:6])
   finally:rt.authorized_point=None
  ad=np.sum(np.asarray(row['response_coordinate_gradients'])*d,axis=(1,2,3));fd=(obs[1]-obs[0])/(2*h);np.testing.assert_allclose(fd,ad,atol=2e-10,rtol=2e-6);self.assertTrue(source174.loss_chain_check(row,p.shape)['within_declared_chain_tolerance']);self.assertEqual(rt.calls['backward_calls'],8);rt.verify(True)
 def test_partial_coordinate_AD_exception_restores_and_counts(self):
  rt,root,p,pt,nt=self.runtime();old=torch.autograd.grad;count=[0]
  def grad(*a,**kw):
   count[0]+=1
   if count[0]==3:raise RuntimeError('injected partial coordinate AD')
   return old(*a,**kw)
  with patch.object(torch.autograd,'grad',grad):
   with self.assertRaises(RuntimeError):rt.coefficient_record(IDS[0],p,pt[(IDS[0],'zero')],True)
  self.assertEqual(rt.calls['backward_attempts'],3);self.assertEqual(rt.calls['backward_calls'],2);self.assertEqual(rt.calls['response_coordinate_backward_calls'],2);self.assertEqual(rt.calls['response_Jacobian_blocks_completed'],0);self.assertIsNone(rt.authorized_point);rt.verify(True);self.assertIs(rt.rt.model.embed.weight,rt.rt.model.proj.weight)
 def test_final_head_AD_rejected_before_backward(self):
  rt,root,p,pt,nt=self.runtime();rt.batch['folder']=root/'final_step8'
  with self.assertRaises(ValueError):rt.coefficient_record(IDS[0],p,pt[(IDS[0],'zero')],True)
  self.assertEqual(rt.calls['backward_calls'],0)
 def test_saved_current_point_tamper_blocks_before_forward(self):
  rt,root,p,pt,nt=self.runtime();f=root/'step_01_before/ACTUAL_TORCH_FP64_POINTS.npz';f.write_bytes(f.read_bytes()+b'tamper');before=rt.calls['model_forwards']
  with self.assertRaises(ValueError):rt.coefficient_record(IDS[0],p,pt[(IDS[0],'zero')],True)
  self.assertEqual(rt.calls['model_forwards'],before)
 def test_native_failure_restores_chart_and_tied_weights(self):
  rt,root,p,pt,nt=self.runtime();point=nt[(IDS[0],'zero')]
  with patch.object(rt.rt.model,'forward',side_effect=RuntimeError('injected native failure')):
   with self.assertRaises(RuntimeError):rt.native_candidate(IDS[0],point,safe.tensor_hash(point))
  rt.verify(True);self.assertIs(rt.rt.model.embed.weight,rt.rt.model.proj.weight);np.testing.assert_array_equal(rt.rt.params[0].numpy(),0.)
 def test_frozen_reference_mutation_blocks(self):
  rt,root,p,pt,nt=self.runtime();rt.objective_response.add_(1.)
  with self.assertRaises(ValueError):rt.verify()
if __name__=='__main__':unittest.main(verbosity=2)
