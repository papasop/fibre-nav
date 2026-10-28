"""Actual CPU Torch tiny tied-model training/commit tests, not GPT-2 evidence."""
import copy,contextlib,io,tempfile,unittest
from unittest.mock import patch
import torch
from core161 import *
from fixture161 import context,inputs
from torch_smoke155 import ToyRuntime
from runtime161 import Trainer
class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);cls.ctx=context();rt=ToyRuntime(cls.root/'reference',dict(model_forwards=0));double=copy.deepcopy(rt.model).double();batch=rt.tok(rt.anchors)
  with torch.no_grad():logits=double(**batch,use_cache=False).logits[:,-1].numpy()
  cls.pins=dict(revision='synthetic-tiny-only',seed=155,lora_A_sha256='synthetic',chart_shapes=[[DIM]],symbols=[['no',0],['yes',1]],anchors=rt.anchors,response_ids=list(range(6)),sketch_sha256='synthetic');cls.ctx.update(modelpins=cls.pins,baseline_logits=dict(FP32_logits_cast64=rt.logits0.double().numpy(),FP64_logits=logits),baseline=dict(FP32_cached_response_reference=rt.response0.double().numpy().tolist(),FP64_response_reference=logits[:,:6].mean(0).tolist()))
  torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True);cls.chain=previous.constants(cls.ctx['data']);np.savez_compressed(cls.root/'FROZEN_FULL84_CHAIN_CONSTANTS.npz',**cls.chain);cls.points,cls.native,_=seal_head(cls.chain,cls.chain['P0'],cls.root/'initial','fixture',True)
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup()
 def factory(self,path,ctx,chain,points,native,coefs,calls):
  def nf(path,d,phase,counts):rt=ToyRuntime(path,counts);rt.pins=self.pins;return rt
  return Trainer(path,ctx,chain,points,native,coefs,calls,nf)
 def runtime(self):
  with contextlib.redirect_stdout(io.StringIO()):r=self.factory(self.root/('rt_'+self._testMethodName),copy.deepcopy(self.ctx),{n:a.copy() for n,a in self.chain.items()},self.points,self.native,dict(zero=self.chain['P0'].copy()),dict(model_forwards=0,functional_FP64_forwards=0,backward_calls=0,backward_attempts=0))
  return r
 def test_actual_full8_worker2187_forwards1344_backwards_final8_only(self):
  from worker161 import run
  with tempfile.TemporaryDirectory() as t:
   t=Path(t);paths,c=inputs(t/'inputs')
   with contextlib.redirect_stdout(io.StringIO()):s=run(paths,t/'out',c,self.factory,lambda *a:copy.deepcopy(self.ctx))
   self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['new_training_steps'],8);self.assertEqual(s['new_model_forwards'],2187);self.assertEqual(s['new_backward_calls'],1344);self.assertEqual(s['functional_FP64_forwards'],1514);self.assertEqual(s['records_completed'],1092);self.assertTrue(s['head_updated']);self.assertTrue(s['final_head_frozen']);self.assertFalse(s['LC2_closed']);self.assertFalse(s['candidate_selected']);self.assertEqual(s['native_final_round1']['n_evaluated'],84);self.assertLess(s['native_initial_round0']['passed'],84)
   with np.load(t/'out/UPDATE_STEP_08.npz',allow_pickle=False) as z:after=z['after'].copy()
   with np.load(t/'out/FIXED_FINAL_HEAD_STEP8.npz',allow_pickle=False) as z:np.testing.assert_array_equal(after,z['P'])
 def test_manual8_updates_match_real_Torch_Adam_clip_and_projection(self):
  rng=np.random.default_rng(161);p=rng.normal(size=(3,2,6))*.01;p[0,0,0]=.125;m=np.zeros_like(p);v=m.copy();leaf=torch.nn.Parameter(torch.tensor(p.copy(),dtype=torch.float64));opt=torch.optim.Adam([leaf],lr=RATE,betas=(.9,.999),eps=1e-8)
  for step in range(1,9):
   g=rng.normal(size=p.shape)*step;leaf.grad=torch.from_numpy(g.copy());norm=torch.linalg.vector_norm(leaf.grad);leaf.grad.mul_(min(1.,1./float(norm)));opt.step();opt.zero_grad(set_to_none=True)
   with torch.no_grad():leaf.clamp_(-.125,.125)
   p,m,v,_=update(p,g,m,v,step);np.testing.assert_allclose(leaf.detach().numpy(),p,rtol=1e-14,atol=1e-17)
 def test_midbatch_failure_retains_partial_counts_and_no_false_final(self):
  from worker161 import run
  with tempfile.TemporaryDirectory() as t:
   t=Path(t);paths,c=inputs(t/'inputs')
   def factory(*args):
    r=self.factory(*args);original=r.coefficient_record;counter=[0]
    def fail(*a,**kw):
     counter[0]+=1
     if counter[0]==13:raise RuntimeError('intentional partial gradient batch')
     return original(*a,**kw)
    r.coefficient_record=fail;return r
   with contextlib.redirect_stdout(io.StringIO()):s=run(paths,t/'out',c,factory,lambda *a:copy.deepcopy(self.ctx))
   self.assertFalse(s['complete']);self.assertEqual(s['new_training_steps'],0);self.assertEqual(s['records_completed'],96);self.assertEqual(s['new_backward_calls'],24);self.assertFalse(s['final_head_frozen']);self.assertFalse(s['LC2_closed']);self.assertFalse((t/'out/FIXED_FINAL_HEAD_STEP8.npz').exists())
 def test_same_function_coefficient_AD_directional_difference(self):
  r=self.runtime();sid=registry()[0]['state_id'];p=self.chain['P0'].copy();pt,_=mapped(self.chain,p,sid);row=r.coefficient_record(sid,p,pt,True);rng=np.random.default_rng(12);d=rng.normal(size=p.shape);d/=np.linalg.norm(d);h=1e-5;values=[]
  for sign in (-1,1):q=p+sign*h*d;point,_=mapped(self.chain,q,sid);o=r.coefficient_record(sid,q,point,False);values.append(sum(o['objective_parts'].values()))
  want=np.sum((np.asarray(row['response_gradient'])+row['guard_gradient'])*d);self.assertAlmostEqual((values[1]-values[0])/(2*h),want,delta=1e-5);r.verify(True)
 def test_changed_expected_point_blocks_before_model_measurement(self):
  r=self.runtime();sid=registry()[0]['state_id'];pt,_=mapped(self.chain,self.chain['P0'],sid);pt[0]+=.001;old=r.calls['model_forwards']
  with self.assertRaises(ValueError):r.coefficient_record(sid,self.chain['P0'],pt,True)
  self.assertEqual(r.calls['model_forwards'],old)
 def test_native_candidate_commit_repeat_and_restore_tied_weights(self):
  r=self.runtime();sid=registry()[0]['state_id'];p=self.native[(sid,'zero')];a=r.native_candidate(sid,p,safe.tensor_hash(p));b=r.native_candidate(sid,p,safe.tensor_hash(p));self.assertTrue(archive155.repeat_check(a['metrics'],b['metrics'])['within_replay_tolerance']);self.assertIs(r.rt.model.embed.weight,r.rt.model.proj.weight);np.testing.assert_array_equal(r.rt.params[0].numpy(),0.);r.verify(True)
 def test_native_failure_restores_physical_chart(self):
  r=self.runtime();sid=registry()[0]['state_id'];p=self.native[(sid,'zero')]
  with patch.object(r.rt.model,'forward',side_effect=RuntimeError('injected')):
   with self.assertRaises(RuntimeError):r.native_candidate(sid,p,safe.tensor_hash(p))
  np.testing.assert_array_equal(r.rt.params[0].numpy(),0.);r.verify(True)
 def test_wrong_native_hash_and_fixed_reference_mutation_block(self):
  r=self.runtime();sid=registry()[0]['state_id']
  with self.assertRaises(ValueError):r.native_candidate(sid,self.native[(sid,'zero')],'wrong')
  r.objective_response.add_(1.)
  with self.assertRaises(ValueError):r.verify()
 def test_FP64_exception_restores_functional_and_tied_state(self):
  r=self.runtime();sid=registry()[0]['state_id'];pt,_=mapped(self.chain,self.chain['P0'],sid)
  with patch.object(r.rt.model,'forward',side_effect=RuntimeError('injected')):
   with self.assertRaises(RuntimeError):r.coefficient_record(sid,self.chain['P0'],pt,True)
  self.assertIs(r.rt.model.embed.weight,r.rt.model.proj.weight);r.verify(True)
if __name__=='__main__':unittest.main(verbosity=2)
