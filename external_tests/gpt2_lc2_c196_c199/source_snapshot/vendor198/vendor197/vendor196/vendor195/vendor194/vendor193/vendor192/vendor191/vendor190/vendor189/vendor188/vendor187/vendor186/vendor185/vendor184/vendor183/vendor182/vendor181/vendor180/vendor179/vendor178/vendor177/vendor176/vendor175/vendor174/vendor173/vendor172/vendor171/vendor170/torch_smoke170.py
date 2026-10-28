"""Actual tiny CPU Torch tests. All upstream contexts are synthetic injections."""
import copy,contextlib,io,tempfile,unittest
from unittest.mock import patch
import torch
from core170 import *
from runtime170 import Relinearised
from worker170 import run
class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  import torch_smoke161 as parent_smoke
  cls.prior=parent_smoke.Checks;cls.prior.setUpClass();cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);cls.ctx=copy.deepcopy(cls.prior.ctx);cls.chain=cls.prior.chain;p=cls.chain['P0'].copy();cls.ctx['initial_head']=p;cls.ctx['C169_receipt']=dict(decision=prior.DECISION,synthetic_upstream_injected=True)
  def nf(path,d,phase,calls):
   from torch_smoke155 import ToyRuntime
   r=ToyRuntime(path,calls);r.pins=cls.prior.pins;return r
  cls.native_factory=staticmethod(nf);folder=cls.root/'reference';folder.mkdir();np.savez_compressed(folder/'FROZEN_FULL84_CHAIN_CONSTANTS.npz',**cls.chain);pt,nt,_=seal_head(cls.chain,p,folder/'anchor','anchor',True);calls=dict(model_forwards=0,functional_FP64_forwards=0,backward_calls=0,backward_attempts=0)
  with contextlib.redirect_stdout(io.StringIO()):rt=Relinearised(folder/'model_runtime',copy.deepcopy(cls.ctx),cls.chain,pt,nt,dict(zero=p),calls,native_factory=nf)
  rt.preseal(p,pt,folder/'anchor','anchor');rt.activate('anchor',p);rr=[rt.coefficient_record(sid,p,pt[(sid,'zero')],False) for sid in IDS];nr=[rt.native_candidate(sid,nt[(sid,'zero')],safe.tensor_hash(nt[(sid,'zero')])) for sid in IDS];cls.ctx.update(C168_FP64_anchor_records=rr,C168_native_anchor_records={'0':copy.deepcopy(nr),'1':copy.deepcopy(nr)})
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup();cls.prior.tearDownClass()
 def factory(self,*a):return Relinearised(*a,native_factory=self.native_factory)
 def markers(self,root):
  root=Path(root);root.mkdir();paths={}
  for k in INPUT_KEYS:q=root/(k+'.zip');q.write_bytes(b'Synthetic upstream context injection only');paths[k]=q
  c=contract();c.update(production=False,pins={k:dict(path=str(q),sha256=sha(q)) for k,q in paths.items()});return paths,c
 def runtime(self):
  root=self.root/self._testMethodName;root.mkdir();p=self.ctx['initial_head'].copy();np.savez_compressed(root/'FROZEN_FULL84_CHAIN_CONSTANTS.npz',**self.chain);pt,nt,_=seal_head(self.chain,p,root/'batch','batch',True)
  with contextlib.redirect_stdout(io.StringIO()):rt=self.factory(root/'model_runtime',copy.deepcopy(self.ctx),self.chain,pt,nt,dict(zero=p),dict(model_forwards=0,functional_FP64_forwards=0,backward_calls=0,backward_attempts=0))
  rt.preseal(p,pt,root/'batch','batch');rt.activate('batch',p);return rt,root,p,pt,nt
 def test_full_actual_Torch_eight_LP_training_and_fixed_final8(self):
  with tempfile.TemporaryDirectory() as t:
   t=Path(t);paths,c=self.markers(t/'inputs')
   with contextlib.redirect_stdout(io.StringIO()):s=run(paths,t/'out',c,self.factory,lambda *a:copy.deepcopy(self.ctx))
   self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['new_training_steps'],8);self.assertEqual(s['new_model_forwards'],2187);self.assertEqual(s['new_backward_calls'],1344);self.assertEqual(s['functional_FP64_forwards'],1514);self.assertEqual(s['records_completed'],1092);self.assertEqual(s['actual_FP64_point_tensors_saved'],756);self.assertEqual(s['nonzero_head_updates'],8);self.assertEqual(s['new_linear_program_calls'],8);self.assertFalse(s['all_fifteen_originals_independently_bound']);self.assertFalse(s['LC2_closed']);self.assertFalse(s['candidate_selected'])
   with np.load(t/'out/UPDATE_STEP_08.npz',allow_pickle=False) as z:after=z['after'].copy()
   with np.load(t/'out/FIXED_FINAL_HEAD_STEP8.npz',allow_pickle=False) as z:np.testing.assert_array_equal(z['P'],after)
   for k in range(1,9):
    c=clean_json((t/'out'/('step_%02d_before'%k)/'EXACT_RECORDED_DIRECTION_CHECK.json').read_bytes());self.assertTrue(c['strict_recorded_common_descent']);self.assertEqual(len(c['all84_exact_contractions']),84)
 def test_unresolved_direction_stops_without_update_or_retry(self):
  with tempfile.TemporaryDirectory() as t:
   t=Path(t);paths,c=self.markers(t/'inputs');calls=[]
   class Result(dict):
    def __getattr__(self,k):return self[k]
   def backend(cost,**kw):calls.append(1);return Result(x=np.zeros(len(cost)),success=False,status=1)
   with contextlib.redirect_stdout(io.StringIO()):s=run(paths,t/'out',c,self.factory,lambda *a:copy.deepcopy(self.ctx),backend)
   self.assertFalse(s['complete']);self.assertEqual(s['new_training_steps'],0);self.assertEqual(s['new_linear_program_calls'],1);self.assertEqual(len(calls),1);self.assertFalse(s['final_head_frozen']);self.assertFalse((t/'out/FIXED_FINAL_HEAD_STEP8.npz').exists());self.assertEqual(s['new_backward_calls'],168)
 def test_initial_native_drift_blocks_before_gradients(self):
  with tempfile.TemporaryDirectory() as t:
   t=Path(t);paths,c=self.markers(t/'inputs');ctx=copy.deepcopy(self.ctx);ctx['C168_native_anchor_records']['0'][0]['metrics']['observed'][0]+=.01
   with contextlib.redirect_stdout(io.StringIO()):s=run(paths,t/'out',c,self.factory,lambda *a:ctx)
   self.assertFalse(s['complete']);self.assertEqual(s['records_completed'],1);self.assertEqual(s['new_backward_calls'],0);self.assertEqual(s['new_linear_program_calls'],0)
 def test_initial_FP64_drift_blocks_before_first_LP(self):
  with tempfile.TemporaryDirectory() as t:
   t=Path(t);paths,c=self.markers(t/'inputs');ctx=copy.deepcopy(self.ctx);ctx['C168_FP64_anchor_records'][0]['observed'][0]+=.001
   with contextlib.redirect_stdout(io.StringIO()):s=run(paths,t/'out',c,self.factory,lambda *a:ctx)
   self.assertFalse(s['complete']);self.assertEqual(s['records_completed'],85);self.assertEqual(s['new_training_steps'],0);self.assertEqual(s['new_linear_program_calls'],0)
 def test_mid_gradient_failure_preserves_partial_records(self):
  with tempfile.TemporaryDirectory() as t:
   t=Path(t);paths,c=self.markers(t/'inputs')
   def factory(*args):
    r=self.factory(*args);orig=r.coefficient_record;counter=[0]
    def record(*a,**k):
     counter[0]+=1
     if counter[0]==13:raise RuntimeError('injected partial gradient batch')
     return orig(*a,**k)
    r.coefficient_record=record;return r
   with contextlib.redirect_stdout(io.StringIO()):s=run(paths,t/'out',c,factory,lambda *a:copy.deepcopy(self.ctx))
   self.assertFalse(s['complete']);self.assertEqual(s['records_completed'],96);self.assertEqual(s['new_backward_calls'],24);self.assertEqual(s['new_linear_program_calls'],0)
 def test_same_FP64_response_gradient_direction_matches_finite_difference(self):
  r,root,p,pt,nt=self.runtime();row=r.coefficient_record(IDS[0],p,pt[(IDS[0],'zero')],True);d=np.ones_like(p);h=1e-6;values=[]
  for sign in (-1,1):
   q=p+sign*h*d;point,e=r.map_head(torch.tensor(q,dtype=torch.float64),IDS[0]);r.authorized_point=point_hash(point.numpy())
   try:o,kl=r.full_forward(point,False)
   finally:r.authorized_point=None
   values.append(float(r.terms(o,kl,point,e,IDS[0])['response_loss']))
  ad=float(np.sum(np.asarray(row['response_gradient'])*d));self.assertAlmostEqual((values[1]-values[0])/(2*h),ad,delta=1e-5*max(1.,abs(ad)));r.verify(True)
 def test_saved_actual_point_tamper_blocks_before_model(self):
  r,root,p,pt,nt=self.runtime();f=root/'batch/ACTUAL_TORCH_FP64_POINTS.npz';f.write_bytes(f.read_bytes()+b'tampered');before=r.calls['model_forwards']
  with self.assertRaises(ValueError):r.coefficient_record(IDS[0],p,pt[(IDS[0],'zero')],True)
  self.assertEqual(r.calls['model_forwards'],before)
 def test_native_failure_restores_tied_weights_and_physical_chart(self):
  r,root,p,pt,nt=self.runtime();point=nt[(IDS[0],'zero')]
  with patch.object(r.rt.model,'forward',side_effect=RuntimeError('injected native failure')):
   with self.assertRaises(RuntimeError):r.native_candidate(IDS[0],point,safe.tensor_hash(point))
  r.verify(True);self.assertIs(r.rt.model.embed.weight,r.rt.model.proj.weight);np.testing.assert_array_equal(r.rt.params[0].numpy(),0.)
 def test_functional_failure_restores_tied_weights_and_physical_chart(self):
  r,root,p,pt,nt=self.runtime()
  with patch.object(r.rt.model,'forward',side_effect=RuntimeError('injected FP64 failure')):
   with self.assertRaises(RuntimeError):r.coefficient_record(IDS[0],p,pt[(IDS[0],'zero')],True)
  r.verify(True);self.assertIs(r.rt.model.embed.weight,r.rt.model.proj.weight);np.testing.assert_array_equal(r.rt.params[0].numpy(),0.)
 def test_fixed_reference_mutation_blocks(self):
  r,root,p,pt,nt=self.runtime();r.objective_response.add_(1.)
  with self.assertRaises(ValueError):r.verify()
if __name__=='__main__':unittest.main(verbosity=2)
