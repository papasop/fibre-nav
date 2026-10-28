"""Actual CPU tiny Torch workflows. Upstream certificate is explicitly injected, not science evidence."""
import copy,contextlib,io,tempfile,unittest
from unittest.mock import patch
import torch
from core168 import *
from runtime168 import FixedProbeRuntime
from runtime165 import DirectionRuntime
from worker168 import run
class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  import torch_smoke165 as prior_smoke
  cls.prior=prior_smoke.Checks;cls.prior.setUpClass();cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);cls.ctx=copy.deepcopy(cls.prior.ctx);cls.chain=cls.prior.chain;cls.native_factory=staticmethod(cls.prior.prior.native_factory)
  p=cls.ctx['initial_head'].copy();root=cls.root/'archived_tiny_gradients';root.mkdir();np.savez_compressed(root/'FROZEN_FULL84_CHAIN_CONSTANTS.npz',**cls.chain);pt,nt,_=old.parent.seal_head(cls.chain,p,root/'anchor','anchor',True);calls=dict(model_forwards=0,functional_FP64_forwards=0,backward_attempts=0,backward_calls=0)
  # Only fixture construction evaluates gradients, outside the C168 worker.
  with contextlib.redirect_stdout(io.StringIO()):r=DirectionRuntime(root/'model_runtime',copy.deepcopy(cls.ctx),cls.chain,pt,nt,dict(zero=p),calls,native_factory=cls.native_factory)
  r.preseal(p,pt,root/'anchor','anchor');r.activate('anchor',p);rr=[r.coefficient_record(sid,p,pt[(sid,'zero')],True) for sid in IDS]
  cls.ctx.update(C165_gradient_records=rr,C165_FP64_anchor_records=rr,C165_native_anchor_records=copy.deepcopy(cls.ctx['parent_final_native_records']),C166_receipt=dict(decision='C166_READONLY_DIRECTION_AUDIT_COMPLETE_PARENT_DECISION_RETAINED'),C167_receipt=dict(decision='C167_RECORDED_LINEAR_PROBLEM_DIAGNOSTIC_COMPLETE_NO_MODEL_ACCEPTANCE',classification='RECORDED_GRADIENT_COMMON_DESCENT_CERTIFIED',synthetic_upstream_injection=True),fixed_C167_direction=np.ones_like(p))
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup();cls.prior.tearDownClass()
 def factory(self,*args):return FixedProbeRuntime(*args,native_factory=self.native_factory)
 def markers(self,root):
  root=Path(root);root.mkdir();paths={}
  for k in INPUT_KEYS:p=root/(k+'.zip');p.write_bytes(b'SYNTHETIC tiny upstream injection only');paths[k]=p
  c=contract();c.update(production=False,pins={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()});return paths,c
 def runtime(self):
  root=self.root/self._testMethodName;root.mkdir();p=self.ctx['initial_head'].copy();np.savez_compressed(root/'FROZEN_FULL84_CHAIN_CONSTANTS.npz',**self.chain);pt,nt,_=old.parent.seal_head(self.chain,p,root/'anchor','anchor',True)
  with contextlib.redirect_stdout(io.StringIO()):r=self.factory(root/'model_runtime',copy.deepcopy(self.ctx),self.chain,pt,nt,dict(zero=p),dict(model_forwards=0,functional_FP64_forwards=0,backward_attempts=0,backward_calls=0))
  return r,root,p,pt,nt
 def test_actual_full1848_3699_forwards_0_backwards_all672_saved_points(self):
  with tempfile.TemporaryDirectory() as t:
   t=Path(t);paths,c=self.markers(t/'inputs')
   with contextlib.redirect_stdout(io.StringIO()):s=run(paths,t/'out',c,self.factory,lambda *a:copy.deepcopy(self.ctx))
   self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['new_model_forwards'],3699);self.assertEqual(s['functional_FP64_forwards'],1346);self.assertEqual(s['new_backward_calls'],0);self.assertEqual(s['scalar_backward_attempts'],0);self.assertEqual(s['records_completed'],1848);self.assertEqual(s['actual_FP64_point_tensors_saved'],672);self.assertEqual(s['temporary_probe_heads'],6);self.assertFalse(s['LC2_closed']);self.assertFalse(s['all_thirteen_originals_independently_bound'])
   rec=clean_json((t/'out/ALL6_PROBES_SEALED_BEFORE_FIRST_STATE.json').read_bytes());self.assertEqual(rec['model_forwards_so_far'],3);self.assertEqual(rec['backward_calls_so_far'],0);self.assertEqual(len(rec['actual_point_seals']),8)
   rr=[clean_json(x) for x in (t/'out/ALL1848_RECORDS.jsonl').read_bytes().splitlines()];cache={}
   for row in rr:
    if 'actual_FP64_point_artifact' not in row:continue
    name=row['actual_FP64_point_artifact']
    if name not in cache:
     with np.load(t/'out'/name,allow_pickle=False) as z:cache[name]={k:z[k] for k in z.files}
    self.assertEqual(point_hash(cache[name][row['state_id']]),row['FP64_point_sha256']);self.assertNotIn('response_gradient',row)
   self.assertEqual(len(clean_json((t/'out/ALL504_SIGNED_LOSS_CHANGES.json').read_bytes())),504)
 def test_backward_request_rejected_before_any_model_forward(self):
  r,root,p,pt,nt=self.runtime();r.preseal(p,pt,root/'anchor','anchor');r.activate('anchor',p);n=r.calls['model_forwards']
  with self.assertRaises(ValueError):r.coefficient_record(IDS[0],p,pt[(IDS[0],'zero')],True)
  self.assertEqual(r.calls['model_forwards'],n);self.assertEqual(r.calls['backward_calls'],0)
 def test_saved_actual_FP64_tamper_blocks_before_forward(self):
  r,root,p,pt,nt=self.runtime();r.preseal(p,pt,root/'anchor','anchor');f=root/'anchor/ACTUAL_TORCH_FP64_POINTS.npz';f.write_bytes(f.read_bytes()+b'tampered');n=r.calls['model_forwards']
  with self.assertRaises(ValueError):r.activate('anchor',p)
  self.assertEqual(r.calls['model_forwards'],n)
 def test_wrong_actual_FP64_point_blocks_before_forward(self):
  r,root,p,pt,nt=self.runtime();r.preseal(p,pt,root/'anchor','anchor');r.activate('anchor',p);r.batch['point_ids'][IDS[0]]='0'*64;n=r.calls['model_forwards']
  with self.assertRaises(ValueError):r.coefficient_record(IDS[0],p,pt[(IDS[0],'zero')],False)
  self.assertEqual(r.calls['model_forwards'],n)
 def test_native_exception_restores_chart_and_tied_weights(self):
  r,root,p,pt,nt=self.runtime();point=nt[(IDS[0],'zero')]
  with patch.object(r.rt.model,'forward',side_effect=RuntimeError('injected native failure')):
   with self.assertRaises(RuntimeError):r.native_candidate(IDS[0],point,safe.tensor_hash(point))
  r.verify(True);np.testing.assert_array_equal(r.rt.params[0].numpy(),0.);self.assertIs(r.rt.model.embed.weight,r.rt.model.proj.weight)
 def test_functional_exception_restores_chart_and_tied_weights(self):
  r,root,p,pt,nt=self.runtime();r.preseal(p,pt,root/'anchor','anchor');r.activate('anchor',p)
  with patch.object(r.rt.model,'forward',side_effect=RuntimeError('injected functional failure')):
   with self.assertRaises(RuntimeError):r.coefficient_record(IDS[0],p,pt[(IDS[0],'zero')],False)
  r.verify(True);np.testing.assert_array_equal(r.rt.params[0].numpy(),0.);self.assertIs(r.rt.model.embed.weight,r.rt.model.proj.weight)
 def test_native_anchor_replay_drift_stops_first_state_retains_failure(self):
  with tempfile.TemporaryDirectory() as t:
   t=Path(t);paths,c=self.markers(t/'inputs');ctx=copy.deepcopy(self.ctx);ctx['C165_native_anchor_records']['0'][0]['metrics']['observed'][0]+=.01
   with contextlib.redirect_stdout(io.StringIO()):s=run(paths,t/'out',c,self.factory,lambda *a:ctx)
   self.assertFalse(s['complete']);self.assertEqual(s['records_completed'],1);self.assertEqual(s['new_backward_calls'],0)
 def test_FP64_anchor_replay_drift_stops_before_first_probe(self):
  with tempfile.TemporaryDirectory() as t:
   t=Path(t);paths,c=self.markers(t/'inputs');ctx=copy.deepcopy(self.ctx);ctx['C165_FP64_anchor_records'][0]['observed'][0]+=.001
   with contextlib.redirect_stdout(io.StringIO()):s=run(paths,t/'out',c,self.factory,lambda *a:ctx)
   self.assertFalse(s['complete']);self.assertEqual(s['records_completed'],85);self.assertEqual(s['new_backward_calls'],0)
 def test_reference_mutation_blocks(self):
  r,root,p,pt,nt=self.runtime();r.objective_response.add_(1.)
  with self.assertRaises(ValueError):r.verify()
 def test_new_heads_preserve_actual_Float64_map_no_FP32_cast(self):
  r,root,p,pt,nt=self.runtime();d=np.ones_like(p);heads,panel=step_panel(p,d);q=heads['plus_s3'];point,_=r.map_head(torch.tensor(q,dtype=torch.float64),IDS[0]);want,_=prior.prior.mapping(self.chain,q,IDS[0]);self.assertEqual(point.dtype,torch.float64);np.testing.assert_allclose(point.numpy(),want,rtol=0,atol=1e-12)
if __name__=='__main__':unittest.main(verbosity=2)
