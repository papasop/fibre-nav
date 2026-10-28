"""Actual CPU Torch tiny-model restart, smooth gradient and saved-point checks."""
import copy,contextlib,io,tempfile,unittest
from unittest.mock import patch
import torch
from core163 import *
from fixture161 import context as context161,inputs as inputs161
from torch_smoke155 import ToyRuntime
from runtime161 import Trainer
from runtime163 import SmoothTrainer
from worker161 import run as run161
class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);ctx=context161();rt=ToyRuntime(cls.root/'reference',dict(model_forwards=0));double=copy.deepcopy(rt.model).double();batch=rt.tok(rt.anchors)
  with torch.no_grad():logits=double(**batch,use_cache=False).logits[:,-1].numpy()
  cls.pins=dict(revision='synthetic-tiny-only',seed=155,lora_A_sha256='synthetic',chart_shapes=[[DIM]],symbols=[['no',0],['yes',1]],anchors=rt.anchors,response_ids=list(range(6)),sketch_sha256='synthetic');ctx.update(modelpins=cls.pins,baseline_logits=dict(FP32_logits_cast64=rt.logits0.double().numpy(),FP64_logits=logits),baseline=dict(FP32_cached_response_reference=rt.response0.double().numpy().tolist(),FP64_response_reference=logits[:,:6].mean(0).tolist()))
  def native(path,d,phase,counts):r=ToyRuntime(path,counts);r.pins=cls.pins;return r
  cls.native_factory=staticmethod(native);paths,cc=inputs161(cls.root/'parents')
  with contextlib.redirect_stdout(io.StringIO()):s=run161(paths,cls.root/'parent161',cc,lambda *a:Trainer(*a,native_factory=native),lambda *a:copy.deepcopy(ctx))
  need(s['complete'],'Actual tiny C161 source head: '+str(s['errors']))
  with np.load(cls.root/'parent161/FIXED_FINAL_HEAD_STEP8.npz',allow_pickle=False) as z:p=z['P'].copy()
  ns=clean_json((cls.root/'parent161/ALL336_NATIVE_ENDPOINTS.json').read_bytes());ctx.update(initial_head=p,parent_final_FP64_records=clean_json((cls.root/'parent161/FINAL_FP64_OBJECTIVE_ALL84.json').read_bytes()),parent_final_native_records={str(t):ns['native_final'+str(t)] for t in (0,1)},C161_audit=dict(natives=ns),C162_receipt=dict(decision='C162_READONLY_TRAINING_AUDIT_COMPLETE_PARENT_DECISION_RETAINED'));cls.ctx=ctx;cls.chain=parent.previous.constants(ctx['data'])
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup()
 def factory(self,*args):return SmoothTrainer(*args,native_factory=self.native_factory)
 def new_runtime(self):
  root=self.root/self._testMethodName;root.mkdir();np.savez_compressed(root/'FROZEN_FULL84_CHAIN_CONSTANTS.npz',**self.chain);p=self.ctx['initial_head'].copy();pts,native,_=seal_head(self.chain,p,root/'initial','initial',True)
  with contextlib.redirect_stdout(io.StringIO()):r=self.factory(root/'model_runtime',copy.deepcopy(self.ctx),{n:a.copy() for n,a in self.chain.items()},pts,native,dict(zero=p.copy()),dict(model_forwards=0,functional_FP64_forwards=0,backward_attempts=0,backward_calls=0))
  self.rt_root=root;self.point_maps=pts;self.p=p;return r
 def marker_inputs(self,root):
  root=Path(root);root.mkdir();paths={}
  for k in INPUT_KEYS:p=root/(k+'.zip');p.write_bytes(b'Tiny injected binding only');paths[k]=p
  c=contract();c.update(production=False,pins={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()});return paths,c
 def test_actual_full8_workflow_counts_final_head_and_all756_point_bytes(self):
  from worker163 import run
  with tempfile.TemporaryDirectory() as t:
   t=Path(t);paths,c=self.marker_inputs(t/'inputs')
   with contextlib.redirect_stdout(io.StringIO()):s=run(paths,t/'out',c,self.factory,lambda *a:copy.deepcopy(self.ctx))
   self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['new_training_steps'],8);self.assertEqual(s['new_model_forwards'],2187);self.assertEqual(s['new_backward_calls'],1344);self.assertEqual(s['functional_FP64_forwards'],1514);self.assertEqual(s['actual_FP64_point_tensors_saved'],756);self.assertEqual(s['records_completed'],1092);self.assertTrue(s['initial_FP64_replay_matched_C161_final']);self.assertFalse(s['LC2_closed']);self.assertFalse(s['candidate_selected']);self.assertEqual(s['native_final_round1']['n_evaluated'],84)
   records=[clean_json(line) for line in (t/'out/ALL1092_RECORDS.jsonl').read_bytes().splitlines()];by={r['actual_FP64_point_artifact']:[] for r in records if 'actual_FP64_point_artifact' in r}
   for r in records:
    if 'actual_FP64_point_artifact' in r:by[r['actual_FP64_point_artifact']].append(r)
   self.assertEqual(len(by),9)
   for name,rr in by.items():
    with np.load(t/'out'/name,allow_pickle=False) as z:
     self.assertEqual(set(z.files),set(IDS))
     for row in rr:self.assertEqual(point_hash(z[row['state_id']]),row['FP64_point_sha256'])
   with np.load(t/'out/FIXED_FINAL_HEAD_STEP8.npz',allow_pickle=False) as z:final=z['P'].copy()
   with np.load(t/'out/UPDATE_STEP_08.npz',allow_pickle=False) as z:np.testing.assert_array_equal(final,z['after'])
   with np.load(t/'out/INITIAL_C161_STEP8_AND_FRESH_MOMENTS.npz',allow_pickle=False) as z:np.testing.assert_array_equal(z['P'],self.ctx['initial_head']);np.testing.assert_array_equal(z['m'],0.);np.testing.assert_array_equal(z['v'],0.)
 def test_manual_aggregation_matches_full_actual_Torch_smooth_objective(self):
  r=self.new_runtime();r.prepare_batch(self.p,self.point_maps,self.rt_root/'initial','tiny');rr=[]
  for sid in IDS:rr.append(r.coefficient_record(sid,self.p,self.point_maps[(sid,'zero')],True))
  manual,a=aggregate(rr,self.p.shape);leaf=torch.tensor(self.p.copy(),dtype=torch.float64,requires_grad=True);R=[];G=[]
  for sid in IDS:
   point,e=r.map_head(leaf,sid);r.authorized_point=r.batch['point_ids'][sid]
   try:o,kl=r.full_forward(point,True)
   finally:r.authorized_point=None
   parts=r.terms(o,kl,point,e,sid);R.append(parts['response_loss']);G.append(sum(v for n,v in parts.items() if n!='response_loss'))
  values=torch.stack(R);J=.9*TAU*(torch.logsumexp(values/TAU,0)-np.log(84))+.1*values.mean()+torch.stack(G).mean();actual=torch.autograd.grad(J,leaf)[0].detach().numpy();np.testing.assert_allclose(actual,manual,rtol=1e-9,atol=1e-10);self.assertAlmostEqual(float(J.detach()),a['objective'],delta=1e-10);r.verify(True)
 def test_saved_actual_FP64_artifact_mutation_blocks_before_forward(self):
  r=self.new_runtime();r.prepare_batch(self.p,self.point_maps,self.rt_root/'initial','tiny');file=self.rt_root/'initial/ACTUAL_TORCH_FP64_POINTS.npz';file.write_bytes(file.read_bytes()+b'changed');count=r.calls['model_forwards']
  with self.assertRaises(ValueError):r.coefficient_record(IDS[0],self.p,self.point_maps[(IDS[0],'zero')],True)
  self.assertEqual(r.calls['model_forwards'],count)
 def test_actual_graph_point_must_match_saved_hash_before_forward(self):
  r=self.new_runtime();r.prepare_batch(self.p,self.point_maps,self.rt_root/'initial','tiny');r.batch['point_ids'][IDS[0]]='0'*64;count=r.calls['model_forwards']
  with self.assertRaises(ValueError):r.coefficient_record(IDS[0],self.p,self.point_maps[(IDS[0],'zero')],True)
  self.assertEqual(r.calls['model_forwards'],count);r.verify(True)
 def test_wrong_head_and_skipped_state_block_before_forward(self):
  r=self.new_runtime();r.prepare_batch(self.p,self.point_maps,self.rt_root/'initial','tiny');count=r.calls['model_forwards']
  with self.assertRaises(ValueError):r.coefficient_record(IDS[0],self.p+.001,self.point_maps[(IDS[0],'zero')],True)
  with self.assertRaises(ValueError):r.coefficient_record(IDS[1],self.p,self.point_maps[(IDS[1],'zero')],True)
  self.assertEqual(r.calls['model_forwards'],count)
 def test_C161_FP64_replay_drift_blocks_first_update_and_retains_failed_row(self):
  from worker163 import run
  with tempfile.TemporaryDirectory() as t:
   t=Path(t);paths,c=self.marker_inputs(t/'inputs');ctx=copy.deepcopy(self.ctx);ctx['parent_final_FP64_records'][0]['observed'][0]+=.0001
   with contextlib.redirect_stdout(io.StringIO()):s=run(paths,t/'out',c,self.factory,lambda *a:ctx)
   self.assertFalse(s['complete']);self.assertEqual(s['new_training_steps'],0);self.assertEqual(s['records_completed'],85);self.assertEqual(s['new_backward_calls'],2);self.assertFalse(s['final_head_frozen'])
 def test_C161_native_replay_drift_blocks_before_gradients(self):
  from worker163 import run
  with tempfile.TemporaryDirectory() as t:
   t=Path(t);paths,c=self.marker_inputs(t/'inputs');ctx=copy.deepcopy(self.ctx);ctx['parent_final_native_records']['0'][0]['metrics']['observed'][0]+=.001
   with contextlib.redirect_stdout(io.StringIO()):s=run(paths,t/'out',c,self.factory,lambda *a:ctx)
   self.assertFalse(s['complete']);self.assertEqual(s['new_training_steps'],0);self.assertEqual(s['records_completed'],1);self.assertEqual(s['new_backward_calls'],0)
 def test_native_exception_restores_chart_and_tied_model(self):
  r=self.new_runtime();p=self.point_maps[(IDS[0],'zero')].astype(np.float32)
  with patch.object(r.rt.model,'forward',side_effect=RuntimeError('injected')):
   with self.assertRaises(RuntimeError):r.native_candidate(IDS[0],p,safe.tensor_hash(p))
  np.testing.assert_array_equal(r.rt.params[0].numpy(),0.);self.assertIs(r.rt.model.embed.weight,r.rt.model.proj.weight);r.verify(True)
 def test_functional_exception_restores_chart_and_tied_model(self):
  r=self.new_runtime();r.prepare_batch(self.p,self.point_maps,self.rt_root/'initial','tiny')
  with patch.object(r.rt.model,'forward',side_effect=RuntimeError('injected')):
   with self.assertRaises(RuntimeError):r.coefficient_record(IDS[0],self.p,self.point_maps[(IDS[0],'zero')],True)
  np.testing.assert_array_equal(r.rt.params[0].numpy(),0.);self.assertIs(r.rt.model.embed.weight,r.rt.model.proj.weight);r.verify(True)
 def test_actual_reference_mutation_blocks(self):
  r=self.new_runtime();r.objective_response.add_(1.)
  with self.assertRaises(ValueError):r.verify()
if __name__=='__main__':unittest.main(verbosity=2)
