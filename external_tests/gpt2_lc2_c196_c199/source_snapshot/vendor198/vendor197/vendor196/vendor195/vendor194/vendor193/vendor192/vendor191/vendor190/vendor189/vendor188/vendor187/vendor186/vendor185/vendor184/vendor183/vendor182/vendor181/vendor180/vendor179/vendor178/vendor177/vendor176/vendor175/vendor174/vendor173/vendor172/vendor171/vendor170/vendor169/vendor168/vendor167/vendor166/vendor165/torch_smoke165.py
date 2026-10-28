"""Real tiny CPU Torch model: end-to-end sealed probes and derivative checks."""
import copy,contextlib,io,tempfile,unittest
from unittest.mock import patch
import torch
from core165 import *
from runtime165 import DirectionRuntime
from worker165 import run
from worker163 import run as run163
from runtime163 import SmoothTrainer
class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  import torch_smoke163 as old
  cls.prior=old.Checks;cls.prior.setUpClass();cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);paths={}
  for k in parent.INPUT_KEYS:p=cls.root/(k+'.zip');p.write_bytes(b'Actual tiny test injected parent only');paths[k]=p
  pc=parent.contract();pc.update(production=False,pins={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()})
  with contextlib.redirect_stdout(io.StringIO()):s=run163(paths,cls.root/'C163_actual_tiny',pc,lambda *a:SmoothTrainer(*a,native_factory=cls.prior.native_factory),lambda *a:copy.deepcopy(cls.prior.ctx))
  need(s['complete'],'Actual tiny C163: '+str(s['errors']));ctx=copy.deepcopy(cls.prior.ctx)
  with np.load(cls.root/'C163_actual_tiny/FIXED_FINAL_HEAD_STEP8.npz',allow_pickle=False) as z:p=z['P'].copy()
  native=clean_json((cls.root/'C163_actual_tiny/ALL336_NATIVE_ENDPOINTS.json').read_bytes());ctx.update(initial_head=p,parent_final_FP64_records=clean_json((cls.root/'C163_actual_tiny/FINAL_FP64_OBJECTIVE_ALL84.json').read_bytes()),parent_final_native_records={str(t):native['native_final'+str(t)] for t in (0,1)},C164_receipt=dict(decision='C164_READONLY_TRAINING_AUDIT_COMPLETE_PARENT_DECISION_RETAINED'));cls.ctx=ctx;cls.chain=parent.parent.previous.constants(ctx['data'])
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup();cls.prior.tearDownClass()
 def factory(self,*args):return DirectionRuntime(*args,native_factory=self.prior.native_factory)
 def markers(self,root):
  root=Path(root);root.mkdir();paths={}
  for k in INPUT_KEYS:p=root/(k+'.zip');p.write_bytes(b'Tiny injected ten-original binding');paths[k]=p
  c=contract();c.update(production=False,pins={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()});return paths,c
 def runtime(self):
  root=self.root/self._testMethodName;root.mkdir();np.savez_compressed(root/'FROZEN_FULL84_CHAIN_CONSTANTS.npz',**self.chain);p=self.ctx['initial_head'].copy();pt,nt,_=parent.seal_head(self.chain,p,root/'anchor','anchor',True)
  with contextlib.redirect_stdout(io.StringIO()):r=self.factory(root/'model_runtime',copy.deepcopy(self.ctx),self.chain,pt,nt,dict(zero=p.copy()),dict(model_forwards=0,functional_FP64_forwards=0,backward_attempts=0,backward_calls=0))
  return r,root,p,pt
 def test_actual_full1848_workflow_all6_arms_672_points_and168_backwards(self):
  with tempfile.TemporaryDirectory() as t:
   t=Path(t);paths,c=self.markers(t/'inputs')
   with contextlib.redirect_stdout(io.StringIO()):s=run(paths,t/'out',c,self.factory,lambda *a:copy.deepcopy(self.ctx))
   self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['new_model_forwards'],3699);self.assertEqual(s['new_backward_calls'],168);self.assertEqual(s['functional_FP64_forwards'],1346);self.assertEqual(s['records_completed'],1848);self.assertEqual(s['actual_FP64_point_tensors_saved'],672);self.assertFalse(s['LC2_closed']);self.assertEqual(s['new_training_steps'],0);self.assertFalse(s['candidate_selected']);self.assertEqual(s['temporary_probe_heads'],6)
   rows=[clean_json(x) for x in (t/'out/ALL1848_RECORDS.jsonl').read_bytes().splitlines()];forward=[r for r in rows if 'actual_FP64_point_artifact' in r];self.assertEqual(len(forward),672)
   arrays={}
   for r in forward:
    key=r['actual_FP64_point_artifact']
    if key not in arrays:
     with np.load(t/'out'/key,allow_pickle=False) as z:arrays[key]={n:z[n] for n in z.files}
    self.assertEqual(point_hash(arrays[key][r['state_id']]),r['FP64_point_sha256'])
   receipt=clean_json((t/'out/ALL6_PROBES_SEALED_BEFORE_FIRST_PROBE.json').read_bytes());self.assertEqual(receipt['model_forwards_so_far'],339);self.assertEqual(receipt['backward_calls_so_far'],168);self.assertEqual(len(receipt['all6_head_digests']),6);self.assertEqual(len(clean_json((t/'out/ALL588_NATIVE_REPEAT_CHECKS.json').read_bytes())),588)
 def test_actual_anchor_response_and_guard_gradients_match_same_function_FD(self):
  r,root,p,pt=self.runtime();r.preseal(p,pt,root/'anchor','anchor');r.activate('anchor',p);row=r.coefficient_record(IDS[0],p,pt[(IDS[0],'zero')],True);d=np.ones_like(p);d/=np.linalg.norm(d);values={}
  for sign in (-1,1):
   h=1e-6;q=p+sign*h*d;tp,_=r.map_head(torch.tensor(q,dtype=torch.float64),IDS[0]);r.authorized_point=point_hash(tp.numpy())
   try:o,kl=r.full_forward(tp,False)
   finally:r.authorized_point=None
   _,e=r.map_head(torch.tensor(q,dtype=torch.float64),IDS[0]);parts=r.terms(o,kl,tp,e,IDS[0]);values[sign]={k:float(v.detach()) for k,v in parts.items()}
  for key,terms in [('response_gradient',('response_loss',)),('guard_gradient',PARTS[1:])]:
   fd=(sum(values[1][k] for k in terms)-sum(values[-1][k] for k in terms))/(2e-6);ad=float(np.sum(np.asarray(row[key])*d));self.assertAlmostEqual(fd,ad,delta=1e-6*max(1.,abs(ad)))
 def test_all_batches_can_be_presealed_before_any_probe_but_activated_once(self):
  r,root,p,pt=self.runtime();before=r.calls['model_forwards'];r.preseal(p,pt,root/'anchor','anchor');tp,_,_=parent.seal_head(self.chain,p,root/'second','second',False);r.preseal(p,tp,root/'second','second');self.assertEqual(r.calls['model_forwards'],before);r.activate('anchor',p)
  with self.assertRaises(ValueError):r.activate('second',p)
  for sid in IDS:r.coefficient_record(sid,p,pt[(sid,'zero')],False)
  with self.assertRaises(ValueError):r.activate('anchor',p)
  r.activate('second',p)
 def test_saved_actual_point_tamper_blocks_before_forward(self):
  r,root,p,pt=self.runtime();r.preseal(p,pt,root/'anchor','anchor');f=root/'anchor/ACTUAL_TORCH_FP64_POINTS.npz';f.write_bytes(f.read_bytes()+b'tampered');n=r.calls['model_forwards']
  with self.assertRaises(ValueError):r.activate('anchor',p)
  self.assertEqual(r.calls['model_forwards'],n)
 def test_actual_graph_wrong_point_blocks_before_model(self):
  r,root,p,pt=self.runtime();r.preseal(p,pt,root/'anchor','anchor');r.activate('anchor',p);r.batch['point_ids'][IDS[0]]='0'*64;n=r.calls['model_forwards']
  with self.assertRaises(ValueError):r.coefficient_record(IDS[0],p,pt[(IDS[0],'zero')],False)
  self.assertEqual(r.calls['model_forwards'],n);r.verify(True)
 def test_crossrun_FP64_replay_drift_stops_before_direction_and_retains_row(self):
  with tempfile.TemporaryDirectory() as t:
   t=Path(t);paths,c=self.markers(t/'inputs');ctx=copy.deepcopy(self.ctx);ctx['parent_final_FP64_records'][0]['observed'][0]+=.001
   with contextlib.redirect_stdout(io.StringIO()):s=run(paths,t/'out',c,self.factory,lambda *a:ctx)
   self.assertFalse(s['complete']);self.assertEqual(s['new_backward_calls'],2);self.assertEqual(s['records_completed'],85);self.assertEqual(s['offline_direction_optimization_iterations'],0);self.assertEqual(s['temporary_probe_heads'],0)
 def test_crossrun_native_replay_drift_stops_before_gradients(self):
  with tempfile.TemporaryDirectory() as t:
   t=Path(t);paths,c=self.markers(t/'inputs');ctx=copy.deepcopy(self.ctx);ctx['parent_final_native_records']['0'][0]['metrics']['observed'][0]+=.01
   with contextlib.redirect_stdout(io.StringIO()):s=run(paths,t/'out',c,self.factory,lambda *a:ctx)
   self.assertFalse(s['complete']);self.assertEqual(s['new_backward_calls'],0);self.assertEqual(s['records_completed'],1)
 def test_native_exception_restores_chart_and_tied_model(self):
  r,root,p,pt=self.runtime();q=pt[(IDS[0],'zero')].astype(np.float32)
  with patch.object(r.rt.model,'forward',side_effect=RuntimeError('injected')):
   with self.assertRaises(RuntimeError):r.native_candidate(IDS[0],q,safe.tensor_hash(q))
  np.testing.assert_array_equal(r.rt.params[0].numpy(),0.);self.assertIs(r.rt.model.embed.weight,r.rt.model.proj.weight);r.verify(True)
 def test_native_wrong_physical_point_hash_blocks_before_model(self):
  r,root,p,pt=self.runtime();q=pt[(IDS[0],'zero')].astype(np.float32);digest=safe.tensor_hash(q);q[0]+=.001;n=r.calls['model_forwards']
  with self.assertRaises(ValueError):r.native_candidate(IDS[0],q,digest)
  self.assertEqual(r.calls['model_forwards'],n)
 def test_functional_exception_restores_chart_and_tied_model(self):
  r,root,p,pt=self.runtime();r.preseal(p,pt,root/'anchor','anchor');r.activate('anchor',p)
  with patch.object(r.rt.model,'forward',side_effect=RuntimeError('injected')):
   with self.assertRaises(RuntimeError):r.coefficient_record(IDS[0],p,pt[(IDS[0],'zero')],False)
  np.testing.assert_array_equal(r.rt.params[0].numpy(),0.);self.assertIs(r.rt.model.embed.weight,r.rt.model.proj.weight);r.verify(True)
 def test_reference_mutation_blocks(self):
  r,root,p,pt=self.runtime();r.objective_response.add_(1.)
  with self.assertRaises(ValueError):r.verify()
if __name__=='__main__':unittest.main(verbosity=2)
