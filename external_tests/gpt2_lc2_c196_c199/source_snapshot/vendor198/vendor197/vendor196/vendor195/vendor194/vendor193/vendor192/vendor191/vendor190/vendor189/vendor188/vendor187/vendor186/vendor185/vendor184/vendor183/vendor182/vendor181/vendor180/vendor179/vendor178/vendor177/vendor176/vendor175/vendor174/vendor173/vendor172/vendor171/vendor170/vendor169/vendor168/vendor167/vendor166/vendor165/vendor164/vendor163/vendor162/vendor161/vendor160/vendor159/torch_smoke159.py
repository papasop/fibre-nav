"""Actual Torch CPU tiny tied-model chain tests; not GPT-2 scientific evidence."""
import contextlib,copy,io,tempfile,unittest
from unittest.mock import patch
import torch
from core159 import *
from map159 import torch_constants,torch_map,chain_arithmetic
from runtime159 import Directional
from fixture159 import context,inputs
from torch_smoke155 import ToyRuntime
class Checks(unittest.TestCase):
 def setup_runtime(self,drift=False):
  tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);root=Path(tmp.name);c=context();rt=ToyRuntime(root/'reference',dict(model_forwards=0));model=copy.deepcopy(rt.model).double();batch=rt.tok(rt.anchors)
  with torch.no_grad():logits=model(**batch,use_cache=False).logits[:,-1].numpy()
  pins=dict(revision='synthetic-torch-only',seed=155,lora_A_sha256='synthetic',chart_shapes=[[DIM]],symbols=[0,1],anchors=rt.anchors,response_ids=list(range(6)),sketch_sha256='synthetic');c.update(modelpins=pins,baseline_logits=dict(FP32_logits_cast64=rt.logits0.double().numpy(),FP64_logits=logits),baseline=dict(FP64_response_reference=logits[:,:6].mean(0).tolist(),FP32_cached_response_reference=rt.response0.double().numpy().tolist()))
  if drift:c['baseline_logits']['FP32_logits_cast64']=c['baseline_logits']['FP32_logits_cast64']+.01;c['baseline_logits']['FP64_logits']=logits+.01
  dump(root/'DESIGN_FROZEN_BEFORE_MODEL.json',design());points,directions,_,_=freeze(c['data'],root);calls=dict(model_forwards=0,functional_FP64_forwards=0,backward_attempts=0,backward_calls=0);self.last_calls=calls;self.last_out=root/'actual'
  def factory(path,d,phase,counts):rt=ToyRuntime(path,counts);rt.pins=pins;return rt
  self.native_factory=factory
  with contextlib.redirect_stdout(io.StringIO()):return Directional(root/'actual',c,points,directions,calls,factory),c
 def test_coordinate_AD_matches_independent_tiny_analytic_derivative(self):
  r,_=self.setup_runtime();sid=FOCUS[0];ad=r.ad(sid);emb=r.rt.model.embed.weight.detach().double();ids=r.rt.tok(r.rt.anchors)['input_ids'][:,-1];x=emb[ids]+torch.from_numpy(r.points[(sid,'zero')][:6]);d=torch.from_numpy(r.directions[sid][:6]);want=(((1-torch.tanh(x)**2)*d)@emb.T*r.rt.model.scale.detach().double()).mean(0)[:6].numpy();np.testing.assert_allclose(ad['coordinate_derivatives'],want,rtol=1e-11,atol=1e-15);self.assertEqual(r.calls['backward_calls'],9);r.verify(True)
 def test_three_reference_pullbacks_and_all_FD_scales(self):
  r,_=self.setup_runtime();sid=FOCUS[0];ad=r.ad(sid);panels={n:r.measure(sid,n) for n in LABELS};a=analyse(panels,ad,r.references)
  for row in a['scales']:
   self.assertLess(row['central_minus_AD_Linf'],1e-10)
   for ref in REFERENCES:self.assertLess(abs(row['response_loss'][ref]['central_minus_AD']),1e-5)
  for cc in ad['chain_pullbacks'].values():
   self.assertLess(cc['pullback_vs_direct_Linf'],1e-10);self.assertLess(abs(cc['explicit_dot_minus_alpha']),1e-12);self.assertLess(abs(cc['direct_dot_minus_alpha']),1e-12)
 def test_complete_actual_Torch_27_forward18_reverse_workflow(self):
  from worker159 import run
  r,c=self.setup_runtime(drift=True);tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);root=Path(tmp.name);paths,config=inputs(root/'inputs')
  def factory(out,ctx,points,directions,calls):self.worker_runtime=Directional(out,ctx,points,directions,calls,self.native_factory);return self.worker_runtime
  with contextlib.redirect_stdout(io.StringIO()):s=run(paths,root/'out',config,factory,lambda *args:c)
  self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['new_model_forwards'],27);self.assertEqual(s['new_backward_calls'],18);self.assertFalse(s['LC2_closed']);self.assertFalse(s['real_model_execution_claim']);self.worker_runtime.verify(True)
 def test_detached_alpha_chain_blocks(self):
  r,_=self.setup_runtime();original=__import__('runtime159').torch_map
  def detached(c,sid,alpha):return original(c,sid,alpha.detach())
  with patch('runtime159.torch_map',detached),self.assertRaises(ValueError):r.ad(FOCUS[0])
  r.verify(True)
 def test_full_control_coefficients_are_graph_connected(self):
  c=constants(context()['data']);tc=torch_constants(c);a=torch.tensor(0.,dtype=torch.float64,requires_grad=True);point,p,e=torch_map(tc,FOCUS[0],a);ga,ge,gp=torch.autograd.grad(point[5],(a,e,p));want=c[FOCUS[0]+'__weights'][:,None,None]*ge.numpy()[None,...];np.testing.assert_array_equal(gp.numpy(),want);self.assertEqual(gp.shape,(3,2,6));self.assertIsNone(tc['P0'].grad)
 def test_reversed_cyclic_shift_detected_by_sealed_point_guard(self):
  r,_=self.setup_runtime();original=__import__('runtime159').torch_map
  def wrong(c,sid,alpha):
   _,p,e=original(c,sid,alpha);u=c['V']@(c[sid+'__raw']-c[sid+'__prepared']);return c[sid+'__raw']+c['V'].T@(e[0]*u+e[1]*torch.roll(u,-1)),p,e
  with patch('runtime159.torch_map',wrong),self.assertRaisesRegex(ValueError,'Actual coefficient map point'):r.measure(FOCUS[0],'zero')
 def test_postseal_context_change_stops_before_model_initialization(self):
  r,c=self.setup_runtime();c['data']['d']['V'][0,0]+=1e-5;calls=dict(model_forwards=0,functional_FP64_forwards=0,backward_attempts=0,backward_calls=0)
  with self.assertRaisesRegex(ValueError,'Actual chain constant matches pre-model seal'):Directional(r.out.parent/'changed',c,r.points,r.directions,calls,self.native_factory)
  self.assertEqual(calls['model_forwards'],0)
 def test_frozen_chain_data_mutation_is_detected(self):
  r,_=self.setup_runtime();r.chain['P0'].data.add_(1e-6)
  with self.assertRaises(ValueError):r.verify()
 def test_frozen_numpy_chain_mutation_is_detected(self):
  r,_=self.setup_runtime();r.chain_numpy['deltaP'][0,0,0]+=1e-6
  with self.assertRaises(ValueError):r.verify()
 def test_frozen_FP64_model_data_mutation_is_detected(self):
  r,_=self.setup_runtime();name=next(n for n,p in r.double.items() if p.dtype==torch.float64 and n not in r.chart_names);r.double[name].data.add_(1.)
  with self.assertRaises(ValueError):r.verify(True)
 def test_wrong_dtype_rejected(self):
  r,_=self.setup_runtime()
  with self.assertRaises(ValueError):r.forward(torch.zeros(DIM,dtype=torch.float32),graph=True)
 def test_exception_restores_tied_physical_FP32_model(self):
  r,_=self.setup_runtime()
  with patch.object(r.rt.model,'forward',side_effect=RuntimeError('intentional')):
   with self.assertRaises(RuntimeError):r.ad(FOCUS[0])
  r.verify(True);self.assertIs(r.rt.model.embed.weight,r.rt.model.proj.weight)
 def test_same_run_baseline_repeat_drift_blocks_before_AD(self):
  original=Directional.forward;count=[0]
  def changed(rt,*a,**kw):
   v=original(rt,*a,**kw)
   if kw.get('baseline'):
    count[0]+=1
    if count[0]==2:v=v+1e-6
   return v
  with patch.object(Directional,'forward',changed),self.assertRaisesRegex(ValueError,'Within-run FP64 baseline repeat differs'):self.setup_runtime()
  self.assertEqual(self.last_calls['model_forwards'],3);self.assertEqual(self.last_calls['backward_calls'],0)
 def test_cross_archive_drift_is_descriptive_only(self):
  r,_=self.setup_runtime(drift=True);self.assertFalse(r.baseline_records['native_logits']['passed_original_replay_tolerance']);self.assertFalse(r.baseline_records['FP64_cross_run']['passed_original_replay_tolerance']);r.ad(FOCUS[0]);r.verify(True)
 def test_reference_is_sealed_before_any_focus_and_has_current_protocol(self):
  r,_=self.setup_runtime();a=clean_json(r.reference_file.read_bytes());self.assertEqual(a['protocol'],PROTOCOL);self.assertEqual(a['model_forwards_so_far'],3);self.assertEqual(a['state_probes_so_far'],0);r.references['current_FP64'][0]+=.01
  with self.assertRaises(ValueError):r.verify()
if __name__=='__main__':unittest.main(verbosity=2)
