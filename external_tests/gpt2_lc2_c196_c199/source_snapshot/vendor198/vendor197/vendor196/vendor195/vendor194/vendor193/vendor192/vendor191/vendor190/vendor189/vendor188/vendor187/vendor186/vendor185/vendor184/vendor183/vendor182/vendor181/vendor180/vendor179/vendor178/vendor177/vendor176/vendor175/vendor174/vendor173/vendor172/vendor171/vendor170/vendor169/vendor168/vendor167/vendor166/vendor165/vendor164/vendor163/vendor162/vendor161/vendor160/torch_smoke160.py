"""Actual Torch full84 tiny tied-model baseline; not GPT-2 scientific evidence."""
import contextlib,copy,io,tempfile,unittest
from unittest.mock import patch
import torch
from core160 import *
from runtime160 import Full84
from fixture160 import context,inputs
from torch_smoke155 import ToyRuntime
class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);cls.ctx=context();rt=ToyRuntime(cls.root/'reference',dict(model_forwards=0));double=copy.deepcopy(rt.model).double();batch=rt.tok(rt.anchors)
  with torch.no_grad():logits=double(**batch,use_cache=False).logits[:,-1].numpy()
  cls.pins=dict(revision='synthetic-tiny-only',seed=155,lora_A_sha256='synthetic',chart_shapes=[[DIM]],symbols=[['no',0],['yes',1]],anchors=rt.anchors,response_ids=list(range(6)),sketch_sha256='synthetic');cls.ctx.update(modelpins=cls.pins,baseline_logits=dict(FP32_logits_cast64=rt.logits0.double().numpy(),FP64_logits=logits),baseline=dict(FP32_cached_response_reference=rt.response0.double().numpy().tolist(),FP64_response_reference=logits[:,:6].mean(0).tolist()))
  dump(cls.root/'DESIGN_FROZEN_BEFORE_MODEL.json',design());cls.chain,cls.points,cls.native,cls.coefs,cls.seals=freeze(cls.ctx,cls.root)
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup()
 def setup_runtime(self):
  calls=dict(model_forwards=0,functional_FP64_forwards=0,backward_calls=0,backward_attempts=0)
  def factory(path,d,phase,counts):rt=ToyRuntime(path,counts);rt.pins=self.pins;return rt
  self.native_factory=factory
  with contextlib.redirect_stdout(io.StringIO()):r=Full84(self.root/('rt_'+self._testMethodName),copy.deepcopy(self.ctx),{n:a.copy() for n,a in self.chain.items()},self.points,self.native,self.coefs,calls,factory)
  return r
 def test_actual_complete84_worker2019_forwards168_backwards(self):
  from worker160 import run
  with tempfile.TemporaryDirectory() as td:
   td=Path(td);paths,c=inputs(td/'inputs')
   def factory(out,ctx,chain,points,native,coefs,calls):
    def native_factory(path,d,phase,counts):rt=ToyRuntime(path,counts);rt.pins=self.pins;return rt
    self.worker_runtime=Full84(out,ctx,chain,points,native,coefs,calls,native_factory);return self.worker_runtime
   with contextlib.redirect_stdout(io.StringIO()):s=run(paths,td/'out',c,factory,lambda *a:copy.deepcopy(self.ctx))
   self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['new_model_forwards'],2019);self.assertEqual(s['functional_FP64_forwards'],1682);self.assertEqual(s['new_backward_calls'],168);self.assertEqual(s['records_completed'],1008);self.assertTrue(s['full84_FP64_objective_available']);self.assertFalse(s['head_updated']);self.assertFalse(s['LC2_closed']);self.assertEqual(s['native_alpha0_round0']['n_evaluated'],84);self.assertEqual(s['native_alpha0_round1']['n_evaluated'],84)
   for r in s['three_scale_complete_objective_FD']:self.assertLess(abs(r['central_minus_AD']),1e-5)
   self.worker_runtime.verify(True)
 def test_full_FP64_response_read_KL_and_both_gradients_finite(self):
  r=self.setup_runtime();sid=registry()[0]['state_id'];row=r.evaluate(sid,'zero',True);self.assertEqual(len(row['observed']),22);self.assertEqual(set(row['objective_parts']),{'response_loss','read_loss','KL_loss','radius_loss','regularizer'});self.assertEqual(np.shape(row['response_gradient']),(3,2,6));self.assertEqual(np.shape(row['guard_gradient']),(3,2,6));self.assertEqual(r.calls['model_forwards'],5);self.assertEqual(r.calls['backward_calls'],2);r.verify(True)
 def test_full84_objective_parts_match_independent_all_active_numpy_losses(self):
  r=self.setup_runtime();sid=registry()[0]['state_id'];o=torch.tensor(np.r_[np.full(6,.004),np.full(16,.01)],dtype=torch.float64);kl=torch.tensor(.01,dtype=torch.float64);point=torch.full((DIM,),.15,dtype=torch.float64);e=torch.full((2,6),.1,dtype=torch.float64);parts={n:float(v) for n,v in r.terms(o,kl,point,e,sid).items()};expected=loss(o.numpy(),float(kl),point.numpy(),self.chain[sid+'__prepared'],e.numpy(),sid.split('__')[0]);compare(parts,expected,'Independent all-active full objective',atol=1e-10,rtol=1e-9);self.assertTrue(all(v>0 for v in parts.values()))
 def test_native_physical_FP32_commit_and_restore_zero(self):
  r=self.setup_runtime();sid=registry()[0]['state_id'];a=r.native(sid);b=r.native(sid);self.assertTrue(a['actual_physical_FP32_commit']);self.assertTrue(archive155.repeat_check(a['metrics'],b['metrics'])['within_replay_tolerance']);np.testing.assert_array_equal(r.rt.params[0].numpy(),np.zeros(DIM,np.float32));self.assertIs(r.rt.model.embed.weight,r.rt.model.proj.weight);r.verify(True)
 def test_native_exception_still_restores_physical_FP32_chart(self):
  r=self.setup_runtime()
  with patch.object(r.rt.model,'forward',side_effect=RuntimeError('intentional native failure')):
   with self.assertRaises(RuntimeError):r.native(registry()[0]['state_id'])
  np.testing.assert_array_equal(r.rt.params[0].numpy(),np.zeros(DIM,np.float32));r.verify(True)
 def test_full_FP64_exception_restores_tied_model(self):
  r=self.setup_runtime()
  with patch.object(r.rt.model,'forward',side_effect=RuntimeError('intentional functional failure')):
   with self.assertRaises(RuntimeError):r.evaluate(registry()[0]['state_id'],'zero',True)
  self.assertIs(r.rt.model.embed.weight,r.rt.model.proj.weight);r.verify(True)
 def test_objective_reference_normal_mutation_blocks(self):
  r=self.setup_runtime();r.objective_response.add_(1.)
  with self.assertRaises(ValueError):r.verify()
 def test_hidden_data_mutation_in_fixed_references_blocks_full_hash(self):
  r=self.setup_runtime();r.prob64.data.add_(.001)
  with self.assertRaises(ValueError):r.verify(True)
 def test_hidden_chain_mutation_blocks_full_hash(self):
  r=self.setup_runtime();r.chain['P0'].data.add_(.001)
  with self.assertRaises(ValueError):r.verify(True)
 def test_map_must_match_actual_sealed_point(self):
  r=self.setup_runtime();sid=registry()[0]['state_id'];wrong=torch.from_numpy(self.coefs['zero'].copy())+.001
  with self.assertRaises(ValueError):r.map(wrong,sid,'zero')
 def test_detached_graph_does_not_silently_supply_gradients(self):
  r=self.setup_runtime();original=r.map;r.map=lambda *a:(original(*a)[0].detach(),original(*a)[1])
  with self.assertRaises(ValueError):r.evaluate(registry()[0]['state_id'],'zero',True)
  r.verify(True)
if __name__=='__main__':unittest.main(verbosity=2)
