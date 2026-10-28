"""Actual tiny CPU Torch native wrapper test; synthetic base, not GPT-2/HF."""
import sys,types,tempfile,unittest,contextlib
from unittest import mock
from core198 import *
from runtime198 import factory
class Checks(unittest.TestCase):
 def setup_runtime(self):
  import torch
  t=torch
  class Tiny(t.nn.Module):
   def __init__(self):
    super().__init__();self.backbone=t.nn.Parameter(t.tensor(.1));self.chart=t.nn.ModuleDict({'lora_B':t.nn.ModuleDict({'default':t.nn.Linear(DIM,1,bias=False)})});self.register_buffer('offset',t.tensor(0.));self.chart['lora_B']['default'].weight.data.zero_()
   def forward(self,x):return x+self.backbone+self.chart['lora_B']['default'].weight[0,0]+self.offset
  w=types.ModuleType('writer_core')
  def make(*a,**kw):
   model=Tiny().eval()
   for p in model.parameters():p.requires_grad_(False)
   return None,model,[model.chart['lora_B']['default'].weight]
  w.make_model=make
  pins={k:'synthetic' for k in SEMANTIC_PINS};pins.update(chart_shapes=[[1,DIM]],anchors=['synthetic']*12,response_ids=list(range(6)),symbols=[['0',0],['1',1]])
  class Base:
   def __init__(self,out):
    self.out=Path(out);self.tok,self.model,self.params=w.make_model();self.zero=t.zeros(DIM);self.online=False
    with t.no_grad():self.logits0=self.model(t.zeros((12,8)));self.response0=self.logits0[:,:6].mean(0)
    dump(self.out/'MODEL_PINS.json',pins)
   def invariant(self):need(all(not p.requires_grad and p.grad is None for p in self.model.parameters()),'Frozen tiny parameters')
   def assign(self,point):
    with t.no_grad():self.params[0].copy_(t.as_tensor(np.asarray(point).copy()).reshape(1,DIM))
   @t.no_grad()
   def features(self,theta,desired):
    self.assign(theta);al=self.model(t.zeros((12,8)));self.model(t.zeros((16,8)));self.model(t.zeros((4,8)));x=np.zeros(3147);x[3076]=desired;o=np.r_[(al[:,:6].mean(0)-self.response0).double().numpy(),np.zeros(16)];return x,o
   @t.no_grad()
   def metrics(self,point,case,write=True):
    self.assign(point);al=self.model(t.zeros((12,8)));self.model(t.zeros((16,8)));x,y=int(case[-2]),int(case[-1]);truth=[x,y,1,x^y];o=np.r_[(al[:,:6].mean(0)-self.response0).double().numpy(),(2*np.tile(truth,4)-1)*.2];return archive155.full_metric(o,0.,point,case)
  base=types.ModuleType('runtime_base123');base.Runtime=Base;tmp=tempfile.TemporaryDirectory();ctx=dict(modelpins=pins,baseline_logits=dict(FP32_logits_cast64=np.full((12,8),float(np.float32(.099)))),baseline=dict(FP32_cached_response_reference=[float(np.float32(.099))]*6));calls=dict(model_forwards=0,initialization_forwards=0,feature_attempts=0,feature_calls=0,metric_attempts=0,metric_calls=0)
  with mock.patch.dict(sys.modules,{'runtime_base123':base,'writer_core':w}):rt=factory(Path(tmp.name),ctx,calls)
  self.addCleanup(tmp.cleanup);return rt,calls,t
 def test_actual_Torch_forward_counts_feature_metrics_and_restore(self):
  rt,c,t=self.setup_runtime();point=np.zeros(DIM,np.float32);point[0]=.0001;rt.features(point,0);m=rt.metrics(point,'xor_00');self.assertEqual(c['model_forwards'],6);self.assertEqual(c['feature_calls'],1);self.assertEqual(c['metric_calls'],1);self.assertTrue(all(bool((p==0).all()) for p in rt.params));rt.close()
 def test_original_reference_retained_after_fresh_zero_drift(self):
  rt,c,t=self.setup_runtime();r=clean_json((rt.out/'REFERENCE_POLICY.json').read_bytes());self.assertFalse(r['reference_recentered']);self.assertGreater(r['fresh_zero_response_gap_Linf'],0);self.assertAlmostEqual(float(rt.response0[0]),.099,places=6);rt.close()
 def test_grad_enabled_forward_rejected(self):
  rt,c,t=self.setup_runtime()
  with self.assertRaisesRegex(ValueError,'gradient-enabled'):rt.model(t.zeros((1,8)))
  rt.close()
 def test_backbone_byte_mutation_detected(self):
  rt,c,t=self.setup_runtime()
  with t.no_grad():rt.model.backbone.add_(.001)
  with self.assertRaises(ValueError):rt.verify(True)
 def test_buffer_mutation_detected(self):
  rt,c,t=self.setup_runtime();rt.model.offset.add_(1)
  with self.assertRaises(ValueError):rt.verify()
 def test_reference_mutation_detected(self):
  rt,c,t=self.setup_runtime();rt.response0[0]+=.001
  with self.assertRaises(ValueError):rt.verify()
 def test_training_mode_rejected(self):
  rt,c,t=self.setup_runtime();rt.model.train()
  with self.assertRaises(ValueError):rt.verify()
 def test_derivative_teacher_entrypoints_rejected(self):
  rt,c,t=self.setup_runtime()
  for f in (rt.jacobian,rt.jacobian10,rt.teacher):
   with self.assertRaises(ValueError):f()
  rt.close()
if __name__=='__main__':unittest.main(verbosity=2)
