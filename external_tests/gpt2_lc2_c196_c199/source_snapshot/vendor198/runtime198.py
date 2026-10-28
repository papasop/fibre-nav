"""Fresh pinned physical CPU GPT-2; fixed archived references, no derivatives."""
from core198 import *
def frozen_model_digest(rt):
 h=hashlib.sha256()
 for n,p in sorted(list(rt.model.named_parameters())+list(rt.model.named_buffers())):
  if 'lora_B.default.' in n:continue
  a=p.detach().cpu().contiguous().numpy();h.update((n+str(a.dtype)+str(a.shape)).encode());h.update(a.tobytes())
 return h.hexdigest()
def factory(out,ctx,calls):
 import torch
 Path(out).mkdir(parents=True,exist_ok=True)
 import core141
 base_path=core141.ROOT/'vendor140/vendor139/vendor138/vendor137/vendor136/vendor135/vendor134/runtime_vendor'
 sys.path.insert(0,str(base_path))
 from runtime_base123 import Runtime
 import writer_core as w
 class Native(Runtime):
  def __init__(self):
   original=w.make_model
   def instrumented(*args,**kwargs):
    tok,model,params=original(*args,**kwargs);model.register_forward_pre_hook(self.count_forward);return tok,model,params
   w.make_model=instrumented
   try:super().__init__(out)
   finally:w.make_model=original
   self.online=True;self.pins=clean_json((self.out/'MODEL_PINS.json').read_bytes());calls['initialization_forwards']=calls['model_forwards'];compare(calls['model_forwards'],1,'Exactly one native zero initialization')
   for k in SEMANTIC_PINS:compare(self.pins[k],ctx['modelpins'][k],'Fixed model/chart identity/'+k)
   actual_logits=self.logits0.detach().clone();actual_response=self.response0.detach().clone()
   archived_logits=torch.from_numpy(ctx['baseline_logits']['FP32_logits_cast64'].astype(np.float32).copy());archived_response=torch.tensor(ctx['baseline']['FP32_cached_response_reference'],dtype=torch.float32)
   need(archived_logits.shape==actual_logits.shape and archived_response.shape==actual_response.shape,'Fixed reference shapes');need(bool(torch.isfinite(archived_logits).all()) and bool(torch.isfinite(archived_response).all()),'Finite fixed references')
   np.savez_compressed(self.out/'INITIALIZATION_BASELINE_DIAGNOSTIC.npz',fresh_logits=actual_logits.numpy(),fresh_response=actual_response.numpy(),fixed_logits=archived_logits.numpy(),fixed_response=archived_response.numpy())
   dump(self.out/'REFERENCE_POLICY.json',dict(protocol=PROTOCOL,fresh_zero_logits_gap_Linf=float((actual_logits-archived_logits).abs().max()),fresh_zero_response_gap_Linf=float((actual_response-archived_response).abs().max()),fresh_baseline_used_for_scoring=False,reference_recentered=False,archived_logits_sha256=array_digest(archived_logits.numpy()),archived_response_sha256=array_digest(archived_response.numpy())))
   self.logits0=archived_logits;self.response0=archived_response;self.refhash=(array_digest(self.logits0.numpy()),array_digest(self.response0.numpy()));self.frozen_digest=frozen_model_digest(self);self.buffer_versions={n:(id(p),p._version) for n,p in self.model.named_buffers()};self.reference_file_hash=sha(self.out/'REFERENCE_POLICY.json');self.verify(True)
  def count_forward(self,module,args):
   need(not torch.is_grad_enabled(),'C198 forbids gradient-enabled model forwards');calls['model_forwards']+=1
  def verify(self,rehash=False):
   self.invariant();need(all(not m.training for m in self.model.modules()),'Frozen model must remain in eval mode')
   compare({n:(id(p),p._version) for n,p in self.model.named_buffers()},self.buffer_versions,'Frozen model buffers');compare((array_digest(self.logits0.numpy()),array_digest(self.response0.numpy())),self.refhash,'Fixed objective references');compare(sha(self.out/'REFERENCE_POLICY.json'),self.reference_file_hash)
   if rehash:compare(frozen_model_digest(self),self.frozen_digest,'Entire frozen backbone/A/buffer byte identity')
  def features(self,theta,desired):
   calls['feature_attempts']+=1;before=calls['model_forwards']
   try:
    x,o=super().features(theta,desired);compare(calls['model_forwards']-before,3,'Three native feature calls');calls['feature_calls']+=1;return x,o
   finally:self.assign(self.zero);self.verify()
  def metrics(self,point,case,write=True):
   calls['metric_attempts']+=1;before=calls['model_forwards']
   try:
    r=super().metrics(point,case,write);exact(np.concatenate([p.detach().numpy().reshape(-1) for p in self.params]),point,'Actual committed physical coordinates');compare(calls['model_forwards']-before,2,'Two native metric calls');calls['metric_calls']+=1
    return archive155.full_metric(r['observed'],r['KL'],point,case) if write else r
   finally:self.assign(self.zero);self.verify()
  def jacobian(self,*a,**k):raise ValueError('C198 forbids Jacobians')
  def jacobian10(self,*a,**k):raise ValueError('C198 forbids derivatives')
  def teacher(self,*a,**k):raise ValueError('C198 forbids teacher/repair/backtracking')
  def close(self):self.assign(self.zero);self.verify(True)
 return Native()
