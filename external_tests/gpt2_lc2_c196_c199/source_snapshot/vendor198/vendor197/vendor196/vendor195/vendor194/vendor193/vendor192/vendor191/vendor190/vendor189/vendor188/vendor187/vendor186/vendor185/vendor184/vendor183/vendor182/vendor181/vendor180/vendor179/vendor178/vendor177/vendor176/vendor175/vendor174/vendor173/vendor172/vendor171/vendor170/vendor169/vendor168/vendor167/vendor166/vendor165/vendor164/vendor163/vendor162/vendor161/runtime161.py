"""Differentiable FP64 training of coefficients; unchanged frozen physical model."""
from core161 import *
from runtime160 import Full84
class Trainer(Full84):
 def __init__(self,*args,**kwargs):
  import torch
  torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
  super().__init__(*args,**kwargs);receipt=clean_json(self.reference_file.read_bytes());receipt['protocol']=PROTOCOL;dump(self.reference_file,receipt);self.reference_file_hash=sha(self.reference_file);self.verify(True)
 def coefficient_record(self,sid,p,expected,gradient):
  torch=self.torch;need(p.dtype==np.float64,'Actual FP64 coefficients')
  with torch.enable_grad() if gradient else torch.no_grad():
   leaf=torch.tensor(p.copy(),dtype=torch.float64,requires_grad=gradient);c=self.chain;w=c[sid+'__weights'];e=(w[0]*leaf[0]+w[1]*leaf[1])+w[2]*leaf[2];u=c['V']@(c[sid+'__raw']-c[sid+'__prepared']);point=c[sid+'__raw']+c['V'].T@(e[0]*u+e[1]*torch.roll(u,1));close(point.detach().numpy(),expected,'Independent NumPy/Torch current map',1e-12)
   o,kl=self.full_forward(point,gradient);parts=self.terms(o,kl,point,e,sid);row=dict(state_id=sid,case=sid.split('__')[0],parameters_digest=parameter_hash(p),FP64_point_sha256=point_hash(point.detach().numpy()),sealed_numpy_point_sha256=point_hash(expected),map_Linf=float(np.max(abs(point.detach().numpy()-expected))),observed=o.detach().numpy().tolist(),KL=float(kl.detach()),objective_parts={n:float(v.detach()) for n,v in parts.items()},FP64_acceptance_evaluated=False)
   compare(row['objective_parts'],loss(row['observed'],row['KL'],point.detach().numpy(),self.chain_numpy[sid+'__prepared'],e.detach().numpy(),row['case']),'Independent training objective',atol=1e-10,rtol=1e-9)
   if gradient:
    for name,v,retain in [('response_gradient',parts['response_loss'],True),('guard_gradient',sum(v for n,v in parts.items() if n!='response_loss'),False)]:
     self.calls['backward_attempts']+=1;g=torch.autograd.grad(v,leaf,retain_graph=retain,create_graph=False,allow_unused=False)[0];self.calls['backward_calls']+=1;need(bool(torch.isfinite(g).all()) and g.shape==leaf.shape,'Finite full coefficient gradient');row[name]=g.detach().numpy().tolist()
    need(leaf.grad is None,'No accumulated coefficient gradients')
  self.verify();return row
 def native_candidate(self,sid,point,digest):
  torch=self.torch;need(point.dtype==np.float32,'Physical FP32 commit only');compare(safe.tensor_hash(point),digest,'Sealed physical proposal');outs=[]
  try:
   self.rt.assign(point);exact(np.concatenate([q.detach().numpy().reshape(-1) for q in self.rt.params]),point,'Physical commit byte identity')
   with torch.no_grad():
    for batch in self.endpoint.batches:
     logits=self.rt.model(**batch,use_cache=False).logits;need(logits.dtype==torch.float32,'Actual physical FP32 logits');last=batch['attention_mask'].sum(1)-1;outs.append(logits[torch.arange(len(last)),last])
    al,rl=outs;o=torch.cat([al[:,self.rt.responseids].mean(0)-self.native_response,rl[:,self.rt.readids[1]]-rl[:,self.rt.readids[0]]]);kl=(self.prob32*(self.logprob32-al.log_softmax(-1))).sum(-1).mean().clamp_min(0.)
   met=archive155.full_metric(o.double().numpy(),float(kl),point,sid.split('__')[0]);qual=archive155.endpoint_gates(met,sid.split('__')[0],point,self.chain_numpy[sid+'__prepared'].astype(np.float32),True)
   return dict(state_id=sid,case=sid.split('__')[0],metrics=met,qualification=qual,actual_physical_FP32_commit=True,preparation_provenance_from_archive_only=True)
  finally:self.rt.assign(self.zero_native);self.verify()
def factory(out,ctx,c,points,native,coefs,calls):return Trainer(out,ctx,c,points,native,coefs,calls)
