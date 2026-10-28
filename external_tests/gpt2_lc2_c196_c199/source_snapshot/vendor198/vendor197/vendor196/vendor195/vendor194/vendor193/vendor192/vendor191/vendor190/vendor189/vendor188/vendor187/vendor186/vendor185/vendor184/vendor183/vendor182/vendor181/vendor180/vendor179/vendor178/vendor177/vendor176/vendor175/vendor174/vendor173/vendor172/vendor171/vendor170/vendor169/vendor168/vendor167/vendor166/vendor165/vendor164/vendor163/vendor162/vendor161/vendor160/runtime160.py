"""Full FP64 loss/gradient endpoint and sealed physical FP32 commits; no optimizer."""
from core160 import *
from runtime158 import Directional as Base
from map159 import torch_constants
class Full84(Base):
 def __init__(self,out,context,c,points,native,coefs,calls,native_factory=None):
  self.chain_numpy=c
  with np.load(Path(out).parent/'FROZEN_FULL84_CHAIN_CONSTANTS.npz',allow_pickle=False) as z:
   compare(set(z.files),set(c),'Sealed full84 constant inventory')
   for n,a in c.items():exact(a,z[n],'Actual full84 constants before model/'+n)
  super().__init__(out,context,points,{},calls,native_factory)
  self.native_points=native;self.coefs=coefs;self.chain=torch_constants(c);self.chain_ids={n:array_digest(a) for n,a in c.items()};self.chain_versions={n:(id(p),p._version) for n,p in self.chain.items()};self.native_ids={k:safe.tensor_hash(a) for k,a in native.items()};self.point_ids={k:point_hash(a) for k,a in points.items()};self.coefficient_ids={n:parameter_hash(a) for n,a in coefs.items()}
  torch=self.torch;self.objective_response=torch.from_numpy(np.asarray(context['baseline']['FP32_cached_response_reference'],np.float64).copy());self.objective_logits=torch.from_numpy(context['baseline_logits']['FP32_logits_cast64'].copy());self.prob64=self.objective_logits.softmax(-1);self.logprob64=self.objective_logits.log_softmax(-1);self.native_logits=self.objective_logits.float();self.native_response=self.objective_response.float();self.prob32=self.native_logits.softmax(-1);self.logprob32=self.native_logits.log_softmax(-1);self.reference_tensors={n:getattr(self,n) for n in ('objective_response','objective_logits','prob64','logprob64','native_logits','native_response','prob32','logprob32')};self.reference_tensor_ids={n:array_digest(p.numpy()) for n,p in self.reference_tensors.items()};self.reference_versions={n:(id(p),p._version) for n,p in self.reference_tensors.items()}
  self.zero_native=np.concatenate([p.numpy().reshape(-1) for p in self.physical_chart]).copy();receipt=clean_json(self.reference_file.read_bytes());receipt.update(protocol=PROTOCOL,objective_reference='Fixed original cached C155 FP32 response/logits; fresh FP64 baseline descriptive only');dump(self.reference_file,receipt);self.reference_file_hash=sha(self.reference_file);self.verify(True)
 def verify(self,rehash=False):
  super().verify(rehash)
  if hasattr(self,'chain'):
   for n,p in self.chain.items():
    need(not p.requires_grad and p.grad is None and (id(p),p._version)==self.chain_versions[n],'Frozen full84 constant changed or acquired gradients')
    if rehash:compare(array_digest(p.detach().numpy()),self.chain_ids[n],'Actual full84 constant/'+n)
   if rehash:compare({n:array_digest(a) for n,a in self.chain_numpy.items()},self.chain_ids,'Original full84 constant bytes')
  if hasattr(self,'reference_tensors'):
   for n,p in self.reference_tensors.items():
    need((id(p),p._version)==self.reference_versions[n] and not p.requires_grad and p.grad is None,'Fixed objective reference mutated')
    if rehash:compare(array_digest(p.numpy()),self.reference_tensor_ids[n],'Fixed objective reference/'+n)
  if hasattr(self,'coefficient_ids'):compare({n:parameter_hash(a) for n,a in self.coefs.items()},self.coefficient_ids,'All seven frozen coefficient tensors')
 def map(self,p,sid,label):
  torch=self.torch;c=self.chain;w=c[sid+'__weights'];e=(w[0]*p[0]+w[1]*p[1])+w[2]*p[2];u=c['V']@(c[sid+'__raw']-c[sid+'__prepared']);point=c[sid+'__raw']+c['V'].T@(e[0]*u+e[1]*torch.roll(u,1));compare(point_hash(point.detach().numpy()),self.point_ids[(sid,label)],'Actual complete objective map matches seal');return point,e
 def full_forward(self,point,graph=False):
  torch=self.torch;need(point.dtype==torch.float64 and tuple(point.shape)==(DIM,) and bool(torch.isfinite(point).all()),'Finite FP64 full endpoint point');mapping=dict(self.double);mapping.update(self.replacements(self.rt.model,self.rt.params,point));self.active=True;self.allow_chart_graph=graph;outs=[]
  try:
   for batch in self.endpoint.batches:
    logits=self.functional_call(self.rt.model,mapping,(),dict(batch,use_cache=False),strict=False).logits;need(logits.dtype==torch.float64,'Full84 actual FP64 logits');last=batch['attention_mask'].sum(1)-1;outs.append(logits[torch.arange(len(last)),last])
   al,rl=outs;observed=torch.cat([al[:,self.rt.responseids].mean(0)-self.objective_response,rl[:,self.rt.readids[1]]-rl[:,self.rt.readids[0]]]);kl=(self.prob64*(self.logprob64-al.log_softmax(-1))).sum(-1).mean().clamp_min(0.);need(bool(torch.isfinite(observed).all()) and bool(torch.isfinite(kl)),'Finite complete84 response/read/KL');need(observed.requires_grad==graph,'Full endpoint graph connection differs')
  finally:self.active=False;self.allow_chart_graph=False;self.verify()
  return observed,kl
 def terms(self,o,kl,point,e,sid):
  torch=self.torch;case=sid.split('__')[0];need(case in ('xor_00','xor_01','xor_11'),'Training classes only');x,y=int(case[-2]),int(case[-1]);sign=torch.tensor(2*np.tile([x,y,1,x^y],4)-1,dtype=torch.float64)
  return dict(response_loss=torch.relu(o[:6].abs().max()/.0018-1).square(),read_loss=torch.relu(1-sign*o[6:]/.075).square().mean(),KL_loss=torch.relu(kl/.0045-1).square(),radius_loss=torch.relu(torch.linalg.vector_norm(point-self.chain[sid+'__prepared'])/20-1).square(),regularizer=.001*e.square().sum()/e.shape[-1])
 def evaluate(self,sid,label,gradient=False):
  torch=self.torch
  with torch.enable_grad() if gradient else torch.no_grad():
   p=torch.tensor(self.coefs[label],dtype=torch.float64,requires_grad=gradient);point,e=self.map(p,sid,label);o,kl=self.full_forward(point,graph=gradient);parts=self.terms(o,kl,point,e,sid);row=dict(state_id=sid,case=sid.split('__')[0],label=label,FP64_point_sha256=point_hash(point.detach().numpy()),parameters_digest=parameter_hash(p.detach().numpy()),observed=o.detach().numpy().tolist(),KL=float(kl.detach()),objective_parts={n:float(v.detach()) for n,v in parts.items()},FP64_acceptance_evaluated=False)
   compare(row['objective_parts'],loss(row['observed'],row['KL'],point.detach().numpy(),self.chain_numpy[sid+'__prepared'],e.detach().numpy(),row['case']),'Independent complete Torch/NumPy losses',atol=1e-10,rtol=1e-9)
   if gradient:
    R=parts['response_loss'];G=sum(v for n,v in parts.items() if n!='response_loss')
    for name,v,retain in [('response_gradient',R,True),('guard_gradient',G,False)]:
     self.calls['backward_attempts']+=1;g=torch.autograd.grad(v,p,retain_graph=retain,create_graph=False,allow_unused=False)[0];self.calls['backward_calls']+=1;need(bool(torch.isfinite(g).all()) and g.shape==p.shape,'Finite complete coefficient gradient');row[name]=g.detach().numpy().tolist()
    need(p.grad is None,'No accumulated coefficient gradients')
  self.verify();return row
 def native(self,sid,label='zero'):
  torch=self.torch;need(label=='zero','Only fixed native alpha0 qualification');p=self.native_points[(sid,label)];compare(safe.tensor_hash(p),self.native_ids[(sid,label)],'Actual sealed native point');outs=[]
  try:
   self.rt.assign(p);flat=np.concatenate([q.detach().numpy().reshape(-1) for q in self.rt.params]);exact(flat,p,'Actual physical FP32 commit')
   with torch.no_grad():
    for batch in self.endpoint.batches:
     logits=self.rt.model(**batch,use_cache=False).logits;need(logits.dtype==torch.float32,'Actual native FP32 logits');last=batch['attention_mask'].sum(1)-1;outs.append(logits[torch.arange(len(last)),last])
    al,rl=outs;o=torch.cat([al[:,self.rt.responseids].mean(0)-self.native_response,rl[:,self.rt.readids[1]]-rl[:,self.rt.readids[0]]]);kl=(self.prob32*(self.logprob32-al.log_softmax(-1))).sum(-1).mean().clamp_min(0.);cross64=(self.prob64*(self.logprob64-al.double().log_softmax(-1))).sum(-1).mean().clamp_min(0.)
   met=archive155.full_metric(o.double().numpy(),float(kl),p,sid.split('__')[0]);qual=archive155.endpoint_gates(met,sid.split('__')[0],p,self.chain_numpy[sid+'__prepared'].astype(np.float32),True)
   return dict(state_id=sid,case=sid.split('__')[0],label=label,metrics=met,qualification=qual,KL_recomputed_FP64_descriptive=float(cross64),actual_physical_FP32_commit=True,preparation_provenance_from_archive_only=True)
  finally:self.rt.assign(self.zero_native);self.verify()
def factory(out,context,c,points,native,coefs,calls):return Full84(out,context,c,points,native,coefs,calls)
