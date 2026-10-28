"""Real same-run FP64 model, complete coefficient chain, joint scalar-loss pullbacks."""
from core159 import *
from map159 import torch_constants,torch_map,chain_arithmetic
from runtime158 import Directional as Base
class Directional(Base):
 def __init__(self,out,context,points,directions,calls,native_factory=None):
  self.chain_numpy=constants(context['data'])
  with np.load(Path(out).parent/'FROZEN_CHAIN_CONSTANTS.npz',allow_pickle=False) as sealed:
   compare(set(sealed.files),set(self.chain_numpy),'Sealed coefficient constant inventory')
   for name,a in self.chain_numpy.items():exact(a,sealed[name],'Actual chain constant matches pre-model seal/'+name)
  super().__init__(out,context,points,directions,calls,native_factory)
  self.chain=torch_constants(self.chain_numpy);self.chain_ids={n:array_digest(a) for n,a in self.chain_numpy.items()};self.chain_versions={n:p._version for n,p in self.chain.items()};self.chain_tensor_ids={n:id(p) for n,p in self.chain.items()}
  receipt=clean_json(self.reference_file.read_bytes());receipt.update(protocol=PROTOCOL,C158_reference_rule_reused=True,C159_scope=design()['scope']);dump(self.reference_file,receipt);self.reference_file_hash=sha(self.reference_file)
  file=self.out/'BASELINE_RUNTIME_PROVENANCE.json';provenance=clean_json(file.read_bytes());provenance.update(protocol=PROTOCOL,continuous_precommit_FP64_coefficient_chain=True,exact_FP32_commit_derivative_claim=False);dump(file,provenance);self.verify(True)
 def verify(self,rehash=False):
  super().verify(rehash)
  if hasattr(self,'chain'):
   for n,p in self.chain.items():
    need(id(p)==self.chain_tensor_ids[n] and p._version==self.chain_versions[n] and not p.requires_grad and p.grad is None,'Frozen chain tensor changed');compare(array_digest(p.detach().numpy()),self.chain_ids[n],'Actual chain constant bytes/'+n)
   compare({n:array_digest(a) for n,a in self.chain_numpy.items()},self.chain_ids,'Actual NumPy chain constants')
 def point(self,sid,n,alpha=None):
  torch=self.torch
  if alpha is None:alpha=torch.tensor(float(LABELS[n]),dtype=torch.float64)
  point,p,e=torch_map(self.chain,sid,alpha);compare(point_hash(point.detach().numpy()),point_hash(self.points[(sid,n)]),'Actual coefficient map point matches pre-model seal');return point,p,e
 def measure(self,sid,label):
  torch=self.torch
  with torch.no_grad():point,p,e=self.point(sid,label);r=self.forward(point).numpy()
  return dict(state_id=sid,label=label,FP64_point_sha256=point_hash(point.numpy()),actual_coefficient_map_used=True,absolute_response=r.tolist(),responses={n:(r-a).tolist() for n,a in self.references.items()},response_losses={n:response_loss(r-a) for n,a in self.references.items()})
 def ad(self,sid):
  torch=self.torch
  with torch.enable_grad():
   alpha=torch.tensor(0.,dtype=torch.float64,requires_grad=True);point,p,e=self.point(sid,'zero',alpha);absolute=self.forward(point,graph=True);losses={n:torch.relu((absolute-torch.from_numpy(self.references[n])).abs().max()/.0018-1).square() for n in REFERENCES};outputs=[absolute[i] for i in range(6)]+[losses[n] for n in REFERENCES];values=[];chain={}
   for i,v in enumerate(outputs):
    self.calls['backward_attempts']+=1
    gg=torch.autograd.grad(v,(alpha,) if i<6 else (alpha,e,p),retain_graph=i<len(outputs)-1,create_graph=False,allow_unused=False);self.calls['backward_calls']+=1;need(all(bool(torch.isfinite(g).all()) for g in gg),'Finite alpha/control gradients');values.append(float(gg[0].detach()))
    if i>=6:chain[REFERENCES[i-6]]=chain_arithmetic(values[-1],gg[1].detach().numpy(),gg[2].detach().numpy(),self.chain_numpy,sid)
   need(alpha.grad is None and p.grad_fn is not None and e.grad_fn is not None,'Graph-connected non-accumulating coefficient chain');r=absolute.detach().numpy();result=dict(state_id=sid,label='zero',alpha=0.,FP64_point_sha256=point_hash(point.detach().numpy()),absolute_response=r.tolist(),coordinate_derivatives=values[:6],response_loss_derivatives=dict(zip(REFERENCES,values[6:])),response_losses={n:float(v.detach()) for n,v in losses.items()},chain_pullbacks=chain,coefficient_shape=list(p.shape),effective_shape=list(e.shape),only_alpha_and_temporary_control_chain_differentiated=True,physical_parameters_differentiated=False,exact_FP32_commit_derivative_claim=False)
   del absolute,outputs,losses,alpha,point,p,e,gg
  self.verify();return result
def factory(out,context,points,directions,calls):return Directional(out,context,points,directions,calls)
