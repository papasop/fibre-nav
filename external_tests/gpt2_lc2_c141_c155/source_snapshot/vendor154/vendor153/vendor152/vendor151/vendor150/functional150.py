"""Real CPU scalar AD to effective routed controls with explicit linear pullback."""
from core150 import *
import torch
from functional146 import Endpoint

def torch_proposal(prepared,raw,V,e):
    u=V@(raw.double()-prepared.double());s=e*u if e.ndim==1 else e[0]*u+e[1]*torch.roll(u,1);return (raw.double()+V.T@s).float()

def loss_parts(o,kl,point,prepared,e,case):
    signs=torch.as_tensor(2*np.array(safe.target_bits(case,True)*4)-1,dtype=torch.float64,device=o.device)
    return dict(response_loss=torch.relu(o[:6].double().abs().amax()/.0018-1).square(),read_loss=torch.relu(1-signs*o[6:].double()/.075).square().mean(),KL_loss=torch.relu(kl.double()/.0045-1).square(),radius_loss=torch.relu(torch.linalg.vector_norm(point.double()-prepared.double())/20-1).square(),regularizer=.001*e.square().sum()/e.shape[-1])

def scalar_gradients(endpoint,point,prepared,e,case,on_attempt=None,on_complete=None):
    o,kl=endpoint(point);parts=loss_parts(o,kl,point,prepared,e,case);R=parts['response_loss'];G=sum(v for k,v in parts.items() if k!='response_loss');need(bool(torch.isfinite(R)) and bool(torch.isfinite(G)),'Finite scalar losses')
    def derivative(x,retain):
        if on_attempt:on_attempt()
        g=torch.autograd.grad(x,e,retain_graph=retain,create_graph=False)[0]
        if on_complete:on_complete()
        need(bool(torch.isfinite(g).all()),'Finite effective gradient');return g.detach().cpu().numpy()
    gr=derivative(R,True);gg=derivative(G,False);return dict(observed=o.detach().double().cpu().tolist(),KL=float(kl.detach().cpu()),objective_parts={k:float(v.detach().cpu()) for k,v in parts.items()},response_gradient=gr.tolist(),guard_gradient=gg.tolist())

class Backend:
    def __init__(self,rt,d,phase,calls):
        self.rt=rt;self.d=d;self.phase=phase;self.endpoint=Endpoint(rt);self.V=torch.as_tensor(d['V'],dtype=torch.float64,device=rt.device);self.constants={sid:(torch.as_tensor(p['prepared'],dtype=torch.float32,device=rt.device),torch.as_tensor(p['raw'],dtype=torch.float32,device=rt.device)) for sid,p in d['points'].items()};self.z=0;self.w=0;self.attempts=0;self.b=0;self.batches=0;self.error=0.;self.per_arm={arm:dict(graph_forwards=0,backwards=0) for arm in TRAIN_ARMS}
    def point(self,sid,p,arm):
        en=effective(p.detach().cpu().numpy(),self.d['weights'][sid],arm);e=torch.tensor(en,dtype=torch.float64,device=self.rt.device,requires_grad=p.requires_grad);theta,raw=self.constants[sid];return torch_proposal(theta,raw,self.V,e),e
    def proposed(self,p,sid,arm):
        with torch.no_grad():return self.point(sid,torch.as_tensor(p,dtype=torch.float64,device=self.rt.device),arm)[0].cpu().numpy()
    def parity(self,sp,p,arm):
        sid=sp['state_id']
        with torch.no_grad():point,e=self.point(sid,torch.as_tensor(p,dtype=torch.float64,device=self.rt.device),arm);pn=point.cpu().numpy();err=self.phase.functional(sid,p,arm,pn);o,k=self.endpoint(point)
        if self.phase.phase=='functional_zero':self.z+=1
        else:self.w+=1
        self.error=max(self.error,err);return dict(state_id=sid,arm=arm,point_sha256=safe.tensor_hash(pn),observed=o.double().cpu().tolist(),KL=float(k.cpu()),point_parity_error_Linf=err)
    def gradient(self,p,step,arm,out,budget):
        out=Path(out);out.mkdir(parents=True,exist_ok=True);pp=torch.tensor(p,dtype=torch.float64,device=self.rt.device,requires_grad=True);rows=[]
        def attempt():self.attempts+=1
        def done():self.b+=1;self.per_arm[arm]['backwards']+=1
        for sp in registry():
            budget();sid=sp['state_id'];point,e=self.point(sid,pp,arm);pn=point.detach().cpu().numpy();err=self.phase.functional(sid,p,arm,pn);self.error=max(self.error,err);r=scalar_gradients(self.endpoint,point,self.constants[sid][0],e,sp['case'],attempt,done);self.per_arm[arm]['graph_forwards']+=1;w=self.d['weights'][sid]
            for key in ('response_gradient','guard_gradient'):r['effective_'+key]=r[key];r[key]=pullback(np.asarray(r[key],float),w,arm).tolist()
            compare(r['objective_parts'],loss(r['observed'],r['KL'],pn,self.d['points'][sid]['prepared'],e.detach().cpu().numpy(),sp['case']),'Torch/NumPy losses',atol=1e-10,rtol=1e-9);rows.append(dict(**sp,arm=arm,**r,routing_weights=w.tolist(),parameters_digest=row_digest(p),effective_control_digest=row_digest(e.detach().cpu().numpy()),functional_point_sha256=safe.tensor_hash(pn),point_parity_error_Linf=err));dump(out/'GRADIENT_PROGRESS.json',rows);self.endpoint.verify();need(pp.grad is None,'No accumulated parameter gradient')
        self.batches+=1;g,agg=aggregate(rows,p.shape);dump(out/'AGGREGATION.json',agg);dump(out/'GRADIENT_PROGRESS.json',rows);return g,rows
    def verify(self,rehash=False,check_chart=True):self.endpoint.verify(rehash=rehash,check_chart=check_chart)
    def stats(self):return dict(functional_endpoint_calls=self.endpoint.calls,functional_zero_calls=self.z,functional_warm_calls=self.w,full84_gradient_batches=self.batches,scalar_backward_attempts=self.attempts,scalar_backward_calls=self.b,maximum_point_parity_error_Linf=self.error,per_arm=self.per_arm,torch_optimizer_objects=0,GPT2_optimizer_updates=0,LoRA_A_optimizer_updates=0,base_writer_updates=0)

def factory(rt,d,phase,calls):return Backend(rt,d,phase,calls)
