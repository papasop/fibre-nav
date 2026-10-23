"""Actual CPU functional endpoint, scalar AD to effective control, exact linear routing pullback."""
from core148 import *
import torch
from functional146 import Endpoint,torch_proposal,scalar_gradients

class Backend:
    def __init__(self,rt,d,phase,calls):
        self.rt=rt;self.d=d;self.phase=phase;self.endpoint=Endpoint(rt);self.V=torch.as_tensor(d['V'],dtype=torch.float64,device=rt.device);self.weights={sid:torch.as_tensor(w,dtype=torch.float64,device=rt.device) for sid,w in d['weights'].items()};self.constants={sid:(torch.as_tensor(p['prepared'],dtype=torch.float32,device=rt.device),torch.as_tensor(p['raw'],dtype=torch.float32,device=rt.device)) for sid,p in d['points'].items()};self.z=0;self.w=0;self.attempts=0;self.b=0;self.batches=0;self.error=0.;self.per_arm={arm:dict(graph_forwards=0,backwards=0) for arm in TRAIN_ARMS}
    def point(self,sid,params,arm):
        # Commit the effective float64 coefficient map in NumPy identically to the frozen writer.
        # AD targets this effective leaf; the fixed affine routing pullback is explicit below.
        aa=effective(params.detach().cpu().numpy(),self.d['weights'][sid],arm)
        a=torch.tensor(aa,dtype=torch.float64,device=self.rt.device,requires_grad=params.requires_grad);p,r=self.constants[sid];return torch_proposal(p,r,self.V,a),a
    def proposed(self,params,sid,arm):
        with torch.no_grad():return self.point(sid,torch.as_tensor(params,dtype=torch.float64,device=self.rt.device),arm)[0].cpu().numpy()
    def parity(self,sp,params,arm):
        sid=sp['state_id']
        with torch.no_grad():point,a=self.point(sid,torch.as_tensor(params,dtype=torch.float64,device=self.rt.device),arm);pn=point.cpu().numpy();err=self.phase.functional(sid,params,arm,pn);o,k=self.endpoint(point)
        if self.phase.phase=='functional_zero':self.z+=1
        else:self.w+=1
        self.error=max(self.error,err);return dict(state_id=sid,arm=arm,point_sha256=safe.tensor_hash(pn),observed=o.double().cpu().tolist(),KL=float(k.cpu()),point_parity_error_Linf=err)
    def gradient(self,params,step,arm,out,budget):
        out=Path(out);out.mkdir(parents=True,exist_ok=True);pp=torch.tensor(params,dtype=torch.float64,device=self.rt.device,requires_grad=True);rows=[]
        def attempt():self.attempts+=1
        def done():self.b+=1;self.per_arm[arm]['backwards']+=1
        for sp in registry():
            budget();sid=sp['state_id'];point,a=self.point(sid,pp,arm);pn=point.detach().cpu().numpy();err=self.phase.functional(sid,params,arm,pn);self.error=max(self.error,err);r=scalar_gradients(self.endpoint,point,self.constants[sid][0],a,sp['case'],attempt,done);self.per_arm[arm]['graph_forwards']+=1;w=self.d['weights'][sid]
            for key in ('response_gradient','guard_gradient'):r['effective_'+key]=r[key];r[key]=lift_gradient(np.array(r[key],float),w,arm).tolist()
            compare(r['objective_parts'],loss(r['observed'],r['KL'],pn,self.d['points'][sid]['prepared'],a.detach().cpu().numpy(),sp['case']),'Torch/NumPy objective',atol=1e-10,rtol=1e-9);rows.append(dict(**sp,arm=arm,**r,routing_weights=w.tolist(),parameters_digest=row_digest(params),effective_control_digest=row_digest(a.detach().cpu().numpy()),functional_point_sha256=safe.tensor_hash(pn),point_parity_error_Linf=err));dump(out/'GRADIENT_PROGRESS.json',rows);self.endpoint.verify();need(pp.grad is None,'No accumulated optimizer gradient')
        self.batches+=1;g,agg=aggregate(rows,params.shape);dump(out/'AGGREGATION.json',agg);dump(out/'GRADIENT_PROGRESS.json',rows);return g,rows
    def verify(self,rehash=False,check_chart=True):self.endpoint.verify(rehash=rehash,check_chart=check_chart)
    def stats(self):return dict(functional_endpoint_calls=self.endpoint.calls,functional_zero_calls=self.z,functional_warm_calls=self.w,full84_gradient_batches=self.batches,scalar_backward_attempts=self.attempts,scalar_backward_calls=self.b,maximum_point_parity_error_Linf=self.error,per_arm=self.per_arm,torch_optimizer_objects=0,GPT2_optimizer_updates=0,LoRA_A_optimizer_updates=0,base_writer_updates=0)

def factory(rt,d,phase,calls):return Backend(rt,d,phase,calls)
