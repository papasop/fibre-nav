"""Two scalar derivatives per state, no response-coordinate Jacobian construction."""
import torch
from core146 import *
from functional143 import Endpoint,torch_proposal

def loss_parts(observed,kl,point,prepared,a,case):
    need(case in ('xor_00','xor_01','xor_11'),'Excluded class forbidden')
    q=observed[:6].double().abs().amax()/RESPONSE_SAFETY
    signs=torch.as_tensor(2*np.array(safe.target_bits(case,True)*4)-1,dtype=torch.float64,device=observed.device)
    return dict(response_loss=torch.relu(q-1).square(),read_loss=torch.relu(1-signs*observed[6:].double()/MARGIN_SAFETY).square().mean(),KL_loss=torch.relu(kl.double()/KL_SAFETY-1).square(),radius_loss=torch.relu(torch.linalg.vector_norm(point.double()-prepared.double())/RADIUS-1).square(),regularizer=L2*a.square().mean())

def scalar_gradients(endpoint,point,prepared,a,case,on_attempt=None,on_complete=None):
    o,kl=endpoint(point);parts=loss_parts(o,kl,point,prepared,a,case);response=parts['response_loss'];guard=sum(v for k,v in parts.items() if k!='response_loss')
    need(bool(torch.isfinite(response)) and bool(torch.isfinite(guard)),'Nonfinite training scalar')
    def derivative(x,retain):
        if on_attempt:on_attempt()
        g=torch.autograd.grad(x,a,retain_graph=retain,create_graph=False)[0]
        if on_complete:on_complete()
        need(bool(torch.isfinite(g).all()),'Nonfinite training gradient');return g.detach().cpu().numpy()
    gr=derivative(response,True);gg=derivative(guard,False)
    return dict(observed=o.detach().double().cpu().tolist(),KL=float(kl.detach().cpu()),objective_parts={k:float(v.detach().cpu()) for k,v in parts.items()},response_gradient=gr.tolist(),guard_gradient=gg.tolist())

class Backend:
    def __init__(self,rt,data,phase,calls):
        self.rt=rt;self.data=data;self.phase=phase;self.calls=calls;self.endpoint=Endpoint(rt);self.V=torch.as_tensor(data['V'],dtype=torch.float64,device=rt.device);self.zero_calls=0;self.warm_calls=0;self.batches=0;self.attempts=0;self.backwards=0;self.error=0.
        self.constants={sid:(torch.as_tensor(p['prepared'],dtype=torch.float32,device=rt.device),torch.as_tensor(p['raw'],dtype=torch.float32,device=rt.device)) for sid,p in data['points'].items()}
    def point(self,sid,a):
        p,r=self.constants[sid];return torch_proposal(p,r,self.V,a)
    def parity(self,sp,a):
        sid=sp['state_id']
        with torch.no_grad():
            aa=torch.as_tensor(a,dtype=torch.float64,device=self.rt.device);point=self.point(sid,aa);p=point.cpu().numpy();err=self.phase.functional(sid,a,p);o,k=self.endpoint(point)
        if self.phase.phase=='functional_zero':self.zero_calls+=1
        else:self.warm_calls+=1
        self.error=max(self.error,err);return dict(state_id=sid,point_sha256=safe.tensor_hash(p),point_parity_error_Linf=err,observed=o.double().cpu().tolist(),KL=float(k.cpu()))
    def gradient(self,a,step,out,budget):
        out=Path(out);out.mkdir(parents=True,exist_ok=True);aa=torch.tensor(a,dtype=torch.float64,device=self.rt.device,requires_grad=True);rows=[]
        def attempt():self.attempts+=1
        def complete():self.backwards+=1
        for sp in registry():
            budget();sid=sp['state_id'];point=self.point(sid,aa);pn=point.detach().cpu().numpy();err=self.phase.functional(sid,a,pn);self.error=max(self.error,err);r=scalar_gradients(self.endpoint,point,self.constants[sid][0],aa,sp['case'],attempt,complete)
            compare(r['objective_parts'],objective(r['observed'],r['KL'],pn,self.data['points'][sid]['prepared'],a,sp['case']),'Torch/NumPy objective',atol=1e-10,rtol=1e-9)
            rows.append(dict(**sp,**r,coefficient_digest=row_digest(a),functional_point_sha256=safe.tensor_hash(pn),point_parity_error_Linf=err));dump(out/'GRADIENT_PROGRESS.json',rows);self.endpoint.verify();need(aa.grad is None,'Leaf gradient accumulation forbidden')
        self.batches+=1;g,agg=aggregate(rows,len(a))
        for r,w,v in zip(rows,agg['response_weights'],agg['state_gradient_contributions']):r.update(response_weight=w,guard_weight=1/N,gradient_contribution=v)
        dump(out/'GRADIENT_PROGRESS.json',rows);dump(out/'AGGREGATION.json',agg);return g,rows
    def proposed(self,a,sid):
        with torch.no_grad():return self.point(sid,torch.as_tensor(a,dtype=torch.float64,device=self.rt.device)).cpu().numpy()
    def verify(self,rehash=False,check_chart=True):self.endpoint.verify(rehash=rehash,check_chart=check_chart)
    def stats(self):return dict(functional_endpoint_calls=self.endpoint.calls,functional_zero_calls=self.zero_calls,functional_warm_calls=self.warm_calls,full84_gradient_batches=self.batches,scalar_backward_attempts=self.attempts,scalar_backward_calls=self.backwards,maximum_point_parity_error_Linf=self.error,torch_optimizer_objects=0,GPT2_optimizer_updates=0,LoRA_A_optimizer_updates=0,base_writer_updates=0)
def factory(rt,data,phase,calls):return Backend(rt,data,phase,calls)
