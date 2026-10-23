"""Two scalar gradients per state at one frozen control; no optimizer or fit."""
import torch
from core145 import *
from functional143 import Endpoint,torch_proposal,loss_parts

def scalar_gradients(endpoint,point,prepared,a,case,on_attempt=None,on_complete=None):
    observed,kl=endpoint(point);terms=loss_parts(observed,kl,point,prepared,a,case);original=sum(terms.values());ratio=observed[:6].double().abs().amax()/.002
    need(bool(torch.isfinite(original)) and bool(torch.isfinite(ratio)),'Nonfinite diagnostic scalar')
    if on_attempt:on_attempt()
    ga=torch.autograd.grad(original,a,retain_graph=True,create_graph=False)[0]
    if on_complete:on_complete()
    if on_attempt:on_attempt()
    gq=torch.autograd.grad(ratio,a,retain_graph=False,create_graph=False)[0]
    if on_complete:on_complete()
    need(bool(torch.isfinite(ga).all()) and bool(torch.isfinite(gq).all()),'Nonfinite diagnostic gradient')
    return dict(observed=observed.detach().double().cpu().tolist(),KL=float(kl.detach().cpu()),original_objective_parts={k:float(v.detach().cpu()) for k,v in terms.items()},original_objective=float(original.detach().cpu()),response_ratio=float(ratio.detach().cpu()),original_gradient=ga.detach().cpu().tolist(),response_gradient=gq.detach().cpu().tolist())

class Backend:
    def __init__(self,rt,data,phase,calls):
        self.rt=rt;self.data=data;self.phase=phase;self.calls=calls;self.endpoint=Endpoint(rt);self.V=torch.as_tensor(data['V'],dtype=torch.float64,device=rt.device);self.parity_count=0;self.gradient_records=0;self.backward_attempts=0;self.backward_completed=0;self.point_error=0.
        self.constants={sid:(torch.as_tensor(p['prepared'],dtype=torch.float32,device=rt.device),torch.as_tensor(p['raw'],dtype=torch.float32,device=rt.device)) for sid,p in data['points'].items()}
    def point(self,sid,a):
        prepared,raw=self.constants[sid];return torch_proposal(prepared,raw,self.V,a)
    def parity(self,sp):
        sid=sp['state_id']
        with torch.no_grad():
            aa=torch.as_tensor(self.data['a'],dtype=torch.float64,device=self.rt.device);point=self.point(sid,aa);p=point.cpu().numpy();err=self.phase.functional(sid,self.data['a'],p,'parity');o,k=self.endpoint(point)
        self.parity_count+=1;self.point_error=max(self.point_error,err)
        return dict(state_id=sid,point_sha256=safe.tensor_hash(p),point_parity_error_Linf=err,observed=o.double().cpu().tolist(),KL=float(k.cpu()))
    def gradient(self,sp,baseline):
        sid=sp['state_id'];aa=torch.tensor(self.data['a'],dtype=torch.float64,device=self.rt.device,requires_grad=True);point=self.point(sid,aa);p=point.detach().cpu().numpy();err=self.phase.functional(sid,self.data['a'],p,'gradient')
        # Compare this exact graph forward before differentiating it.
        captured={}
        def checked_endpoint(point):
            observed,kl=self.endpoint(point);close(observed.detach().double().cpu().tolist(),baseline['observed'],'Gradient/frozen functional forward parity',2e-6);need(abs(float(kl.detach().cpu())-baseline['KL'])<=2e-7,'Gradient/frozen functional KL parity');captured['ok']=True;return observed,kl
        def attempt():self.backward_attempts+=1
        def complete():self.backward_completed+=1
        rec=scalar_gradients(checked_endpoint,point,self.constants[sid][0],aa,sp['case'],attempt,complete);self.gradient_records+=1;self.point_error=max(self.point_error,err);self.endpoint.verify();need(aa.grad is None,'Unexpected accumulated leaf gradient')
        return dict(**sp,baseline_coefficient_digest=row_digest(self.data['a']),functional_point_sha256=safe.tensor_hash(p),point_parity_error_Linf=err,forward_parity_checked_before_backwards=bool(captured.get('ok')),**rec)
    def verify(self,rehash=False,check_chart=True):self.endpoint.verify(rehash=rehash,check_chart=check_chart)
    def stats(self):return dict(functional_endpoint_calls=self.endpoint.calls,functional_parity_records=self.parity_count,gradient_state_records=self.gradient_records,scalar_backward_attempts=self.backward_attempts,scalar_backward_calls=self.backward_completed,maximum_point_parity_error_Linf=self.point_error,optimizer_objects=0,optimizer_updates=0,GPT2_optimizer_updates=0,LoRA_A_optimizer_updates=0,base_writer_updates=0)

def factory(rt,data,phase,calls):return Backend(rt,data,phase,calls)
