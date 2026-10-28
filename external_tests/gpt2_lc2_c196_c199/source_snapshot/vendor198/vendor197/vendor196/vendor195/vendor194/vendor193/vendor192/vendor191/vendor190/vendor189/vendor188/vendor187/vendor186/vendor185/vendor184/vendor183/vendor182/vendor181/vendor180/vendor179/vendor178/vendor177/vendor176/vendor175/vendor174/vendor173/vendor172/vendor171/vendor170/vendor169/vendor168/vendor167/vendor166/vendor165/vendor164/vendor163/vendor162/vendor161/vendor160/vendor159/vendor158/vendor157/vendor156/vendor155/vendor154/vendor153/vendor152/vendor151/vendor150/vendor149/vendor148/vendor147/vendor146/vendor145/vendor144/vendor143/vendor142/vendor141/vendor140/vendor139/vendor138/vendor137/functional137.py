"""Graph-connected temporary chart replacements; only shared control is differentiated."""
import torch
from torch.func import functional_call
from core137 import *

def torch_proposal(prepared,raw,v,a):
    # No detach/copy_/Parameter mutation in the differentiable proposal.
    z=v@(raw.double()-prepared.double());return (raw.double()+v.T@(a*z)).float()

def loss_parts(observed,kl,point,prepared,a,case):
    rv=torch.relu(observed[:6].double().abs()/RESPONSE_SAFETY-1).square()
    signs=torch.as_tensor(2*np.array(safe.target_bits(case,True)*4)-1,dtype=torch.float64,device=observed.device)
    read=torch.relu(1-signs*observed[6:].double()/MARGIN_SAFETY).square()
    radius=torch.relu(torch.linalg.vector_norm(point.double()-prepared.double())/RADIUS-1).square()
    return dict(response_loss=rv.mean()+rv.max(),read_loss=read.mean(),KL_loss=torch.relu(kl.double()/KL_SAFETY-1).square(),radius_loss=radius,regularizer=L2*a.square().mean())

def replacements(model,params,point):
    ids={id(p) for p in params};named=[(n,p) for n,p in model.named_parameters() if id(p) in ids]
    need([id(p) for _,p in named]==[id(p) for p in params],'Chart parameter order differs')
    need(point.ndim==1 and point.numel()==sum(p.numel() for p in params),'Functional chart shape differs')
    result={};at=0
    for name,p in named:result[name]=point[at:at+p.numel()].reshape(p.shape);at+=p.numel()
    return result

def fixed_digest(model,chart_names):
    h=hashlib.sha256()
    for kind,items in (('parameter',model.named_parameters()),('buffer',model.named_buffers())):
        for name,p in items:
            if kind=='parameter' and name in chart_names:continue
            t=p.detach().cpu().contiguous();h.update((kind+name+str(t.dtype)+str(tuple(t.shape))).encode());h.update(t.numpy().tobytes())
    return h.hexdigest()

class Endpoint:
    def __init__(self,rt):
        self.rt=rt;self.calls=0;self.objects={n:p for n,p in rt.model.named_parameters()};self.chart_names={n for n,p in rt.model.named_parameters() if any(p is q for q in rt.params)}
        need(len(self.chart_names)==len(rt.params),'Chart identity inventory differs');self.chart_values=[p.detach().clone() for p in rt.params];self.digest=fixed_digest(rt.model,self.chart_names)
        self.batches=[rt.tok(prompts,return_tensors='pt',padding=True,truncation=True,max_length=48).to(rt.device) for prompts in (rt.anchors,rt.read_prompts)]
        self.p0=rt.logits0.detach().softmax(-1);self.logp0=rt.logits0.detach().log_softmax(-1)
    def verify(self,rehash=False,check_chart=True):
        need(all(self.objects[n] is p for n,p in self.rt.model.named_parameters()),'Functional parameter identity not restored');self.rt.invariant()
        if check_chart:need(all(torch.equal(p,old) for p,old in zip(self.rt.params,self.chart_values)),'Functional training mutated physical chart')
        if rehash:need(fixed_digest(self.rt.model,self.chart_names)==self.digest,'Frozen backbone/A/buffers changed')
    def __call__(self,point):
        need(point.dtype==torch.float32 and bool(torch.isfinite(point).all()),'Finite float32 functional point required');mapping=replacements(self.rt.model,self.rt.params,point);outputs=[]
        try:
            for batch in self.batches:
                logits=functional_call(self.rt.model,mapping,(),dict(batch,use_cache=False),strict=False).logits;last=batch['attention_mask'].sum(1)-1;outputs.append(logits[torch.arange(len(last),device=point.device),last])
        finally:self.verify()
        al,rl=outputs;rl=rl[:,self.rt.readids];o=torch.cat([al[:,self.rt.responseids].mean(0)-self.rt.response0,rl[:,1]-rl[:,0]])
        kl=(self.p0*(self.logp0-al.log_softmax(-1))).sum(-1).mean().clamp_min(0.);self.calls+=1;return o,kl

class Backend:
    def __init__(self,rt,data,phase,calls):
        self.rt=rt;self.data=data;self.phase=phase;self.calls=calls;self.endpoint=Endpoint(rt);self.V=torch.as_tensor(data['V'],dtype=torch.float64,device=rt.device);self.backward_attempts=0;self.backward_completed=0;self.batches=0;self.zero_calls=0;self.warm_calls=0;self.maximum_point_parity_error=0.
        self.constants={sid:(torch.as_tensor(p['prepared'],dtype=torch.float32,device=rt.device),torch.as_tensor(p['raw'],dtype=torch.float32,device=rt.device)) for sid,p in data['points'].items()}
    def point(self,sid,a):
        prepared,raw=self.constants[sid];return torch_proposal(prepared,raw,self.V,a)
    def zero(self,spec,a):
        sid=spec['state_id']
        with torch.no_grad():
            aa=torch.as_tensor(a,dtype=torch.float64,device=self.rt.device);point=self.point(sid,aa);p=point.cpu().numpy();err=self.phase.functional(sid,a,p);o,kl=self.endpoint(point)
        if self.phase.phase=='functional_warm':self.warm_calls+=1
        else:self.zero_calls+=1
        self.maximum_point_parity_error=max(self.maximum_point_parity_error,err)
        return dict(state_id=sid,observed=o.double().cpu().tolist(),KL=float(kl.cpu()),point_sha256=safe.tensor_hash(p),point_parity_error_Linf=err)
    def gradient(self,a,step,out,budget):
        out=Path(out);out.mkdir(parents=True,exist_ok=True);aa=torch.tensor(a,dtype=torch.float64,device=self.rt.device,requires_grad=True);total=np.zeros_like(a);rows=[]
        for spec in registry():
            budget();sid=spec['state_id'];point=self.point(sid,aa);pn=point.detach().cpu().numpy();err=self.phase.functional(sid,a,pn);self.maximum_point_parity_error=max(self.maximum_point_parity_error,err)
            o,kl=self.endpoint(point);terms=loss_parts(o,kl,point,self.constants[sid][0],aa,spec['case']);loss=sum(terms.values());need(bool(torch.isfinite(loss)),'Nonfinite training objective')
            self.backward_attempts+=1;g=torch.autograd.grad(loss,aa,retain_graph=False,create_graph=False)[0];self.backward_completed+=1;need(bool(torch.isfinite(g).all()),'Nonfinite shared-control gradient');gn=g.detach().cpu().numpy();total+=gn
            row=dict(state_id=sid,case=spec['case'],C137_membership='training',observed=o.detach().double().cpu().tolist(),KL=float(kl.detach().cpu()),functional_point_sha256=safe.tensor_hash(pn),point_parity_error_Linf=err,coefficient_digest=row_digest(a),objective_parts={k:float(v.detach().cpu()) for k,v in terms.items()},objective=float(loss.detach().cpu()),state_gradient=gn.tolist())
            expected=objective(row['observed'],row['KL'],pn,self.data['points'][sid]['prepared'],a,spec['case']);compare(row['objective_parts'],expected,'Training objective parts',atol=1e-10,rtol=1e-9)
            rows.append(row);dump(out/'GRADIENT_PROGRESS.json',rows);self.endpoint.verify();need(aa.grad is None,'Unexpected accumulated leaf gradient');del point,pn,o,kl,terms,loss,g
        self.batches+=1;g,agg=aggregate([r['objective'] for r in rows],[r['state_gradient'] for r in rows])
        for r,w in zip(rows,agg['weights']):r.update(aggregate_weight=w,gradient_contribution=(w*np.array(r['state_gradient'])).tolist())
        dump(out/'AGGREGATION.json',agg);dump(out/'GRADIENT_PROGRESS.json',rows);return g,rows
    def verify(self,rehash=False,check_chart=True):self.endpoint.verify(rehash=rehash,check_chart=check_chart)
    def proposed(self,a,sid):
        with torch.no_grad():return self.point(sid,torch.as_tensor(a,dtype=torch.float64,device=self.rt.device)).cpu().numpy()
    def stats(self):return dict(functional_endpoint_calls=self.endpoint.calls,functional_zero_calls=self.zero_calls,functional_warm_calls=self.warm_calls,scalar_backward_attempts=self.backward_attempts,scalar_backward_calls=self.backward_completed,full12_gradient_batches=self.batches,maximum_point_parity_error_Linf=self.maximum_point_parity_error,torch_optimizer_objects=0,GPT2_optimizer_updates=0,LoRA_A_optimizer_updates=0,base_writer_updates=0,new_Jacobians=0,new_SVD_calls=0)

def factory(rt,data,phase,calls):return Backend(rt,data,phase,calls)
