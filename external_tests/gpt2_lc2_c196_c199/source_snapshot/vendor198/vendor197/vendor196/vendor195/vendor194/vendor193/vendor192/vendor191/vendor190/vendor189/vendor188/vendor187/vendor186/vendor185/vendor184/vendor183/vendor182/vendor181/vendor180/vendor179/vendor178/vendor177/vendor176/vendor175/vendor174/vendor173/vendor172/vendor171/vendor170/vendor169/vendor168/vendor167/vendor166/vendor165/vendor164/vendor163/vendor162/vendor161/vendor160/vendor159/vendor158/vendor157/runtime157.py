"""Graph-connected scalar-alpha FP64 temporary functional calls; physical model never updated."""
from core157 import *

class Directional:
    def __init__(self,out,context,points,directions,calls,native_factory=None):
        import torch
        from torch.func import functional_call
        from functional143 import Endpoint,replacements
        from baseline157 import record_comparison,runtime_provenance
        self.torch=torch;self.functional_call=functional_call;self.replacements=replacements;self.calls=calls;self.context=context;self.points=points;self.directions=directions
        self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True)
        if native_factory is None:
            import runtime155
            compare(Path(runtime155.__file__).resolve(),ROOT/'vendor156/vendor155/runtime155.py','Pinned native factory file')
            native_factory=runtime155.factory
        class NoPhysicalWrites:
            def consume(self,*a,**k):raise ValueError('C157 forbids native chart commits or metric proposals')
        self.rt=native_factory(self.out,context['data']['d'],NoPhysicalWrites(),calls)
        need(calls['model_forwards']==1,'Exactly one native zero-chart initialization forward')
        runtime_provenance(self.out,torch,self.rt.pins,context['modelpins'])
        for k in ('revision','seed','lora_A_sha256','chart_shapes','symbols','anchors','response_ids','sketch_sha256'):
            compare(self.rt.pins[k],context['modelpins'][k],'Pinned model identity/'+k)
        self.endpoint=Endpoint(self.rt);self.chart_names=self.endpoint.chart_names
        self.objects=dict(self.rt.model.named_parameters());self.buffers=dict(self.rt.model.named_buffers())
        self.double={n:(p.detach().double().clone() if p.is_floating_point() else p.detach().clone()) for n,p in list(self.objects.items())+list(self.buffers.items())}
        self.double_ids={n:id(p) for n,p in self.double.items()};self.double_versions={n:p._version for n,p in self.double.items()}
        self.physical_chart=[p.detach().clone() for p in self.rt.params];self.active=False;self.allow_chart_graph=False
        self.hook=self.rt.model.register_forward_pre_hook(self.inside_forward)
        self.local=np.asarray(context['baseline']['FP64_response_reference'],float);self.shared=np.asarray(context['baseline']['FP32_cached_response_reference'],float)
        native=self.rt.logits0.detach().double().cpu().numpy();expected=context['baseline_logits']['FP32_logits_cast64']
        record_comparison(self.out,'NATIVE_FP32_BASELINE_LOGITS',self.rt.logits0.detach().cpu().numpy(),expected)
        record_comparison(self.out,'NATIVE_FP32_CACHED_RESPONSE',self.rt.response0.detach().cpu().numpy(),self.shared)
        need(native.shape==expected.shape and np.max(abs(native-expected))<=REPLAY_TOL,'Native FP32 baseline logits replay differs')
        need(np.max(abs(self.rt.response0.detach().double().cpu().numpy()-self.shared))<=REPLAY_TOL,'Native cached FP32 response replay differs')
        with torch.no_grad():al=self.forward(torch.zeros(DIM,dtype=torch.float64),baseline=True)
        baseline=al.cpu().numpy();want=context['baseline_logits']['FP64_logits']
        record_comparison(self.out,'FUNCTIONAL_FP64_BASELINE_LOGITS',baseline,want)
        record_comparison(self.out,'FUNCTIONAL_FP64_RESPONSE_REFERENCE',baseline[:,self.rt.responseids].mean(0),self.local)
        need(baseline.shape==want.shape and np.max(abs(baseline-want))<=REPLAY_TOL,'FP64 zero-chart baseline logits replay differs')
        need(np.max(abs(baseline[:,self.rt.responseids].mean(0)-self.local))<=REPLAY_TOL,'FP64 response-reference replay differs')
        np.savez_compressed(self.out/'REPLAYED_ZERO_BASELINE_LOGITS.npz',native_FP32_exact_cast64=native,functional_FP64=baseline)
        dump(self.out/'BASELINE_REPLAY.json',dict(original_references_retained=True,native_FP32_logits_Linf=float(np.max(abs(native-expected))),FP64_logits_Linf=float(np.max(abs(baseline-want))),replay_tolerance=REPLAY_TOL,torch_version=str(torch.__version__),FP64_is_ground_truth=False));self.verify(True)
    def inside_forward(self,module,args):
        if not self.active:return
        torch=self.torch
        need(all(p.dtype==torch.float64 for p in module.parameters() if p.is_floating_point()),'Actual functional parameters must be FP64')
        need(all(p.dtype==torch.float64 for p in module.buffers() if p.is_floating_point()),'Actual functional buffers must be FP64')
        for n,p in module.named_parameters():
            need(p.requires_grad==(self.allow_chart_graph and n in self.chart_names),'Only temporary chart graph may require grad')
        self.calls['functional_FP64_forwards']+=1
    def verify(self,rehash=False):
        self.endpoint.verify(rehash=rehash)
        need(all(self.objects[n] is p for n,p in self.rt.model.named_parameters()),'Physical parameter identities not restored')
        need(all(self.buffers[n] is p for n,p in self.rt.model.named_buffers()),'Physical buffer identities not restored')
        need(all(p.dtype==self.torch.float32 for p in self.rt.model.parameters()),'Physical model dtype changed')
        need(all(self.torch.equal(p,q) for p,q in zip(self.rt.params,self.physical_chart)),'Physical chart mutated')
        need(all(id(p)==self.double_ids[n] and p._version==self.double_versions[n] and not p.requires_grad and p.grad is None for n,p in self.double.items()),'Frozen FP64 parameters mutated or acquired gradients')
    def forward(self,point,baseline=False,graph=False):
        torch=self.torch;need(point.dtype==torch.float64 and tuple(point.shape)==(DIM,) and bool(torch.isfinite(point).all()),'Finite differentiable FP64 point required')
        mapping=dict(self.double);mapping.update(self.replacements(self.rt.model,self.rt.params,point));self.active=True;self.allow_chart_graph=graph
        try:
            batch=self.endpoint.batches[0];logits=self.functional_call(self.rt.model,mapping,(),dict(batch,use_cache=False),strict=False).logits
            need(logits.dtype==torch.float64,'Actual logits must be FP64');last=batch['attention_mask'].sum(1)-1;al=logits[torch.arange(len(last),device=point.device),last]
            result=al if baseline else al[:,self.rt.responseids].mean(0)
            need(bool(torch.isfinite(result).all()),'Nonfinite FP64 model output');need(result.requires_grad==graph,'Functional graph connection differs')
        finally:self.active=False;self.allow_chart_graph=False;self.verify()
        return result
    def measure(self,sid,label):
        torch=self.torch;point=torch.from_numpy(self.points[(sid,label)])
        with torch.no_grad():r=self.forward(point).cpu().numpy()
        return dict(state_id=sid,label=label,FP64_point_sha256=point_hash(point.numpy()),absolute_response=r.tolist(),response_local=(r-self.local).tolist(),response_shared=(r-self.shared).tolist(),response_losses=losses(r,self.local,self.shared))
    def ad(self,sid):
        torch=self.torch
        with torch.enable_grad():
            alpha=torch.tensor(0.,dtype=torch.float64,requires_grad=True);base=torch.from_numpy(self.points[(sid,'zero')]);direction=torch.from_numpy(self.directions[sid]);point=base+alpha*direction
            absolute=self.forward(point,graph=True);local=absolute-torch.from_numpy(self.local);shared=absolute-torch.from_numpy(self.shared)
            response_losses={k:torch.relu(r.abs().max()/.0018-1).square() for k,r in [('local',local),('shared',shared)]}
            outputs=[absolute[i] for i in range(6)]+[response_losses['local'],response_losses['shared']];values=[]
            for i,v in enumerate(outputs):
                self.calls['backward_attempts']+=1;g=torch.autograd.grad(v,alpha,retain_graph=i<len(outputs)-1,create_graph=False,allow_unused=False)[0];self.calls['backward_calls']+=1
                need(g.shape==() and bool(torch.isfinite(g)),'Finite scalar directional derivative required');values.append(float(g.detach()))
            need(alpha.grad is None,'No accumulated alpha gradient');r=absolute.detach().cpu().numpy()
            result=dict(state_id=sid,alpha=0.,FP64_point_sha256=point_hash(point.detach().numpy()),absolute_response=r.tolist(),coordinate_derivatives=values[:6],response_loss_derivatives=dict(zip(('local','shared'),values[6:])),response_losses={k:float(v.detach()) for k,v in response_losses.items()},only_scalar_alpha_differentiated=True,physical_parameters_differentiated=False)
            del absolute,outputs,response_losses,point,alpha
        self.verify();return result

def factory(out,context,points,directions,calls):return Directional(out,context,points,directions,calls)
