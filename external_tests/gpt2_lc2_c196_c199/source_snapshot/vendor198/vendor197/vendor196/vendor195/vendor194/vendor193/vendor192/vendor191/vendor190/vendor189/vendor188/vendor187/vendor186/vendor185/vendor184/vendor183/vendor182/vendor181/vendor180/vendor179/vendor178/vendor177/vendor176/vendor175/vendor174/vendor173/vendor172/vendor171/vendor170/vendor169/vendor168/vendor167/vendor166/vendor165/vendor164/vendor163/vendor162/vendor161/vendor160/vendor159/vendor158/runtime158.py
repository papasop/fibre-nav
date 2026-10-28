"""One frozen runtime, FP64 functional chart, three separately frozen references; scalar alpha AD only."""
from core158 import *
from runtime157 import Directional as PreviousDirectional
from baseline158 import record,provenance
class Directional(PreviousDirectional):
    def __init__(self,out,context,points,directions,calls,native_factory=None):
        import torch
        from torch.func import functional_call
        from functional143 import Endpoint,replacements
        self.torch=torch;self.functional_call=functional_call;self.replacements=replacements;self.calls=calls;self.context=context;self.points=points;self.directions=directions;self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True)
        if native_factory is None:
            import runtime155
            compare(Path(runtime155.__file__).resolve(),ROOT/'vendor157/vendor156/vendor155/runtime155.py','Pinned native runtime path');native_factory=runtime155.factory
        class NoPhysicalWrites:
            def consume(self,*a,**k):raise ValueError('C158 forbids native proposal commits')
        self.rt=native_factory(self.out,context['data']['d'],NoPhysicalWrites(),calls);need(calls['model_forwards']==1,'One native initialization forward')
        provenance(self.out,torch,self.rt.pins,context['modelpins'])
        for k in ('revision','seed','lora_A_sha256','chart_shapes','symbols','anchors','response_ids','sketch_sha256'):compare(self.rt.pins[k],context['modelpins'][k],'Pinned semantic model identity/'+k)
        self.endpoint=Endpoint(self.rt);self.chart_names=self.endpoint.chart_names;self.objects=dict(self.rt.model.named_parameters());self.buffers=dict(self.rt.model.named_buffers());self.double={n:(p.detach().double().clone() if p.is_floating_point() else p.detach().clone()) for n,p in list(self.objects.items())+list(self.buffers.items())};self.double_ids={n:id(p) for n,p in self.double.items()};self.double_versions={n:p._version for n,p in self.double.items()};self.physical_chart=[p.detach().clone() for p in self.rt.params];self.active=False;self.allow_chart_graph=False;self.hook=self.rt.model.register_forward_pre_hook(self.inside_forward)
        self.double_digest=self.hash_double();self.baseline_records={}
        self.baseline_records['native_logits']=record(self.out,'NATIVE_FP32_BASELINE_LOGITS',self.rt.logits0.detach().numpy(),context['baseline_logits']['FP32_logits_cast64'])
        self.baseline_records['native_cached_response']=record(self.out,'NATIVE_FP32_CACHED_RESPONSE',self.rt.response0.detach().numpy(),np.asarray(context['baseline']['FP32_cached_response_reference'],float))
        need(self.baseline_records['native_logits']['shapes_equal'] and self.baseline_records['native_logits']['all_values_finite'],'Native baseline shape/nonfinite mismatch')
        need(self.baseline_records['native_cached_response']['shapes_equal'] and self.baseline_records['native_cached_response']['all_values_finite'],'Native cached reference shape/nonfinite mismatch')
        with torch.no_grad():
            first=self.forward(torch.zeros(DIM,dtype=torch.float64),baseline=True);current=first[:,self.rt.responseids].mean(0).cpu().numpy();repeat=self.forward(torch.zeros(DIM,dtype=torch.float64),baseline=True)
        first=first.cpu().numpy();repeat=repeat.cpu().numpy();need(np.isfinite(first).all() and np.isfinite(repeat).all(),'Finite same-run FP64 zero-chart baselines')
        self.baseline_records['FP64_cross_run']=record(self.out,'FUNCTIONAL_FP64_BASELINE_LOGITS',first,context['baseline_logits']['FP64_logits'])
        need(self.baseline_records['FP64_cross_run']['shapes_equal'] and self.baseline_records['FP64_cross_run']['all_values_finite'],'FP64 baseline shape/nonfinite mismatch')
        self.baseline_records['FP64_repeat']=record(self.out,'WITHIN_RUN_FP64_ZERO_BASELINE_REPEAT',repeat,first,'within_run_repeat')
        need(self.baseline_records['FP64_repeat']['passed_original_replay_tolerance'],'Within-run FP64 baseline repeat differs; stop before AD')
        self.references={'current_FP64':current.copy(),'archived_FP64':np.asarray(context['baseline']['FP64_response_reference'],float).copy(),'archived_FP32':np.asarray(context['baseline']['FP32_cached_response_reference'],float).copy()}
        for name,a in self.references.items():farray(a,(6,));need(a.dtype==np.float64,'FP64 fixed reference storage')
        self.reference_ids={n:hashlib.sha256(a.tobytes()).hexdigest() for n,a in self.references.items()}
        self.reference_file=self.out/'REFERENCES_FROZEN_BEFORE_FOCUS.json';dump(self.reference_file,dict(protocol=PROTOCOL,rule=design()['reference_rule'],references={n:a.tolist() for n,a in self.references.items()},reference_tensor_sha256=self.reference_ids,model_forwards_so_far=calls['model_forwards'],state_probes_so_far=0,original_archived_references_retained=True,old_scores_recomputed_or_reclassified=False));self.reference_file_hash=sha(self.reference_file);need(calls['model_forwards']==3,'Freeze references after exactly three initialization forwards');self.verify(True)
    def hash_double(self):
        h=hashlib.sha256()
        for n,p in sorted(self.double.items()):
            a=p.detach().cpu().contiguous().numpy();h.update((n+str(a.dtype)+str(a.shape)).encode());h.update(memoryview(a).cast('B'))
        return h.hexdigest()
    def verify(self,rehash=False):
        super().verify(rehash)
        if rehash and hasattr(self,'double_digest'):compare(self.hash_double(),self.double_digest,'Actual complete frozen FP64 parameter/buffer bytes')
        if hasattr(self,'reference_ids'):
            compare({n:hashlib.sha256(a.tobytes()).hexdigest() for n,a in self.references.items()},self.reference_ids,'Actual three reference tensors')
            compare(sha(self.reference_file),self.reference_file_hash,'Reference seal file')
    def measure(self,sid,label):
        torch=self.torch;point=torch.from_numpy(self.points[(sid,label)])
        with torch.no_grad():r=self.forward(point).cpu().numpy()
        return dict(state_id=sid,label=label,FP64_point_sha256=point_hash(point.numpy()),absolute_response=r.tolist(),responses={n:(r-a).tolist() for n,a in self.references.items()},response_losses={n:response_loss(r-a) for n,a in self.references.items()})
    def ad(self,sid):
        torch=self.torch
        with torch.enable_grad():
            alpha=torch.tensor(0.,dtype=torch.float64,requires_grad=True);point=torch.from_numpy(self.points[(sid,'zero')])+alpha*torch.from_numpy(self.directions[sid]);absolute=self.forward(point,graph=True)
            response_losses={n:torch.relu((absolute-torch.from_numpy(self.references[n])).abs().max()/.0018-1).square() for n in REFERENCES};outputs=[absolute[i] for i in range(6)]+[response_losses[n] for n in REFERENCES];values=[]
            for i,v in enumerate(outputs):
                self.calls['backward_attempts']+=1;g=torch.autograd.grad(v,alpha,retain_graph=i<len(outputs)-1,create_graph=False,allow_unused=False)[0];self.calls['backward_calls']+=1;need(g.shape==() and bool(torch.isfinite(g)),'Finite scalar alpha derivative');values.append(float(g.detach()))
            need(alpha.grad is None,'No accumulated alpha gradient');r=absolute.detach().numpy();result=dict(state_id=sid,label='zero',alpha=0.,FP64_point_sha256=point_hash(point.detach().numpy()),absolute_response=r.tolist(),coordinate_derivatives=values[:6],response_loss_derivatives=dict(zip(REFERENCES,values[6:])),response_losses={n:float(v.detach()) for n,v in response_losses.items()},only_scalar_alpha_differentiated=True,physical_parameters_differentiated=False)
            del absolute,outputs,response_losses,point,alpha
        self.verify();return result
def factory(out,context,points,directions,calls):return Directional(out,context,points,directions,calls)
