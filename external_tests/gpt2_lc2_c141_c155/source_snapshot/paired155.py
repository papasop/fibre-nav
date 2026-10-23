"""Actual no-grad functional calls; precision casts never commit physical FP64 weights."""
import torch
from torch.func import functional_call
from core155 import *
from functional143 import Endpoint,replacements,fixed_digest

class Paired:
    def __init__(self,rt,calls):
        self.rt=rt;self.calls=calls;self.endpoint=Endpoint(rt);self.chart_names=self.endpoint.chart_names;self.chart_order=[n for n,p in rt.model.named_parameters() if n in self.chart_names];self.objects=dict(rt.model.named_parameters());self.buffers=dict(rt.model.named_buffers());self.double={n:(p.detach().double().clone() if p.is_floating_point() else p.detach().clone()) for n,p in list(self.objects.items())+list(self.buffers.items())};self.double_ids={n:id(p) for n,p in self.double.items()};self.double_versions={n:p._version for n,p in self.double.items()};self.native_reference=rt.response0.detach().double().clone();self.cross_logits=rt.logits0.detach().double().clone();self.active_dtype=None
        self.hook=rt.model.register_forward_pre_hook(self.validate_inside_forward)
        with torch.no_grad():self.baseline_logits=self.forward64(torch.zeros(DIM,dtype=torch.float32),anchors_only=True);self.baseline_response=self.baseline_logits[:,rt.responseids].mean(0);self.calls['FP64_baseline_forwards']+=1
        self.offset=(self.baseline_response-self.native_reference).cpu().numpy();dump(rt.out/'FP64_BASELINE_PROVENANCE.json',dict(float32_parameter_values_cast_exactly_to_float64=True,FP64_chart_source='original all-zero float32 chart cast to float64',FP32_response_reference=self.native_reference.cpu().tolist(),FP64_response_reference=self.baseline_response.cpu().tolist(),response_reference_offset=self.offset.tolist(),FP32_logits_recomputed_in_FP64_for_cross_KL=True,FP64_is_not_ground_truth=True,precision_diagnostic_only=True));np.savez_compressed(rt.out/'FP64_BASELINE_LOGITS.npz',FP32_logits_cast64=self.cross_logits.cpu().numpy(),FP64_logits=self.baseline_logits.cpu().numpy())
    def validate_inside_forward(self,module,args):
        if self.active_dtype is None:return
        need(all(p.dtype==self.active_dtype and not p.requires_grad for p in module.parameters() if p.is_floating_point()),'Actual functional parameter dtype/gradient differs')
        need(all(b.dtype==self.active_dtype for b in module.buffers() if b.is_floating_point()),'Actual functional buffer precision differs');self.calls['functional_FP64_forwards' if self.active_dtype==torch.float64 else 'functional_FP32_forwards']+=1
    def verify(self,rehash=False):
        self.endpoint.verify(rehash=rehash);need(all(self.objects[n] is p for n,p in self.rt.model.named_parameters()),'Physical parameter identity changed');need(all(self.buffers[n] is p for n,p in self.rt.model.named_buffers()),'Physical buffer identity changed');need(all(p.dtype==torch.float32 for p in self.rt.model.parameters()),'Physical FP32 model dtype changed');need(all(id(p)==self.double_ids[n] and p._version==self.double_versions[n] and not p.requires_grad for n,p in self.double.items()),'Frozen diagnostic FP64 parameter changed')
    def sync_native_chart(self,point):
        expected=torch.as_tensor(point,dtype=torch.float32,device=self.rt.device);actual=torch.cat([p.detach().reshape(-1) for p in self.rt.params]);need(torch.equal(actual,expected),'Native/functional physical point differs');self.endpoint.chart_values=[p.detach().clone() for p in self.rt.params];self.verify()
    def forward64(self,point,anchors_only=False):
        need(point.dtype==torch.float32 and tuple(point.shape)==(DIM,) and bool(torch.isfinite(point).all()),'Sealed FP32 chart required before exact FP64 cast');mapping=dict(self.double);mapping.update(replacements(self.rt.model,self.rt.params,point.double()));out=[];self.active_dtype=torch.float64
        try:
            with torch.no_grad():
                for batch in self.endpoint.batches[:1] if anchors_only else self.endpoint.batches:
                    logits=functional_call(self.rt.model,mapping,(),dict(batch,use_cache=False),strict=False).logits;need(logits.dtype==torch.float64 and not logits.requires_grad,'Actual FP64 logits required');last=batch['attention_mask'].sum(1)-1;out.append(logits[torch.arange(len(last),device=point.device),last])
        finally:self.active_dtype=None;self.verify()
        return out[0] if anchors_only else out
    def functional32(self,point,case):
        self.sync_native_chart(point);self.active_dtype=torch.float32
        try:
            with torch.no_grad():o,kl=self.endpoint(torch.as_tensor(point,dtype=torch.float32,device=self.rt.device))
        finally:self.active_dtype=None
        self.calls['functional_FP32_metrics']+=1;return full_metric(o.double().cpu().numpy(),float(kl.cpu()),point,case)
    def precision64(self,point,case):
        self.sync_native_chart(point)
        with torch.no_grad():
            al,rl=self.forward64(torch.as_tensor(point,dtype=torch.float32,device=self.rt.device));absolute=al[:,self.rt.responseids].mean(0);margin=rl[:,self.rt.readids[1]]-rl[:,self.rt.readids[0]];local=torch.cat([absolute-self.baseline_response,margin]);cross=torch.cat([absolute-self.native_reference,margin]);logp=al.log_softmax(-1)
            kl_local=(self.baseline_logits.softmax(-1)*(self.baseline_logits.log_softmax(-1)-logp)).sum(-1).mean().clamp_min(0.);kl_cross=(self.cross_logits.softmax(-1)*(self.cross_logits.log_softmax(-1)-logp)).sum(-1).mean().clamp_min(0.)
        self.calls['FP64_focus_metrics']+=1;return dict(point_sha256=safe.tensor_hash(point),observed_local_reference=local.cpu().tolist(),observed_FP32_response_reference=cross.cpu().tolist(),KL_local_reference=float(kl_local.cpu()),KL_FP32_logits_reference_recomputed64=float(kl_cross.cpu()),response_reference_offset=self.offset.tolist(),response_baseline_decomposition_residual_Linf=float(torch.max(abs(cross[:6]-local[:6]-(self.baseline_response-self.native_reference))).cpu()),precision='float64 temporary functional model on exact float32 parameter values',acceptance_evaluated=False,training_qualified=False)
def factory(rt,calls):return Paired(rt,calls)
