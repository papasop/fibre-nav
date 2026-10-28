"""Mandatory real Torch CPU functional FP64 alpha-AD tests on a tiny tied-weight model, not GPT-2 evidence."""
import contextlib,copy,io,tempfile,unittest
from unittest.mock import patch
import torch
from core157 import *
from runtime157 import Directional
from torch_smoke155 import ToyRuntime

class Checks(unittest.TestCase):
    def setup_runtime(self,corrupt_baseline=False,corrupt_native=False):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);out=Path(temp.name);self.last_out=out/'actual'
        counter=dict(model_forwards=0);reference=ToyRuntime(out/'reference',counter);double=copy.deepcopy(reference.model).double();batch=reference.tok(reference.anchors)
        with torch.no_grad():logits=double(**batch,use_cache=False).logits[:,-1].numpy()
        pins=dict(revision='synthetic-torch-only',seed=155,lora_A_sha256='synthetic',chart_shapes=[[DIM]],symbols=[0,1],anchors=reference.anchors,response_ids=list(range(6)),sketch_sha256='synthetic')
        context=dict(data=dict(d={}),modelpins=pins,baseline_logits=dict(FP32_logits_cast64=reference.logits0.double().numpy(),FP64_logits=logits),baseline=dict(FP64_response_reference=logits[:,:6].mean(0).tolist(),FP32_cached_response_reference=reference.response0.double().numpy().tolist()))
        points={};directions={}
        for j,sid in enumerate(FOCUS):
            z=np.zeros(DIM,np.float32);z[:6]=np.linspace(.03,.08,6);p=z.copy();p[:6]+=np.float32(1e-5*(j+1));directions[sid]=p.astype(float)-z.astype(float)
            for n,a in LABELS.items():points[(sid,n)]=chord(z,p,a)
        if corrupt_baseline:context['baseline_logits']['FP64_logits']=logits.copy()+.01
        if corrupt_native:
            value=context['baseline_logits']['FP32_logits_cast64'].copy();value[0,0]=float(np.float32(value[0,0]+.01));context['baseline_logits']['FP32_logits_cast64']=value
        calls=dict(model_forwards=0,functional_FP64_forwards=0,backward_attempts=0,backward_calls=0);self.last_calls=calls
        def factory(path,d,phase,calls):rt=ToyRuntime(path,calls);rt.pins=pins;return rt
        self.native_factory=factory
        with contextlib.redirect_stdout(io.StringIO()):return Directional(out/'actual',context,points,directions,calls,factory),context
    def test_real_FP64_coordinate_AD_matches_independent_analytic_derivative(self):
        runtime,_=self.setup_runtime();sid=FOCUS[0];ad=runtime.ad(sid);rt=runtime.rt
        emb=rt.model.embed.weight.detach().double();ids=rt.tok(rt.anchors)['input_ids'][:,-1];x=emb[ids]+torch.from_numpy(runtime.points[(sid,'zero')][:6]);d=torch.from_numpy(runtime.directions[sid][:6])
        derivative=(((1-torch.tanh(x)**2)*d)@emb.T*rt.model.scale.detach().double()).mean(0)[:6].numpy()
        np.testing.assert_allclose(ad['coordinate_derivatives'],derivative,rtol=1e-12,atol=1e-15);self.assertEqual(runtime.calls['backward_calls'],8);runtime.verify(True)
    def test_real_same_function_FD_all_scales_both_references(self):
        runtime,c=self.setup_runtime();sid=FOCUS[0];ad=runtime.ad(sid);rows={n:runtime.measure(sid,n) for n in LABELS};result=analyse(rows,ad,runtime.local,runtime.shared)
        for row in result['scales']:
            self.assertLess(row['central_minus_AD_Linf'],1e-10)
            for ref in ['local','shared']:self.assertLess(abs(row['response_loss'][ref]['central_minus_AD']),1e-7)
        self.assertTrue(all(not p.requires_grad and p.grad is None and p.dtype==torch.float32 for p in runtime.rt.model.parameters()));self.assertIs(runtime.rt.model.embed.weight,runtime.rt.model.proj.weight)
    def test_detached_point_breaks_graph_and_is_rejected(self):
        runtime,_=self.setup_runtime();original=runtime.replacements
        runtime.replacements=lambda model,params,p:original(model,params,p.detach())
        with self.assertRaises(ValueError):runtime.ad(FOCUS[0])
        runtime.verify(True);self.assertEqual(runtime.calls['backward_calls'],0)
    def test_float32_point_not_silently_used_for_AD(self):
        runtime,_=self.setup_runtime()
        with self.assertRaises(ValueError):runtime.forward(torch.zeros(DIM,dtype=torch.float32),graph=True)
    def test_exception_restores_actual_tied_FP32_parameters(self):
        runtime,_=self.setup_runtime()
        with patch.object(runtime.rt.model,'forward',side_effect=RuntimeError('intentional smoke exception')):
            with self.assertRaises(RuntimeError):runtime.ad(FOCUS[0])
        runtime.verify(True);self.assertIs(runtime.rt.model.embed.weight,runtime.rt.model.proj.weight)
    def test_frozen_double_mutation_detected(self):
        runtime,_=self.setup_runtime();name=next(n for n,p in runtime.double.items() if p.dtype==torch.float64 and n not in runtime.chart_names)
        runtime.double[name].add_(1.)
        with self.assertRaises(ValueError):runtime.verify()
    def test_wrong_baseline_blocks_before_directional_backwards(self):
        with self.assertRaisesRegex(ValueError,'baseline logits replay differs'):self.setup_runtime(corrupt_baseline=True)
        self.assertEqual(self.last_calls['backward_calls'],0);self.assertEqual(self.last_calls['model_forwards'],2)
        report=clean_json((self.last_out/'FUNCTIONAL_FP64_BASELINE_LOGITS.json').read_bytes());self.assertFalse(report['passed_original_replay_tolerance']);self.assertGreater(report['maximum_absolute_error'],.009)
    def test_native_baseline_drift_saved_with_one_forward_zero_backwards(self):
        with self.assertRaisesRegex(ValueError,'Native FP32 baseline logits replay differs'):self.setup_runtime(corrupt_native=True)
        self.assertEqual(self.last_calls['model_forwards'],1);self.assertEqual(self.last_calls['backward_calls'],0);report=clean_json((self.last_out/'NATIVE_FP32_BASELINE_LOGITS.json').read_bytes());self.assertFalse(report['passed_original_replay_tolerance']);self.assertTrue((self.last_out/'NATIVE_FP32_BASELINE_LOGITS.npz').is_file());self.assertTrue((self.last_out/'NATIVE_FP32_CACHED_RESPONSE.json').is_file());self.assertTrue((self.last_out/'BASELINE_RUNTIME_PROVENANCE.json').is_file());self.assertFalse((self.last_out/'FUNCTIONAL_FP64_BASELINE_LOGITS.json').exists())
    def test_actual_model_dtype_hook_runs_on_graph_forward(self):
        runtime,_=self.setup_runtime();before=runtime.calls['functional_FP64_forwards'];runtime.ad(FOCUS[1]);self.assertEqual(runtime.calls['functional_FP64_forwards'],before+1);runtime.verify(True)
    def test_complete_worker_with_real_Torch_and_exact26_16_counts(self):
        import contextlib,io
        from fixture157 import setup_inputs
        from worker157 import run
        reference,c=self.setup_runtime();c['data'].update(points={n+'__'+sid:p.astype(np.float32) for (sid,n),p in reference.points.items()},parent=dict(decision='SYNTHETIC_TORCH_ONLY'))
        c['replay_absolute']={(sid,n):np.array(reference.measure(sid,n)['absolute_response']) for sid in FOCUS for n in ('zero','plus_one')};c['archived_gradient_arithmetic']=dict(archived_FP32_response_gradient_dot_actual_delta={sid:0. for sid in FOCUS})
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);paths,config=setup_inputs(Path(tmp.name)/'inputs')
        def load(*args):return c
        def factory(out,context,points,directions,calls):
            self.actual_worker_runtime=Directional(out,context,points,directions,calls,self.native_factory);return self.actual_worker_runtime
        with contextlib.redirect_stdout(io.StringIO()):s=run(paths,Path(tmp.name)/'out',config,factory,load)
        self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['new_model_forwards'],26);self.assertEqual(s['new_backward_calls'],16);self.assertFalse(s['real_model_execution_claim']);self.assertFalse(s['LC2_closed']);self.actual_worker_runtime.verify(True)
if __name__=='__main__':unittest.main(verbosity=2)
