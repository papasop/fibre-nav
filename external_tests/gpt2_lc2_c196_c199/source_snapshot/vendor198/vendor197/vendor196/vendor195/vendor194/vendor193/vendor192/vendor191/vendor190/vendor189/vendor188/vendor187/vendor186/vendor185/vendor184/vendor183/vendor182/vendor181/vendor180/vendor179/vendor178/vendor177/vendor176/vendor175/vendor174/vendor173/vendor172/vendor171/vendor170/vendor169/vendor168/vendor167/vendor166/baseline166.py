from core166 import *
from baseline158 import record
def baseline_audit(z,ctx,out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);first=None;diagnostics={}
 configs=[('native_logits','NATIVE_FP32_BASELINE_LOGITS',ctx['baseline_logits']['FP32_logits_cast64'],'cross_run_descriptive_only'),('native_cached_response','NATIVE_FP32_CACHED_RESPONSE',np.asarray(ctx['baseline']['FP32_cached_response_reference'],float),'cross_run_descriptive_only'),('FP64_cross_run','FUNCTIONAL_FP64_BASELINE_LOGITS',ctx['baseline_logits']['FP64_logits'],'cross_run_descriptive_only'),('FP64_repeat','WITHIN_RUN_FP64_ZERO_BASELINE_REPEAT',None,'within_run_repeat')]
 for key,name,expected,role in configs:
  a=z.arrays('model_runtime/'+name+'.npz');compare(set(a),{'actual','archived'});expected=first if key=='FP64_repeat' else expected;exact(a['archived'],expected,'C163 unchanged reference/'+key)
  with contextlib.redirect_stdout(__import__('io').StringIO()):r=record(out,name,a['actual'],a['archived'],role)
  compare(z.json('model_runtime/'+name+'.json'),r,'C163 baseline arithmetic');diagnostics[key]=r
  if key=='FP64_cross_run':first=a['actual']
 need(diagnostics['FP64_repeat']['passed_original_replay_tolerance'],'Within-run FP64 zero repeat')
 refs=z.json('model_runtime/REFERENCES_FROZEN_BEFORE_FOCUS.json');compare(refs['protocol'],parent.PROTOCOL);compare(refs['model_forwards_so_far'],3);compare(refs['state_probes_so_far'],0);R={n:farray(np.asarray(v,np.float64),(6,)) for n,v in refs['references'].items()};compare(set(R),{'current_FP64','archived_FP64','archived_FP32'});compare(refs['reference_tensor_sha256'],{n:hashlib.sha256(a.tobytes()).hexdigest() for n,a in R.items()});exact(R['archived_FP32'],np.asarray(ctx['baseline']['FP32_cached_response_reference'],float),'Original cached response');exact(R['archived_FP64'],np.asarray(ctx['baseline']['FP64_response_reference'],float),'Original double reference');close(R['current_FP64'],first[:,ctx['modelpins']['response_ids']].mean(0),'Same-run zero reference identity',1e-12)
 return diagnostics
