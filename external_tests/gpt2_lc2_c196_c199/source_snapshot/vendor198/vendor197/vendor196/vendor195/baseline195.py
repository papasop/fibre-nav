"""Independent baseline/reference arithmetic; archived objective references unchanged."""
from core195 import *
from baseline158 import record
def audit(z,ctx,out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);first=None;diag={}
 for key,name,expected,role in [('native_logits','NATIVE_FP32_BASELINE_LOGITS',ctx['baseline_logits']['FP32_logits_cast64'],'cross_run_descriptive_only'),('native_cached_response','NATIVE_FP32_CACHED_RESPONSE',np.asarray(ctx['baseline']['FP32_cached_response_reference'],float),'cross_run_descriptive_only'),('FP64_cross_run','FUNCTIONAL_FP64_BASELINE_LOGITS',ctx['baseline_logits']['FP64_logits'],'cross_run_descriptive_only'),('FP64_repeat','WITHIN_RUN_FP64_ZERO_BASELINE_REPEAT',None,'within_run_repeat')]:
  a=z.arrays('model_runtime/'+name+'.npz');compare(set(a),{'actual','archived'});expected=first if key=='FP64_repeat' else expected;exact(a['archived'],expected,'Original objective reference/'+key)
  with contextlib.redirect_stdout(__import__('io').StringIO()):r=record(out,name,a['actual'],a['archived'],role)
  compare(z.json('model_runtime/'+name+'.json'),r);diag[key]=r
  if key=='FP64_cross_run':first=a['actual']
 need(diag['FP64_repeat']['passed_original_replay_tolerance'],'Original within-run zero baseline repeat');r=z.json('model_runtime/REFERENCES_FROZEN_BEFORE_FOCUS.json');compare(r['protocol'],prior.PROTOCOL);compare(r['model_forwards_so_far'],3);compare(r['state_probes_so_far'],0);refs={n:farray(np.asarray(v,np.float64),(6,)) for n,v in r['references'].items()};compare(set(refs),{'current_FP64','archived_FP64','archived_FP32'});compare(r['reference_tensor_sha256'],{n:hashlib.sha256(a.tobytes()).hexdigest() for n,a in refs.items()});exact(refs['archived_FP32'],np.asarray(ctx['baseline']['FP32_cached_response_reference'],float));exact(refs['archived_FP64'],np.asarray(ctx['baseline']['FP64_response_reference'],float));close(refs['current_FP64'],first[:,ctx['modelpins']['response_ids']].mean(0),'Descriptive same-run zero reference',1e-12);return diag
