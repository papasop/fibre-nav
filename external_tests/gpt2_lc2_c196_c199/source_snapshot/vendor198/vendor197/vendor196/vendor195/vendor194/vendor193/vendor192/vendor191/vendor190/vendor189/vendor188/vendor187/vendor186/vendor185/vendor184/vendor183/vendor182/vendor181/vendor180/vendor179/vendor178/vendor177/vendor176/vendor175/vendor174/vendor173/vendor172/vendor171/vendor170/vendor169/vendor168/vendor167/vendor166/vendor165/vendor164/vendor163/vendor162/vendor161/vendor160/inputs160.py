"""Bind five original archives, retain C159 derivative records as archived measurements."""
from core160 import *
from inputs159 import load as load159
from audit147 import source_check
from map159 import chain_arithmetic
from baseline158 import record

def audit159(path,context,out,c):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 with archive155.prior.no_models(),safe.no_solvers(),contextlib.closing(Archive(path)) as z:
  count=z.verify_manifest();source_check(z,c['source159'],'C159');s=z.json('SUMMARY.json');compare(s['protocol'],previous.PROTOCOL);compare(s['complete'],True);compare(s['decision'],'C159_COEFFICIENT_CHAIN_FP64_DIAGNOSTIC_COMPLETE_NO_CANDIDATE_SELECTED');compare(s['synthetic_fixture'],not c['production']);compare(s['real_model_execution_claim'],c['production']);compare(s['errors'],[])
  for k,v in dict(new_model_forwards=27,new_backward_calls=18,scalar_backward_attempts=18,focus_training_states_evaluated=2,new_training_steps=0,new_head_fits=0,excluded_input_states_evaluated=0).items():compare(s[k],v,'C159 denominator/'+k)
  for k in ('LC2_closed','LC2_single_fold_passed','candidate_selected','writer_deployed','LC2_acceptance_thresholds_changed','C157_decision_reclassified','C158_decision_reclassified','AD_accuracy_certified'):compare(s[k],False)
  compare(s['original_inputs_unchanged'],{k:True for k in previous.INPUT_KEYS});links=z.json('ORIGINAL_INPUTS.json')
  for k in previous.INPUT_KEYS:compare(links[k]['sha256'],c['pins'][k]['sha256'],'C159 original link/'+k)
  compare(z.json('DESIGN_FROZEN_BEFORE_MODEL.json'),previous.design());cc=previous.constants(context['data']);saved=z.arrays('FROZEN_CHAIN_CONSTANTS.npz');compare(set(saved),set(cc))
  for n,a in cc.items():exact(saved[n],a,'C159 full coefficient constants/'+n)
  pointarrays=z.arrays('ALL14_FP64_CHAIN_POINTS.npz');bindings=z.json('POINT_BINDINGS_AND_PATH_DIFFERENCES.json');compare(len(bindings),14);points={}
  for b in bindings:
   sid,n=b['state_id'],b['label'];need(sid in previous.FOCUS and n in LABELS,'Fixed two-state C159 points');p=pointarrays[sid+'__'+n];compare(point_hash(p),b['FP64_point_sha256']);want,pp,ee=previous.numpy_map(cc,sid,LABELS[n]);close(p,want,'C159 continuous coefficient formula',MAP_TOL);points[(sid,n)]=p
  compare(set(points),{(sid,n) for sid in previous.FOCUS for n in LABELS});compare(set(pointarrays),{sid+'__'+n for sid,n in points})
  for n,h in z.json('SEALED_BEFORE_MODEL.json')['artifact_seals'].items():compare(hashlib.sha256(z.raw(n)).hexdigest(),h,'C159 actual artifact seal')
  refs=z.json('model_runtime/REFERENCES_FROZEN_BEFORE_FOCUS.json');compare(refs['protocol'],previous.PROTOCOL);compare(refs['model_forwards_so_far'],3);compare(refs['state_probes_so_far'],0);R={n:farray(np.asarray(v,float),(6,)) for n,v in refs['references'].items()};compare(set(R),set(previous.REFERENCES));compare(refs['reference_tensor_sha256'],{n:hashlib.sha256(a.tobytes()).hexdigest() for n,a in R.items()});exact(R['archived_FP32'],np.asarray(context['baseline']['FP32_cached_response_reference'],float),'Unchanged C159 FP32 reference');exact(R['archived_FP64'],np.asarray(context['baseline']['FP64_response_reference'],float),'Unchanged C159 FP64 reference')
  base={};first=None
  for key,label,role,expected in [('native_logits','NATIVE_FP32_BASELINE_LOGITS','cross_run_descriptive_only',context['baseline_logits']['FP32_logits_cast64']),('native_cached_response','NATIVE_FP32_CACHED_RESPONSE','cross_run_descriptive_only',np.asarray(context['baseline']['FP32_cached_response_reference'],float)),('FP64_cross_run','FUNCTIONAL_FP64_BASELINE_LOGITS','cross_run_descriptive_only',context['baseline_logits']['FP64_logits']),('FP64_repeat','WITHIN_RUN_FP64_ZERO_BASELINE_REPEAT','within_run_repeat',None)]:
   ar=z.arrays('model_runtime/'+label+'.npz');compare(set(ar),{'actual','archived'})
   if key=='FP64_repeat':expected=first
   exact(ar['archived'],expected,'C159 fixed archived baseline/'+key)
   with contextlib.redirect_stdout(__import__('io').StringIO()):rr=record(out,label,ar['actual'],ar['archived'],role)
   compare(z.json('model_runtime/'+label+'.json'),rr,'C159 baseline arithmetic');compare(s['cross_run_baseline_diagnostics'][key],rr);base[key]=rr
   if key=='FP64_cross_run':first=ar['actual']
  need(base['FP64_repeat']['passed_original_replay_tolerance'],'C159 within-run zero-chart repeat');close(R['current_FP64'],first[:,context['modelpins']['response_ids']].mean(0),'C159 fresh zero-chart reference',1e-12)
  expected_requests=[dict(stage='preflight',state_id=sid,label=n) for sid in previous.FOCUS for n in ('zero','plus_one')]
  for sid in previous.FOCUS:expected_requests.append(dict(stage='AD',state_id=sid,label='zero'));expected_requests.extend(dict(stage='finite',state_id=sid,label=n) for n in LABELS);expected_requests.extend(dict(stage='repeat',state_id=sid,label=n) for n in ('zero','plus_one'))
  rows=z.json('ALL_FORWARD_AND_AD_RECORDS.json');request=z.json('REQUESTS_FROZEN.json');compare(request,expected_requests,'Exact C159 prescribed request order');compare([{k:r[k] for k in ('stage','state_id','label')} for r in rows],request,'All24 C159 ordered records');compare(len(rows),24);by={(r['stage'],r['state_id'],r['label']):r for r in rows};diagnostics=[]
  for row in rows:compare(row['FP64_point_sha256'],point_hash(points[(row['state_id'],row['label'])]));farray(np.asarray(row['absolute_response'],float),(6,))
  for sid in previous.FOCUS:
   ad=by[('AD',sid,'zero')];panels={n:by[('finite',sid,n)] for n in LABELS};chain={}
   for name in previous.REFERENCES:
    v=ad['chain_pullbacks'][name];re=chain_arithmetic(ad['response_loss_derivatives'][name],v['effective_gradient'],v['direct_coefficient_gradient'],cc,sid);compare(re,v,'C159 independent stored pullback');chain[name]=re
   repeats=[]
   for n in ('zero','plus_one'):
    diff=float(max(abs(np.asarray(by[('repeat',sid,n)]['absolute_response'])-panels[n]['absolute_response'])));need(diff<=WITHIN_RUN_TOL,'C159 repeated point');repeats.append(dict(label=n,absolute_response_change_Linf=diff));close(by[('preflight',sid,n)]['absolute_response'],panels[n]['absolute_response'],'C159 endpoint preflight',WITHIN_RUN_TOL)
   close(ad['absolute_response'],panels['zero']['absolute_response'],'C159 AD/no-grad center',WITHIN_RUN_TOL);diagnostics.append(dict(state_id=sid,**previous.analyse(panels,ad,R),chain_pullbacks=chain,repeat_comparisons=repeats,archived_FP32_coefficient_response_gradient_dot_update=context['archived_gradient_arithmetic']['archived_FP32_response_gradient_dot_actual_delta'][sid],archived_gradient_not_same_precision_or_path=True))
  compare(s['diagnostics'],diagnostics,'Complete C159 derivative arithmetic');compare(z.json('TWO_STATE_SAME_RUN_AD_FINITE_DIAGNOSTICS.json'),diagnostics);dump(out/'C159_AUDIT_RECEIPT.json',dict(members_hashed=count,all24_records_and14_points_bound=True,all_six_scalar_control_gradient_pullbacks_recomputed=True,gradients_independently_recomputed=False,decision_retained=s['decision'],LC2_closed=False));return dict(decision=s['decision'],diagnostics=diagnostics)

def load(paths,out,c=None,budget=lambda:None):
 c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);compare(set(paths),set(INPUT_KEYS),'Exactly five original inputs')
 for k,p in paths.items():compare(sha(p),c['pins'][k]['sha256'],'Original whole ZIP before payload/'+k)
 for n,h in c['source159'].items():compare(sha(ROOT/'vendor159'/n),h,'Trusted C159 source/'+n)
 context=load159({k:paths[k] for k in previous.INPUT_KEYS},out/'C155_C156_C157_C158',c['parent'],budget);budget();context['completed159']=audit159(paths['C159'],context,out/'C159',c);budget();return context
