"""Pinned C155-C158 archive and arithmetic binding only; no input-ZIP source execution."""
from core159 import *
from inputs158 import load as load158
from audit147 import source_check
from baseline158 import record

def audit158(path,context,out,c):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 with previous.previous.parent.parent.prior.no_models(),safe.no_solvers(),contextlib.closing(Archive(path)) as z:
  count=z.verify_manifest();source_check(z,c['source158'],'C158');s=z.json('SUMMARY.json');compare(s['protocol'],previous.PROTOCOL);compare(s['complete'],True);compare(s['decision'],'C158_SAME_RUN_TWO_STATE_FP64_DIAGNOSTIC_COMPLETE_NO_CANDIDATE_SELECTED');compare(s['synthetic_fixture'],not c['production']);compare(s['real_model_execution_claim'],c['production']);compare(s['errors'],[])
  for k,v in dict(new_model_forwards=27,new_backward_calls=18,scalar_backward_attempts=18,focus_training_states_evaluated=2,new_training_steps=0,new_head_fits=0,excluded_input_states_evaluated=0).items():compare(s[k],v,'C158 denominator/'+k)
  for k in ('LC2_closed','LC2_single_fold_passed','candidate_selected','writer_deployed','LC2_acceptance_thresholds_changed','C157_decision_reclassified','AD_accuracy_certified'):compare(s[k],False)
  compare(s['C157_original_decision'],'C157_INCOMPLETE_NO_DIRECTIONAL_CONCLUSION');compare(s['original_inputs_unchanged'],{k:True for k in previous.INPUT_KEYS});links=z.json('ORIGINAL_INPUTS.json')
  for k in previous.INPUT_KEYS:compare(links[k]['sha256'],c['pins'][k]['sha256'],'C158 original links/'+k)
  dump(out/'DESIGN_FROZEN_BEFORE_MODEL.json',previous.design());points,dirs,requests,seals=previous.freeze(context['data'],out);compare(z.json('DESIGN_FROZEN_BEFORE_MODEL.json'),previous.design());compare(z.json('REQUESTS_FROZEN.json'),requests);compare(z.json('POINT_BINDINGS_AND_PATH_DIFFERENCES.json'),clean_json((out/'POINT_BINDINGS_AND_PATH_DIFFERENCES.json').read_bytes()));a=z.arrays('ALL14_FP64_CHORD_POINTS.npz');compare(set(a),{sid+'__'+n for sid,n in points})
  for (sid,n),p in points.items():exact(a[sid+'__'+n],p,'C158 full physical point')
  ds=z.arrays('TWO_FIXED_DIRECTIONS.npz');compare(set(ds),set(dirs))
  for sid,p in dirs.items():exact(ds[sid],p,'C158 frozen direction')
  for n,h in z.json('SEALED_BEFORE_MODEL.json')['artifact_seals'].items():compare(hashlib.sha256(z.raw(n)).hexdigest(),h,'C158 frozen artifact')
  base={};first=None
  for key,label,role,expected in [('native_logits','NATIVE_FP32_BASELINE_LOGITS','cross_run_descriptive_only',context['baseline_logits']['FP32_logits_cast64']),('native_cached_response','NATIVE_FP32_CACHED_RESPONSE','cross_run_descriptive_only',np.asarray(context['baseline']['FP32_cached_response_reference'],float)),('FP64_cross_run','FUNCTIONAL_FP64_BASELINE_LOGITS','cross_run_descriptive_only',context['baseline_logits']['FP64_logits']),('FP64_repeat','WITHIN_RUN_FP64_ZERO_BASELINE_REPEAT','within_run_repeat',None)]:
   ar=z.arrays('model_runtime/'+label+'.npz');compare(set(ar),{'actual','archived'})
   if key=='FP64_repeat':expected=first
   exact(ar['archived'],expected,'C158 stored reference/'+key)
   with contextlib.redirect_stdout(__import__('io').StringIO()):rr=record(out,label,ar['actual'],ar['archived'],role)
   compare(z.json('model_runtime/'+label+'.json'),rr,'C158 saved baseline arithmetic');compare(s['cross_run_baseline_diagnostics'][key],rr);base[key]=rr
   if key=='FP64_cross_run':first=ar['actual']
  need(base['FP64_repeat']['passed_original_replay_tolerance'],'C158 same-run baseline repeat');refs=z.json('model_runtime/REFERENCES_FROZEN_BEFORE_FOCUS.json');compare(refs['protocol'],previous.PROTOCOL);compare(refs['model_forwards_so_far'],3);compare(refs['state_probes_so_far'],0);references={n:farray(np.asarray(a,float),(6,)) for n,a in refs['references'].items()};compare(set(references),set(REFERENCES));compare(refs['reference_tensor_sha256'],{n:hashlib.sha256(a.tobytes()).hexdigest() for n,a in references.items()})
  exact(references['archived_FP64'],np.asarray(context['baseline']['FP64_response_reference'],float),'Unchanged archived FP64 reference');exact(references['archived_FP32'],np.asarray(context['baseline']['FP32_cached_response_reference'],float),'Unchanged archived FP32 reference');close(references['current_FP64'],first[:,context['modelpins']['response_ids']].mean(0),'Fresh FP64 zero-chart mean',1e-12)
  rows=z.json('ALL_FORWARD_AND_AD_RECORDS.json');compare([{k:r[k] for k in ('stage','state_id','label')} for r in rows],requests,'All24 C158 actual records');by={(r['stage'],r['state_id'],r['label']):r for r in rows};diagnostics=[];endpoints={}
  for row in rows:
   sid,n=row['state_id'],row['label'];compare(row['FP64_point_sha256'],point_hash(points[(sid,n)]));farray(np.asarray(row['absolute_response'],float),(6,))
   if 'response_losses' in row:
    for name in REFERENCES:compare(row['response_losses'][name],response_loss(np.asarray(row['absolute_response'])-references[name]),'C158 response-loss arithmetic',atol=1e-12,rtol=1e-9)
  for sid in FOCUS:
   ad=by[('AD',sid,'zero')];panels={n:by[('finite',sid,n)] for n in LABELS};repeats=[]
   close(ad['absolute_response'],panels['zero']['absolute_response'],'C158 same-run AD center',WITHIN_RUN_TOL)
   for n in ('zero','plus_one'):
    ref=by[('preflight',sid,n)];close(ref['absolute_response'],panels[n]['absolute_response'],'C158 endpoint preflight',WITHIN_RUN_TOL);rep=by[('repeat',sid,n)];diff=float(max(abs(np.asarray(rep['absolute_response'])-panels[n]['absolute_response'])));need(diff<=WITHIN_RUN_TOL,'C158 final repeat');repeats.append(dict(label=n,absolute_response_change_Linf=diff));endpoints[(sid,n)]=np.asarray(ref['absolute_response'],float)
   diagnostics.append(dict(state_id=sid,**previous.analyse(panels,ad,references),repeat_comparisons=repeats,archived_FP32_coefficient_response_gradient_dot_update=context['archived_gradient_arithmetic']['archived_FP32_response_gradient_dot_actual_delta'][sid],archived_gradient_not_same_precision_or_path=True))
  compare(diagnostics,s['diagnostics'],'C158 recomputed full derivative arithmetic');compare(z.json('TWO_STATE_SAME_RUN_AD_FINITE_DIAGNOSTICS.json'),diagnostics);compare(z.json('model_runtime/BASELINE_RUNTIME_PROVENANCE.json')['archived_C155_MODEL_PINS'],context['modelpins']);dump(out/'C158_READONLY_AUDIT_RECEIPT.json',dict(members_hashed=count,all24_records_verified=True,all14_points_verified=True,all_three_references_verified=True,all_derivative_difference_arithmetic_recomputed=True,model_measurements_and_gradients_independently_recomputed=False,decision_retained=s['decision'],LC2_closed=False));return dict(decision=s['decision'],endpoints=endpoints,diagnostics=diagnostics)

def load(paths,out,c=None,budget=lambda:None):
 c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);compare(set(paths),set(INPUT_KEYS),'Exactly four original archives')
 for k,p in paths.items():compare(sha(p),c['pins'][k]['sha256'],'Fixed whole original before payload/'+k)
 for n,h in c['source158'].items():compare(sha(ROOT/'vendor158'/n),h,'Trusted delivered C158 source/'+n)
 context=load158({k:paths[k] for k in previous.INPUT_KEYS},out/'C155_C156_C157',c['parent'],budget);budget();context['completed158']=audit158(paths['C158'],context,out/'C158',c);budget();return context
