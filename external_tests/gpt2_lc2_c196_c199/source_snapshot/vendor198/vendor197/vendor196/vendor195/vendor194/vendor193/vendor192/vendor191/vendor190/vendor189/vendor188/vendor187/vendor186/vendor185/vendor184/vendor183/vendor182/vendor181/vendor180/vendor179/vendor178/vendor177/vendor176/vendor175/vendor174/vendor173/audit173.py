"""Independent C172 original tensor/gradient/direction/probe/gate arithmetic."""
from core173 import *
from audit147 import source_check
from audit164 import seal,actual_batch
from baseline173 import audit as baseline_audit
from recompute173 import reconstruct
def audit(path,ctx,out,c,budget=lambda:None):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 with contextlib.closing(Archive(path)) as z:
  members=z.verify_manifest();source_check(z,c['source172'],'C172');s=z.json('SUMMARY.json');compare(s['protocol'],prior.PROTOCOL);compare(s['complete'],True);compare(s['errors'],[]);compare(s['stage'],'complete');compare(s['decision'],'C172_FIXED_FINAL_HEAD_FINITE_STEPS_COMPLETE_NO_CANDIDATE_SELECTED');compare(s['synthetic_fixture'],not c['production']);compare(s['real_model_execution_claim'],c['production']);compare(s['scope'],prior.SCOPE)
  for k,v in dict(new_training_steps=0,new_head_fits=0,nonzero_training_updates=0,new_model_parameter_gradients=0,new_Jacobians=0,new_SVD_calls=0,temporary_probe_heads=6,new_states_prepared=0,confirmation_states_evaluated=0,excluded_input_states_evaluated=0,new_model_forwards=3699,functional_FP64_forwards=1346,new_backward_calls=168,scalar_backward_attempts=168,records_completed=1848,requests_consumed=1848,actual_FP64_point_tensors_saved=672).items():compare(s[k],v,'Original C172 count/'+k)
  for k in ('linear_systems_solved','solver_internal_linear_system_count'):compare(s[k],None,'Unknown original LP internals stay null')
  for k in ('candidate_selected','best_checkpoint_selected','writer_deployed','thresholds_changed','parent_decisions_reclassified','archived_source_executed','chronology_or_nonaccess_proved','preparation_replay_performed','FP64_acceptance_evaluated','guard_preservation_certified','solver_optimality_or_infeasibility_certified','curvature_or_convergence_certified','causal_explanation_established','LC2_closed','LC2_single_fold_passed'):compare(s[k],False,'Original scope/'+k)
  for k in ('all6_sealed_before_first_probe','gradient_batch_precedes_probe_sealing','all_requested_states_and_arms_retained','native_repeats_consistent','FP64_anchor_repeat_consistent','initial_native_and_FP64_replay_matched_C170'):compare(s[k],True)
  if c['production']:compare(s['all_seventeen_originals_independently_bound'],True)
  compare(s['C171_parent_decision'],ctx['C171_receipt']['decision']);compare(s['original_inputs_unchanged'],{k:True for k in prior.INPUT_KEYS});links=z.json('ORIGINAL_INPUTS.json')
  for k in prior.INPUT_KEYS:compare(links[k]['sha256'],c['pins'][k]['sha256'],'C172 original ancestry/'+k)
  compare(z.json('DESIGN_FROZEN_BEFORE_MODEL.json'),prior.design());compare(z.json('ALL1848_REQUESTS_FROZEN.json'),prior.requests());chain=control.source168.old.parent.parent.previous.constants(ctx['data']);saved=z.arrays('FROZEN_FULL84_CHAIN_CONSTANTS.npz');compare(set(saved),set(chain))
  for n,a in chain.items():exact(saved[n],a,'Original complete constant/'+n)
  p=ctx['C170_final_head'].copy();a=z.arrays('ANCHOR_C170_FIXED_FINAL_HEAD.npz');compare(set(a),{'P'});exact(a['P'],p,'Exact original C170 step8 head');compare(s['original_C170_fixed_final_head_digest'],parameter_hash(p))
  if c['production']:compare(p.shape,(3,2,36));compare(parameter_hash(p),prior.design()['initial_head_digest'])
  refs=z.arrays('FIXED_ARCHIVED_OBJECTIVE_REFERENCES.npz');compare(set(refs),{'response','baseline_logits'});exact(refs['response'],np.asarray(ctx['baseline']['FP32_cached_response_reference'],float));exact(refs['baseline_logits'],ctx['baseline_logits']['FP32_logits_cast64'])
  before=z.json('ANCHOR_SEALED_BEFORE_MODEL.json');compare(before['model_forwards_so_far'],0);compare(before['backward_calls_so_far'],0);compare(before['probe_heads_not_yet_computed'],True);compare(set(before['artifact_seals']),{'DESIGN_FROZEN_BEFORE_MODEL.json','ANCHOR_C170_FIXED_FINAL_HEAD.npz','FROZEN_FULL84_CHAIN_CONSTANTS.npz','FIXED_ARCHIVED_OBJECTIVE_REFERENCES.npz','ALL1848_REQUESTS_FROZEN.json'})
  for n,h in before['artifact_seals'].items():compare(hashlib.sha256(z.raw(n)).hexdigest(),h,'Before-model artifact/'+n)
  baseline=baseline_audit(z,ctx,out/'baseline');rows=[clean_json(line) for line in z.raw('ALL1848_RECORDS.jsonl').splitlines()];compare([{k:r[k] for k in ('stage','arm','state_id')} for r in rows],prior.requests(),'Complete ordered1848 records');groups={}
  for r in rows:compare(r['case'],r['state_id'].split('__')[0]);groups.setdefault((r['stage'],r['arm']),[]).append(r)
  pts,nt=seal(z,'anchor',chain,p,'C170_fixed_final_anchor',True);rpt,_=seal(z,'anchor_repeat',chain,p,'anchor_repeat',False);points={'anchor':pts,'anchor_repeat':rpt};native_points={'anchor':nt};actuals={};batches={};point_rows=[]
  for i,sid in enumerate(IDS):compare(safe.tensor_hash(nt[sid]),ctx['C170_final_native']['0'][i]['metrics']['theta_sha256'],'Original C170 final native identity')
  def continuous(rr,head,folder,tag,F,B,gradient):
   compare([r['state_id'] for r in rr],IDS);actual,re=actual_batch(z,folder,head,points[folder],tag,F,B,budget);actuals[folder]=actual;batches[folder]=re
   for r in rr:
    budget();sid=r['state_id'];compare(r['parameters_digest'],parameter_hash(head));compare(r['sealed_numpy_point_sha256'],point_hash(points[folder][sid]));compare(r['FP64_point_sha256'],point_hash(actual[sid]));compare(r['actual_FP64_point_artifact'],folder+'/ACTUAL_TORCH_FP64_POINTS.npz');err=float(np.max(abs(actual[sid]-points[folder][sid])));compare(r['map_Linf'],err,atol=1e-18,rtol=1e-12);compare(r['FP64_acceptance_evaluated'],False);o=farray(np.asarray(r['observed'],np.float64),(22,));kl=float(r['KL']);need(math.isfinite(kl) and kl>=0,'Finite archived KL');_,e=mapping(chain,head,sid);parts=losses(o,kl,actual[sid],chain[sid+'__prepared'],e,r['case']);compare(r['objective_parts'],parts,'Independent observed loss arithmetic',atol=1e-10,rtol=1e-9)
    if gradient:
     for key in ('response_gradient','guard_gradient'):farray(np.asarray(r[key],np.float64),head.shape)
    else:need('response_gradient' not in r and 'guard_gradient' not in r,'Only anchor has new AD records')
    point_rows.append(dict(folder=folder,state_id=sid,head_sha256=parameter_hash(head),actual_FP64_point_sha256=point_hash(actual[sid]),NumPy_point_sha256=point_hash(points[folder][sid]),recomputed_map_Linf=err))
  anchor=groups[('FP64_anchor','anchor')];continuous(anchor,p,'anchor','FP64_anchor_gradient',3,0,True)
  for i,r in enumerate(anchor):
   old=ctx['C170_final_FP64'][i];re=dict(observed_Linf=float(np.max(abs(np.asarray(r['observed'])-old['observed']))),KL_absolute=abs(r['KL']-old['KL']));re['within_original_replay_tolerance']=re['observed_Linf']<=2e-9 and re['KL_absolute']<=2e-9;compare(r['C170_final_FP64_replay'],re);need(re['within_original_replay_tolerance'],'Original C170 FP64 replay')
  compare(z.json('ALL84_FRESH_GRADIENT_RECORDS.json'),stripped(anchor),'Full84 gradient duplicate');pr=problem(anchor,p);saved=z.arrays('CURRENT_ANCHOR_LINEAR_PROBLEM.npz');compare(set(saved),set(pr))
  for n,a in pr.items():exact(saved[n],a,'Independent fresh full84 problem/'+n)
  a=z.arrays('ONE_FRESH_FIXED_DIRECTION.npz');compare(set(a),{'anchor','direction'});exact(a['anchor'],p,'Direction anchor');d=a['direction'];farray(d,p.shape);cert=certificate(pr,d);compare(z.json('EXACT_RECORDED_DIRECTION_CHECK.json'),cert,'Independent exact recorded-gradient witness');need(not len(pr['active']) or cert['strict_recorded_common_descent'],'Nonempty exact recorded descent');compare(s['exact_recorded_common_descent'],cert['strict_recorded_common_descent']);compare(s['active_response_states'],len(pr['active']));sol=z.json('SOLVER_DIAGNOSTIC.json');nc=int(len(pr['active'])>0);compare(s['new_linear_program_calls'],nc);compare(sol['calls'],nc);compare(sol['empty_active_noop'],not bool(nc));compare(s['empty_active_zero_direction'],not bool(nc));need(isinstance(sol['warnings'],list) and all(isinstance(v,str) for v in sol['warnings']),'Original solver warning list')
  if nc:
   raw=z.json('RAW_PRIMAL_SOLVER_RESULT.json');compare(sol['solver_result'],raw);x=farray(np.asarray(raw['x'],np.float64),(p.size+1,));bounded=np.minimum(1.,np.maximum(-1.,x[:-1]));bounded[~pr['free']]=0.;exact(d.ravel(),bounded,'Original raw primal maps to fixed direction');compare(sol['raw_claimed_margin'],float(x[-1]));compare(sol['direction_bound_adjustment_Linf'],float(np.max(abs(d.ravel()-x[:-1]))));rawmargin=float(x[-1]);violation=float(max(0.,np.max(pr['A']@x[:-1]+x[-1])))
  else:compare(sol,dict(calls=0,solver_result=None,warnings=[],empty_active_noop=True));need(not np.any(d),'Empty active zero direction');rawmargin=None;violation=None
  solver=dict(recorded_LP_calls=nc,raw_solver_claimed_margin=rawmargin,raw_solver_maximum_linear_violation=violation,exact_recorded_margin_lower=cert['normalized_margin_lower'],solver_optimality_or_infeasibility_certified=False,original_solver_internal_linear_system_count=None,new_LP_calls=0,new_linear_solves=0)
  heads,pan=panel(p,d);compare(z.json('FIXED6_STEP_PANEL_BEFORE_PROBES.json'),pan,'Independent fixed signed panel');compare(s['positive_steps'],list(H));probes={}
  for name,q in heads.items():
   points[name],native_points[name]=seal(z,name,chain,q,name,True);probes[name]=groups[('FP64_probe',name)];continuous(probes[name],q,name,name,339,168,False)
  repeat=groups[('FP64_anchor_repeat','anchor')];continuous(repeat,p,'anchor_repeat','FP64_anchor_repeat',3,0,False);ordered=[batches[n] for n in ['anchor','anchor_repeat']+ARM_NAMES];compare(z.json('ALL8_ACTUAL_FP64_POINT_SEALS.json'),ordered);compare(len(point_rows),672);receipt=z.json('ALL6_PROBES_SEALED_BEFORE_FIRST_PROBE.json');compare(receipt['model_forwards_so_far'],339);compare(receipt['backward_calls_so_far'],168);compare(receipt['anchor_gradient_records_completed'],84);compare(receipt['probe_records_completed'],0);compare(receipt['all6_head_digests'],{n:parameter_hash(q) for n,q in heads.items()});compare(receipt['actual_point_seals'],ordered);compare(receipt['all6_before_first_probe'],True);compare(set(receipt['artifact_seals']),{'ONE_FRESH_FIXED_DIRECTION.npz','CURRENT_ANCHOR_LINEAR_PROBLEM.npz','ALL84_FRESH_GRADIENT_RECORDS.json','EXACT_RECORDED_DIRECTION_CHECK.json','SOLVER_DIAGNOSTIC.json','FIXED6_STEP_PANEL_BEFORE_PROBES.json','ALL8_ACTUAL_FP64_POINT_SEALS.json'})
  for n,h in receipt['artifact_seals'].items():compare(hashlib.sha256(z.raw(n)).hexdigest(),h,'Sealed probe prerequisite/'+n)
  native={}
  for name in ['anchor']+ARM_NAMES:
   for rnd in (0,1):
    stage=('native_anchor' if name=='anchor' else 'native_probe')+str(rnd);rr=groups[(stage,name)];compare([r['state_id'] for r in rr],IDS)
    for i,r in enumerate(rr):
     budget();sid=r['state_id'];pt=native_points[name][sid];o=farray(np.asarray(r['metrics']['observed'],np.float64),(22,));kl=float(r['metrics']['KL']);need(math.isfinite(kl) and kl>=0,'Finite native KL');met=archive155.full_metric(o,kl,pt,r['case']);qual=archive155.endpoint_gates(met,r['case'],pt,chain[sid+'__prepared'].astype(np.float32),True);compare(r['metrics'],met,'Native metric arithmetic');compare(r['qualification'],qual,'All original native gates')
     if c['production']:compare(r['actual_physical_FP32_commit'],True)
     if name=='anchor':
      re=archive155.repeat_check(ctx['C170_final_native'][str(rnd)][i]['metrics'],met);compare(r['C170_final_native_replay'],re);need(re['within_replay_tolerance'] and re['same_point_identity'],'Original C170 native replay')
    native[(name,rnd)]=rr
  diag,fd,changes,fr,nr=reconstruct(anchor,repeat,probes,d,native,pan,p,heads)
  for n,v in [('DIAGNOSTIC_SUMMARY.json',diag),('ALL252_CENTRAL_DIFFERENCES_VS_FRESH_AD.json',fd),('ALL504_SIGNED_LOSS_CHANGES.json',changes),('ALL84_FP64_ANCHOR_REPEATS.json',fr),('ALL588_NATIVE_REPEAT_CHECKS.json',nr),('ALL1176_NATIVE_ENDPOINTS.json',{n+'__round'+str(t):stripped(rr) for (n,t),rr in native.items()}),('ALL6_FP64_PROBE_RECORDS.json',{n:stripped(rr) for n,rr in probes.items()})]:compare(z.json(n),v,'Original diagnostic duplicate reconstructed/'+n)
  compare(s['diagnostics'],diag,'C172 complete scientific diagnostic summary')
  for n,v in [('ALL672_ACTUAL_FP64_POINT_AUDIT_ROWS.json',point_rows),('ALL8_ACTUAL_POINT_SEALS_RECHECKED.json',ordered),('EXACT_RECORDED_DIRECTION_RECOMPUTED.json',cert),('ORIGINAL_SOLVER_VECTOR_BINDING.json',solver),('PARENT_DIAGNOSTIC_SUMMARY_RECOMPUTED.json',diag),('ALL252_CENTRAL_DIFFERENCES_RECOMPUTED.json',fd),('ALL504_SIGNED_LOSS_CHANGES_RECOMPUTED.json',changes),('ALL84_FP64_REPEATS_RECOMPUTED.json',fr),('ALL588_NATIVE_REPEATS_RECOMPUTED.json',nr)]:dump(out/n,v)
  return dict(summary=s,members_hashed=members,chain=chain,head=p,heads=heads,direction=d,panel=pan,anchor=anchor,repeat=repeat,probes=probes,native=native,points=points,actual_points=actuals,native_points=native_points,problem=pr,certificate=cert,solver=solver,diagnostics=diag,fd=fd,changes=changes,baseline=baseline,maximum_map_error=max(r['recomputed_map_Linf'] for r in point_rows))
