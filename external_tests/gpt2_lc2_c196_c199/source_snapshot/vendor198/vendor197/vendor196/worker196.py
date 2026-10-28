"""Fixed8-step all-coordinate offline training; fixed final step8 qualification."""
import argparse,resource,traceback
from core196 import *
from inputs196 import load
from solver174 import solve
from endpoint196 import precision_margins

def run(paths,out,c=None,runtime_factory=None,input_loader=None,solver_backend=None):
 c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);started=time.monotonic();errors=[];stage='registered';rt=None;at=0;records=[];history=[];actual=[];completed_seals=[];calls=dict(model_forwards=0,functional_FP64_forwards=0,backward_attempts=0,backward_calls=0,response_coordinate_backward_calls=0,response_Jacobian_blocks_completed=0)
 s=dict(protocol=PROTOCOL,complete=False,decision='C196_INCOMPLETE_NO_FINAL_QUALIFICATION',synthetic_fixture=not c['production'],real_model_execution_claim=False,training_states_requested=84,new_training_steps=0,new_head_fits=0,nonzero_head_updates=0,new_linear_program_calls=0,solver_internal_linear_system_count=None,linear_systems_solved=None,new_model_parameter_gradients=0,new_Jacobians=0,new_model_parameter_Jacobians=0,new_response_coordinate_Jacobian_rows=0,new_SVD_calls=0,new_states_prepared=0,confirmation_states_evaluated=0,excluded_input_states_evaluated=0,candidate_selected=False,best_checkpoint_selected=False,writer_deployed=False,thresholds_changed=False,parent_decisions_reclassified=False,archived_source_executed=False,preparation_replay_performed=False,FP64_acceptance_evaluated=False,final_checkpoint_step=8,prior_C176_C178_C180_C182_C184_C186_C188_C190_C192_C194_training_steps=80,cumulative_C176_C178_C180_C182_C184_C186_C188_C190_C192_C194_C196_training_steps=80,final_head_frozen=False,all_active_batch_recorded_affine_checks_verified=False,guard_preservation_certified=False,solver_optimality_or_infeasibility_certified=False,curvature_or_convergence_certified=False,causal_explanation_established=False,chronology_or_nonaccess_proved=False,LC2_closed=False,LC2_single_fold_passed=False,scope=SCOPE)
 def budget():need(time.monotonic()-started<TIMEOUT,'C196 fixed240-minute worker budget')
 def checkpoint():s.update(new_Jacobians=calls['response_Jacobian_blocks_completed'],new_response_coordinate_Jacobian_rows=calls['response_coordinate_backward_calls'],stage=stage,new_model_forwards=calls['model_forwards'],functional_FP64_forwards=calls['functional_FP64_forwards'],new_backward_calls=calls['backward_calls'],scalar_backward_attempts=calls['backward_attempts'],records_completed=len(records),requests_consumed=at,elapsed_worker_seconds=time.monotonic()-started,process_peak_RSS_KiB=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss));dump(out/'SUMMARY.json',dict(s,errors=errors))
 def authorize(kind,t,sid):
  nonlocal at
  budget();need(at<len(rq),'No unregistered measurement');compare(rq[at],dict(stage=kind,step=t,state_id=sid),'All1176 fixed requests');at+=1
 def save(kind,t,r):
  records.append(dict(stage=kind,step=t,**r))
  with (out/'ALL1176_RECORDS.jsonl').open('a') as f:f.write(json.dumps(records[-1],ensure_ascii=False,allow_nan=False)+'\n')
  if len(records)%12==0:checkpoint()
 def rehash(folder,re,p):
  compare(parameter_hash(p),re['head_digest'],'Sealed head identity');compare(clean_json((folder/'SEALED.json').read_bytes()),re)
  for n,h in re['artifact_seals'].items():compare(sha(folder/n),h,'Sealed point/head/'+n)
 try:
  need(not(out/'SUMMARY.json').exists(),'Fresh output only');dump(out/'DESIGN_FROZEN_BEFORE_MODEL.json',design());checkpoint();compare(set(paths),set(INPUT_KEYS),'Exactly41 fixed originals')
  for k,path in paths.items():compare(sha(path),c['pins'][k]['sha256'],'Fixed original before loading/'+k)
  for n,h in c['source195'].items():compare(sha(ROOT/'vendor195'/n),h,'Trusted C195 source/'+n)
  stage='forty_one_original_binding';ctx=(load if input_loader is None else input_loader)(paths,out/'binding',c,budget);compare(ctx['C195_receipt']['decision'],prior.DECISION);s.update(C195_parent_decision=ctx['C195_receipt']['decision'],all_forty_one_originals_independently_bound=input_loader is None);chain=control.source168.old.parent.parent.previous.constants(ctx['data']);p=ctx['C194_final_head'].copy();anchor=p.copy()
  if c['production']:compare(p.shape,(3,2,36));compare(parameter_hash(p),design()['anchor_head_digest'])
  s['initial_C194_fixed_final_head_digest']=parameter_hash(p);np.savez_compressed(out/'INITIAL_C194_FIXED_FINAL_HEAD.npz',P=p);np.savez_compressed(out/'FROZEN_FULL84_CHAIN_CONSTANTS.npz',**chain);np.savez_compressed(out/'FIXED_ARCHIVED_OBJECTIVE_REFERENCES.npz',response=np.asarray(ctx['baseline']['FP32_cached_response_reference'],float),baseline_logits=ctx['baseline_logits']['FP32_logits_cast64']);rq=requests();dump(out/'ALL1176_REQUESTS_FROZEN.json',rq);dump(out/'ORIGINAL_INPUTS.json',{k:dict(path=str(v),sha256=sha(v)) for k,v in paths.items()});pts,initial,ire=seal_head(chain,p,out/'initial_head','original_C194_final',True);initial_ids={k:safe.tensor_hash(v) for k,v in initial.items()}
  for i,sid in enumerate(IDS):compare(initial_ids[(sid,'zero')],ctx['C194_final_native']['0'][i]['metrics']['theta_sha256'],'Original C194 anchor physical identity')
  fixed={n:sha(out/n) for n in ('DESIGN_FROZEN_BEFORE_MODEL.json','INITIAL_C194_FIXED_FINAL_HEAD.npz','FROZEN_FULL84_CHAIN_CONSTANTS.npz','FIXED_ARCHIVED_OBJECTIVE_REFERENCES.npz','ALL1176_REQUESTS_FROZEN.json')};dump(out/'SEALED_BEFORE_MODEL.json',dict(artifact_seals=fixed,model_forwards_so_far=0,backward_calls_so_far=0));fixed['SEALED_BEFORE_MODEL.json']=sha(out/'SEALED_BEFORE_MODEL.json');chain_ids={n:array_digest(a) for n,a in chain.items()}
  def unchanged():
   budget()
   for n,h in fixed.items():compare(sha(out/n),h,'Fixed initial artifact/'+n)
   compare({n:array_digest(a) for n,a in chain.items()},chain_ids,'Frozen full84 constants');need({k:safe.tensor_hash(v) for k,v in initial.items()}==initial_ids,'Initial native arrays unchanged');rehash(out/'initial_head',ire,anchor)
   for folder,seal in completed_seals:
    for n,h in seal.items():compare(sha(folder/n),h,'Committed immutable update/'+n)
  if runtime_factory is None:
   from runtime196 import factory
   runtime_factory=factory
  stage='same_run_pinned_model';rt=runtime_factory(out/'model_runtime',ctx,chain,pts,initial,dict(zero=anchor.copy()),calls);s['real_model_execution_claim']=c['production'];natives={};stage='native_initial0';rr=[]
  for i,sid in enumerate(IDS):
   authorize(stage,0,sid);r=rt.native_candidate(sid,initial[(sid,'zero')],initial_ids[(sid,'zero')]);re=archive155.repeat_check(ctx['C194_final_native']['0'][i]['metrics'],r['metrics']);r['C194_final_native_replay']=re;rr.append(r);save(stage,0,r);need(re['within_replay_tolerance'] and re['same_point_identity'],'Original C194 native replay before any gradient')
  natives[stage]=rr;initial_FP64=None;s['native_initial_round0']=native_summary(rr);unchanged();rt.verify(True)
  for t in range(STEPS):
   stage='training';folder=out/('step_%02d_before'%(t+1));bp,_,br=seal_head(chain,p,folder,'training_before_local_step%d'%(t+1));actual.append(rt.preseal(p,bp,folder,'training_before_local_step%d'%(t+1)));rt.activate(folder.name,p);dump(out/'ALL10_ACTUAL_FP64_POINT_SEALS.json',actual);rr=[]
   for i,sid in enumerate(IDS):
    authorize(stage,t,sid);r=rt.coefficient_record(sid,p,bp[(sid,'zero')],True)
    if t==0:
     old=ctx['C194_final_FP64'][i];re=dict(observed_Linf=float(np.max(abs(np.asarray(r['observed'])-old['observed']))),KL_absolute=abs(r['KL']-old['KL']));re['within_original_replay_tolerance']=re['observed_Linf']<=2e-9 and re['KL_absolute']<=2e-9;r['C194_final_FP64_replay']=re
    rr.append(r);save(stage,t,r)
    if t==0:need(re['within_original_replay_tolerance'],'Original C194 FP64 replay before first LP/update')
   rt.verify_batch(p);rehash(folder,br,p)
   if t==0:initial_FP64=rr
   pr=problem(rr,p);np.savez_compressed(folder/'CURRENT_RECORDED_LINEAR_PROBLEM.npz',**pr);dump(folder/'ALL84_CURRENT_GRADIENT_ROWS.json',rr);stage='offline_all_coordinate_LP_step%d'%(t+1);checkpoint()
   def attempted():s['new_linear_program_calls']+=1;checkpoint()
   result=solve(pr,solver_backend,attempted,lambda r:dump(folder/'RAW_PRIMAL_SOLVER_RESULT.json',r));d=result.pop('direction');np.savez_compressed(folder/'FIXED_STEP_DIRECTION.npz',direction=d);dump(folder/'SOLVER_DIAGNOSTIC.json',result);cert=exact_certificate(pr,d);dump(folder/'EXACT_RECORDED_DIRECTION_CHECK.json',cert);budget();need(not len(pr['active']) or cert['strict_recorded_active_margin'],'No positive exact recorded all-coordinate margin: stop without fallback or infeasibility claim');new,upd=update(p,d);np.savez_compressed(folder/'PROPOSED_NEXT_HEAD_BEFORE_COMMIT.npz',P=new);actual_cert=exact_certificate(pr,d,new);dump(folder/'EXACT_ACTUAL_HEAD_DIFFERENCE_BEFORE_COMMIT.json',actual_cert);need(not len(pr['active']) or actual_cert['strict_recorded_active_margin'],'Actual binary64 next head must have positive recorded affine cap margin');budget()
   np.savez_compressed(out/('UPDATE_STEP_%02d.npz'%(t+1)),before=p,direction=d,after=new);step_files=['HEAD.npz','POINT_BINDINGS.json','ACTUAL_TORCH_FP64_POINTS.npz','ACTUAL_TORCH_FP64_POINT_BINDINGS.json','ACTUAL_FP64_POINTS_SEALED_BEFORE_MEASUREMENTS.json','SEALED.json','CURRENT_RECORDED_LINEAR_PROBLEM.npz','ALL84_CURRENT_GRADIENT_ROWS.json','FIXED_STEP_DIRECTION.npz','SOLVER_DIAGNOSTIC.json','EXACT_RECORDED_DIRECTION_CHECK.json','PROPOSED_NEXT_HEAD_BEFORE_COMMIT.npz','EXACT_ACTUAL_HEAD_DIFFERENCE_BEFORE_COMMIT.json']
   if result['calls']:step_files+=['RAW_PRIMAL_SOLVER_RESULT.json']
   sealed={n:sha(folder/n) for n in step_files};seal=dict(step=t+1,head_before=parameter_hash(p),head_after=parameter_hash(new),artifact_seals=sealed,update_artifact='UPDATE_STEP_%02d.npz'%(t+1),update_artifact_sha256=sha(out/('UPDATE_STEP_%02d.npz'%(t+1))),model_forwards_so_far=calls['model_forwards'],backward_calls_so_far=calls['backward_calls'],next_head_not_yet_measured=True);dump(folder/'UPDATE_COMMITTED_AND_SEALED.json',seal);completed_seals.append((folder,dict(sealed,**{'UPDATE_COMMITTED_AND_SEALED.json':sha(folder/'UPDATE_COMMITTED_AND_SEALED.json')})));fixed['UPDATE_STEP_%02d.npz'%(t+1)]=seal['update_artifact_sha256'];agg=objective(rr);item=dict(step=t+1,head_before=parameter_hash(p),head_after=parameter_hash(new),current_objective=agg,active_state_ids=cert['active_state_ids'],exact_direction_minimum_cap_slack=cert['minimum_normalized_recorded_cap_slack'],exact_actual_head_minimum_cap_slack=actual_cert['minimum_normalized_recorded_cap_slack'],empty_active_noop=result['empty_active_noop'],solver_calls=result['calls'],all1008_signed_rows_retained=True,update=upd);history.append(item);dump(out/'ALL8_UPDATE_HISTORY.json',history);s['nonzero_head_updates']+=int(np.any(new!=p));p=new;s.update(new_training_steps=t+1,cumulative_C176_C178_C180_C182_C184_C186_C188_C190_C192_C194_C196_training_steps=80+t+1,new_head_fits=1);unchanged();rt.verify(True);checkpoint();print('C196_STEP',t+1,'/8','active',len(pr['active']),'exact_actual_margin',actual_cert['strict_recorded_active_margin'],'step',STEP,'update_Linf',upd['update_Linf'],flush=True)
  stage='freeze_fixed_final_step8';fp,final,fre=seal_head(chain,p,out/'final_step8','fixed_final_step8',True);rp,_,rre=seal_head(chain,p,out/'final_step8_repeat','fixed_final_step8_repeat',False);actual.append(rt.preseal(p,fp,out/'final_step8','fixed_final_step8'));actual.append(rt.preseal(p,rp,out/'final_step8_repeat','fixed_final_step8_repeat'));dump(out/'ALL10_ACTUAL_FP64_POINT_SEALS.json',actual);np.savez_compressed(out/'FIXED_FINAL_HEAD_STEP8.npz',P=p);dump(out/'FINAL_HEAD_FROZEN_BEFORE_EVALUATION.json',dict(step=8,head_digest=parameter_hash(p),selected_from_candidates=False,offline_updates_completed=s['new_training_steps'],record_count_before_final_test=len(records),artifact_seals={n:sha(out/n) for n in ('FIXED_FINAL_HEAD_STEP8.npz','ALL8_UPDATE_HISTORY.json','ALL10_ACTUAL_FP64_POINT_SEALS.json')}));fixed['FINAL_HEAD_FROZEN_BEFORE_EVALUATION.json']=sha(out/'FINAL_HEAD_FROZEN_BEFORE_EVALUATION.json');fixed['FIXED_FINAL_HEAD_STEP8.npz']=sha(out/'FIXED_FINAL_HEAD_STEP8.npz');fixed['ALL8_UPDATE_HISTORY.json']=sha(out/'ALL8_UPDATE_HISTORY.json');fixed['ALL10_ACTUAL_FP64_POINT_SEALS.json']=sha(out/'ALL10_ACTUAL_FP64_POINT_SEALS.json');s.update(final_head_frozen=True,final_head_digest=parameter_hash(p));final_ids={k:safe.tensor_hash(v) for k,v in final.items()};final_frames={}
  for stage,folder,pt in [('final_FP64','final_step8',fp),('final_FP64_repeat','final_step8_repeat',rp)]:
   rt.activate(folder,p);rr=[]
   for sid in IDS:authorize(stage,8,sid);r=rt.coefficient_record(sid,p,pt[(sid,'zero')],False);rr.append(r);save(stage,8,r)
   final_frames[stage]=rr;rt.verify_batch(p);unchanged();rehash(out/'final_step8',fre,p);rehash(out/'final_step8_repeat',rre,p)
  s['initial_FP64_objective']=history[0]['current_objective'];s['final_FP64_objective']=objective(final_frames['final_FP64']);fr=[]
  for a,b in zip(final_frames['final_FP64'],final_frames['final_FP64_repeat']):
   compare(a['state_id'],b['state_id']);compare(a['FP64_point_sha256'],b['FP64_point_sha256']);err=float(np.max(abs(np.asarray(a['observed'])-b['observed'])));ke=abs(a['KL']-b['KL']);need(err<=2e-9 and ke<=2e-9,'Fixed final same-run FP64 repeat');fr.append(dict(state_id=a['state_id'],observed_Linf=err,KL_absolute=ke,within_original_repeat_tolerance=True))
  dump(out/'ALL84_FINAL_FP64_REPEATS.json',fr);dump(out/'FINAL_FP64_RECORDS_ALL84.json',final_frames)
  for stage,t,points,ids in [('native_final0',8,final,final_ids),('native_initial1',0,initial,initial_ids),('native_final1',8,final,final_ids)]:
   rr=[]
   for i,sid in enumerate(IDS):
    authorize(stage,t,sid);r=rt.native_candidate(sid,points[(sid,'zero')],ids[(sid,'zero')])
    if stage=='native_initial1':r['C194_final_native_replay']=archive155.repeat_check(ctx['C194_final_native']['1'][i]['metrics'],r['metrics'])
    rr.append(r);save(stage,t,r)
    if stage=='native_initial1':need(r['C194_final_native_replay']['within_replay_tolerance'] and r['C194_final_native_replay']['same_point_identity'],'Original C194 repeated native replay')
   natives[stage]=rr;unchanged();rehash(out/'final_step8',fre,p);need({k:safe.tensor_hash(v) for k,v in final.items()}==final_ids,'Final native arrays unchanged');rt.verify(True);checkpoint()
  nr=[]
  for arm in ('initial','final'):
   for a,b in zip(natives['native_'+arm+'0'],natives['native_'+arm+'1']):
    r=archive155.repeat_check(a['metrics'],b['metrics']);need(r['within_replay_tolerance'] and r['same_point_identity'],'Native repeat identity and original gates');nr.append(dict(arm=arm,state_id=a['state_id'],**r))
  dump(out/'ALL168_NATIVE_REPEAT_CHECKS.json',nr);dump(out/'ALL336_NATIVE_ENDPOINTS.json',natives)
  for arm in ('initial','final'):
   for rnd in (0,1):s['native_'+arm+'_round'+str(rnd)]=native_summary(natives['native_'+arm+str(rnd)])
  s['paired_by_round']={str(t):paired(natives['native_initial'+str(t)],natives['native_final'+str(t)]) for t in (0,1)};s['endpoint_precision_margins']=precision_margins(natives,initial_FP64,final_frames['final_FP64'],out);s['final_training_all84_passed_both_rounds']=all(s['native_final_round'+str(t)]['passed']==84 for t in (0,1));compare(at,RECORDS);compare(len(records),RECORDS);compare(calls['model_forwards'],FORWARDS);compare(calls['functional_FP64_forwards'],FUNCTIONAL);compare(calls['backward_calls'],BACKWARDS);compare(calls['backward_attempts'],BACKWARDS);compare(calls['response_Jacobian_blocks_completed'],672);compare(calls['response_coordinate_backward_calls'],4032);compare(rt.point_tensors_saved,840);compare(s['new_training_steps'],8);compare(s['new_linear_program_calls'],sum(h['solver_calls'] for h in history));need(s['new_linear_program_calls']<=8,'At most one LP per current batch');rt.verify_batch(p);unchanged();rehash(out/'final_step8',fre,p);rehash(out/'final_step8_repeat',rre,p);rt.verify(True);s.update(complete=True,decision='C196_FIXED_FINAL_TRAINING_ALL84_PASS_NO_CONFIRMATION' if s['final_training_all84_passed_both_rounds'] else 'C196_FIXED_FINAL_TRAINING_CONSTRAINT_FAILURE_NO_CONFIRMATION',native_repeats_consistent=True,FP64_final_repeat_consistent=True,actual_FP64_point_tensors_saved=840,actual_FP64_points_checked_before_model_forward=True,initial_native_and_FP64_replay_matched_C194=True,all_active_batch_recorded_affine_checks_verified=True,all_requested_states_retained=True,LP_signed_rows_per_step=1008);stage='complete'
  (out/'REPORT_ZH.md').write_text('# C196：固定8步全响应坐标续训\n\n'+json.dumps(dict(history=history,final_training_all84_passed_both_rounds=s['final_training_all84_passed_both_rounds'],paired=s['paired_by_round'],endpoint_precision_margins=s['endpoint_precision_margins']),ensure_ascii=False,indent=2)+'\n\n固定第8步头接受两轮原生84态验收与FP64重复；全部失败保留。每步全1008条响应线性化约束，guard仅记录。精确证书只关于归档binary64算术，不认证真实导数、有限误差界、收敛、泛化或LC2闭合。该拟合过程显式使用离线Jacobian与LP，不构成在线免校正LC1执行。\n')
 except BaseException:errors.append(traceback.format_exc());print(errors[-1],flush=True)
 finally:
  checks={}
  for k in INPUT_KEYS:
   try:checks[k]=sha(paths[k])==c['pins'][k]['sha256']
   except Exception:checks[k]=False
  s['original_inputs_unchanged']=checks
  if not all(checks.values()):s['complete']=False;errors.append('Original input missing or changed')
  if rt is not None:
   try:rt.verify(True)
   except BaseException:s['complete']=False;errors.append(traceback.format_exc())
  if not s['complete']:s['decision']='C196_INCOMPLETE_NO_FINAL_QUALIFICATION'
  checkpoint()
 return dict(s,errors=errors)
if __name__=='__main__':
 ap=argparse.ArgumentParser()
 for k in INPUT_KEYS:ap.add_argument('--'+k.lower(),required=True)
 ap.add_argument('--out',required=True);a=ap.parse_args();s=run({k:getattr(a,k.lower()) for k in INPUT_KEYS},a.out);print(json.dumps(s,ensure_ascii=False,indent=2),flush=True)
 if not s['complete']:raise SystemExit(1)
