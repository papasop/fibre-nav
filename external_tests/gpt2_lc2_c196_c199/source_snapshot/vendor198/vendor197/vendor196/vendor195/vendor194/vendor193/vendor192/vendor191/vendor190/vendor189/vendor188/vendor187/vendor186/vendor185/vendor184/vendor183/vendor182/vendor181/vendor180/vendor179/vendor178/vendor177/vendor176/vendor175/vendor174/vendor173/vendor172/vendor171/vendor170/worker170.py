"""Fixed eight-step offline training, exact direction checks, fixed final head."""
import argparse,resource,traceback
from core170 import *
from inputs170 import load
from solver170 import solve
def run(paths,out,c=None,runtime_factory=None,input_loader=None,solver_backend=None):
 c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);started=time.monotonic();errors=[];stage='registered';rt=None;at=0;records=[];history=[];calls=dict(model_forwards=0,functional_FP64_forwards=0,backward_attempts=0,backward_calls=0)
 s=dict(protocol=PROTOCOL,complete=False,decision='C170_INCOMPLETE_NO_FINAL_QUALIFICATION',synthetic_fixture=not c['production'],real_model_execution_claim=False,training_states_requested=84,new_training_steps=0,new_head_fits=0,nonzero_head_updates=0,new_linear_program_calls=0,solver_internal_linear_system_count=None,linear_systems_solved=None,new_model_parameter_gradients=0,new_Jacobians=0,new_SVD_calls=0,new_states_prepared=0,confirmation_states_evaluated=0,excluded_input_states_evaluated=0,candidate_selected=False,best_checkpoint_selected=False,writer_deployed=False,thresholds_changed=False,parent_decisions_reclassified=False,archived_source_executed=False,preparation_replay_performed=False,FP64_acceptance_evaluated=False,final_checkpoint_step=8,final_head_frozen=False,all_active_batch_recorded_directions_strictly_verified=False,guard_preservation_certified=False,solver_optimality_or_infeasibility_certified=False,LC2_closed=False,LC2_single_fold_passed=False,scope=SCOPE)
 def budget():need(time.monotonic()-started<TIMEOUT,'C170 fixed180-minute worker budget')
 def checkpoint():s.update(stage=stage,new_model_forwards=calls['model_forwards'],functional_FP64_forwards=calls['functional_FP64_forwards'],new_backward_calls=calls['backward_calls'],scalar_backward_attempts=calls['backward_attempts'],records_completed=len(records),requests_consumed=at,elapsed_worker_seconds=time.monotonic()-started,process_peak_RSS_KiB=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss));dump(out/'SUMMARY.json',dict(s,errors=errors))
 def authorize(kind,t,sid):
  nonlocal at
  budget();need(at<len(rq),'No unregistered model measurement');compare(rq[at],dict(stage=kind,step=t,state_id=sid),'All1092 fixed requests');at+=1
 def save(kind,t,r):
  row=dict(stage=kind,step=t,**r);records.append(row)
  with (out/'ALL1092_RECORDS.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
  if len(records)%12==0:checkpoint()
 def rehash(folder,re,p):
  compare(parameter_hash(p),re['head_digest']);compare(clean_json((folder/'SEALED.json').read_bytes()),re)
  for n,h in re['artifact_seals'].items():compare(sha(folder/n),h,'Sealed batch/'+n)
 try:
  need(not(out/'SUMMARY.json').exists(),'Fresh output only');dump(out/'DESIGN_FROZEN_BEFORE_MODEL.json',design());checkpoint();stage='fifteen_original_binding';ctx=(load if input_loader is None else input_loader)(paths,out/'binding',c,budget);compare(ctx['C169_receipt']['decision'],prior.DECISION);s.update(C169_parent_decision=ctx['C169_receipt']['decision'],all_fifteen_originals_independently_bound=input_loader is None);chain=source168.old.parent.parent.previous.constants(ctx['data']);p=ctx['initial_head'].copy();anchor=p.copy()
  if c['production']:compare(p.shape,(3,2,36))
  s['initial_C163_fixed_final_head_digest']=parameter_hash(p);np.savez_compressed(out/'INITIAL_C163_FIXED_FINAL_HEAD.npz',P=p);np.savez_compressed(out/'FROZEN_FULL84_CHAIN_CONSTANTS.npz',**chain);np.savez_compressed(out/'FIXED_ARCHIVED_OBJECTIVE_REFERENCES.npz',response=np.asarray(ctx['baseline']['FP32_cached_response_reference'],float),baseline_logits=ctx['baseline_logits']['FP32_logits_cast64']);rq=requests();dump(out/'ALL1092_REQUESTS_FROZEN.json',rq);dump(out/'ORIGINAL_INPUTS.json',{k:dict(path=str(v),sha256=sha(v)) for k,v in paths.items()});pts,initial,ire=seal_head(chain,p,out/'initial_head','initial0',True)
  for i,sid in enumerate(IDS):compare(safe.tensor_hash(initial[(sid,'zero')]),ctx['C168_native_anchor_records']['0'][i]['metrics']['theta_sha256'],'Original C168 anchor physical identity')
  fixed={n:sha(out/n) for n in ('DESIGN_FROZEN_BEFORE_MODEL.json','INITIAL_C163_FIXED_FINAL_HEAD.npz','FROZEN_FULL84_CHAIN_CONSTANTS.npz','FIXED_ARCHIVED_OBJECTIVE_REFERENCES.npz','ALL1092_REQUESTS_FROZEN.json')};dump(out/'SEALED_BEFORE_MODEL.json',dict(artifact_seals=fixed,model_forwards_so_far=0));fixed['SEALED_BEFORE_MODEL.json']=sha(out/'SEALED_BEFORE_MODEL.json');chain_ids={n:array_digest(a) for n,a in chain.items()};initial_ids={k:safe.tensor_hash(v) for k,v in initial.items()}
  def unchanged():
   for n,h in fixed.items():compare(sha(out/n),h,'Fixed before-model artifact/'+n)
   compare({n:array_digest(a) for n,a in chain.items()},chain_ids);need({k:safe.tensor_hash(v) for k,v in initial.items()}==initial_ids,'Actual initial native proposals unchanged');rehash(out/'initial_head',ire,anchor)
  if runtime_factory is None:
   from runtime170 import factory
   runtime_factory=factory
  stage='same_run_model_and_reference';rt=runtime_factory(out/'model_runtime',ctx,chain,pts,initial,dict(zero=anchor.copy()),calls);s['real_model_execution_claim']=c['production'];natives={};stage='native_initial0';rr=[]
  for i,sid in enumerate(IDS):
   authorize(stage,0,sid);r=rt.native_candidate(sid,initial[(sid,'zero')],initial_ids[(sid,'zero')]);re=archive155.repeat_check(ctx['C168_native_anchor_records']['0'][i]['metrics'],r['metrics']);r['C168_anchor_native_replay']=re;rr.append(r);save(stage,0,r);need(re['within_replay_tolerance'] and re['same_point_identity'],'Initial native replay before training')
  natives[stage]=rr;s['native_initial_round0']=native_summary(rr);actual=[];unchanged();rt.verify(True)
  for t in range(STEPS):
   stage='training';folder=out/('step_%02d_before'%(t+1));bp,_,br=seal_head(chain,p,folder,'step%d_before'%(t+1));actual.append(rt.preseal(p,bp,folder,'training_before_local_step%d'%(t+1)));rt.activate(folder.name,p);dump(out/'ALL9_ACTUAL_FP64_POINT_SEALS.json',actual);rr=[]
   for i,sid in enumerate(IDS):
    authorize(stage,t,sid);r=rt.coefficient_record(sid,p,bp[(sid,'zero')],True)
    if t==0:
     old=ctx['C168_FP64_anchor_records'][i];re=dict(observed_Linf=float(max(abs(np.asarray(r['observed'])-old['observed']))),KL_absolute=abs(r['KL']-old['KL']));re['within_original_replay_tolerance']=re['observed_Linf']<=2e-9 and re['KL_absolute']<=2e-9;r['C168_anchor_FP64_replay']=re
    rr.append(r);save(stage,t,r)
    if t==0:need(re['within_original_replay_tolerance'],'C168 FP64 anchor replay before first LP')
   rt.verify_batch(p);rehash(folder,br,p);pr=problem(rr,p);np.savez_compressed(folder/'CURRENT_RECORDED_LINEAR_PROBLEM.npz',**pr);dump(folder/'ALL84_CURRENT_GRADIENT_ROWS.json',rr);stage='offline_LP_and_exact_direction_step%d'%(t+1);checkpoint()
   def attempted():s['new_linear_program_calls']+=1;checkpoint()
   result=solve(pr,solver_backend,attempted,lambda r:dump(folder/'RAW_PRIMAL_SOLVER_RESULT.json',r));d=result.pop('direction');np.savez_compressed(folder/'FIXED_STEP_DIRECTION.npz',direction=d);dump(folder/'SOLVER_DIAGNOSTIC.json',result);cert=verify_direction(pr,d);dump(folder/'EXACT_RECORDED_DIRECTION_CHECK.json',cert);budget();need(not len(pr['active']) or cert['strict_recorded_common_descent'],'No positive exact recorded direction; stop without fallback/step search');new,upd=update(p,d.reshape(p.shape));agg=objective(rr);np.savez_compressed(out/('UPDATE_STEP_%02d.npz'%(t+1)),before=p,direction=d.reshape(p.shape),after=new);item=dict(step=t+1,head_before=parameter_hash(p),head_after=parameter_hash(new),current_objective=agg,active_state_ids=cert['active_state_ids'],exact_normalized_margin_lower=cert['normalized_margin_lower'],empty_active_noop=result['empty_active_noop'],solver_calls=result['calls'],update=upd);history.append(item);dump(out/'ALL8_UPDATE_HISTORY.json',history);s['nonzero_head_updates']+=int(np.any(new!=p));p=new;s.update(new_training_steps=t+1,new_head_fits=1);unchanged();rt.verify(True);checkpoint();print('C170_STEP',t+1,'/8','active',len(pr['active']),'exact_positive',cert['strict_recorded_common_descent'],'step',STEP,'update_Linf',upd['update_Linf'],flush=True)
  stage='freeze_fixed_final_step8';fp,final,fre=seal_head(chain,p,out/'final_step8','fixed_final_step8',True);actual.append(rt.preseal(p,fp,out/'final_step8','fixed_final_local_step8'));dump(out/'ALL9_ACTUAL_FP64_POINT_SEALS.json',actual);rt.activate('final_step8',p);final_ids={k:safe.tensor_hash(v) for k,v in final.items()};np.savez_compressed(out/'FIXED_FINAL_HEAD_STEP8.npz',P=p);dump(out/'FINAL_HEAD_FROZEN_BEFORE_EVALUATION.json',dict(step=8,head_digest=parameter_hash(p),selected_from_candidates=False,optimizer_updates_completed=s['new_training_steps'],record_count_before_final_test=len(records)));s.update(final_head_frozen=True,final_head_digest=parameter_hash(p));stage='final_FP64';rr=[]
  for sid in IDS:authorize(stage,8,sid);r=rt.coefficient_record(sid,p,fp[(sid,'zero')],False);rr.append(r);save(stage,8,r)
  s['initial_FP64_objective']=history[0]['current_objective'];s['final_FP64_objective']=objective(rr);dump(out/'FINAL_FP64_OBJECTIVE_ALL84.json',rr);rt.verify_batch(p);unchanged();rehash(out/'final_step8',fre,p)
  for stage,t,points,ids in [('native_final0',8,final,final_ids),('native_initial1',0,initial,initial_ids),('native_final1',8,final,final_ids)]:
   rr=[]
   for i,sid in enumerate(IDS):
    authorize(stage,t,sid);r=rt.native_candidate(sid,points[(sid,'zero')],ids[(sid,'zero')])
    if stage=='native_initial1':r['C168_anchor_native_replay']=archive155.repeat_check(ctx['C168_native_anchor_records']['1'][i]['metrics'],r['metrics'])
    rr.append(r);save(stage,t,r)
    if stage=='native_initial1':need(r['C168_anchor_native_replay']['within_replay_tolerance'] and r['C168_anchor_native_replay']['same_point_identity'],'Repeated original native anchor replay')
   natives[stage]=rr;unchanged();rehash(out/'final_step8',fre,p);rt.verify(True);checkpoint()
  repeats=[]
  for arm in ('initial','final'):
   for a,b in zip(natives['native_'+arm+'0'],natives['native_'+arm+'1']):
    r=archive155.repeat_check(a['metrics'],b['metrics']);need(r['within_replay_tolerance'] and r['same_point_identity'],'Original native repeat/gate check');repeats.append(dict(arm=arm,state_id=a['state_id'],**r))
  dump(out/'ALL168_NATIVE_REPEAT_CHECKS.json',repeats);dump(out/'ALL336_NATIVE_ENDPOINTS.json',natives)
  for arm in ('initial','final'):
   for rnd in (0,1):s['native_'+arm+'_round'+str(rnd)]=native_summary(natives['native_'+arm+str(rnd)])
  s['paired_by_round']={str(t):paired(natives['native_initial'+str(t)],natives['native_final'+str(t)]) for t in (0,1)};s['final_training_all84_passed_both_rounds']=all(s['native_final_round'+str(t)]['passed']==84 for t in (0,1));compare(at,1092);compare(len(records),1092);compare(calls['model_forwards'],2187);compare(calls['functional_FP64_forwards'],1514);compare(calls['backward_calls'],1344);compare(calls['backward_attempts'],1344);compare(rt.point_tensors_saved,756);compare(s['new_training_steps'],8);need(s['new_linear_program_calls']<=8,'Only one LP per scheduled batch');rt.verify_batch(p);unchanged();rehash(out/'final_step8',fre,p);rt.verify(True);s.update(complete=True,decision='C170_FIXED_FINAL_TRAINING_ALL84_PASS_NO_CONFIRMATION' if s['final_training_all84_passed_both_rounds'] else 'C170_FIXED_FINAL_TRAINING_CONSTRAINT_FAILURE_NO_CONFIRMATION',native_repeats_consistent=True,actual_FP64_point_tensors_saved=756,actual_FP64_points_checked_before_model_forward=True,initial_native_and_FP64_replay_matched_C168=True,all_active_batch_recorded_directions_strictly_verified=True,all_requested_states_retained=True);stage='complete'
  (out/'REPORT_ZH.md').write_text('# C170：固定8步重新线性化离线训练\n\n'+json.dumps(dict(history=history,final_training_all84_passed_both_rounds=s['final_training_all84_passed_both_rounds'],paired=s['paired_by_round']),ensure_ascii=False,indent=2)+'\n\n仅固定第8步头接受两轮原生84态验收，不选择中间头。所有训练、失败、救回和破坏态保留。LP仅针对当前响应损失梯度，guard损失与梯度记录但不作为LP约束。精确方向核验只关于记录的binary64输入，不认证真实导数、最优性、有限写入、泛化或LC2闭合。\n')
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
  if not s['complete']:s['decision']='C170_INCOMPLETE_NO_FINAL_QUALIFICATION'
  checkpoint()
 return dict(s,errors=errors)
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in INPUT_KEYS:p.add_argument('--'+k.lower(),required=True)
 p.add_argument('--out',required=True);a=p.parse_args();s=run({k:getattr(a,k.lower()) for k in INPUT_KEYS},a.out);print(json.dumps(s,ensure_ascii=False,indent=2),flush=True)
 if not s['complete']:raise SystemExit(1)
