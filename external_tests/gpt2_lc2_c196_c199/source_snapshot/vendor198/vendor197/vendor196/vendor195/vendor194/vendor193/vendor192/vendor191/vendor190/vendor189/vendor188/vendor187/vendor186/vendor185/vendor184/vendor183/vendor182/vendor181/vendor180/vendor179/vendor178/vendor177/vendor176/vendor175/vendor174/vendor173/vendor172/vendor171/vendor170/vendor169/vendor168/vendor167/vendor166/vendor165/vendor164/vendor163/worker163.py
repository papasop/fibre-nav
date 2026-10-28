"""Fixed smooth-risk fit from C161 step8; archive actual FP64 points."""
import argparse,resource,traceback
from core163 import *
from inputs163 import load
def run(paths,out,c=None,runtime_factory=None,input_loader=None):
 c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);started=time.monotonic();stage='registered';errors=[];rt=None;records=[];at=0;history=[];calls=dict(model_forwards=0,functional_FP64_forwards=0,backward_attempts=0,backward_calls=0)
 s=dict(protocol=PROTOCOL,complete=False,decision='C163_INCOMPLETE_NO_FINAL_QUALIFICATION',synthetic_fixture=not c['production'],real_model_execution_claim=False,training_states_requested=N,new_training_steps=0,new_head_fits=0,head_updated=False,final_checkpoint_step=STEPS,final_head_frozen=False,new_model_parameter_gradients=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,new_states_prepared=0,confirmation_states_evaluated=0,excluded_input_states_evaluated=0,candidate_selected=False,best_checkpoint_selected=False,writer_deployed=False,thresholds_changed=False,parent_decisions_reclassified=False,preparation_replay_performed=False,FP64_acceptance_evaluated=False,LC2_closed=False,LC2_single_fold_passed=False,scope=design()['scope'])
 def budget():need(time.monotonic()-started<TIMEOUT,'C163 fixed180-minute worker budget')
 def checkpoint():
  s.update(stage=stage,new_model_forwards=calls['model_forwards'],functional_FP64_forwards=calls['functional_FP64_forwards'],new_backward_calls=calls['backward_calls'],scalar_backward_attempts=calls['backward_attempts'],records_completed=len(records),requests_consumed=at,elapsed_worker_seconds=time.monotonic()-started,process_peak_RSS_KiB=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss));dump(out/'SUMMARY.json',dict(s,errors=errors))
 def save(kind,step,row):
  record=dict(stage=kind,step=step,**row);records.append(record)
  with (out/'ALL1092_RECORDS.jsonl').open('a') as f:f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
  if len(records)%12==0:checkpoint()
 def authorize(kind,step,sid):
  nonlocal at
  budget();need(at<len(rq),'No extra measurement');compare(rq[at],dict(stage=kind,state_id=sid,step=step),'Exact fixed measurement order');at+=1
 def checkseal(folder,receipt,p):
  compare(parameter_hash(p),receipt['head_digest'],'Frozen current head bytes')
  for n,h in receipt['artifact_seals'].items():compare(sha(folder/n),h,'Current batch artifact/'+n)
  compare(clean_json((folder/'SEALED.json').read_bytes()),receipt,'Current batch seal receipt')
 try:
  need(not (out/'SUMMARY.json').exists(),'Fresh output only');dump(out/'DESIGN_FROZEN_BEFORE_MODEL.json',design());checkpoint();stage='eight_original_archive_binding';ctx=(load if input_loader is None else input_loader)(paths,out/'binding',c,budget);compare(ctx['C162_receipt']['decision'],'C162_READONLY_TRAINING_AUDIT_COMPLETE_PARENT_DECISION_RETAINED');s['C162_parent_decision']=ctx['C162_receipt']['decision'];chain=parent.previous.constants(ctx['data']);p=ctx['initial_head'].copy();m=np.zeros_like(p);v=np.zeros_like(p);np.savez_compressed(out/'INITIAL_C161_STEP8_AND_FRESH_MOMENTS.npz',P=p,m=m,v=v);s['initial_C161_step8_head_digest']=parameter_hash(p);s['optimizer_moments_restarted']=True;s['smooth_temperature']=TAU
  if c['production']:compare(p.shape,(3,2,36),'Production full216 coefficient head')
  np.savez_compressed(out/'FROZEN_FULL84_CHAIN_CONSTANTS.npz',**chain);np.savez_compressed(out/'FIXED_ARCHIVED_OBJECTIVE_REFERENCES.npz',response=np.asarray(ctx['baseline']['FP32_cached_response_reference'],float),baseline_logits=ctx['baseline_logits']['FP32_logits_cast64']);rq=requests();compare(len(rq),1092);dump(out/'ALL1092_REQUESTS_FROZEN.json',rq);points,initial,initial_receipt=seal_head(chain,p,out/'initial_head','initial0',True)
  for sp in registry():sid=sp['state_id'];compare(safe.tensor_hash(initial[(sid,'zero')]),ctx['C161_audit']['natives']['native_final0'][IDS.index(sid)]['metrics']['theta_sha256'],'Exact C161 step8 initial physical proposal')
  basefiles=['DESIGN_FROZEN_BEFORE_MODEL.json','FROZEN_FULL84_CHAIN_CONSTANTS.npz','FIXED_ARCHIVED_OBJECTIVE_REFERENCES.npz','ALL1092_REQUESTS_FROZEN.json','INITIAL_C161_STEP8_AND_FRESH_MOMENTS.npz'];seals={n:sha(out/n) for n in basefiles};dump(out/'SEALED_BEFORE_MODEL.json',dict(artifact_seals=seals,model_forwards_so_far=0));seals['SEALED_BEFORE_MODEL.json']=sha(out/'SEALED_BEFORE_MODEL.json');chain_ids={n:array_digest(a) for n,a in chain.items()};initial_ids={k:safe.tensor_hash(a) for k,a in initial.items()};anchor=p.copy()
  def unchanged():
   for n,h in seals.items():compare(sha(out/n),h,'Before-model constant/'+n)
   compare({n:array_digest(a) for n,a in chain.items()},chain_ids,'Actual fixed input tensors');need({k:safe.tensor_hash(a) for k,a in initial.items()}==initial_ids,'Actual initial native proposal tensors');checkseal(out/'initial_head',initial_receipt,anchor)
  if runtime_factory is None:
   from runtime163 import factory
   runtime_factory=factory
  stage='same_run_fixed_model_and_reference';unchanged();rt=runtime_factory(out/'model_runtime',ctx,chain,points,initial,dict(zero=anchor),calls);s['real_model_execution_claim']=c['production'];checkpoint();natives={};stage='native_initial0';rr=[]
  for sp in registry():
   sid=sp['state_id'];authorize(stage,0,sid);row=rt.native_candidate(sid,initial[(sid,'zero')],initial_ids[(sid,'zero')]);re=archive155.repeat_check(ctx['parent_final_native_records']['0'][IDS.index(sid)]['metrics'],row['metrics']);row['C161_final_native_replay']=re;rr.append(row);save(stage,0,row);need(re['within_replay_tolerance'] and re['same_point_identity'],'C161 final/current initial native replay before training')
  natives[stage]=rr;s['native_initial_round0']=native_summary(rr);unchanged();rt.verify(True);checkpoint()
  actual_seals=[]
  for t in range(STEPS):
   stage='training';folder=out/('step_%02d_before'%(t+1));batchpoints,_,receipt=seal_head(chain,p,folder,'step%d_before'%(t+1));checkseal(folder,receipt,p);actual_seals.append(rt.prepare_batch(p,batchpoints,folder,'training_before_local_step%d'%(t+1)));dump(out/'ALL9_ACTUAL_FP64_POINT_SEALS.json',actual_seals);gradrows=[]
   for sp in registry():
    sid=sp['state_id'];authorize(stage,t,sid);row=rt.coefficient_record(sid,p,batchpoints[(sid,'zero')],True);compare(row['parameters_digest'],receipt['head_digest']);
    if t==0:
     old=ctx['parent_final_FP64_records'][IDS.index(sid)];re=dict(observed_Linf=float(np.max(abs(np.asarray(row['observed'])-old['observed']))),KL_absolute=abs(row['KL']-old['KL']));re['within_declared_replay_tolerance']=re['observed_Linf']<=2e-9 and re['KL_absolute']<=2e-9;row['C161_final_FP64_replay']=re
    gradrows.append(row);save(stage,t,row)
    if t==0:need(re['within_declared_replay_tolerance'],'C161 final/current initial FP64 replay before first update')
   checkseal(folder,receipt,p);rt.verify_batch(p);g,agg=aggregate(gradrows,p.shape);new,mm,vv,diag=update(p,g,m,v,t+1)
   # Independent complete84 smooth-response coefficient aggregation.
   close(independent_gradient(gradrows,p.shape),g,'Independent complete84 smoothed gradient',1e-10)
   np.savez_compressed(out/('UPDATE_STEP_%02d.npz'%(t+1)),before=p,gradient=g,m_before=m,v_before=v,after=new,m_after=mm,v_after=vv)
   item=dict(step=t+1,head_before=parameter_hash(p),head_after=parameter_hash(new),aggregation=agg,optimizer=diag);history.append(item);dump(out/'ALL8_UPDATE_HISTORY.json',history);p,m,v=new,mm,vv;s.update(new_training_steps=t+1,new_head_fits=1,head_updated=bool(np.any(p!=anchor)));unchanged();rt.verify(True);checkpoint();print('C163_STEP',t+1,'/8','J_smooth_before',agg['objective'],'J_original_hardmax_before',agg['original_hardmax_objective'],'gradient_L2',diag['gradient_L2'],'update_Linf',diag['update_Linf'],flush=True)
  stage='freeze_exact_final_step8';finalpoints,final,final_receipt=seal_head(chain,p,out/'final_step8','fixed_final_step8',True);actual_seals.append(rt.prepare_batch(p,finalpoints,out/'final_step8','fixed_final_local_step8'));dump(out/'ALL9_ACTUAL_FP64_POINT_SEALS.json',actual_seals);final_ids={k:safe.tensor_hash(a) for k,a in final.items()};s['final_head_frozen']=True;s['final_head_digest']=parameter_hash(p);np.savez_compressed(out/'FIXED_FINAL_HEAD_STEP8.npz',P=p);dump(out/'FINAL_HEAD_FROZEN_BEFORE_EVALUATION.json',dict(step=8,head_digest=parameter_hash(p),selected_from_candidates=False,optimizer_updates_completed=s['new_training_steps'],record_count_before_final_test=len(records)));checkseal(out/'final_step8',final_receipt,p);stage='final_FP64';rr=[]
  for sp in registry():sid=sp['state_id'];authorize(stage,STEPS,sid);r=rt.coefficient_record(sid,p,finalpoints[(sid,'zero')],False);rr.append(r);save(stage,STEPS,r)
  s['initial_FP64_objective']=history[0]['aggregation'];s['final_FP64_objective']=objective(rr);s['objective_change_final_minus_initial']=s['final_FP64_objective']['objective']-s['initial_FP64_objective']['objective'];dump(out/'FINAL_FP64_OBJECTIVE_ALL84.json',rr);checkseal(out/'final_step8',final_receipt,p);rt.verify_batch(p);unchanged();rt.verify(True);checkpoint()
  for stage,t,pts,ids in [('native_final0',STEPS,final,final_ids),('native_initial1',0,initial,initial_ids),('native_final1',STEPS,final,final_ids)]:
   rr=[]
   for sp in registry():
    sid=sp['state_id'];authorize(stage,t,sid);r=rt.native_candidate(sid,pts[(sid,'zero')],ids[(sid,'zero')])
    if stage=='native_initial1':r['C161_final_native_replay']=archive155.repeat_check(ctx['parent_final_native_records']['1'][IDS.index(sid)]['metrics'],r['metrics'])
    rr.append(r);save(stage,t,r)
    if stage=='native_initial1':need(r['C161_final_native_replay']['within_replay_tolerance'] and r['C161_final_native_replay']['same_point_identity'],'Second initial native/C161 replay')
   natives[stage]=rr;unchanged();checkseal(out/'final_step8',final_receipt,p);rt.verify(True);checkpoint()
  repeat=[]
  for arm in ('initial','final'):
   for a,b in zip(natives['native_'+arm+'0'],natives['native_'+arm+'1']):
    r=archive155.repeat_check(a['metrics'],b['metrics']);need(r['within_replay_tolerance'] and r['same_point_identity'],'Native same point/gates repeat disagreement');repeat.append(dict(arm=arm,state_id=a['state_id'],**r))
  dump(out/'ALL168_NATIVE_REPEAT_CHECKS.json',repeat);s['native_final_round0']=native_summary(natives['native_final0']);s['native_initial_round1']=native_summary(natives['native_initial1']);s['native_final_round1']=native_summary(natives['native_final1']);s['paired_by_round']={str(t):paired(natives['native_initial'+str(t)],natives['native_final'+str(t)]) for t in (0,1)};s['native_repeats_consistent']=True;s['final_training_all84_passed_both_rounds']=all(native_summary(natives['native_final'+str(t)])['passed']==N for t in (0,1));dump(out/'ALL336_NATIVE_ENDPOINTS.json',natives)
  compare(at,1092);compare(len(records),1092);compare(calls['model_forwards'],REQUESTED_FORWARDS);compare(calls['functional_FP64_forwards'],FUNCTIONAL_FORWARDS);compare(calls['backward_calls'],REQUESTED_BACKWARDS);compare(calls['backward_attempts'],REQUESTED_BACKWARDS);compare(s['new_training_steps'],8);compare(rt.point_tensors_saved,756);rt.verify_batch(p);unchanged();checkseal(out/'final_step8',final_receipt,p);rt.verify(True);s.update(complete=True,decision='C163_FIXED_FINAL_TRAINING_ALL84_PASS_NO_CONFIRMATION' if s['final_training_all84_passed_both_rounds'] else 'C163_FIXED_FINAL_TRAINING_CONSTRAINT_FAILURE_NO_CONFIRMATION',all_requested_states_retained=True,actual_FP64_point_tensors_saved=756,actual_FP64_points_checked_before_model_forward=True,original_hardmax_objective_initial=s['initial_FP64_objective']['original_hardmax_objective'],original_hardmax_objective_final=s['final_FP64_objective']['original_hardmax_objective'],initial_FP64_replay_matched_C161_final=True,initial_native_replay_matched_C161_final=True);stage='complete'
  (out/'REPORT_ZH.md').write_text('# C163：从C161第8步头开始的平滑风险训练\n\n固定温度0.01、8步Adam，动量清零；原硬最大值目标并列报告。只验收固定最终头的两轮完整84态原生FP32，保留失败、救回与破坏态。实际756个Torch FP64前向点坐标在对应批次测量前封存，前向前核对实际图点与保存字节。\n\n聚合改变、额外训练与动量重启不是匹配因果对照，不能宣称优化器优越。没有收敛、可行性、泛化或LC2闭合证明。没有新确认或排除输入测试；所有原判定保留。\n')
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
  if not s['complete']:s['decision']='C163_INCOMPLETE_NO_FINAL_QUALIFICATION'
  checkpoint()
 return dict(s,errors=errors)
if __name__=='__main__':
 parser=argparse.ArgumentParser()
 for k in INPUT_KEYS:parser.add_argument('--'+k.lower(),required=True)
 parser.add_argument('--out',required=True);a=parser.parse_args();s=run({k:getattr(a,k.lower()) for k in INPUT_KEYS},a.out);print(json.dumps(s,ensure_ascii=False,indent=2),flush=True)
 if not s['complete']:raise SystemExit(1)
