"""One fresh anchor gradient/LP; fixed signed probes, no training or selection."""
import argparse,resource,traceback
from core172 import *
from inputs172 import load
from solver170 import solve
from diagnostics172 import analyse
def run(paths,out,c=None,runtime_factory=None,input_loader=None,solver_backend=None):
 c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);started=time.monotonic();errors=[];stage='registered';calls=dict(model_forwards=0,functional_FP64_forwards=0,backward_attempts=0,backward_calls=0);rt=None;records=[];at=0
 s=dict(protocol=PROTOCOL,complete=False,decision='C172_INCOMPLETE_NO_FINITE_STEP_CONCLUSION',synthetic_fixture=not c['production'],real_model_execution_claim=False,new_training_steps=0,new_head_fits=0,nonzero_training_updates=0,new_model_parameter_gradients=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=None,solver_internal_linear_system_count=None,new_linear_program_calls=0,temporary_probe_heads=0,new_states_prepared=0,confirmation_states_evaluated=0,excluded_input_states_evaluated=0,candidate_selected=False,best_checkpoint_selected=False,writer_deployed=False,thresholds_changed=False,parent_decisions_reclassified=False,archived_source_executed=False,chronology_or_nonaccess_proved=False,preparation_replay_performed=False,FP64_acceptance_evaluated=False,guard_preservation_certified=False,solver_optimality_or_infeasibility_certified=False,curvature_or_convergence_certified=False,causal_explanation_established=False,LC2_closed=False,LC2_single_fold_passed=False,scope=SCOPE)
 def budget():need(time.monotonic()-started<TIMEOUT,'C172 fixed180-minute worker budget')
 def checkpoint():s.update(stage=stage,new_model_forwards=calls['model_forwards'],functional_FP64_forwards=calls['functional_FP64_forwards'],new_backward_calls=calls['backward_calls'],scalar_backward_attempts=calls['backward_attempts'],records_completed=len(records),requests_consumed=at,elapsed_worker_seconds=time.monotonic()-started,process_peak_RSS_KiB=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss));dump(out/'SUMMARY.json',dict(s,errors=errors))
 def authorize(kind,arm,sid):
  nonlocal at
  budget();need(at<len(rq),'No unregistered measurement');compare(rq[at],dict(stage=kind,arm=arm,state_id=sid),'All1848 fixed request order');at+=1
 def save(kind,arm,row):
  records.append(dict(stage=kind,arm=arm,**row))
  with (out/'ALL1848_RECORDS.jsonl').open('a') as f:f.write(json.dumps(records[-1],ensure_ascii=False,allow_nan=False)+'\n')
  if len(records)%12==0:checkpoint()
 def rehash(folder,receipt,p):
  compare(parameter_hash(p),receipt['head_digest'],'Immutable sealed head')
  for n,h in receipt['artifact_seals'].items():compare(sha(folder/n),h,'Immutable point/head artifact/'+n)
  compare(clean_json((folder/'SEALED.json').read_bytes()),receipt)
 try:
  need(not(out/'SUMMARY.json').exists(),'Fresh output only');dump(out/'DESIGN_FROZEN_BEFORE_MODEL.json',design());checkpoint();compare(set(paths),set(INPUT_KEYS),'Exactly17 original ZIPs')
  for k,path in paths.items():compare(sha(path),c['pins'][k]['sha256'],'Fixed whole input before loading/'+k)
  for n,h in c['source171'].items():compare(sha(ROOT/'vendor171'/n),h,'Trusted source/'+n)
  stage='seventeen_original_binding';ctx=(load if input_loader is None else input_loader)(paths,out/'binding',c,budget);compare(ctx['C171_receipt']['decision'],prior.DECISION);s.update(C171_parent_decision=ctx['C171_receipt']['decision'],all_seventeen_originals_independently_bound=input_loader is None)
  chain=control.source168.old.parent.parent.previous.constants(ctx['data']);p=ctx['C170_final_head'].copy()
  if c['production']:compare(p.shape,(3,2,36));compare(parameter_hash(p),design()['initial_head_digest'])
  s['original_C170_fixed_final_head_digest']=parameter_hash(p);np.savez_compressed(out/'ANCHOR_C170_FIXED_FINAL_HEAD.npz',P=p);np.savez_compressed(out/'FROZEN_FULL84_CHAIN_CONSTANTS.npz',**chain);np.savez_compressed(out/'FIXED_ARCHIVED_OBJECTIVE_REFERENCES.npz',response=np.asarray(ctx['baseline']['FP32_cached_response_reference'],float),baseline_logits=ctx['baseline_logits']['FP32_logits_cast64']);rq=requests();dump(out/'ALL1848_REQUESTS_FROZEN.json',rq);dump(out/'ORIGINAL_INPUTS.json',{k:dict(path=str(v),sha256=sha(v)) for k,v in paths.items()});points,native,receipt=seal_head(chain,p,out/'anchor','C170_fixed_final_anchor',True);repeat_points,_,repeat_receipt=seal_head(chain,p,out/'anchor_repeat','anchor_repeat',False)
  physical_ids={k:safe.tensor_hash(a) for k,a in native.items()}
  for i,sid in enumerate(IDS):compare(physical_ids[(sid,'zero')],ctx['C170_final_native']['0'][i]['metrics']['theta_sha256'],'Original C170 final physical identity')
  fixed={n:sha(out/n) for n in ('DESIGN_FROZEN_BEFORE_MODEL.json','ANCHOR_C170_FIXED_FINAL_HEAD.npz','FROZEN_FULL84_CHAIN_CONSTANTS.npz','FIXED_ARCHIVED_OBJECTIVE_REFERENCES.npz','ALL1848_REQUESTS_FROZEN.json')};dump(out/'ANCHOR_SEALED_BEFORE_MODEL.json',dict(artifact_seals=fixed,model_forwards_so_far=0,backward_calls_so_far=0,probe_heads_not_yet_computed=True));fixed['ANCHOR_SEALED_BEFORE_MODEL.json']=sha(out/'ANCHOR_SEALED_BEFORE_MODEL.json');chain_ids={n:array_digest(a) for n,a in chain.items()}
  def unchanged():
   for n,h in fixed.items():compare(sha(out/n),h,'Fixed artifact/'+n)
   compare({n:array_digest(a) for n,a in chain.items()},chain_ids,'Full chain constant bytes');rehash(out/'anchor',receipt,p);rehash(out/'anchor_repeat',repeat_receipt,p);need({k:safe.tensor_hash(a) for k,a in native.items()}==physical_ids,'Anchor native bytes unchanged')
  if runtime_factory is None:
   from runtime172 import factory
   runtime_factory=factory
  stage='same_run_pinned_model';rt=runtime_factory(out/'model_runtime',ctx,chain,points,native,dict(zero=p.copy()),calls);s['real_model_execution_claim']=c['production'];actual=[rt.preseal(p,points,out/'anchor','FP64_anchor_gradient'),rt.preseal(p,repeat_points,out/'anchor_repeat','FP64_anchor_repeat')];native_rows={};stage='native_anchor0';rr=[]
  for i,sid in enumerate(IDS):
   authorize(stage,'anchor',sid);r=rt.native_candidate(sid,native[(sid,'zero')],physical_ids[(sid,'zero')]);re=archive155.repeat_check(ctx['C170_final_native']['0'][i]['metrics'],r['metrics']);r['C170_final_native_replay']=re;rr.append(r);save(stage,'anchor',r);need(re['within_replay_tolerance'] and re['same_point_identity'],'Original C170 native replay before gradients')
  native_rows[('anchor',0)]=rr;stage='FP64_anchor';rt.activate('anchor',p);anchor=[]
  for i,sid in enumerate(IDS):
   authorize(stage,'anchor',sid);r=rt.coefficient_record(sid,p,points[(sid,'zero')],True);prev=ctx['C170_final_FP64'][i];re=dict(observed_Linf=float(np.max(abs(np.asarray(r['observed'])-prev['observed']))),KL_absolute=abs(r['KL']-prev['KL']));re['within_original_replay_tolerance']=re['observed_Linf']<=2e-9 and re['KL_absolute']<=2e-9;r['C170_final_FP64_replay']=re;anchor.append(r);save(stage,'anchor',r);need(re['within_original_replay_tolerance'],'Original C170 FP64 replay before LP')
  rt.verify_batch(p);unchanged();rt.verify(True);stage='one_offline_response_LP';pr=problem(anchor,p);np.savez_compressed(out/'CURRENT_ANCHOR_LINEAR_PROBLEM.npz',**pr);dump(out/'ALL84_FRESH_GRADIENT_RECORDS.json',anchor);checkpoint()
  def attempted():s['new_linear_program_calls']+=1;checkpoint()
  result=solve(pr,solver_backend,attempted,lambda r:dump(out/'RAW_PRIMAL_SOLVER_RESULT.json',r));d=result.pop('direction').reshape(p.shape);np.savez_compressed(out/'ONE_FRESH_FIXED_DIRECTION.npz',anchor=p,direction=d);dump(out/'SOLVER_DIAGNOSTIC.json',result);cert=control.verify_direction(pr,d.ravel());dump(out/'EXACT_RECORDED_DIRECTION_CHECK.json',cert);need(not len(pr['active']) or cert['strict_recorded_common_descent'],'Nonpositive or unresolved direction: no fallback or probe');s.update(active_response_states=len(pr['active']),exact_recorded_common_descent=cert['strict_recorded_common_descent'],empty_active_zero_direction=result['empty_active_noop']);budget()
  stage='freeze_all6_before_any_probe';heads,panel=step_panel(p,d);dump(out/'FIXED6_STEP_PANEL_BEFORE_PROBES.json',panel);all_points={};all_native={};all_native_ids={};head_receipts={}
  for a in panel['arms']:
   name=a['name'];pt,nt,re=seal_head(chain,heads[name],out/name,name,True);all_points[name]=pt;all_native[name]=nt;all_native_ids[name]={k:safe.tensor_hash(v) for k,v in nt.items()};head_receipts[name]=re;actual.append(rt.preseal(heads[name],pt,out/name,name))
  dump(out/'ALL8_ACTUAL_FP64_POINT_SEALS.json',actual);probe_seal=dict(model_forwards_so_far=calls['model_forwards'],backward_calls_so_far=calls['backward_calls'],anchor_gradient_records_completed=84,probe_records_completed=0,all6_head_digests={n:parameter_hash(q) for n,q in heads.items()},actual_point_seals=actual,all6_before_first_probe=True,artifact_seals={n:sha(out/n) for n in ('ONE_FRESH_FIXED_DIRECTION.npz','CURRENT_ANCHOR_LINEAR_PROBLEM.npz','ALL84_FRESH_GRADIENT_RECORDS.json','EXACT_RECORDED_DIRECTION_CHECK.json','SOLVER_DIAGNOSTIC.json','FIXED6_STEP_PANEL_BEFORE_PROBES.json','ALL8_ACTUAL_FP64_POINT_SEALS.json')});dump(out/'ALL6_PROBES_SEALED_BEFORE_FIRST_PROBE.json',probe_seal);sealed_sha=sha(out/'ALL6_PROBES_SEALED_BEFORE_FIRST_PROBE.json');direction_id=array_digest(d);s.update(temporary_probe_heads=6,all6_sealed_before_first_probe=True,gradient_batch_precedes_probe_sealing=True,positive_steps=list(H))
  def check_all():
   unchanged();compare(sha(out/'ALL6_PROBES_SEALED_BEFORE_FIRST_PROBE.json'),sealed_sha,'Fixed all6 probe receipt');compare(array_digest(d),direction_id,'One direction unchanged');compare(calls['backward_calls'],168,'Only original anchor gradients');compare(calls['backward_attempts'],168)
   for n,h in probe_seal['artifact_seals'].items():compare(sha(out/n),h,'Frozen probe prerequisite/'+n)
   for name in heads:rehash(out/name,head_receipts[name],heads[name]);need({k:safe.tensor_hash(v) for k,v in all_native[name].items()}==all_native_ids[name],'Frozen native probe bytes')
  check_all();probes={};stage='FP64_probe'
  for a in panel['arms']:
   name=a['name'];q=heads[name];check_all();rt.activate(name,q);rr=[]
   for sid in IDS:authorize(stage,name,sid);r=rt.coefficient_record(sid,q,all_points[name][(sid,'zero')],False);rr.append(r);save(stage,name,r)
   rt.verify_batch(q);probes[name]=rr;print('C172_FP64_PROBE',name,'alpha',a['alpha'],'all84 complete',flush=True)
  stage='FP64_anchor_repeat';check_all();rt.activate('anchor_repeat',p);repeat=[]
  for sid in IDS:authorize(stage,'anchor',sid);r=rt.coefficient_record(sid,p,repeat_points[(sid,'zero')],False);repeat.append(r);save(stage,'anchor',r)
  rt.verify_batch(p)
  for stage,arms,rnd in [('native_probe0',panel['arms'],0),('native_anchor1',[dict(name='anchor')],1),('native_probe1',panel['arms'],1)]:
   for a in arms:
    name=a['name'];nt=native if name=='anchor' else all_native[name];check_all();rr=[]
    for i,sid in enumerate(IDS):
     authorize(stage,name,sid);r=rt.native_candidate(sid,nt[(sid,'zero')],(physical_ids if name=='anchor' else all_native_ids[name])[(sid,'zero')])
     if name=='anchor':r['C170_final_native_replay']=archive155.repeat_check(ctx['C170_final_native']['1'][i]['metrics'],r['metrics'])
     rr.append(r);save(stage,name,r)
     if name=='anchor':need(r['C170_final_native_replay']['within_replay_tolerance'] and r['C170_final_native_replay']['same_point_identity'],'Original C170 repeated native replay')
    native_rows[(name,rnd)]=rr;rt.verify(True);print('C172_NATIVE',name,'round',rnd,'passed',native_summary(rr)['passed'],'/84',flush=True)
  stage='all84_all6_diagnostics';diag,fd,changes,fr,nr=analyse(anchor,repeat,probes,d,native_rows,panel,p,heads);compare(len(fd),252);compare(len(changes),504);compare(len(fr),84);compare(len(nr),588)
  for n,v in [('ALL252_CENTRAL_DIFFERENCES_VS_FRESH_AD.json',fd),('ALL504_SIGNED_LOSS_CHANGES.json',changes),('ALL84_FP64_ANCHOR_REPEATS.json',fr),('ALL588_NATIVE_REPEAT_CHECKS.json',nr),('ALL1176_NATIVE_ENDPOINTS.json',{name+'__round'+str(t):rr for (name,t),rr in native_rows.items()}),('ALL6_FP64_PROBE_RECORDS.json',probes),('DIAGNOSTIC_SUMMARY.json',diag)]:dump(out/n,v)
  check_all();rt.verify_batch(p);rt.verify(True);compare(at,RECORDS);compare(len(records),RECORDS);compare(calls['model_forwards'],FORWARDS);compare(calls['functional_FP64_forwards'],FUNCTIONAL);compare(calls['backward_calls'],BACKWARDS);compare(calls['backward_attempts'],BACKWARDS);compare(rt.point_tensors_saved,672);compare(s['new_linear_program_calls'],int(len(pr['active'])>0));s.update(complete=True,decision='C172_FIXED_FINAL_HEAD_FINITE_STEPS_COMPLETE_NO_CANDIDATE_SELECTED',actual_FP64_point_tensors_saved=672,all_requested_states_and_arms_retained=True,native_repeats_consistent=True,FP64_anchor_repeat_consistent=True,initial_native_and_FP64_replay_matched_C170=True,diagnostics=diag);stage='complete'
  (out/'REPORT_ZH.md').write_text('# C172：C170固定最终头的有限步长探测\n\n'+json.dumps(dict(panel=panel,diagnostics=diag),ensure_ascii=False,indent=2)+'\n\n一次全84态新梯度、最多一次响应LP；全部六臂在首次探测前封存，梯度早于封存。全部原生失败和两个轮次保留。不选择步长、头或部署；不改变父判定，所有状态均为已暴露训练态，不计LC2通过。\n')
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
  if not s['complete']:s['decision']='C172_INCOMPLETE_NO_FINITE_STEP_CONCLUSION'
  checkpoint()
 return dict(s,errors=errors)
if __name__=='__main__':
 ap=argparse.ArgumentParser()
 for k in INPUT_KEYS:ap.add_argument('--'+k.lower(),required=True)
 ap.add_argument('--out',required=True);a=ap.parse_args();s=run({k:getattr(a,k.lower()) for k in INPUT_KEYS},a.out);print(json.dumps(s,ensure_ascii=False,indent=2),flush=True)
 if not s['complete']:raise SystemExit(1)
