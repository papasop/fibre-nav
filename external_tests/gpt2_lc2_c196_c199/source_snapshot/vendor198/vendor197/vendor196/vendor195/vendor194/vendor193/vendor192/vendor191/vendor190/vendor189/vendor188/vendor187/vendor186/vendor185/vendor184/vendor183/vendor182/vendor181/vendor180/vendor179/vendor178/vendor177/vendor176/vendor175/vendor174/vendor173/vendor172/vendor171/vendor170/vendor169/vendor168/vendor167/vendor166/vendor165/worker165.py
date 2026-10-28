"""Full84 freshly computed scalar gradients, one fixed direction, six fixed probes."""
import argparse,resource,traceback
from core165 import *
from inputs165 import load
from diagnostics165 import analyse

def run(paths,out,c=None,runtime_factory=None,input_loader=None):
 c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);started=time.monotonic();errors=[];stage='registered';calls=dict(model_forwards=0,functional_FP64_forwards=0,backward_attempts=0,backward_calls=0);rt=None;records=[];at=0
 s=dict(protocol=PROTOCOL,complete=False,decision='C165_INCOMPLETE_NO_DIRECTIONAL_CONCLUSION',synthetic_fixture=not c['production'],real_model_execution_claim=False,new_training_steps=0,new_head_fits=0,new_model_parameter_gradients=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,offline_direction_optimization_iterations=0,temporary_probe_heads=0,new_states_prepared=0,confirmation_states_evaluated=0,excluded_input_states_evaluated=0,candidate_selected=False,best_checkpoint_selected=False,writer_deployed=False,thresholds_changed=False,parent_decisions_reclassified=False,archived_source_executed=False,chronology_or_nonaccess_proved=False,preparation_replay_performed=False,FP64_acceptance_evaluated=False,LC2_closed=False,LC2_single_fold_passed=False,scope=design()['scope'])
 def budget():need(time.monotonic()-started<TIMEOUT,'C165 fixed180-minute worker budget')
 def checkpoint():s.update(stage=stage,new_model_forwards=calls['model_forwards'],functional_FP64_forwards=calls['functional_FP64_forwards'],new_backward_calls=calls['backward_calls'],scalar_backward_attempts=calls['backward_attempts'],records_completed=len(records),requests_consumed=at,elapsed_worker_seconds=time.monotonic()-started,process_peak_RSS_KiB=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss));dump(out/'SUMMARY.json',dict(s,errors=errors))
 def authorize(kind,arm,sid):
  nonlocal at
  budget();need(at<len(rq),'No unregistered endpoint');compare(rq[at],dict(stage=kind,arm=arm,state_id=sid),'Exact fixed1848 request sequence');at+=1
 def save(kind,arm,row):
  records.append(dict(stage=kind,arm=arm,**row))
  with (out/'ALL1848_RECORDS.jsonl').open('a') as f:f.write(json.dumps(records[-1],ensure_ascii=False,allow_nan=False)+'\n')
  if len(records)%12==0:checkpoint()
 def rehash(folder,receipt,p):
  compare(parameter_hash(p),receipt['head_digest'],'Exact immutable probe head')
  for n,h in receipt['artifact_seals'].items():compare(sha(folder/n),h,'Immutable probe artifact/'+n)
  compare(clean_json((folder/'SEALED.json').read_bytes()),receipt)
 try:
  need(not (out/'SUMMARY.json').exists(),'Fresh output only');dump(out/'DESIGN_FROZEN_BEFORE_MODEL.json',design());checkpoint();stage='ten_original_readonly_binding';ctx=(load if input_loader is None else input_loader)(paths,out/'binding',c,budget);compare(ctx['C164_receipt']['decision'],'C164_READONLY_TRAINING_AUDIT_COMPLETE_PARENT_DECISION_RETAINED');s['C164_parent_decision']=ctx['C164_receipt']['decision'];chain=parent.parent.previous.constants(ctx['data']);p=ctx['initial_head'].copy();s['initial_C163_fixed_final_head_digest']=parameter_hash(p)
  if c['production']:compare(p.shape,(3,2,36),'Production full216 head')
  np.savez_compressed(out/'FROZEN_FULL84_CHAIN_CONSTANTS.npz',**chain);np.savez_compressed(out/'ANCHOR_C163_FIXED_FINAL_HEAD.npz',P=p);rq=requests();dump(out/'ALL1848_REQUESTS_FROZEN.json',rq);dump(out/'ORIGINAL_INPUTS.json',{k:dict(path=str(v),sha256=sha(v)) for k,v in paths.items()});points,native,receipt=parent.seal_head(chain,p,out/'anchor','C163_fixed_final_anchor',True)
  for i,sid in enumerate(IDS):compare(safe.tensor_hash(native[(sid,'zero')]),ctx['parent_final_native_records']['0'][i]['metrics']['theta_sha256'],'Exact C163 final anchor native proposal')
  files=['DESIGN_FROZEN_BEFORE_MODEL.json','FROZEN_FULL84_CHAIN_CONSTANTS.npz','ANCHOR_C163_FIXED_FINAL_HEAD.npz','ALL1848_REQUESTS_FROZEN.json'];fixed={n:sha(out/n) for n in files};dump(out/'SEALED_BEFORE_MODEL.json',dict(artifact_seals=fixed,model_forwards_so_far=0));fixed['SEALED_BEFORE_MODEL.json']=sha(out/'SEALED_BEFORE_MODEL.json');chain_ids={n:array_digest(a) for n,a in chain.items()};physical_ids={k:safe.tensor_hash(a) for k,a in native.items()}
  def unchanged():
   for n,h in fixed.items():compare(sha(out/n),h,'Original constant file/'+n)
   compare({n:array_digest(a) for n,a in chain.items()},chain_ids,'Original complete chain');rehash(out/'anchor',receipt,p)
  if runtime_factory is None:
   from runtime165 import factory
   runtime_factory=factory
  stage='same_run_pinned_model';unchanged();rt=runtime_factory(out/'model_runtime',ctx,chain,points,native,dict(zero=p.copy()),calls);s['real_model_execution_claim']=c['production'];actual=[rt.preseal(p,points,out/'anchor','anchor_gradient')];dump(out/'ALL8_ACTUAL_FP64_POINT_SEALS.json',actual);native_rows={};stage='native_anchor0';rr=[]
  for i,sid in enumerate(IDS):
   authorize(stage,'anchor',sid);r=rt.native_candidate(sid,native[(sid,'zero')],physical_ids[(sid,'zero')]);re=archive155.repeat_check(ctx['parent_final_native_records']['0'][i]['metrics'],r['metrics']);r['C163_final_native_replay']=re;rr.append(r);save(stage,'anchor',r);need(re['within_replay_tolerance'] and re['same_point_identity'],'Initial native C163 replay before gradients')
  native_rows[('anchor',0)]=rr;rt.activate('anchor',p);stage='anchor_gradient';grads=[]
  for i,sid in enumerate(IDS):
   authorize(stage,'anchor',sid);r=rt.coefficient_record(sid,p,points[(sid,'zero')],True);old=ctx['parent_final_FP64_records'][i];re=dict(observed_Linf=float(np.max(abs(np.asarray(r['observed'])-old['observed']))),KL_absolute=abs(r['KL']-old['KL']));re['within_original_replay_tolerance']=re['observed_Linf']<=2e-9 and re['KL_absolute']<=2e-9;r['C163_final_FP64_replay']=re;grads.append(r);save(stage,'anchor',r);need(re['within_original_replay_tolerance'],'C163 FP64 replay before direction computation')
  rt.verify_batch(p);unchanged();rt.verify(True);stage='fixed_offline_direction';d,ds,history=direction(grads,p);np.savez_compressed(out/'ONE_FIXED_DIRECTION.npz',direction=d,anchor=p);dump(out/'DIRECTION_DIAGNOSTICS.json',ds);dump(out/'ALL4096_DIRECTION_ITERATIONS.json',history);s.update(offline_direction_optimization_iterations=len(history),direction_computed_from_training_measurements=True,computed_strict_common_response_descent=ds['computed_strict_common_response_descent'],direction_zero=ds['direction_zero']);heads=probe_heads(p,d);all_points={};all_native={};all_native_ids={};head_receipts={};stage='seal_all6_before_any_probe'
  direction_identity=array_digest(d);direction_hashes={n:sha(out/n) for n in ('ONE_FIXED_DIRECTION.npz','DIRECTION_DIAGNOSTICS.json','ALL4096_DIRECTION_ITERATIONS.json')}
  for arm in ARMS:
   name=arm['name'];folder=out/name;q=heads[name];pt,nt,re=parent.seal_head(chain,q,folder,name,True);all_points[name]=pt;all_native[name]=nt;all_native_ids[name]={key:safe.tensor_hash(a) for key,a in nt.items()};head_receipts[name]=re;actual.append(rt.preseal(q,pt,folder,name));rehash(folder,re,q)
  repeat_points,_,repeat_receipt=parent.seal_head(chain,p,out/'anchor_repeat','anchor_repeat',False);actual.append(rt.preseal(p,repeat_points,out/'anchor_repeat','anchor_repeat'));dump(out/'ALL8_ACTUAL_FP64_POINT_SEALS.json',actual);dump(out/'ALL6_PROBES_SEALED_BEFORE_FIRST_PROBE.json',dict(model_forwards_so_far=calls['model_forwards'],backward_calls_so_far=calls['backward_calls'],all6_head_digests={name:parameter_hash(q) for name,q in heads.items()},direction_artifact_seals=direction_hashes,head_seals=head_receipts,actual_point_seals=actual,all6_before_first_probe=True));seal_path=out/'ALL6_PROBES_SEALED_BEFORE_FIRST_PROBE.json';sealed_sha=sha(seal_path);s['temporary_probe_heads']=6
  def check_all():
   unchanged();compare(array_digest(d),direction_identity,'Frozen in-memory direction');compare(sha(seal_path),sealed_sha,'All6-before-probe receipt')
   for n,h in direction_hashes.items():compare(sha(out/n),h,'Frozen training-derived direction/'+n)
   for name in heads:rehash(out/name,head_receipts[name],heads[name])
  probes={};stage='FP64_probe'
  for arm in ARMS:
   name=arm['name'];q=heads[name];check_all();rt.activate(name,q);rr=[]
   for sid in IDS:authorize(stage,name,sid);r=rt.coefficient_record(sid,q,all_points[name][(sid,'zero')],False);rr.append(r);save(stage,name,r)
   rt.verify_batch(q);probes[name]=rr;print('C165_FP64_PROBE',name,'all84 complete',flush=True)
  stage='FP64_anchor_repeat';check_all();rt.activate('anchor_repeat',p);repeat=[]
  for sid in IDS:authorize(stage,'anchor',sid);r=rt.coefficient_record(sid,p,repeat_points[(sid,'zero')],False);repeat.append(r);save(stage,'anchor',r)
  rt.verify_batch(p)
  for stage,arms,rnd in [('native_probe0',ARMS,0),('native_anchor1',[dict(name='anchor')],1),('native_probe1',ARMS,1)]:
   for arm in arms:
    name=arm['name'];nt=native if name=='anchor' else all_native[name];check_all();rr=[]
    for i,sid in enumerate(IDS):
     authorize(stage,name,sid);r=rt.native_candidate(sid,nt[(sid,'zero')],(physical_ids if name=='anchor' else all_native_ids[name])[(sid,'zero')])
     if name=='anchor':r['C163_final_native_replay']=archive155.repeat_check(ctx['parent_final_native_records']['1'][i]['metrics'],r['metrics'])
     rr.append(r);save(stage,name,r)
     if name=='anchor':need(r['C163_final_native_replay']['within_replay_tolerance'] and r['C163_final_native_replay']['same_point_identity'],'Second native C163 replay')
    native_rows[(name,rnd)]=rr;rt.verify(True);print('C165_NATIVE',name,'round',rnd,'passed',parent.native_summary(rr)['passed'],'/84',flush=True)
  stage='fixed_full84_diagnostics';diag,fd,fr,nr=analyse(grads,repeat,probes,d,ds,native_rows);compare(len(fd),252);compare(len(fr),84);compare(len(nr),588);dump(out/'ALL252_CENTRAL_DIFFERENCE_CHECKS.json',fd);dump(out/'ALL84_FP64_ANCHOR_REPEATS.json',fr);dump(out/'ALL588_NATIVE_REPEAT_CHECKS.json',nr);dump(out/'ALL1176_NATIVE_ENDPOINTS.json',{name+'__round'+str(t):rr for (name,t),rr in native_rows.items()});dump(out/'ALL6_FP64_PROBE_RECORDS.json',probes);dump(out/'DIAGNOSTIC_SUMMARY.json',diag);check_all();rt.verify_batch(p);rt.verify(True)
  compare(at,RECORDS);compare(len(records),RECORDS);compare(calls['model_forwards'],REQUESTED_FORWARDS);compare(calls['functional_FP64_forwards'],FUNCTIONAL_FORWARDS);compare(calls['backward_calls'],REQUESTED_BACKWARDS);compare(calls['backward_attempts'],REQUESTED_BACKWARDS);compare(rt.point_tensors_saved,672);s.update(complete=True,decision='C165_FIXED_DIRECTION_PROBES_COMPLETE_NO_CANDIDATE_SELECTED',all_requested_states_and_arms_retained=True,all6_probes_sealed_before_first_probe=True,actual_FP64_point_tensors_saved=672,actual_FP64_points_checked_before_model_forward=True,initial_native_and_FP64_replay_matched_C163=True,native_repeats_consistent=True,FP64_anchor_repeat_consistent=True,diagnostics=diag);stage='complete'
  (out/'REPORT_ZH.md').write_text('# C165：共同响应下降方向与固定小步探测\n\n'+json.dumps(dict(direction=ds,diagnostics=diag),ensure_ascii=False,indent=2)+'\n\n方向来自本轮训练态的写后观测及梯度，仅为离线诊断，不能算免校正写入或LC2通过。方向计算只针对原R>0的全部状态，并在固定双侧内部坐标子空间中进行；没有对全部保护门槛的一阶可行性证明，全部84态的真实原生门槛另行测量。有限迭代未找到共同下降不能证明不可行，中心差分分歧也不能单独证明AD错误。所有6臂完整保留，不选步长、不选头、不部署。\n')
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
  if not s['complete']:s['decision']='C165_INCOMPLETE_NO_DIRECTIONAL_CONCLUSION'
  checkpoint()
 return dict(s,errors=errors)
if __name__=='__main__':
 parser=argparse.ArgumentParser()
 for k in INPUT_KEYS:parser.add_argument('--'+k.lower(),required=True)
 parser.add_argument('--out',required=True);a=parser.parse_args();s=run({k:getattr(a,k.lower()) for k in INPUT_KEYS},a.out);print(json.dumps(s,ensure_ascii=False,indent=2),flush=True)
 if not s['complete']:raise SystemExit(1)
