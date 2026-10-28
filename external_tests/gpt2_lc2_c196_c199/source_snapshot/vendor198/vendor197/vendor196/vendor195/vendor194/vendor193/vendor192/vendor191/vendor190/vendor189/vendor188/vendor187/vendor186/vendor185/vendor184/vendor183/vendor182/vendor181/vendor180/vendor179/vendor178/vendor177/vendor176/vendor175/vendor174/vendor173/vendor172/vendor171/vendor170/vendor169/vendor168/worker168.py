"""Fixed C167 direction; six sealed finite probes, all84 states, no AD or solve."""
import argparse,resource,traceback
from core168 import *
from inputs168 import load
from diagnostics168 import analyse

def run(paths,out,c=None,runtime_factory=None,input_loader=None):
 c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);started=time.monotonic();errors=[];stage='registered';calls=dict(model_forwards=0,functional_FP64_forwards=0,backward_attempts=0,backward_calls=0);rt=None;records=[];at=0
 s=dict(protocol=PROTOCOL,complete=False,decision='C168_INCOMPLETE_NO_FINITE_PROBE_CONCLUSION',synthetic_fixture=not c['production'],real_model_execution_claim=False,new_training_steps=0,new_head_fits=0,new_model_parameter_gradients=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,new_linear_program_calls=0,new_direction_optimization_iterations=0,temporary_probe_heads=0,new_states_prepared=0,confirmation_states_evaluated=0,excluded_input_states_evaluated=0,candidate_selected=False,best_checkpoint_selected=False,writer_deployed=False,thresholds_changed=False,parent_decisions_reclassified=False,archived_source_executed=False,chronology_or_nonaccess_proved=False,preparation_replay_performed=False,FP64_acceptance_evaluated=False,new_AD_evaluated=False,LC2_closed=False,LC2_single_fold_passed=False,scope=SCOPE)
 def budget():need(time.monotonic()-started<TIMEOUT,'C168 fixed180-minute worker budget')
 def checkpoint():s.update(stage=stage,new_model_forwards=calls['model_forwards'],functional_FP64_forwards=calls['functional_FP64_forwards'],new_backward_calls=calls['backward_calls'],scalar_backward_attempts=calls['backward_attempts'],records_completed=len(records),requests_consumed=at,elapsed_worker_seconds=time.monotonic()-started,process_peak_RSS_KiB=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss));dump(out/'SUMMARY.json',dict(s,errors=errors))
 def authorize(kind,arm,sid):
  nonlocal at
  budget();need(at<len(rq),'No unregistered finite endpoint');compare(rq[at],dict(stage=kind,arm=arm,state_id=sid),'Exact1848 request order');at+=1
 def save(kind,arm,row):
  records.append(dict(stage=kind,arm=arm,**row))
  with (out/'ALL1848_RECORDS.jsonl').open('a') as f:f.write(json.dumps(records[-1],ensure_ascii=False,allow_nan=False)+'\n')
  if len(records)%12==0:checkpoint()
 def rehash(folder,receipt,p):
  compare(parameter_hash(p),receipt['head_digest'],'Immutable fixed head')
  for n,h in receipt['artifact_seals'].items():compare(sha(folder/n),h,'Fixed head artifact/'+n)
  compare(clean_json((folder/'SEALED.json').read_bytes()),receipt)
 try:
  need(not(out/'SUMMARY.json').exists(),'Fresh output only');dump(out/'DESIGN_FROZEN_BEFORE_MODEL.json',design());checkpoint();stage='thirteen_original_binding_and_exact_certificate';ctx=(load if input_loader is None else input_loader)(paths,out/'binding',c,budget);compare(ctx['C167_receipt']['classification'],'RECORDED_GRADIENT_COMMON_DESCENT_CERTIFIED');s.update(C165_parent_decision='C165_FIXED_DIRECTION_PROBES_COMPLETE_NO_CANDIDATE_SELECTED',C166_parent_decision=ctx['C166_receipt']['decision'],C167_parent_decision=ctx['C167_receipt']['decision'],exact_recorded_gradient_certificate_reverified=True,all_thirteen_originals_independently_bound=input_loader is None)
  chain=old.parent.parent.previous.constants(ctx['data']);p=ctx['initial_head'].copy();d=ctx['fixed_C167_direction'].copy();s['initial_C163_fixed_final_head_digest']=parameter_hash(p);s['C167_direction_digest']=array_digest(d)
  if c['production']:compare(p.shape,(3,2,36))
  heads,panel=step_panel(p,d);np.savez_compressed(out/'FROZEN_FULL84_CHAIN_CONSTANTS.npz',**chain);np.savez_compressed(out/'ANCHOR_C163_FIXED_FINAL_HEAD.npz',P=p);np.savez_compressed(out/'ONE_FIXED_C167_DIRECTION.npz',direction=d,anchor=p);dump(out/'FIXED6_STEP_PANEL_BEFORE_MODEL.json',panel);rq=requests();dump(out/'ALL1848_REQUESTS_FROZEN.json',rq);dump(out/'ORIGINAL_INPUTS.json',{k:dict(path=str(v),sha256=sha(v)) for k,v in paths.items()})
  points,native,receipt=old.parent.seal_head(chain,p,out/'anchor','C163_fixed_final_anchor',True)
  for i,sid in enumerate(IDS):compare(safe.tensor_hash(native[(sid,'zero')]),ctx['C165_native_anchor_records']['0'][i]['metrics']['theta_sha256'],'Exact C165 same-anchor physical proposal')
  all_points={};all_native={};all_native_ids={};head_receipts={}
  for arm in panel['arms']:
   name=arm['name'];pt,nt,re=old.parent.seal_head(chain,heads[name],out/name,name,True);all_points[name]=pt;all_native[name]=nt;all_native_ids[name]={k:safe.tensor_hash(a) for k,a in nt.items()};head_receipts[name]=re
  repeat_points,_,repeat_receipt=old.parent.seal_head(chain,p,out/'anchor_repeat','anchor_repeat',False)
  files=['DESIGN_FROZEN_BEFORE_MODEL.json','FROZEN_FULL84_CHAIN_CONSTANTS.npz','ANCHOR_C163_FIXED_FINAL_HEAD.npz','ONE_FIXED_C167_DIRECTION.npz','FIXED6_STEP_PANEL_BEFORE_MODEL.json','ALL1848_REQUESTS_FROZEN.json'];fixed={n:sha(out/n) for n in files};dump(out/'ALL6_NUMPY_HEADS_SEALED_BEFORE_MODEL.json',dict(artifact_seals=fixed,all6_head_digests={n:parameter_hash(q) for n,q in heads.items()},head_seals=head_receipts,model_forwards_so_far=0,backward_calls_so_far=0));fixed['ALL6_NUMPY_HEADS_SEALED_BEFORE_MODEL.json']=sha(out/'ALL6_NUMPY_HEADS_SEALED_BEFORE_MODEL.json');chain_ids={n:array_digest(a) for n,a in chain.items()};direction_id=array_digest(d);physical_ids={k:safe.tensor_hash(a) for k,a in native.items()}
  def unchanged():
   for n,h in fixed.items():compare(sha(out/n),h,'Frozen pre-model artifact/'+n)
   compare(array_digest(d),direction_id,'Original C167 direction unchanged');compare({n:array_digest(a) for n,a in chain.items()},chain_ids,'Frozen complete chain');rehash(out/'anchor',receipt,p);rehash(out/'anchor_repeat',repeat_receipt,p)
   for name in heads:rehash(out/name,head_receipts[name],heads[name])
  unchanged();s.update(temporary_probe_heads=6,positive_steps=panel['positive_steps'],all6_NumPy_heads_sealed_before_model=True)
  if runtime_factory is None:
   from runtime168 import factory
   runtime_factory=factory
  stage='same_run_pinned_model_and_actual_point_seals';rt=runtime_factory(out/'model_runtime',ctx,chain,points,native,dict(zero=p.copy()),calls);s['real_model_execution_claim']=c['production'];actual=[rt.preseal(p,points,out/'anchor','FP64_anchor')]
  for arm in panel['arms']:
   name=arm['name'];actual.append(rt.preseal(heads[name],all_points[name],out/name,name))
  actual.append(rt.preseal(p,repeat_points,out/'anchor_repeat','anchor_repeat'));dump(out/'ALL8_ACTUAL_FP64_POINT_SEALS.json',actual);dump(out/'ALL6_PROBES_SEALED_BEFORE_FIRST_STATE.json',dict(model_forwards_so_far=calls['model_forwards'],backward_calls_so_far=calls['backward_calls'],all6_head_digests={n:parameter_hash(q) for n,q in heads.items()},actual_point_seals=actual,all6_before_first_state=True));seal_path=out/'ALL6_PROBES_SEALED_BEFORE_FIRST_STATE.json';sealed_sha=sha(seal_path)
  def check_all():unchanged();compare(sha(seal_path),sealed_sha,'All6 actual points before first-state receipt');need(calls['backward_calls']==calls['backward_attempts']==0,'No C168 backward allowed')
  native_rows={};stage='native_anchor0';rr=[]
  for i,sid in enumerate(IDS):
   authorize(stage,'anchor',sid);r=rt.native_candidate(sid,native[(sid,'zero')],physical_ids[(sid,'zero')]);re=archive155.repeat_check(ctx['C165_native_anchor_records']['0'][i]['metrics'],r['metrics']);r['C165_native_anchor_replay']=re;rr.append(r);save(stage,'anchor',r);need(re['within_replay_tolerance'] and re['same_point_identity'],'Initial native replay before all fixed probes')
  native_rows[('anchor',0)]=rr;stage='FP64_anchor';rt.activate('anchor',p);anchor=[]
  for i,sid in enumerate(IDS):
   authorize(stage,'anchor',sid);r=rt.coefficient_record(sid,p,points[(sid,'zero')],False);prev=ctx['C165_FP64_anchor_records'][i];re=dict(observed_Linf=float(max(abs(np.asarray(r['observed'])-prev['observed']))),KL_absolute=abs(r['KL']-prev['KL']));re['within_original_replay_tolerance']=re['observed_Linf']<=2e-9 and re['KL_absolute']<=2e-9;r['C165_FP64_anchor_replay']=re;anchor.append(r);save(stage,'anchor',r);need(re['within_original_replay_tolerance'],'C165 same-anchor FP64 replay before probes')
  rt.verify_batch(p);check_all();probes={};stage='FP64_probe'
  for arm in panel['arms']:
   name=arm['name'];q=heads[name];check_all();rt.activate(name,q);rr=[]
   for sid in IDS:authorize(stage,name,sid);r=rt.coefficient_record(sid,q,all_points[name][(sid,'zero')],False);rr.append(r);save(stage,name,r)
   rt.verify_batch(q);probes[name]=rr;print('C168_FP64_PROBE',name,'alpha',arm['alpha'],'all84 complete',flush=True)
  stage='FP64_anchor_repeat';check_all();rt.activate('anchor_repeat',p);repeat=[]
  for sid in IDS:authorize(stage,'anchor',sid);r=rt.coefficient_record(sid,p,repeat_points[(sid,'zero')],False);repeat.append(r);save(stage,'anchor',r)
  rt.verify_batch(p)
  for stage,arms,rnd in [('native_probe0',panel['arms'],0),('native_anchor1',[dict(name='anchor')],1),('native_probe1',panel['arms'],1)]:
   for arm in arms:
    name=arm['name'];nt=native if name=='anchor' else all_native[name];check_all();rr=[]
    for i,sid in enumerate(IDS):
     authorize(stage,name,sid);r=rt.native_candidate(sid,nt[(sid,'zero')],(physical_ids if name=='anchor' else all_native_ids[name])[(sid,'zero')])
     if name=='anchor':r['C165_native_anchor_replay']=archive155.repeat_check(ctx['C165_native_anchor_records']['1'][i]['metrics'],r['metrics'])
     rr.append(r);save(stage,name,r)
     if name=='anchor':need(r['C165_native_anchor_replay']['within_replay_tolerance'] and r['C165_native_anchor_replay']['same_point_identity'],'Repeated C165 native anchor replay')
    native_rows[(name,rnd)]=rr;rt.verify(True);print('C168_NATIVE',name,'round',rnd,'passed',old.parent.native_summary(rr)['passed'],'/84',flush=True)
  stage='all84_all6_fixed_diagnostics';diag,fd,changes,fr,nr=analyse(anchor,repeat,probes,d,ctx['C165_gradient_records'],native_rows,panel);compare(len(fd),252);compare(len(changes),504);compare(len(fr),84);compare(len(nr),588);dump(out/'ALL252_CENTRAL_DIFFERENCES_VS_ARCHIVED_AD.json',fd);dump(out/'ALL504_SIGNED_LOSS_CHANGES.json',changes);dump(out/'ALL84_FP64_ANCHOR_REPEATS.json',fr);dump(out/'ALL588_NATIVE_REPEAT_CHECKS.json',nr);dump(out/'ALL1176_NATIVE_ENDPOINTS.json',{name+'__round'+str(t):rr for (name,t),rr in native_rows.items()});dump(out/'ALL6_FP64_PROBE_RECORDS.json',probes);dump(out/'DIAGNOSTIC_SUMMARY.json',diag);check_all();rt.verify_batch(p);rt.verify(True)
  compare(at,RECORDS);compare(len(records),RECORDS);compare(calls['model_forwards'],REQUESTED_FORWARDS);compare(calls['functional_FP64_forwards'],FUNCTIONAL_FORWARDS);compare(calls['backward_calls'],0);compare(calls['backward_attempts'],0);compare(rt.point_tensors_saved,672);s.update(complete=True,decision='C168_FIXED_CERTIFIED_DIRECTION_PROBES_COMPLETE_NO_CANDIDATE_SELECTED',all_requested_states_and_arms_retained=True,all6_actual_point_batches_sealed_before_first_state=True,actual_FP64_point_tensors_saved=672,actual_FP64_points_checked_before_model_forward=True,initial_native_and_FP64_replay_matched_C165=True,native_repeats_consistent=True,FP64_anchor_repeat_consistent=True,diagnostics=diag);stage='complete'
  (out/'REPORT_ZH.md').write_text('# C168：固定C167方向的有限写入探测\n\n'+json.dumps(dict(steps=panel,diagnostics=diag),ensure_ascii=False,indent=2)+'\n\n保留全部84态、六臂、两个原生测量轮次和失败态。不选步长、头或部署。精确证书只关于归档梯度；本轮实际模型测量另行报告。当前均为已暴露训练态，不构成新的确认或排除输入泛化证据，LC2仍开放。\n')
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
  if not s['complete']:s['decision']='C168_INCOMPLETE_NO_FINITE_PROBE_CONCLUSION'
  checkpoint()
 return dict(s,errors=errors)
if __name__=='__main__':
 ap=argparse.ArgumentParser()
 for k in INPUT_KEYS:ap.add_argument('--'+k.lower(),required=True)
 ap.add_argument('--out',required=True);a=ap.parse_args();s=run({k:getattr(a,k.lower()) for k in INPUT_KEYS},a.out);print(json.dumps(s,ensure_ascii=False,indent=2),flush=True)
 if not s['complete']:raise SystemExit(1)
