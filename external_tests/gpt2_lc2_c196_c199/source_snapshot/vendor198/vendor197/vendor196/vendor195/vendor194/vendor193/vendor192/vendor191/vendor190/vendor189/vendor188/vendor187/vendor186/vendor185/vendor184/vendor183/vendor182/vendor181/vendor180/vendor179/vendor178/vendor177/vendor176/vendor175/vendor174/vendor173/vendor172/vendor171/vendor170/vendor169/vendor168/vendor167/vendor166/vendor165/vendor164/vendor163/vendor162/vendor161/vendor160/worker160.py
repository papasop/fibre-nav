"""Full84 baseline: frozen coefficients, complete loss/gradient/FD/native denominators."""
import argparse,resource,traceback
from core160 import *
from inputs160 import load

def run(paths,out,c=None,runtime_factory=None,input_loader=None):
 c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);start=time.monotonic();stage='registered';errors=[];calls=dict(model_forwards=0,functional_FP64_forwards=0,backward_attempts=0,backward_calls=0);runtime=None;at=0;records=[];aggregations={}
 s=dict(protocol=PROTOCOL,complete=False,decision='C160_INCOMPLETE_NO_BASELINE_ACCEPTED',synthetic_fixture=not c['production'],real_model_execution_claim=False,training_states_requested=N,training_states_gradient_evaluated=0,full84_FP64_objective_available=False,new_training_steps=0,new_head_fits=0,new_model_parameter_gradients=0,new_parameter_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,new_states_prepared=0,new_model_features_acquired=0,confirmation_states_evaluated=0,excluded_input_states_evaluated=0,candidate_selected=False,writer_deployed=False,head_updated=False,coefficient_head_frozen=True,fresh_native_baseline_substituted_for_objective=False,preparation_replay_performed=False,FP64_acceptance_evaluated=False,exact_FP32_commit_derivative_certified=False,AD_accuracy_certified=False,curvature_bound_established=False,causal_failure_explanation_established=False,LC2_acceptance_thresholds_changed=False,parent_decisions_reclassified=False,LC2_closed=False,LC2_single_fold_passed=False,scope=design()['scope'])
 def budget():need(time.monotonic()-start<TIMEOUT,'C160 fixed120-minute worker budget')
 def checkpoint():
  s.update(stage=stage,new_model_forwards=calls['model_forwards'],functional_FP64_forwards=calls['functional_FP64_forwards'],new_backward_calls=calls['backward_calls'],scalar_backward_attempts=calls['backward_attempts'],records_completed=len(records),requests_consumed=at,elapsed_worker_seconds=time.monotonic()-start,process_peak_RSS_KiB=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss));dump(out/'SUMMARY.json',dict(s,errors=errors))
 try:
  need(not (out/'SUMMARY.json').exists(),'Fresh output directory');dump(out/'DESIGN_FROZEN_BEFORE_MODEL.json',design());checkpoint();stage='five_original_archive_binding';context=(load if input_loader is None else input_loader)(paths,out/'binding',c,budget);s['C159_original_decision']=context['completed159']['decision'];compare(s['C159_original_decision'],'C159_COEFFICIENT_CHAIN_FP64_DIAGNOSTIC_COMPLETE_NO_CANDIDATE_SELECTED');budget();stage='all588_points_and1008_requests_sealed_before_model';chain,points,native,coefs,seals=freeze(context,out);point_ids={k:point_hash(a) for k,a in points.items()};native_ids={k:safe.tensor_hash(a) for k,a in native.items()};chain_ids={n:array_digest(a) for n,a in chain.items()};rq=requests();need(len(rq)==1008,'Full1008 request registry')
  def unchanged():
   for n,h in seals.items():compare(sha(out/n),h,'Before-model artifact/'+n)
   need({k:point_hash(a) for k,a in points.items()}==point_ids,'All actual FP64 points unchanged');need({k:safe.tensor_hash(a) for k,a in native.items()}==native_ids,'All actual original FP32 points unchanged');compare({n:array_digest(a) for n,a in chain.items()},chain_ids,'All actual chain constants unchanged')
  def authorize(kind,sid,label):
   nonlocal at
   budget();compare(rq[at],dict(stage=kind,state_id=sid,label=label),'Exact requested order');at+=1;compare(point_hash(points[(sid,label)]),point_ids[(sid,label)],'Current sealed FP64 point');compare(safe.tensor_hash(native[(sid,label)]),native_ids[(sid,label)],'Current sealed original FP32 point')
  def save(kind,row):
   record=dict(stage=kind,**row);records.append(record)
   with (out/'ALL1008_RECORDS.jsonl').open('a') as f:f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
   if len(records)%12==0:checkpoint()
  if runtime_factory is None:
   from runtime160 import factory
   runtime_factory=factory
  unchanged();stage='fixed_model_and_archived_objective_reference';runtime=runtime_factory(out/'model_runtime',context,chain,points,native,coefs,calls);s['real_model_execution_claim']=c['production'];checkpoint();native_rounds=[];stage='native_FP32_alpha0_round0';rr=[]
  for sp in registry():sid=sp['state_id'];authorize('native0',sid,'zero');r=runtime.native(sid);rr.append(r);save('native0',r)
  native_rounds.append(rr);s['native_alpha0_round0']=native_summary(rr);unchanged();runtime.verify(True);checkpoint();stage='all84_complete_FP64_coefficient_gradients';gradrows=[]
  for sp in registry():
   sid=sp['state_id'];authorize('AD',sid,'zero');r=runtime.evaluate(sid,'zero',True);compare(r['FP64_point_sha256'],point_ids[(sid,'zero')]);need(r['parameters_digest']==parameter_hash(coefs['zero']),'Exact fixed gradient coefficient head');gradrows.append(r);save('AD',r);s['training_states_gradient_evaluated']=len(gradrows)
  gradient,agg=aggregate(gradrows,coefs['zero'].shape);s['gradient_baseline_aggregation']=agg;s['aggregate_gradient_L2']=float(np.linalg.norm(gradient));s['aggregate_gradient_dot_original_delta']=float(np.sum(gradient*chain['deltaP']));np.savez_compressed(out/'COMPLETE84_GRADIENT_BASELINE.npz',aggregate_gradient=gradient,original_deltaP=chain['deltaP'],response_gradients=np.array([r['response_gradient'] for r in gradrows]),guard_gradients=np.array([r['guard_gradient'] for r in gradrows]));dump(out/'ALL84_GRADIENT_BASELINE.json',gradrows);unchanged();runtime.verify(True);checkpoint();panelrows={};stage='seven_complete84_FP64_objective_panels'
  for label in LABELS:
   rr=[]
   for sp in registry():sid=sp['state_id'];authorize('finite',sid,label);r=runtime.evaluate(sid,label,False);rr.append(r);save('finite',r)
   panelrows[label]=rr;aggregations[label]=objective(rr);dump(out/('panel_'+label+'.json'),dict(rows=rr,aggregation=aggregations[label]));unchanged();runtime.verify(True);checkpoint();print('C160_FULL84_PANEL',label,'J',aggregations[label]['objective'],'worst',aggregations[label]['maximizing_state_ids'],flush=True)
  for a,b in zip(gradrows,panelrows['zero']):close(a['observed'],b['observed'],'Full84 AD/no-grad center',WITHIN_RUN_TOL);need(abs(a['KL']-b['KL'])<=WITHIN_RUN_TOL,'Full84 AD/no-grad KL center')
  compare(objective(gradrows),aggregations['zero'],'Complete objective AD/no-grad scalar arithmetic',atol=1e-9,rtol=1e-9);s['full84_FP64_objective_available']=True;stage='all84_FP64_zero_plus_one_repeats';repeat_rows=[]
  for label in ('zero','plus_one'):
   for sp,old in zip(registry(),panelrows[label]):
    sid=sp['state_id'];authorize('repeat',sid,label);r=runtime.evaluate(sid,label,False);diff=float(max(abs(np.asarray(r['observed'])-old['observed'])));kd=abs(r['KL']-old['KL']);need(diff<=WITHIN_RUN_TOL and kd<=WITHIN_RUN_TOL,'Within-run full84 FP64 repeat changed');repeat_rows.append(dict(state_id=sid,label=label,observed_Linf=diff,KL_absolute=kd));save('repeat',r)
   unchanged();runtime.verify(True);checkpoint()
  stage='native_FP32_alpha0_round1';rr=[]
  for sp in registry():sid=sp['state_id'];authorize('native1',sid,'zero');r=runtime.native(sid);rr.append(r);save('native1',r)
  native_rounds.append(rr);s['native_alpha0_round1']=native_summary(rr);nr=[]
  for a,b in zip(*native_rounds):re=archive155.repeat_check(a['metrics'],b['metrics']);need(re['within_replay_tolerance'] and re['same_point_identity'],'Original native FP32 repeat/gate identity failed');nr.append(dict(state_id=a['state_id'],**re))
  dump(out/'ALL84_NATIVE_REPEAT_COMPARISONS.json',nr);dump(out/'ALL168_FP64_REPEAT_COMPARISONS.json',repeat_rows);fd=[];dot=s['aggregate_gradient_dot_original_delta']
  for h,m,p in SCALES:
   r=previous.previous.previous.parent.fd(aggregations['zero']['objective'],aggregations[m]['objective'],aggregations[p]['objective'],h);fd.append(dict(**r,AD_directional=dot,central_minus_AD=r['central']-dot,positive_minus_AD=r['positive']-dot,negative_minus_AD=r['negative']-dot))
  s['three_scale_complete_objective_FD']=fd;s['all_panel_objectives']=aggregations;s['native_archived_anchor_all84_passed_both_rounds']=all(native_summary(x)['passed']==N for x in native_rounds);s['native_repeats_consistent']=True;s['maximum_FP64_repeat_observed_Linf']=max(r['observed_Linf'] for r in repeat_rows);s['maximum_FP64_repeat_KL_absolute']=max(r['KL_absolute'] for r in repeat_rows);dump(out/'COMPLETE_OBJECTIVE_DIRECTIONAL_DIAGNOSTIC.json',dict(gradient_aggregation=agg,all_panel_objectives=aggregations,aggregate_gradient_dot_original_delta=dot,all_three_scales=fd,no_derivative_agreement_threshold=True,all_states_retained=True))
  unchanged();runtime.verify(True);compare(at,1008);compare(len(records),1008);compare(calls['model_forwards'],2019);compare(calls['functional_FP64_forwards'],1682);compare(calls['backward_calls'],168);compare(calls['backward_attempts'],168);compare(s['training_states_gradient_evaluated'],84);s.update(complete=True,decision='C160_FULL84_FP64_OBJECTIVE_GRADIENT_BASELINE_COMPLETE_NO_TRAINING',all_requested_states_and_scales_retained=True);stage='complete'
  (out/'REPORT_ZH.md').write_text('# C160：完整84态FP64目标与梯度基线\n\n不更新原头；完整84态响应、读取、KL、半径、正则与最坏态聚合均保留。使用固定原C155缓存参考；所有步长均报告，不选择最优结果。原生FP32的两轮资格表独立保存，不用FP64结果替代。准备态有效性是原归档来源，未重新准备。\n\n完成只表示固定基线完成，不等于原头全部通过、梯度绝对正确、离散FP32提交可微、训练收敛或LC2闭合。无训练、选头、部署、新确认或排除输入评估。\n')
 except BaseException:errors.append(traceback.format_exc());print(errors[-1],flush=True)
 finally:
  checks={}
  for k in INPUT_KEYS:
   try:checks[k]=sha(paths[k])==c['pins'][k]['sha256']
   except Exception:checks[k]=False
  s['original_inputs_unchanged']=checks
  if not all(checks.values()):s['complete']=False;errors.append('Original input missing or changed')
  if runtime is not None:
   try:runtime.verify(True)
   except BaseException:s['complete']=False;errors.append(traceback.format_exc())
  if not s['complete']:s['decision']='C160_INCOMPLETE_NO_BASELINE_ACCEPTED'
  checkpoint()
 return dict(s,errors=errors)
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for key in INPUT_KEYS:p.add_argument('--'+key.lower(),required=True)
 p.add_argument('--out',required=True);a=p.parse_args();s=run({k:getattr(a,k.lower()) for k in INPUT_KEYS},a.out);print(json.dumps(s,ensure_ascii=False,indent=2),flush=True)
 if not s['complete']:raise SystemExit(1)
