"""Fixed12 full84 offline updates; exact-max response plus full84 guard penalties."""
import argparse,traceback
from core146 import *
from inputs146 import load

def run(paths,out,c=None,runtime_factory=None,backend_factory=None,input_loader=None):
    c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);started=time.monotonic();stage='registered';d=None;backend=None;a=None;errors=[];pre=[];zeros=[];warms=[];updates=[];records=[];seals={}
    calls=dict(model_forwards=0,initialisation_model_forwards=0,preflight_metric_attempts=0,preflight_metrics=0,path_metric_attempts=0,path_metrics=0)
    s=dict(protocol=PROTOCOL,complete=False,decision='C146_INCOMPLETE_FIXED_FINAL_NOT_QUALIFIED',synthetic_fixture=not c['production'],real_model_execution_claim=False,training_states=84,all84_remain_exposed_training=True,requested_training_steps=12,requested_scalar_backward_calls=2016,requested_preflight_records=336,requested_final_endpoint_records=504,requested_model_forwards=4033,confirmation_states_evaluated=0,excluded_input_states_evaluated=0,new_states_prepared=0,new_model_features_acquired=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,full_response_Jacobian_constructed=False,candidate_selected=False,writer_deployed=False,C145_probe_selected=False,best_checkpoint_selected=False,recovery_checkpoint_selected=False,thresholds_changed=False,parent_decision_reclassified=False,archived_source_executed=False,teacher_calls=0,backtracking_steps=0,online_response_repair_steps=0,GPT2_optimizer_updates=0,LoRA_A_optimizer_updates=0,base_writer_updates=0,functional_zero_parity_complete_before_gradients=False,functional_warm_parity_complete_before_gradients=False,step12_frozen_before_final_evaluation=False,fixed_final_training_qualification_passed=False,LC2_closed=False,LC2_single_fold_passed=False,AD_accuracy_independently_certified=False,scope=design()['scope'])
    def budget():need(time.monotonic()-started<TIMEOUT,'C146 fixed180-minute worker budget exceeded')
    def checkpoint(full=False):
        stats=backend.stats() if backend is not None else {};s.update(stage=stage,elapsed_worker_seconds=time.monotonic()-started,runtime_calls=calls,new_model_forwards=calls['model_forwards'],gradient_stats=stats,new_backward_calls=stats.get('scalar_backward_calls',0),new_training_steps=len(updates),completed_training_steps=len(updates),new_head_fits=int(bool(updates)),preflight_records=len(pre),zero_parity_records=len(zeros),warm_parity_records=len(warms),final_endpoint_records=len(records));dump(out/'SUMMARY.json',dict(s,errors=errors))
        if full:
            for n,obj in [('PREFLIGHT_PROGRESS.json',pre),('ZERO_FUNCTIONAL_PROGRESS.json',zeros),('WARM_FUNCTIONAL_PROGRESS.json',warms),('TRAINING_PROGRESS.json',updates),('FINAL_ENDPOINT_PROGRESS.json',records)]:dump(out/n,obj)
    def seal(n):seals[n]=sha(out/n)
    def unchanged():
        for n,h in seals.items():need(sha(out/n)==h,'Frozen artifact changed/'+n)
        if d is not None:compare(frozen_digest(d),d['fixed_identities'],'Frozen writer/basis/inputs/C143 controls')
    try:
        need(not (out/'SUMMARY.json').exists(),'Fresh C146 output required');dump(out/'DESIGN_FROZEN_BEFORE_MODEL.json',design());seal('DESIGN_FROZEN_BEFORE_MODEL.json');checkpoint();stage='six_original_archive_bindings'
        d=(load if input_loader is None else input_loader)(paths,out,c);d['fixed_identities']=frozen_digest(d);budget()
        for n in ('INPUTS_FROZEN.json','ALL84_TRAINING_REGISTRY_FROZEN.json','FROZEN_INPUT_IDENTITIES.json','MODEL_PINS_REFERENCE.json','C145_BOUND_RECEIPT.json','ALL84_TRAINING_INPUTS.npz','INITIAL_C143_STEP12_CONTROL.npz','FROZEN_SHARED_ROW_BASIS.npz','FROZEN_ORIGINAL_WRITER.npz','FROZEN_ORIGINAL_CONTROL.npz'):
            need((out/n).is_file(),'Missing frozen binding/'+n);seal(n)
        need(set(d['points'])=={r['state_id'] for r in registry()},'Only all84 declared training states');rank=len(d['a']);need(rank==36 if c['production'] else 0<rank<=36,'Shared control dimension');a=d['a'].copy();m=np.zeros_like(a);v=np.zeros_like(a);zero=np.zeros_like(a);farray(a,(rank,));s.update(trainable_shared_coordinates=rank,initial_C143_control_digest=row_digest(a),training_response_safety=RESPONSE_SAFETY,acceptance_response_budget=.002,C145_parent_decision=d['C145_summary']['decision']);phase=Phase(d);stage='model_initialisation';checkpoint()
        actual=runtime_factory is None and backend_factory is None and input_loader is None
        if runtime_factory is None:
            from runtime146 import factory
            runtime_factory=factory
        rt=runtime_factory(out/'model_runtime',d,phase,calls);s['real_model_execution_claim']=actual;compare({k:v for k,v in rt.pins.items() if k!='torch'},{k:v for k,v in d['modelpins'].items() if k!='torch'},'Pinned model identity');stage='all84_four_native_preflights';checkpoint();native={}
        for sp in registry():
            budget();sid=sp['state_id'];p=d['points'][sid]
            for kind,point,write in [('prepared',p['prepared'],False),('raw',p['raw'],True),('C140',proposal(p['prepared'],p['raw'],d['V'],d['C140_a'])[0],True),('C143',proposal(p['prepared'],p['raw'],d['V'],d['a'])[0],True)]:
                phase.authorize((sid,kind),point,sp['case'],write);met=rt.metrics(point,sp['case'],write);g,_,_=native_gates(met,sp['case'],point,write)
                if not write:need(all(g.values()),'Invalid preparation retained and blocks before training')
                else:endpoint_gates(met,sp['case'],point,p['prepared'],True)
                comparisons=[repeat_check(met,old) for old in d['refs'][(sid,kind)]];pre.append(dict(**sp,kind=kind,metrics=met,comparisons=comparisons));checkpoint();need(all(q['same_point_identity'] and q['within_replay_tolerance'] for q in comparisons),'Native original replay drift blocks before gradients')
                native[(sid,kind)]=met
            checkpoint(True);print('C146_PREFLIGHT',sid,'OK',flush=True)
        unchanged();phase.begin_zero();stage='all84_zero_functional_parity'
        if backend_factory is None:
            from functional146 import factory
            backend_factory=factory
        backend=backend_factory(rt,d,phase,calls)
        for sp in registry():
            budget();sid=sp['state_id'];r=backend.parity(sp,zero);compare(r['state_id'],sid);compare(r['point_sha256'],safe.tensor_hash(d['points'][sid]['raw']));err=close(r['observed'],native[(sid,'raw')]['observed'],'Zero functional/native observations',2e-6);kl=abs(r['KL']-native[(sid,'raw')]['KL']);need(kl<=2e-7,'Zero functional/native KL');zeros.append(dict(r,observed_parity_Linf=err,KL_parity_absolute_error=kl));checkpoint()
        need(backend.stats()['scalar_backward_calls']==0,'No backward before all84 zero parity');s['functional_zero_parity_complete_before_gradients']=True;phase.begin_warm();stage='all84_C143_functional_parity'
        for sp in registry():
            budget();sid=sp['state_id'];r=backend.parity(sp,a);compare(r['state_id'],sid);need(0<=r['point_parity_error_Linf']<=2e-6,'Warm point parity');err=close(r['observed'],native[(sid,'C143')]['observed'],'Warm functional/native observations',2e-6);kl=abs(r['KL']-native[(sid,'C143')]['KL']);need(kl<=2e-7,'Warm functional/native KL');warms.append(dict(r,observed_parity_Linf=err,KL_parity_absolute_error=kl));checkpoint()
        need(backend.stats()['scalar_backward_calls']==0,'No backward before all84 warm parity');backend.verify(rehash=True);unchanged();s['functional_warm_parity_complete_before_gradients']=True;phase.begin_training();stage='fixed12_full84_worst_response_training';checkpoint(True)
        for step in range(STEPS):
            budget();phase.begin_batch(step,a);folder=out/'training'/f'step{step+1:02d}';g,rows=backend.gradient(a,step,folder,budget);farray(g,a.shape);gg,agg=aggregate(rows,rank);close(g,gg,'All84 aggregate gradient',1e-12)
            for sp,r,w,contrib in zip(registry(),rows,agg['response_weights'],agg['state_gradient_contributions']):
                for k,val in sp.items():compare(r[k],val,'Training state binding/'+k)
                compare(r['coefficient_digest'],row_digest(a));p=d['points'][sp['state_id']];point=proposal(p['prepared'],p['raw'],d['V'],a)[0];compare(r['functional_point_sha256'],safe.tensor_hash(point));compare(r['objective_parts'],objective(r['observed'],r['KL'],point,p['prepared'],a,sp['case']),atol=1e-10,rtol=1e-9);compare(r['response_weight'],w);compare(r['guard_weight'],1/N);close(r['gradient_contribution'],contrib,'Response/guard contribution',1e-12)
            phase.end_batch();new,mm,vv,info=update(a,g,m,v,step+1);name=f'training/step{step+1:02d}/UPDATE.npz';folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/name,before_a=a,before_m=m,before_v=v,full_weighted_gradient=g,after_a=new,after_m=mm,after_v=vv);dump(folder/'OBSERVATIONS.json',rows);dump(folder/'AGGREGATION.json',agg)
            updates.append(dict(**info,aggregate_objective_before_update=agg['objective'],maximum_response_loss=agg['maximum_response_loss'],mean_response_loss=agg['mean_response_loss'],mean_guard_loss=agg['mean_guard_loss'],maximizing_state_ids=agg['maximizing_state_ids'],response_weights=agg['response_weights'],before_digest=row_digest(a),after_digest=row_digest(new),update_file=name,update_sha256=sha(out/name),n_training_records=N));a,m,v=new,mm,vv;np.savez_compressed(out/'LATEST_PARTIAL_CONTROL.npz',a=a,m=m,v=v,completed_steps=np.array([step+1],np.int64));unchanged();checkpoint(True);print('C146_FIXED_UPDATE',step+1,'/12 max_response_loss',agg['maximum_response_loss'],'mean_guard_loss',agg['mean_guard_loss'],'gradient_L2',info['gradient_L2'],flush=True)
        need(backend.stats()['scalar_backward_calls']==2016 and backend.stats()['full84_gradient_batches']==12,'Fixed training derivative denominator');backend.verify(rehash=True);unchanged();stage='freeze_only_step12';checkpoint();np.savez_compressed(out/'FINAL_SHARED_CONTROL_STEP12.npz',a=a);seal('FINAL_SHARED_CONTROL_STEP12.npz');points={};outcomes=[]
        for sp in registry():
            sid=sp['state_id'];p=d['points'][sid];trained,diag=proposal(p['prepared'],p['raw'],d['V'],a);warm,wd=proposal(p['prepared'],p['raw'],d['V'],d['a']);close(backend.proposed(a,sid),trained,'Final NumPy/Torch map parity',2e-6)
            for arm,point,dd in [(EVAL_ARMS[0],p['raw'].copy(),dict(original_raw_bytes_retained=True)),(EVAL_ARMS[1],warm,wd),(EVAL_ARMS[2],trained,diag)]:points[(sid,arm)]=point;outcomes.append(dict(**sp,arm=arm,point_sha256=safe.tensor_hash(point),proposal_diagnostics=dd))
        np.savez_compressed(out/'ALL252_FROZEN_FINAL_POINTS.npz',**{sid+'__'+arm:p for (sid,arm),p in points.items()});dump(out/'ALL252_FROZEN_FINAL_OUTCOMES.json',outcomes)
        for n in ('ALL252_FROZEN_FINAL_POINTS.npz','ALL252_FROZEN_FINAL_OUTCOMES.json'):seal(n)
        dump(out/'STEP12_FREEZE_BEFORE_EVALUATION.json',dict(completed_steps=12,control_digest=row_digest(a),artifact_seals=dict(seals),final_endpoint_measurements_so_far=0,checkpoint_choice='Fixed step12 only',no_best_or_recovery_selection=True));seal('STEP12_FREEZE_BEFORE_EVALUATION.json');phase.freeze(points);s['step12_frozen_before_final_evaluation']=True;stage='two_round504_native_endpoints';checkpoint();first={};drifts=[];repeats=[]
        for rnd in (0,1):
            for sp in registry():
                sid=sp['state_id'];p=d['points'][sid]
                for arm in EVAL_ARMS:
                    budget();point=points[(sid,arm)];phase.authorize((rnd,sid,arm),point,sp['case'],True);met=rt.metrics(point,sp['case'],True);g=endpoint_gates(met,sp['case'],point,p['prepared'],True);control=zero if arm==EVAL_ARMS[0] else (d['a'] if arm==EVAL_ARMS[1] else a);parts=objective(met['observed'],met['KL'],point,p['prepared'],control,sp['case']);r=dict(**sp,round=rnd,arm=arm,metrics=met,gates=g,training_objective_parts=parts)
                    if arm in EVAL_ARMS[:2]:
                        q=repeat_check(met,native[(sid,'raw' if arm==EVAL_ARMS[0] else 'C143')]);r['native_control_replay']=q
                        if not(q['same_point_identity'] and q['within_replay_tolerance']):drifts.append(dict(state_id=sid,round=rnd,arm=arm,comparison=q))
                    if rnd==0:first[(sid,arm)]=met
                    else:
                        q=repeat_check(met,first[(sid,arm)]);r['repeat_comparison']=q
                        if not(q['same_point_identity'] and q['within_replay_tolerance']):repeats.append(dict(state_id=sid,arm=arm,comparison=q))
                    records.append(r);dump(out/'final_measurements'/f'record{len(records)-1:03d}.json',r);checkpoint()
                if len(records)%36==0:unchanged();checkpoint(True)
                print('C146_FINAL',rnd,sid,[(r['arm'],r['gates']['complete_pass'],r['metrics']['response_drift']) for r in records[-3:]],flush=True)
        phase.finish();unchanged();backend.verify(rehash=True,check_chart=False);need(calls['preflight_metrics']==336 and calls['path_metrics']==504,'Complete native request denominator');need(calls['model_forwards']==4033 and backend.stats()['functional_endpoint_calls']==1176,'Fixed4033 forward budget');sums,pairs=final_summaries(records);consistent=not drifts and not repeats;passed=consistent and all(sums['all84'][str(j)][EVAL_ARMS[2]]['passed']==84 for j in (0,1))
        for n,obj in [('ALL504_FINAL_TRAINING_ENDPOINTS.json',records),('FINAL_TRAINING_SUMMARIES.json',sums),('PAIRED_VS_RAW_AND_C143.json',pairs),('CONTROL_REPLAY_FAILURES.json',drifts),('REPEATABILITY_FAILURES.json',repeats)]:dump(out/n,obj)
        s.update(complete=True,decision=('C146_FIXED_FINAL_COMPLETE_REPLAY_DRIFT_NO_QUALIFICATION' if not consistent else ('C146_FIXED_FINAL_ALL84_TRAINING_GATES_PASSED' if passed else 'C146_FIXED_FINAL_TRAINING_CONSTRAINT_FAILURE')),fixed_final_training_qualification_passed=bool(passed),repeatable=not repeats,controls_replay_consistent=not drifts,final_training_summaries=sums,paired_vs_raw_and_C143=pairs,next_branch=('RESOLVE_REPLAY_DRIFT' if not consistent else ('PREREGISTER_SEPARATE_UNSEEN_LOCAL_CONFIRMATION' if passed else 'STOP_FIXED12_SCHEDULE_NO_CHECKPOINT_SUBSTITUTION')))
        lines=['# C146：全84态最坏响应训练','','从原C143第12步开始；未采用C145任何探测点；只有一个共享控制向量。12步全部完成后只验收固定第12步。训练响应安全线0.0018不替代原验收线0.002。','','|臂|首轮通过|重复轮通过|首轮最坏响应|','|---|---:|---:|---:|']
        for arm in EVAL_ARMS:lines.append(f"|{arm}|{sums['all84']['0'][arm]['passed']}/84|{sums['all84']['1'][arm]['passed']}/84|{sums['all84']['0'][arm]['maximum_response_drift']}|")
        lines+=['','判定：'+s['decision'],'后续：'+s['next_branch'],'','目标：0.9倍全84态最大响应超额平方，加0.1倍均值，加全84态保护损失均值。精确max并列时取平均次梯度。这是有限加权目标，不保证逐步下降、可行性或约束优先级。所有失败、破坏和获救均保留。','','本次含离线训练；只更新共享控制，不训练GPT2主干/LoRA-A/原writer/V或特征。无新完整响应Jacobian/SVD/求解/教师/回溯/在线修复。完整训练通过也不代表新状态或排除输入泛化；两轮为同84态重复。LC2与单折均未闭合。'];(out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');stage='complete'
    except BaseException:errors.append(traceback.format_exc());s.update(complete=False,decision='C146_INCOMPLETE_FIXED_FINAL_NOT_QUALIFIED',fixed_final_training_qualification_passed=False);print(errors[-1],flush=True)
    finally:
        checks={}
        for k,p in paths.items():
            try:checks[k]=sha(p)==c['pins'][k]['sha256']
            except Exception:checks[k]=False
        s['original_inputs_unchanged']=checks
        if not all(checks.values()):s.update(complete=False,decision='C146_INCOMPLETE_FIXED_FINAL_NOT_QUALIFIED',fixed_final_training_qualification_passed=False);errors.append('Original input changed')
        checkpoint(True)
    return dict(s,errors=errors)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in INPUT_KEYS:p.add_argument('--'+k.lower(),required=True)
    p.add_argument('--out',required=True);args=p.parse_args();s=run({k:Path(getattr(args,k.lower())) for k in INPUT_KEYS},args.out);print(json.dumps(s,ensure_ascii=False,indent=2),flush=True)
    if not s['complete']:raise SystemExit(1)
