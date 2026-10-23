"""Fixed12 full36 offline updates; final checkpoint only, three native arms."""
import argparse,traceback
from core140 import *
from inputs140 import load

def final_summaries(records):
    result={};pairs={}
    for group in ('all36','historical12','C139_exposed_kernel24','xor_00','xor_01','xor_11'):
        expected=36 if group=='all36' else (24 if group=='C139_exposed_kernel24' else 12);result[group]={};pairs[group]={}
        for rnd in (0,1):
            subset=[r for r in records if r['round']==rnd and (group=='all36' or r.get('training_origin')==group or r['case']==group)];by={(r['state_id'],r['arm']):r for r in subset};result[group][str(rnd)]={};pairs[group][str(rnd)]={}
            for arm in EVAL_ARMS:
                rr=[r for r in subset if r['arm']==arm];bad=Counter(k for r in rr for k in r['recomputed_gates']['failed_gates'])
                item=dict(n_requested=expected,n_evaluated=len(rr),passed=sum(r['recomputed_gates']['complete_pass'] for r in rr),failed_ids=[r['state_id'] for r in rr if not r['recomputed_gates']['complete_pass']],failed_gates=dict(sorted(bad.items())),maximum_response_drift=max((r['metrics']['response_drift'] for r in rr),default=None),within_training_response_safety=sum(r['metrics']['response_drift']<=RESPONSE_SAFETY for r in rr),mean_training_objective=float(np.mean([r['training_objective'] for r in rr])) if rr else None)
                if group=='all36' and len(rr)==N:item['aggregate_training_objective']=aggregate([r['training_objective'] for r in rr],np.zeros((N,1)))[1]['objective']
                result[group][str(rnd)][arm]=item;pairs[group][str(rnd)][arm]={}
                for control in EVAL_ARMS[:2]:pairs[group][str(rnd)][arm][control]=dict(rescued_ids=[r['state_id'] for r in rr if r['recomputed_gates']['complete_pass'] and not by[(r['state_id'],control)]['recomputed_gates']['complete_pass']],destroyed_ids=[r['state_id'] for r in rr if not r['recomputed_gates']['complete_pass'] and by[(r['state_id'],control)]['recomputed_gates']['complete_pass']])
    return result,pairs

def run(paths,out,c=None,runtime_factory=None,backend_factory=None,input_loader=None):
    c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);started=time.monotonic();stage='registered';errors=[];replay=[];zeros=[];warms=[];updates=[];records=[];seals={};data=None;backend=None;a=None
    calls=dict(model_forwards=0,initialisation_model_forwards=0,preflight_metric_attempts=0,preflight_metrics=0,path_metric_attempts=0,path_metrics=0)
    s=dict(protocol=PROTOCOL,complete=False,decision='C140_INCOMPLETE_FIXED_FINAL_NOT_QUALIFIED',synthetic_fixture=not c['production'],real_model_execution_claim=False,training_states=N,all36_used_for_training=True,independent_confirmation_states=0,confirmation_states_evaluated=0,excluded_input_states_evaluated=0,
        requested_training_steps=STEPS,requested_scalar_backward_calls=STEPS*N,requested_final_endpoint_records=216,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,candidate_selected=False,writer_deployed=False,best_checkpoint_selected=False,recovery_checkpoint_selected=False,thresholds_changed=False,parent_decision_reclassified=False,LC2_closed=False,LC2_single_fold_passed=False,
        new_states_prepared=0,new_model_features_acquired=0,archived_source_executed=False,teacher_calls=0,backtracking_steps=0,online_response_repair_steps=0,GPT2_optimizer_updates=0,LoRA_A_optimizer_updates=0,base_writer_updates=0,
        functional_zero_parity_complete_before_gradients=False,functional_warm_parity_complete_before_gradients=False,step12_frozen_before_final_evaluation=False,AD_accuracy_independently_certified=False,forward_error_bound_established=False,scope=design()['scope'])
    def budget():need(time.monotonic()-started<TIMEOUT,'C140 fixed50-minute worker budget reached')
    def checkpoint():
        stats=backend.stats() if backend is not None else {};s.update(stage=stage,elapsed_worker_seconds=time.monotonic()-started,runtime_calls=calls,new_model_forwards=calls['model_forwards'],gradient_stats=stats,new_backward_calls=stats.get('scalar_backward_calls',0),new_training_steps=len(updates),completed_training_steps=len(updates),new_head_fits=int(bool(updates)),preflight_records=len(replay),zero_parity_records=len(zeros),warm_parity_records=len(warms),final_endpoint_records=len(records))
        dump(out/'SUMMARY.json',dict(s,errors=errors))
        for n,x in (('PREFLIGHT_PROGRESS.json',replay),('ZERO_FUNCTIONAL_PROGRESS.json',zeros),('WARM_FUNCTIONAL_PROGRESS.json',warms),('TRAINING_PROGRESS.json',updates),('FINAL_ENDPOINT_PROGRESS.json',records)):dump(out/n,x)
    def seal(n):seals[n]=sha(out/n)
    def unchanged():
        for n,h in seals.items():need(sha(out/n)==h,'Frozen artifact changed: '+n)
        if data is not None:compare(frozen_digest(data),data['frozen_identites'],'Frozen shared basis/writer/training inputs')
    def check(m,old):
        q=safe.repeat_comparison(m,old);return q,bool(q['within_replay_tolerance'] and q['same_point_identity'])
    try:
        need(not (out/'SUMMARY.json').exists(),'Fresh C140 output required');dump(out/'DESIGN_FROZEN_BEFORE_MODEL.json',design());seal('DESIGN_FROZEN_BEFORE_MODEL.json');checkpoint();stage='eight_original_archive_bindings'
        data=(load if input_loader is None else input_loader)(paths,out,c);data['frozen_identites']=frozen_digest(data);budget()
        for n in ('INPUTS_FROZEN.json','ALL36_TRAINING_REGISTRY_FROZEN.json','ALL36_TRAINING_INPUTS.npz','MODEL_PINS_REFERENCE.json','FROZEN_SHARED_ROW_BASIS.npz','INITIAL_C137_STEP24_CONTROL.npz','FROZEN_ORIGINAL_WRITER.npz','FROZEN_ORIGINAL_CONTROL.npz','FROZEN_INPUT_IDENTITIES.json','C139_BOUND_RECEIPT.json'):
            need((out/n).is_file(),'Missing frozen binding '+n);seal(n)
        rank=len(data['V']);need(rank==36 if c['production'] else 0<rank<=36,'Declared shared-control dimension');a=data['initial_a'].copy();m=np.zeros_like(a);v=np.zeros_like(a);zero=np.zeros_like(a);farray(a,(rank,))
        need(set(data['points'])=={r['state_id'] for r in registry()},'All36 and only36 training states');s.update(trainable_shared_coordinates=rank,C139_parent_decision=data['C139_summary']['decision'],initialization='Exact C137 step24; zero Adam moments',training_response_safety=RESPONSE_SAFETY,acceptance_response_budget=BUDGET);phase=Phase(data);stage='model_initialisation';checkpoint();actual=runtime_factory is None and backend_factory is None and input_loader is None
        if runtime_factory is None:
            from runtime140 import factory
            runtime_factory=factory
        rt=runtime_factory(out/'model_runtime',data,phase,calls);s['real_model_execution_claim']=actual;compare({k:v for k,v in rt.pins.items() if k!='torch'},{k:v for k,v in data['modelpins'].items() if k!='torch'},'Pinned model identity');stage='all36_known_native_preflight';checkpoint();raw_native={};warm_native={}
        for sp in registry():
            budget();sid=sp['state_id'];p=data['points'][sid];rr=[]
            for kind,point,write in (('prepared',p['prepared'],False),('raw',p['raw'],True),('warm',proposal(p['prepared'],p['raw'],data['V'],a)[0],True)):
                phase.authorize((kind,sid),point,sp['case'],write);met=rt.metrics(point,sp['case'],write)
                if not write:need(safe.metric_gates(met,sp['case'],point,False),'Known training preparation failed; no deletion')
                else:parent.gate_record(met,sp['case'],point,p['prepared'])
                qs=[check(met,old) for old in data['native_refs'][(sid,kind)]];replay.append(dict(**sp,kind=kind,metrics=met,comparisons=[q for q,ok in qs]));checkpoint();need(all(ok for q,ok in qs),'Known replay drift blocks before gradients')
                if kind=='raw':raw_native[sid]=met
                if kind=='warm':warm_native[sid]=met
            print('C140_PREFLIGHT',sid,'OK',flush=True)
        unchanged();phase.begin_zero();stage='all36_zero_functional_parity'
        if backend_factory is None:
            from functional140 import factory
            backend_factory=factory
        backend=backend_factory(rt,data,phase,calls)
        for sp in registry():
            budget();sid=sp['state_id'];r=backend.zero(sp,zero);compare(r['state_id'],sid);compare(r['point_sha256'],safe.tensor_hash(data['points'][sid]['raw']));err=close(r['observed'],raw_native[sid]['observed'],'Zero functional/native parity',2e-6);ke=abs(r['KL']-raw_native[sid]['KL']);need(ke<=2e-7,'Zero KL parity');zeros.append(dict(r,observed_parity_Linf=err,KL_parity_absolute_error=ke));checkpoint()
        need(backend.stats()['scalar_backward_calls']==0,'No backward before complete zero parity');s['functional_zero_parity_complete_before_gradients']=True;phase.begin_warm();stage='all36_warm_functional_parity';checkpoint()
        for sp in registry():
            budget();sid=sp['state_id'];r=backend.zero(sp,a);compare(r['state_id'],sid);warm=proposal(data['points'][sid]['prepared'],data['points'][sid]['raw'],data['V'],a)[0];need(0<=r['point_parity_error_Linf']<=2e-6,'Warm point parity');err=close(r['observed'],warm_native[sid]['observed'],'Warm functional/native parity',2e-6);ke=abs(r['KL']-warm_native[sid]['KL']);need(ke<=2e-7,'Warm KL parity');warms.append(dict(r,native_point_sha256=safe.tensor_hash(warm),observed_parity_Linf=err,KL_parity_absolute_error=ke));checkpoint()
        need(backend.stats()['scalar_backward_calls']==0,'No backward before complete warm parity');backend.verify(rehash=True);unchanged();phase.begin_training();s['functional_warm_parity_complete_before_gradients']=True;stage='fixed12_full36_training';checkpoint()
        for step in range(STEPS):
            budget();phase.begin_batch(step,a);folder=out/'training'/f'step{step+1:02d}';g,rows=backend.gradient(a,step,folder,budget);farray(g,a.shape);need([r['state_id'] for r in rows]==[r['state_id'] for r in registry()],'All36 ordered state gradients required')
            need(all(r['coefficient_digest']==row_digest(a) for r in rows),'Wrong shared vector in gradient');gg,agg=aggregate([r['objective'] for r in rows],[r['state_gradient'] for r in rows]);close(g,gg,'Full36 weighted gradient',1e-12)
            for sp,r,w in zip(registry(),rows,agg['weights']):
                compare(r['case'],sp['case']);compare(r['C140_membership'],'training');point=proposal(data['points'][sp['state_id']]['prepared'],data['points'][sp['state_id']]['raw'],data['V'],a)[0];compare(r['objective_parts'],objective(r['observed'],r['KL'],point,data['points'][sp['state_id']]['prepared'],a,sp['case']),atol=1e-10,rtol=1e-9);compare(r['objective'],sum(r['objective_parts'].values()),atol=1e-10,rtol=1e-9);compare(r['aggregate_weight'],w);close(r['gradient_contribution'],w*np.array(r['state_gradient']),'Weighted gradient row',1e-12)
            phase.end_batch();new,mm,vv,info=adam_step(a,g,m,v,step+1);name=f'training/step{step+1:02d}/UPDATE.npz';np.savez_compressed(out/name,before_a=a,before_m=m,before_v=v,full_weighted_gradient=g,after_a=new,after_m=mm,after_v=vv);dump(folder/'OBSERVATIONS.json',rows);dump(folder/'AGGREGATION.json',agg)
            updates.append(dict(**info,aggregate_objective_before_update=agg['objective'],state_weights=agg['weights'],before_digest=row_digest(a),after_digest=row_digest(new),mean_objective_before_update=agg['mean_state_loss'],maximum_objective_before_update=agg['maximum_state_loss'],update_file=name,update_sha256=sha(out/name),n_training_records=N));a,m,v=new,mm,vv;np.savez_compressed(out/'LATEST_PARTIAL_CONTROL.npz',a=a,m=m,v=v,completed_steps=np.array([step+1],np.int64));unchanged();checkpoint();print('C140_FIXED_UPDATE',step+1,'/12 max_loss',agg['maximum_state_loss'],'mean_loss',agg['mean_state_loss'],'gradient_L2',info['gradient_L2'],flush=True)
        need(backend.stats()['scalar_backward_calls']==432 and backend.stats()['full36_gradient_batches']==STEPS,'Fixed gradient budget differs');backend.verify(rehash=True);unchanged();stage='freeze_only_step12';checkpoint();np.savez_compressed(out/'FINAL_SHARED_CONTROL_STEP12.npz',a=a);seal('FINAL_SHARED_CONTROL_STEP12.npz');points={};arrays={};outcomes=[]
        for sp in registry():
            sid=sp['state_id'];p=data['points'][sid];trained,diag=proposal(p['prepared'],p['raw'],data['V'],a);warm,wd=proposal(p['prepared'],p['raw'],data['V'],data['initial_a']);close(backend.proposed(a,sid),trained,'Final NumPy/Torch map parity',2e-6)
            for arm,point,dd in ((EVAL_ARMS[0],p['raw'].copy(),dict(original_raw_bytes_retained=True)),(EVAL_ARMS[1],warm,wd),(EVAL_ARMS[2],trained,diag)):
                points[(sid,arm)]=point;arrays[sid+'__'+arm]=point;outcomes.append(dict(**sp,arm=arm,point_sha256=safe.tensor_hash(point),proposal_diagnostics=dd))
        np.savez_compressed(out/'ALL108_FROZEN_FINAL_POINTS.npz',**arrays);dump(out/'ALL108_FROZEN_FINAL_OUTCOMES.json',outcomes)
        for n in ('ALL108_FROZEN_FINAL_POINTS.npz','ALL108_FROZEN_FINAL_OUTCOMES.json'):seal(n)
        dump(out/'STEP12_FREEZE_BEFORE_EVALUATION.json',dict(completed_steps=STEPS,control_sha256=sha(out/'FINAL_SHARED_CONTROL_STEP12.npz'),control_digest=row_digest(a),proposal_seals=dict(seals),checkpoint_choice='Fixed C140 step12 only',no_best_or_recovery_selection=True));seal('STEP12_FREEZE_BEFORE_EVALUATION.json');phase.freeze(points);s['step12_frozen_before_final_evaluation']=True;stage='two_round216_native_training_endpoints';checkpoint();first={};drifts=[];repeats=[]
        for rnd in (0,1):
            for sp in registry():
                sid=sp['state_id'];p=data['points'][sid]
                for arm in EVAL_ARMS:
                    budget();point=points[(sid,arm)];phase.authorize((rnd,sid,arm),point,sp['case'],True);met=rt.metrics(point,sp['case'],True);g=parent.gate_record(met,sp['case'],point,p['prepared']);aa=zero if arm==EVAL_ARMS[0] else (data['initial_a'] if arm==EVAL_ARMS[1] else a);parts=objective(met['observed'],met['KL'],point,p['prepared'],aa,sp['case']);r=dict(**sp,round=rnd,arm=arm,metrics=met,recomputed_gates=g,training_objective_parts=parts,training_objective=sum(parts.values()))
                    if arm in EVAL_ARMS[:2]:
                        q,ok=check(met,raw_native[sid] if arm==EVAL_ARMS[0] else warm_native[sid]);r['native_control_replay']=q
                        if not ok:drifts.append(dict(state_id=sid,arm=arm,round=rnd,comparison=q))
                    if rnd==0:first[(sid,arm)]=met
                    else:
                        q,ok=check(met,first[(sid,arm)]);r['repeat_comparison']=q
                        if not ok:repeats.append(dict(state_id=sid,arm=arm,comparison=q))
                    records.append(r);dump(out/'final_measurements'/f'record{len(records)-1:03d}.json',r);checkpoint()
                print('C140_FINAL',rnd,sid,[(r['arm'],r['recomputed_gates']['complete_pass'],r['metrics']['response_drift']) for r in records[-3:]],flush=True)
        phase.finish();unchanged();backend.verify(rehash=True,check_chart=False);need(calls['preflight_metrics']==108 and calls['path_metrics']==216,'Fixed native denominator differs');need(calls['model_forwards']==calls['initialisation_model_forwards']+2*(108+backend.stats()['functional_endpoint_calls']+216),'Model forward accounting differs')
        sums,pairs=final_summaries(records);qualified=not drifts and not repeats;passed=qualified and all(sums['all36'][str(j)][EVAL_ARMS[2]]['passed']==N for j in (0,1))
        for n,x in (('ALL216_FINAL_TRAINING_ENDPOINTS.json',records),('FINAL_TRAINING_SUMMARIES.json',sums),('PAIRED_VS_RAW_AND_C137.json',pairs),('CONTROL_REPLAY_FAILURES.json',drifts),('REPEATABILITY_FAILURES.json',repeats)):dump(out/n,x)
        s.update(complete=True,decision=('C140_FIXED_FINAL_COMPLETE_REPLAY_DRIFT_NO_QUALIFICATION' if not qualified else ('C140_FIXED_FINAL_ALL36_TRAINING_GATES_PASSED' if passed else 'C140_FIXED_FINAL_TRAINING_CONSTRAINT_FAILURE')),fixed_final_training_qualification_passed=bool(passed),repeatable=not repeats,controls_replay_consistent=not drifts,final_training_summaries=sums,paired_vs_raw_and_C137=pairs,next_branch=('RESOLVE_REPLAY_DRIFT' if not qualified else ('PREREGISTER_SEPARATE_UNSEEN_LOCAL_CONFIRMATION' if passed else 'STOP_FIXED12_SCHEDULE_NO_CHECKPOINT_SUBSTITUTION')))
        report=['# C140：36态共享控制的固定训练','','原12中心与C139全部24态均为训练数据；一个共享向量，从冻结C137第24步开始，重置Adam矩，只验收新第12步。响应训练安全线0.00125，原验收线0.002保持不变。','','|臂|首轮通过/请求|重复轮通过/请求|最大响应漂移|','|---|---:|---:|---:|']
        for arm in EVAL_ARMS:report.append(f"|{arm}|{sums['all36']['0'][arm]['passed']}/36|{sums['all36']['1'][arm]['passed']}/36|{sums['all36']['0'][arm]['maximum_response_drift']}|")
        report+=['','最终判定：'+s['decision'],'后续分支：'+s['next_branch'],'','所有36态固定分母；不按失败态单独拟合，不选中途/最佳/恢复检查点，不删失败，不放宽门槛。无新Jacobian/SVD/求解/教师/回溯/在线修复。离线反传只更新共享控制，原writer/V/主干/A/特征与归一化固定。','','本次完整通过也只建立训练资格；需要另行冻结并做全新局部确认，再执行隔离的排除输入协议。LC2和单折均未闭合。历史C137/C138/C139判定不改写。AD不是独立数值导数精度证明。'];(out/'REPORT_ZH.md').write_text('\n'.join(report)+'\n');stage='complete'
    except BaseException:errors.append(traceback.format_exc());s.update(complete=False,decision='C140_INCOMPLETE_FIXED_FINAL_NOT_QUALIFIED');print(errors[-1],flush=True)
    finally:
        checks={}
        for k,p in paths.items():
            try:checks[k]=sha(p)==c['pins'][k]['sha256']
            except Exception:checks[k]=False
        s['original_inputs_unchanged']=checks
        if not all(checks.values()):s.update(complete=False,decision='C140_INCOMPLETE_FIXED_FINAL_NOT_QUALIFIED');errors.append('Original input changed')
        checkpoint()
    return dict(s,errors=errors)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in INPUT_KEYS:p.add_argument('--'+k.lower(),required=True)
    p.add_argument('--out',required=True);args=p.parse_args();s=run({k:Path(getattr(args,k.lower())) for k in INPUT_KEYS},args.out);print(json.dumps(s,ensure_ascii=False,indent=2),flush=True)
    if not s['complete']:raise SystemExit(1)
