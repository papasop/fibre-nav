"""Exactly sixteen full12 offline updates; only the final vector is qualified."""
import argparse,json,traceback
from collections import Counter
from core136 import *
from inputs136 import load

def final_summaries(records):
    summaries={};pairs={}
    for role in ('all12','geometry_origin_training6','historical_probe_origin_training6'):
        summaries[role]={};pairs[role]={}
        for rnd in (0,1):
            subset=[r for r in records if r['round']==rnd and (role=='all12' or r['role']==('geometry_training' if role=='geometry_origin_training6' else 'historical_seen_center_probe'))]
            raw={r['state_id']:r for r in subset if r['arm']==EVAL_ARMS[0]};summaries[role][str(rnd)]={};pairs[role][str(rnd)]={}
            for arm in EVAL_ARMS:
                q=[r for r in subset if r['arm']==arm];bad=Counter(g for r in q for g in r['recomputed_gates']['failed_gates'])
                summaries[role][str(rnd)][arm]=dict(n_requested=12 if role=='all12' else 6,n_evaluated=len(q),passed=sum(r['recomputed_gates']['complete_pass'] for r in q),maximum_response_drift=max((r['metrics']['response_drift'] for r in q),default=None),failed_gates={k:bad[k] for k in ('preparation','bits_match','output_flip','unselected_preserved','response','KL','exact_read_margin','development_read_margin','proposal_radius')},failed_ids=[r['state_id'] for r in q if not r['recomputed_gates']['complete_pass']],mean_training_objective=float(np.mean([r['training_objective'] for r in q])) if q else None)
                pairs[role][str(rnd)][arm]=dict(rescued_ids=[r['state_id'] for r in q if r['recomputed_gates']['complete_pass'] and not raw[r['state_id']]['recomputed_gates']['complete_pass']],destroyed_ids=[r['state_id'] for r in q if not r['recomputed_gates']['complete_pass'] and raw[r['state_id']]['recomputed_gates']['complete_pass']])
    return summaries,pairs

def run(paths,out,c=None,runtime_factory=None,backend_factory=None):
    c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);started=time.monotonic();stage='registered';errors=[];replay=[];zeros=[];updates=[];records=[];seals={};backend=None;a=None;real=runtime_factory is None and backend_factory is None
    calls=dict(model_forwards=0,initialisation_model_forwards=0,preflight_metric_attempts=0,preflight_metrics=0,path_metric_attempts=0,path_metrics=0)
    s=dict(protocol=PROTOCOL,complete=False,decision='C136_INCOMPLETE_FIXED_FINAL_NOT_QUALIFIED',synthetic_fixture=not c['production'],real_model_execution_claim=False,training_states=12,all12_used_for_training=True,independent_confirmation_states=0,
        requested_training_steps=16,completed_training_steps=0,requested_scalar_backward_calls=192,requested_final_endpoint_records=48,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,
        candidate_selected=False,writer_deployed=False,best_checkpoint_selected=False,recovery_checkpoint_selected=False,thresholds_changed=False,LC2_closed=False,LC2_single_fold_passed=False,parent_decision_reclassified=False,
        excluded_input_states_evaluated=0,confirmation_states_evaluated=0,new_states_prepared=0,archived_source_executed=False,teacher_calls=0,backtracking_steps=0,online_response_repair_steps=0,
        GPT2_optimizer_updates=0,LoRA_A_optimizer_updates=0,base_writer_updates=0,functional_zero_parity_complete_before_gradients=False,step16_frozen_before_final_evaluation=False,
        AD_accuracy_independently_certified=False,forward_error_bound_established=False,
        scope='Offline finite-endpoint fitting of one shared bounded vector on12 historical training states. Both former role groups are training, not independent validation. Only fixed update16 evaluated; no prospective generalisation or LC2 acceptance claim.')
    def budget():need(time.monotonic()-started<TIMEOUT,'C136 fixed50-minute worker budget reached')
    def checkpoint():
        stats=backend.stats() if backend is not None else {};s.update(stage=stage,elapsed_worker_seconds=time.monotonic()-started,runtime_calls=calls,new_model_forwards=calls['model_forwards'],gradient_stats=stats,new_backward_calls=stats.get('scalar_backward_calls',0),completed_training_steps=len(updates),new_training_steps=len(updates),new_head_fits=int(bool(updates)),final_endpoint_records=len(records))
        dump(out/'SUMMARY.json',dict(s,errors=errors));dump(out/'PREFLIGHT_PROGRESS.json',replay);dump(out/'ZERO_FUNCTIONAL_PROGRESS.json',zeros);dump(out/'TRAINING_PROGRESS.json',updates);dump(out/'FINAL_ENDPOINT_PROGRESS.json',records)
    def seal(name):seals[name]=sha(out/name)
    def unchanged():
        for n,h in seals.items():need(sha(out/n)==h,'Frozen C136 artifact changed: '+n)
    def check(m,old):
        q=safe.repeat_comparison(m,old);return q,bool(q['within_replay_tolerance'] and q['same_point_identity'])
    try:
        need(not (out/'SUMMARY.json').exists(),'Fresh C136 output required');dump(out/'DESIGN_FROZEN_BEFORE_MODEL.json',design());seal('DESIGN_FROZEN_BEFORE_MODEL.json');checkpoint()
        stage='four_original_archive_bindings';data=load(paths,out,c);budget()
        for n in ('INPUTS_FROZEN.json','ALL12_TRAINING_REGISTRY_FROZEN.json','MODEL_PINS_REFERENCE.json','FROZEN_SHARED_ROW_BASIS.npz'):seal(n)
        rank=len(data['V']);need(rank>0 and (rank==36 if c['production'] else rank<=36),'Declared shared vector dimension differs');a=np.zeros(rank,np.float64);m=np.zeros_like(a);v=np.zeros_like(a);np.savez_compressed(out/'INITIAL_ZERO_CONTROL.npz',a=a);seal('INITIAL_ZERO_CONTROL.npz')
        s['trainable_shared_coordinates']=rank;phase=Phase(data);stage='model_initialisation';checkpoint()
        if runtime_factory is None:
            from runtime134 import factory
            runtime_factory=factory
        rt=runtime_factory(out/'model_runtime',data,phase,calls);s['real_model_execution_claim']=real
        compare({k:v for k,v in rt.pins.items() if k!='torch'},{k:v for k,v in data['modelpins'].items() if k!='torch'},'Pinned model/chart identity')
        stage='all12_known_point_preflight';checkpoint();fresh_raw={};archived={(r['state_id'],r['arm'],r['round']):r['metrics'] for r in data['records'] if r['status']=='EVALUATED'}
        for spec in registry():
            budget();sid=spec['state_id'];p=data['points'][sid];case=spec['case'];theta=p['prepared'];phase.authorize(('prepared',sid),theta,case,False);met=rt.metrics(theta,case,False);need(safe.metric_gates(met,case,theta,False),'Preparation gate failed')
            q,valid=check(met,p['preparation']);replay.append(dict(**spec,C136_membership='training',kind='prepared',metrics=met,comparisons=[q]));checkpoint();need(valid,'Preparation replay drift blocks before gradients')
            for arm in ARMS[:2]:
                point=data['maps'][(sid,arm)]['point'];phase.authorize(('endpoint',sid,arm),point,case,True);met=rt.metrics(point,case,True);gate_record(met,case,point,theta);refs=[archived[(sid,arm,k)] for k in (0,1)]+[data['C135_controls'][(sid,arm,k)] for k in (0,1)];checks=[check(met,old) for old in refs]
                replay.append(dict(**spec,C136_membership='training',kind='endpoint',arm=arm,metrics=met,comparisons=[q[0] for q in checks]));checkpoint();need(all(q[1] for q in checks),'Endpoint replay drift blocks before gradients')
                if arm==ARMS[0]:fresh_raw[sid]=met
            print('C136_PREFLIGHT',sid,'OK',flush=True)
        unchanged();phase.begin_zero();stage='zero_control_functional_parity'
        if backend_factory is None:
            from functional136 import factory
            backend_factory=factory
        backend=backend_factory(rt,data,phase,calls)
        for spec in registry():
            budget();row=backend.zero(spec,a);need(row['state_id']==spec['state_id'],'Wrong functional zero state');need(row['point_sha256']==safe.tensor_hash(data['points'][spec['state_id']]['raw']),'Zero functional point identity differs');err=close(row['observed'],fresh_raw[spec['state_id']]['observed'],'Functional zero observed parity',2e-6);klerr=abs(row['KL']-fresh_raw[spec['state_id']]['KL']);need(klerr<=2e-7,'Functional zero KL parity differs');zeros.append(dict(row,observed_parity_Linf=err,KL_parity_absolute_error=klerr));checkpoint()
        need(backend.stats()['scalar_backward_calls']==0,'No gradients allowed before complete zero parity');backend.verify(rehash=True);phase.begin_training();s['functional_zero_parity_complete_before_gradients']=True;stage='fixed16_full12_training';checkpoint()
        for step in range(STEPS):
            budget();phase.begin_batch(step,a);g,rows=backend.gradient(a,step,out/'training'/f'step{step+1:02d}',budget);farray(g,a.shape);need([r['state_id'] for r in rows]==[r['state_id'] for r in registry()],'All12 ordered gradient records required')
            need(all(r['coefficient_digest']==row_digest(a) for r in rows),'Gradient evaluated at wrong shared vector');close(np.sum([r['gradient_contribution'] for r in rows],axis=0),g,'Full gradient contribution sum',1e-12);phase.end_batch()
            new,mm,vv,info=adam_step(a,g,m,v,step+1);name=f'training/step{step+1:02d}/UPDATE.npz';np.savez_compressed(out/name,before_a=a,before_m=m,before_v=v,full_mean_gradient=g,after_a=new,after_m=mm,after_v=vv)
            dump(out/'training'/f'step{step+1:02d}'/'OBSERVATIONS.json',rows);update=dict(**info,before_digest=row_digest(a),after_digest=row_digest(new),mean_objective_before_update=float(np.mean([r['objective'] for r in rows])),maximum_objective_before_update=float(max(r['objective'] for r in rows)),update_file=name,update_sha256=sha(out/name),n_training_records=12)
            a,m,v=new,mm,vv;updates.append(update);np.savez_compressed(out/'LATEST_PARTIAL_CONTROL.npz',a=a,m=m,v=v,completed_steps=np.array([step+1],np.int64));checkpoint();print('C136_FIXED_UPDATE',step+1,'/16 mean_loss',update['mean_objective_before_update'],'gradient_L2',info['gradient_L2'],'control_Linf',float(max(abs(a))),flush=True)
        need(backend.stats()['scalar_backward_calls']==192 and backend.stats()['full12_gradient_batches']==16,'Fixed fullbatch gradient budget differs');backend.verify(rehash=True);unchanged();stage='freezing_only_step16';checkpoint()
        np.savez_compressed(out/'FINAL_SHARED_CONTROL_STEP16.npz',a=a);seal('FINAL_SHARED_CONTROL_STEP16.npz');points={};arrays={};outcomes=[]
        for spec in registry():
            sid=spec['state_id'];p=data['points'][sid];trained,diag=proposal(p['prepared'],p['raw'],data['V'],a);close(backend.proposed(a,sid),trained,'Frozen NumPy/Torch step16 map parity',2e-6)
            for arm,point in ((EVAL_ARMS[0],p['raw'].copy()),(EVAL_ARMS[1],trained)):
                points[(sid,arm)]=point;arrays[sid+'__'+arm]=point;outcomes.append(dict(**spec,C136_membership='training',independent_validation_eligible=False,arm=arm,point_sha256=safe.tensor_hash(point),proposal_diagnostics=diag if arm==EVAL_ARMS[1] else {'original_raw_bytes_retained':True}))
        np.savez_compressed(out/'ALL24_FROZEN_FINAL_POINTS.npz',**arrays);dump(out/'ALL24_FROZEN_FINAL_OUTCOMES.json',outcomes)
        for n in ('ALL24_FROZEN_FINAL_POINTS.npz','ALL24_FROZEN_FINAL_OUTCOMES.json'):seal(n)
        dump(out/'STEP16_FREEZE_BEFORE_EVALUATION.json',dict(completed_steps=16,control_sha256=sha(out/'FINAL_SHARED_CONTROL_STEP16.npz'),control_digest=row_digest(a),basis_sha256=sha(out/'FROZEN_SHARED_ROW_BASIS.npz'),training_registry_sha256=sha(out/'ALL12_TRAINING_REGISTRY_FROZEN.json'),proposal_seals=seals,checkpoint_choice='Fixed step16 only',no_best_or_recovery_selection=True));phase.freeze(points);s['step16_frozen_before_final_evaluation']=True;stage='two_round_final_training_qualification';checkpoint();first={};control_failures=[];repeat_failures=[]
        for rnd in (0,1):
            for spec in registry():
                sid=spec['state_id'];p=data['points'][sid]
                for arm in EVAL_ARMS:
                    budget();point=points[(sid,arm)];phase.authorize((rnd,sid,arm),point,spec['case'],True);met=rt.metrics(point,spec['case'],True);g=gate_record(met,spec['case'],point,p['prepared']);aa=np.zeros_like(a) if arm==EVAL_ARMS[0] else a;parts=objective(met['observed'],met['KL'],point,p['prepared'],aa,spec['case']);r=dict(**spec,C136_membership='training',independent_validation_eligible=False,arm=arm,round=rnd,metrics=met,recomputed_gates=g,training_objective_parts=parts,training_objective=sum(parts.values()))
                    if arm==EVAL_ARMS[0]:
                        comp,valid=check(met,fresh_raw[sid]);r['raw_control_replay']=comp
                        if not valid:control_failures.append(dict(state_id=sid,round=rnd,comparison=comp))
                    key=(sid,arm)
                    if rnd==0:first[key]=met
                    else:
                        comp,valid=check(met,first[key]);r['repeat_comparison']=comp
                        if not valid:repeat_failures.append(dict(state_id=sid,arm=arm,comparison=comp))
                    records.append(r);dump(out/'final_measurements'/f'record{len(records)-1:02d}.json',r);checkpoint()
                print('C136_FINAL',rnd,sid,[(r['arm'],r['recomputed_gates']['complete_pass'],r['metrics']['response_drift']) for r in records[-2:]],flush=True)
        phase.finish();unchanged();backend.verify(rehash=True,check_chart=False);need(len(records)==48 and calls['preflight_metrics']==36 and calls['path_metrics']==48,'Complete native endpoint registry required')
        need(calls['model_forwards']==calls['initialisation_model_forwards']+2*(36+backend.stats()['functional_endpoint_calls']+48),'Model forward counter differs');summaries,pairs=final_summaries(records);qualified=not control_failures and not repeat_failures;passed=qualified and all(summaries['all12'][str(rnd)][EVAL_ARMS[1]]['passed']==12 for rnd in (0,1))
        dump(out/'ALL48_FINAL_TRAINING_ENDPOINTS.json',records);dump(out/'FINAL_TRAINING_SUMMARIES.json',summaries);dump(out/'PAIRED_VS_RAW.json',pairs);dump(out/'RAW_CONTROL_FAILURES.json',control_failures);dump(out/'REPEATABILITY_FAILURES.json',repeat_failures)
        s.update(complete=True,decision=('C136_FIXED_FINAL_COMPLETE_REPLAY_DRIFT_NO_QUALIFICATION' if not qualified else ('C136_FIXED_FINAL_ALL12_TRAINING_GATES_PASSED' if passed else 'C136_FIXED_FINAL_TRAINING_CONSTRAINT_FAILURE')),
            fixed_final_training_qualification_passed=bool(passed),repeatable=not repeat_failures,raw_control_replay_consistent=not control_failures,final_training_summaries=summaries,
            next_branch=('RESOLVE_REPLAY_DRIFT' if not qualified else ('PREREGISTER_SEPARATE_NEW_STATE_CONFIRMATION_OF_FROZEN_STEP16' if passed else 'STOP_THIS_FIXED_SCHEDULE_REASSESS_SHARED_PARAMETERISATION')))
        report=['# C136：共享低维向量的有限端点训练','','12态全部是本次训练数据；原“探测6态”仅保留来源标签。一个共享向量，零初始化，固定16步全批次Adam，只验收第16步。','','|来源组（均为训练）|臂|首轮通过/请求|重复轮通过/请求|','|---|---|---:|---:|']
        for role in summaries:
            for arm in EVAL_ARMS:report.append(f"|{role}|{arm}|{summaries[role]['0'][arm]['passed']}/{summaries[role]['0'][arm]['n_requested']}|{summaries[role]['1'][arm]['passed']}/{summaries[role]['1'][arm]['n_requested']}|")
        report+=['','最终判定：'+s['decision'],'后续分支：'+s['next_branch'],'','训练前逐态验证零控制点身份及功能前向一致性。训练通过临时LoRA-B替换求共享控制向量梯度；主干、LoRA-A及原写入器不优化。没有新Jacobian/SVD、教师、回溯或在线修复。','','控制坐标逐步投影到[-1/8,1/8]；提案不做径向裁剪。更新半径超标会进入训练损失并在最终门槛失败，不能被隐藏。只采用固定第16步，不用中途或恢复检查点替代。','','所有通过均为训练资格，不构成新态确认、排除输入泛化、LC2或单折通过。AD遵循float64控制到float32提交的计算图，不是有限精度导数准确性的独立证明；本归档也不能独立证明非访问或测量时序。']
        (out/'REPORT_ZH.md').write_text('\n'.join(report)+'\n',encoding='utf-8');stage='complete'
    except BaseException:errors.append(traceback.format_exc());s.update(complete=False,decision='C136_INCOMPLETE_FIXED_FINAL_NOT_QUALIFIED');print(errors[-1],flush=True)
    finally:
        checks={}
        for k,p in paths.items():
            try:checks[k]=sha(p)==c['pins'][k]['sha256']
            except Exception:checks[k]=False
        s['original_inputs_unchanged']=checks
        if not all(checks.values()):s.update(complete=False,decision='C136_INCOMPLETE_FIXED_FINAL_NOT_QUALIFIED');errors.append('Original input changed')
        checkpoint();print(json.dumps(dict(s,errors=errors),ensure_ascii=False,indent=2),flush=True)
    return dict(s,errors=errors)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in INPUT_KEYS:p.add_argument('--'+k.lower(),required=True)
    p.add_argument('--out',required=True);a=p.parse_args();s=run({k:Path(getattr(a,k.lower())) for k in INPUT_KEYS},a.out);sys.exit(0 if s['complete'] else 1)
