"""Real CPU model observations on an entirely frozen 126-point path registry."""
import argparse,json,traceback
from core134 import *
from inputs134 import load

def run(paths,out,c=None,runtime_factory=None):
    c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);started=time.monotonic();errors=[];stage='registered';records=[];components={};replay=[];rt=None;seals={};real=runtime_factory is None
    calls=dict(model_forwards=0,initialisation_model_forwards=0,preflight_metric_attempts=0,preflight_metrics=0,path_metric_attempts=0,path_metrics=0)
    s=dict(protocol=PROTOCOL,complete=False,decision='C134_INCOMPLETE_NO_CANDIDATE',synthetic_fixture=not c['production'],real_model_execution_claim=False,requested_states=6,path_points_requested=126,path_measurements_requested=252,
        new_Jacobians=0,new_SVD_calls=0,new_backward_calls=0,new_training_steps=0,new_head_fits=0,linear_systems_solved=0,new_states_prepared=0,excluded_input_states_evaluated=0,confirmation_states_evaluated=0,
        candidate_selected=False,step_selected=False,writer_deployed=False,thresholds_changed=False,LC2_closed=False,LC2_single_fold_passed=False,parent_decision_reclassified=False,
        archived_source_executed=False,postwrite_response_repair_steps=0,backtracking_steps=0,teacher_calls=0,all_path_points_frozen_before_model=False,preflight_completed_before_paths=False,
        AD_accuracy_independently_verified=False,physical_curvature_cause_established=False,forward_error_bound_established=False,
        scope='Fixed historical geometry-training centers only; newly measured CPU GPT2 paths with archived C132 derivatives. All fractions descriptive, none selected. No fresh-state, excluded-input or LC2 acceptance claim.')
    def checkpoint():
        s.update(stage=stage,path_measurement_records=len(records),elapsed_worker_seconds=time.monotonic()-started,runtime_calls=calls,new_model_forwards=calls['model_forwards'])
        dump(out/'SUMMARY.json',dict(s,errors=errors));dump(out/'PATH_PROGRESS.json',records);dump(out/'PREFLIGHT_PROGRESS.json',replay)
    def budget():need(time.monotonic()-started<TIMEOUT,'C134 fixed50-minute budget reached')
    def seal(name):seals[name]=sha(out/name)
    def unchanged():
        for name,h in seals.items():need(sha(out/name)==h,'Frozen C134 artifact changed: '+name)
    def check_replay(m,old):
        comparison=safe.repeat_comparison(m,old);return comparison,bool(comparison['within_replay_tolerance'] and comparison['same_point_identity'])
    try:
        need(not (out/'SUMMARY.json').exists(),'Fresh C134 output required');dump(out/'DESIGN_FROZEN_BEFORE_MODEL.json',design());seal('DESIGN_FROZEN_BEFORE_MODEL.json');checkpoint()
        stage='two_original_archive_bindings';data=load(paths,out,c);budget();seal('INPUTS_FROZEN.json');seal('FROZEN_MODEL_PINS_REFERENCE.json')
        stage='freezing_all_path_outcomes';arrays,outcomes,maps=freeze_paths(data);np.savez_compressed(out/'ALL_AVAILABLE_PATH_POINTS.npz',**arrays);dump(out/'ALL126_PATH_OUTCOMES_FROZEN.json',outcomes)
        for n in ('ALL_AVAILABLE_PATH_POINTS.npz','ALL126_PATH_OUTCOMES_FROZEN.json'):seal(n)
        dump(out/'POINT_SEALS_BEFORE_MODEL.json',seals);s['all_path_points_frozen_before_model']=True;phase=Phase(data,maps);checkpoint()
        stage='model_initialisation'
        if runtime_factory is None:
            from runtime134 import factory
            runtime_factory=factory
        rt=runtime_factory(out/'model_runtime',data,phase,calls);s['real_model_execution_claim']=real
        compare({k:v for k,v in rt.pins.items() if k!='torch'},{k:v for k,v in data['modelpins'].items() if k!='torch'},'Pinned model/chart identity')
        dump(out/'MODEL_PIN_COMPARISON.json',dict(passed=True,torch_runtime=rt.pins.get('torch'),torch_archived=data['modelpins'].get('torch'),torch_version_excluded_only=True,known_point_replay_still_required=True))
        stage='fixed_known_point_preflight';checkpoint();fresh_base={};archived_end={(r['state_id'],r['arm'],r['round']):r['metrics'] for r in data['records'] if r['status']=='EVALUATED'}
        for spec in registry():
            sid=spec['state_id'];case=spec['case'];p=data['points'][sid];theta=p['prepared'];budget();phase.authorize(('prepared',sid),theta,case,False);m=rt.metrics(theta,case,False);need(safe.metric_gates(m,case,theta,False),'Prepared-state replay invalid')
            check,valid=check_replay(m,p['preparation']);replay.append(dict(**spec,kind='prepared',metrics=m,comparison=check));checkpoint();need(valid,'Prepared replay drift blocks before path measurements');fresh_base[sid]=m
            for arm in ARMS:
                q=data['maps'][(sid,arm)]['point']
                if q is None:continue
                budget();phase.authorize(('endpoint',sid,arm),q,case,True);m=rt.metrics(q,case,True);gate_record(m,case,q,theta);checks=[check_replay(m,archived_end[(sid,arm,rnd)]) for rnd in (0,1)]
                replay.append(dict(**spec,kind='endpoint',arm=arm,metrics=m,comparisons=[v[0] for v in checks]));checkpoint();need(all(v[1] for v in checks),'Endpoint replay drift blocks before path measurements')
            print('C134_PREFLIGHT',sid,'OK',flush=True)
        unchanged();phase.begin_paths();s['preflight_completed_before_paths']=True;stage='fixed_two_round_path_measurements';checkpoint();first={};repeat_failures=[];known_failures=[]
        for rnd in (0,1):
            for row in outcomes:
                budget();sid=row['state_id'];arm=row['arm'];step=row['step'];alpha=row['alpha'];point=maps[(sid,arm,step)];p=data['points'][sid];base=dict(**row,round=rnd)
                if point is None:rec=dict(**base,status='UNAVAILABLE_COMMON_READ_OPERATOR',metrics=None,recomputed_gates=None,statistics=None)
                else:
                    phase.authorize((rnd,sid,arm,step),point,row['case'],alpha>0);m=rt.metrics(point,row['case'],alpha>0);g=path_gates(m,row['case'],point,p['prepared'],alpha)
                    vals,stats=analyse(data['jacobians'][sid],p['prepared'],data['maps'][(sid,arm)]['point'],point,alpha,fresh_base[sid]['observed'][:10],m['observed'][:10]);prefix=f'record{len(records):03d}';components.update({prefix+'__'+k:v for k,v in vals.items()})
                    rec=dict(**base,status='EVALUATED',metrics=m,recomputed_gates=g,statistics=stats,array_prefix=prefix)
                    if alpha in (0.,1.):
                        reference=p['preparation'] if alpha==0. else archived_end[(sid,arm,0)];comp,valid=check_replay(m,reference);rec['known_point_replay']=comp
                        if not valid:known_failures.append(dict(state_id=sid,arm=arm,step=step,round=rnd,comparison=comp))
                    key=(sid,arm,step)
                    if rnd==0:first[key]=m
                    else:
                        comp,valid=check_replay(m,first[key]);rec['repeat_comparison']=comp
                        if not valid:repeat_failures.append(dict(state_id=sid,arm=arm,step=step,comparison=comp))
                records.append(rec);dump(out/'measurements'/f'record{len(records)-1:03d}.json',rec);checkpoint()
                if step==len(ALPHAS)-1:
                    evaluated=[r for r in records[-len(ALPHAS):] if r['status']=='EVALUATED'];print('C134_CURVE',rnd,sid,arm,'response_Linf',[round(r['statistics']['response_drift_Linf'],8) for r in evaluated],flush=True)
        phase.finish();unchanged();need(len(records)==252,'All252 requested records required');evaluated=sum(r['status']=='EVALUATED' for r in records);need(calls['path_metrics']==evaluated,'Actual path metric count differs')
        need(calls['preflight_metrics']==len(phase.preflight),'Preflight count differs');need(calls['model_forwards']==calls['initialisation_model_forwards']+2*(calls['preflight_metrics']+evaluated),'Model forward counter differs')
        np.savez_compressed(out/'PATH_OBSERVABLE_COMPONENTS.npz',**components);dump(out/'ALL252_PATH_MEASUREMENTS.json',records);dump(out/'REPEATABILITY_FAILURES.json',repeat_failures);dump(out/'KNOWN_POINT_REPLAY_FAILURES.json',known_failures)
        summaries={};curves=[]
        for rnd in (0,1):
            summaries[str(rnd)]={}
            for arm in ARMS:
                summaries[str(rnd)][arm]=[]
                for step,alpha in enumerate(ALPHAS):
                    q=[r for r in records if r['round']==rnd and r['arm']==arm and r['step']==step];ev=[r for r in q if r['status']=='EVALUATED']
                    summaries[str(rnd)][arm].append(dict(step=step,alpha=alpha,evaluation_target='prepared_state' if alpha==0. else 'requested_write',n_requested=6,n_evaluated=len(ev),n_unavailable=6-len(ev),response_passed=sum(r['statistics']['finite_response_within_budget'] for r in ev),all_declared_gates_passed=sum(r['recomputed_gates']['complete_pass'] for r in ev),maximum_response_drift=max((r['statistics']['response_drift_Linf'] for r in ev),default=None),maximum_response_remainder=max((r['statistics']['response_remainder_Linf'] for r in ev),default=None)))
                for spec in registry():
                    q=[r for r in records if r['round']==rnd and r['arm']==arm and r['state_id']==spec['state_id'] and r['status']=='EVALUATED'];departures=[r['alpha'] for r in q if r['statistics']['remainder_exceeds_response_budget']]
                    curves.append(dict(**spec,arm=arm,round=rnd,n_requested=7,n_evaluated=len(q),first_sampled_remainder_over_budget=departures[0] if departures else None,first_sampled_is_not_true_onset_or_selected_step=True,small_fraction_slope_disagreements={str(r['alpha']):r['statistics']['response_slope_disagreement_Linf'] for r in q if r['step'] in (1,2)}))
        dump(out/'PATH_SUMMARIES.json',summaries);dump(out/'ALL36_CURVE_DESCRIPTIONS.json',curves)
        qualified=not repeat_failures and not known_failures;s.update(complete=True,decision='C134_FIXED_PATH_MEASUREMENT_COMPLETE_NO_CANDIDATE' if qualified else 'C134_FIXED_PATH_MEASUREMENT_COMPLETE_REPLAY_DRIFT_NO_INTERPRETATION',
            evaluated_path_measurements=evaluated,unavailable_path_measurements=252-evaluated,repeatable=not repeat_failures,known_point_replay_consistent=not known_failures,path_interpretation_qualified=qualified,path_summaries=summaries,
            next_branch='USE_FIXED_CURVES_TO_SPECIFY_A_SEPARATE_FINITE_WRITE_PARAMETERISATION' if qualified else 'RESOLVE_REPLAY_DRIFT_BEFORE_MECHANISTIC_INTERPRETATION')
        report=['# C134：固定写入路径实测','','6个历史训练中心、3臂、7个固定步长、2轮，共252条请求。起点使用制备态门槛，其余点均用最终写入目标门槛。所有步长描述性报告，不选步长、不生成或修复写入器。','','|轮次|臂|α|响应通过/请求|全部声明门槛通过/请求|最大响应|最大余项|','|---|---|---:|---:|---:|---:|---:|']
        for rnd in ('0','1'):
            for arm,items in summaries[rnd].items():
                for q in items:report.append(f"|{rnd}|{arm}|{q['alpha']}|{q['response_passed']}/6|{q['all_declared_gates_passed']}/6|{q['maximum_response_drift']}|{q['maximum_response_remainder']}|")
        report+=['','路径解释资格：'+str(qualified),'','余项相对于C134新测写前观测与C132归档局部Jacobian定义；包含非线性、归档导数误差、前向误差和提交舍入等影响，不能独立证明曲率或AD错误。有限分数斜率不是精确导数。首次采样越界位置不是真正连续起点，更不是可部署步长。','','无新反传、Jacobian、SVD、拟合、求解或训练；保留C132/C133原判定。不测试新态或排除输入，不构成LC2或单折通过。']
        (out/'REPORT_ZH.md').write_text('\n'.join(report)+'\n',encoding='utf-8');stage='complete'
    except BaseException:errors.append(traceback.format_exc());s.update(complete=False,decision='C134_INCOMPLETE_NO_CANDIDATE');print(errors[-1],flush=True)
    finally:
        checks={}
        for k,p in paths.items():
            try:checks[k]=sha(p)==c['pins'][k]['sha256']
            except Exception:checks[k]=False
        s['original_inputs_unchanged']=checks
        if not all(checks.values()):s.update(complete=False,decision='C134_INCOMPLETE_NO_CANDIDATE');errors.append('Original input changed')
        checkpoint();print(json.dumps(dict(s,errors=errors),ensure_ascii=False,indent=2),flush=True)
    return dict(s,errors=errors)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in INPUT_KEYS:p.add_argument('--'+k.lower(),required=True)
    p.add_argument('--out',required=True);a=p.parse_args();s=run({k:Path(getattr(a,k.lower())) for k in INPUT_KEYS},a.out);sys.exit(0 if s['complete'] else 1)
