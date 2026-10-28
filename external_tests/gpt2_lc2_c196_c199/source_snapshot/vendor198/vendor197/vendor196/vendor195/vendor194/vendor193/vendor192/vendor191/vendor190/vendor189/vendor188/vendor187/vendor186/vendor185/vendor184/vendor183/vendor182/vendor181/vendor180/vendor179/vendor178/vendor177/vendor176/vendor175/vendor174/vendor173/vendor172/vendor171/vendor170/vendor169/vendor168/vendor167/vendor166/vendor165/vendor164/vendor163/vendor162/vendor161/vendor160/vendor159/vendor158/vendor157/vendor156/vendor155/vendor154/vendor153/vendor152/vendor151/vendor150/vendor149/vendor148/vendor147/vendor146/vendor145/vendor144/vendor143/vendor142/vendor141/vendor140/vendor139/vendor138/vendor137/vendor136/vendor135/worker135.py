"""Frozen shared normal multipliers, with raw control and complete denominators."""
import argparse,json,traceback
from collections import Counter
from core135 import *
from inputs135 import load

def summarise(records):
    summaries={};pairs={};raw_index=BETAS.index(1.)
    for role in ROLES:
        summaries[role]={};pairs[role]={}
        for rnd in (0,1):
            summaries[role][str(rnd)]=[];pairs[role][str(rnd)]=[]
            raw={r['state_id']:r for r in records if r['round']==rnd and r['coefficient_index']==raw_index and (role=='all12' or r['role']==role)}
            for i,beta in enumerate(BETAS):
                q=[r for r in records if r['round']==rnd and r['coefficient_index']==i and (role=='all12' or r['role']==role)];bad=Counter(g for r in q for g in r['recomputed_gates']['failed_gates'])
                per_case={case:dict(n_requested=sum(r['case']==case for r in q),passed=sum(r['case']==case and r['recomputed_gates']['complete_pass'] for r in q)) for case in ('xor_00','xor_01','xor_11')}
                summaries[role][str(rnd)].append(dict(coefficient_index=i,beta=beta,n_requested=12 if role=='all12' else 6,n_evaluated=len(q),passed=sum(r['recomputed_gates']['complete_pass'] for r in q),maximum_response_drift=max((r['metrics']['response_drift'] for r in q),default=None),failed_gates={g:bad[g] for g in ('preparation','bits_match','output_flip','unselected_preserved','response','KL','exact_read_margin','development_read_margin','proposal_radius')},failed_ids=[r['state_id'] for r in q if not r['recomputed_gates']['complete_pass']],per_case=per_case))
                pairs[role][str(rnd)].append(dict(coefficient_index=i,beta=beta,n_requested=12 if role=='all12' else 6,n_matching_evaluated=len(q),rescued_ids=[r['state_id'] for r in q if r['recomputed_gates']['complete_pass'] and not raw[r['state_id']]['recomputed_gates']['complete_pass']],destroyed_ids=[r['state_id'] for r in q if not r['recomputed_gates']['complete_pass'] and raw[r['state_id']]['recomputed_gates']['complete_pass']]))
    return summaries,pairs

def historical_rule(summaries,pairs,qualified):
    if not qualified:return []
    return [beta for i,beta in enumerate(BETAS) if beta!=1. and all(summaries['all12'][str(rnd)][i]['passed']==summaries['all12'][str(rnd)][i]['n_evaluated']==12 and len(pairs['all12'][str(rnd)][i]['rescued_ids'])>0 and not pairs['all12'][str(rnd)][i]['destroyed_ids'] for rnd in (0,1))]

def run(paths,out,c=None,runtime_factory=None):
    c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);started=time.monotonic();stage='registered';records=[];replay=[];components={};errors=[];seals={};real=runtime_factory is None
    calls=dict(model_forwards=0,initialisation_model_forwards=0,preflight_metric_attempts=0,preflight_metrics=0,path_metric_attempts=0,path_metrics=0)
    s=dict(protocol=PROTOCOL,complete=False,decision='C135_INCOMPLETE_NO_CANDIDATE',synthetic_fixture=not c['production'],real_model_execution_claim=False,requested_states=12,geometry_training_states=6,historical_probe_states=6,coefficient_points_requested=84,endpoint_measurements_requested=168,
        new_model_forwards=0,new_Jacobians=0,new_SVD_calls=0,new_backward_calls=0,new_training_steps=0,new_head_fits=0,linear_systems_solved=0,new_states_prepared=0,excluded_input_states_evaluated=0,confirmation_states_evaluated=0,
        candidate_selected=False,coefficient_selected=False,writer_deployed=False,thresholds_changed=False,LC2_closed=False,LC2_single_fold_passed=False,parent_decision_reclassified=False,
        archived_source_executed=False,postwrite_response_repair_steps=0,teacher_calls=0,backtracking_steps=0,all_coefficient_points_frozen_before_model=False,preflight_completed_before_coefficients=False,
        physical_curvature_cause_established=False,AD_accuracy_independently_verified=False,forward_error_bound_established=False,
        scope='One fixed shared row-space multiplier family on12 known historical training centers. All coefficients and failures reported; raw beta1 is an incumbent control. No selection, deployment, fresh-state or excluded-input generalisation claim.')
    def checkpoint():
        s.update(stage=stage,endpoint_measurement_records=len(records),elapsed_worker_seconds=time.monotonic()-started,runtime_calls=calls,new_model_forwards=calls['model_forwards'])
        dump(out/'SUMMARY.json',dict(s,errors=errors));dump(out/'ENDPOINT_PROGRESS.json',records);dump(out/'PREFLIGHT_PROGRESS.json',replay)
    def budget():need(time.monotonic()-started<TIMEOUT,'C135 fixed50-minute budget reached')
    def seal(name):seals[name]=sha(out/name)
    def unchanged():
        for n,h in seals.items():need(sha(out/n)==h,'Frozen C135 artifact changed: '+n)
    def replay_check(m,old):
        q=safe.repeat_comparison(m,old);return q,bool(q['within_replay_tolerance'] and q['same_point_identity'])
    try:
        need(not (out/'SUMMARY.json').exists(),'Fresh C135 output required');dump(out/'DESIGN_FROZEN_BEFORE_MODEL.json',design());seal('DESIGN_FROZEN_BEFORE_MODEL.json');checkpoint()
        stage='three_original_archive_bindings';data=load(paths,out,c);budget();seal('INPUTS_FROZEN.json')
        stage='freezing_all84_coefficient_points';arrays,outcomes,maps,parts=freeze_coefficients(data);np.savez_compressed(out/'ALL84_FROZEN_POINTS.npz',**arrays);dump(out/'ALL84_COEFFICIENT_OUTCOMES_FROZEN.json',outcomes)
        for n in ('ALL84_FROZEN_POINTS.npz','ALL84_COEFFICIENT_OUTCOMES_FROZEN.json'):seal(n)
        dump(out/'POINT_SEALS_BEFORE_MODEL.json',seals);phase=Phase(data,maps);s['all_coefficient_points_frozen_before_model']=True;stage='model_initialisation';checkpoint()
        if runtime_factory is None:
            from runtime134 import factory
            runtime_factory=factory
        rt=runtime_factory(out/'model_runtime',data,phase,calls);s['real_model_execution_claim']=real
        compare({k:v for k,v in rt.pins.items() if k!='torch'},{k:v for k,v in data['modelpins'].items() if k!='torch'},'Pinned model/chart identity')
        dump(out/'MODEL_PIN_COMPARISON.json',dict(passed=True,torch_runtime=rt.pins.get('torch'),torch_archived=data['modelpins'].get('torch'),torch_version_excluded_only=True,known_point_replay_required=True))
        stage='all12_known_point_preflight';checkpoint();fresh={};archived={(r['state_id'],r['arm'],r['round']):r['metrics'] for r in data['records'] if r['status']=='EVALUATED'}
        for spec in registry():
            budget();sid=spec['state_id'];case=spec['case'];p=data['points'][sid];theta=p['prepared'];phase.authorize(('prepared',sid),theta,case,False);m=rt.metrics(theta,case,False);need(safe.metric_gates(m,case,theta,False),'Prepared-state replay invalid')
            checks=[replay_check(m,p['preparation'])]
            if sid in data['C134_preparations']:checks.append(replay_check(m,data['C134_preparations'][sid]))
            replay.append(dict(**spec,kind='prepared',metrics=m,comparisons=[q[0] for q in checks]));checkpoint();need(all(q[1] for q in checks),'Prepared replay drift blocks before coefficient measurements');fresh[sid]=m
            for arm in ARMS[:2]:
                q=data['maps'][(sid,arm)]['point'];phase.authorize(('endpoint',sid,arm),q,case,True);m=rt.metrics(q,case,True);gate_record(m,case,q,theta);checks=[replay_check(m,archived[(sid,arm,rnd)]) for rnd in (0,1)]
                if (sid,arm,0) in data['C134_endpoints']:checks += [replay_check(m,data['C134_endpoints'][(sid,arm,rnd)]) for rnd in (0,1)]
                replay.append(dict(**spec,kind='endpoint',arm=arm,metrics=m,comparisons=[v[0] for v in checks]));checkpoint();need(all(v[1] for v in checks),'Endpoint replay drift blocks before coefficient measurements')
            print('C135_PREFLIGHT',sid,'OK',flush=True)
        unchanged();phase.begin_measurements();s['preflight_completed_before_coefficients']=True;stage='two_round_coefficient_endpoints';checkpoint();first={};known_failures=[];repeat_failures=[]
        for rnd in (0,1):
            for row in outcomes:
                budget();sid=row['state_id'];i=row['coefficient_index'];beta=row['beta'];point=maps[(sid,i)];p=data['points'][sid];phase.authorize((rnd,sid,i),point,row['case'],True);m=rt.metrics(point,row['case'],True);g=gate_record(m,row['case'],point,p['prepared'])
                values,stats=local_diagnostic(data['jacobians'].get(sid),parts[(sid,i)],beta,fresh[sid]['observed'][:10],m['observed'][:10]);rec=dict(**row,round=rnd,status='EVALUATED',metrics=m,recomputed_gates=g,local_statistics=stats)
                if values is not None:
                    prefix=f'record{len(records):03d}';rec['array_prefix']=prefix;components.update({prefix+'__'+k:v for k,v in values.items()})
                if beta in (0.,1.):
                    arm=ARMS[1] if beta==0. else ARMS[0];checks=[replay_check(m,archived[(sid,arm,k)]) for k in (0,1)]
                    if (sid,arm,0) in data['C134_endpoints']:checks += [replay_check(m,data['C134_endpoints'][(sid,arm,k)]) for k in (0,1)]
                    rec['known_control_comparisons']=[v[0] for v in checks]
                    if not all(v[1] for v in checks):known_failures.append(dict(state_id=sid,beta=beta,round=rnd,comparisons=rec['known_control_comparisons']))
                key=(sid,i)
                if rnd==0:first[key]=m
                else:
                    comp,valid=replay_check(m,first[key]);rec['repeat_comparison']=comp
                    if not valid:repeat_failures.append(dict(state_id=sid,beta=beta,comparison=comp))
                records.append(rec);dump(out/'measurements'/f'record{len(records)-1:03d}.json',rec);checkpoint()
                if i==len(BETAS)-1:print('C135_COEFFICIENTS',rnd,sid,[(r['beta'],r['recomputed_gates']['complete_pass'],round(r['metrics']['response_drift'],8)) for r in records[-len(BETAS):]],flush=True)
        phase.finish();unchanged();need(len(records)==168 and calls['path_metrics']==168 and calls['preflight_metrics']==36,'Complete fixed endpoint registry required');need(calls['model_forwards']==calls['initialisation_model_forwards']+2*(36+168),'Model forward counter differs')
        np.savez_compressed(out/'GEOMETRY_ONLY_LINEAR_COMPONENTS.npz',**components);dump(out/'ALL168_COEFFICIENT_MEASUREMENTS.json',records);dump(out/'KNOWN_CONTROL_FAILURES.json',known_failures);dump(out/'REPEATABILITY_FAILURES.json',repeat_failures)
        summaries,pairs=summarise(records);qualified=not known_failures and not repeat_failures;feasible=historical_rule(summaries,pairs,qualified)
        dump(out/'COEFFICIENT_SUMMARIES.json',summaries);dump(out/'PAIRED_VS_RAW.json',pairs);dump(out/'HISTORICAL_FEASIBILITY_ONLY.json',dict(noncontrol_coefficients_meeting_fixed_rule=feasible,no_coefficient_selected=True,no_deployment=True,no_LC2_acceptance=True))
        s.update(complete=True,decision=('C135_FIXED_NORMAL_COEFFICIENT_COMPLETE_REPLAY_DRIFT_NO_INTERPRETATION' if not qualified else ('C135_FIXED_NORMAL_COEFFICIENT_COMPLETE_HISTORICAL_FEASIBILITY_ONLY' if feasible else 'C135_FIXED_NORMAL_COEFFICIENT_COMPLETE_NO_QUALIFIED_COEFFICIENT')),
            evaluated_endpoint_measurements=168,repeatable=not repeat_failures,known_control_replay_consistent=not known_failures,interpretation_qualified=qualified,noncontrol_coefficients_meeting_historical_rule=feasible,
            local_derivative_diagnostics=84,historical_probe_local_derivative_diagnostics=0,endpoint_summaries=summaries,
            next_branch=('RESOLVE_REPLAY_DRIFT_BEFORE_MECHANISTIC_INTERPRETATION' if not qualified else ('PREREGISTER_SEPARATE_NEW_STATE_STUDY_WITH_EXPLICIT_FREEZE_RULE' if feasible else 'STOP_FIXED_NORMAL_MULTIPLIER_FAMILY_REVISE_FINITE_WRITE_STRUCTURE')))
        report=['# C135：固定共享法向系数实测','','12个历史训练中心，7个预登记共享系数，两轮168条端点。系数1为原raw字节不变对照，不算新改善；系数0复现原投影。大于1是法向放大。全部系数与失败保留，不选择、不部署。','','|β|全部12态首轮|全部12态重复轮|几何6态首轮|另外6态首轮|首轮最大响应漂移|','|---:|---:|---:|---:|---:|---:|']
        for i,beta in enumerate(BETAS):report.append(f"|{beta}|{summaries['all12']['0'][i]['passed']}/12|{summaries['all12']['1'][i]['passed']}/12|{summaries['geometry_training']['0'][i]['passed']}/6|{summaries['historical_seen_center_probe']['0'][i]['passed']}/6|{summaries['all12']['0'][i]['maximum_response_drift']}|")
        report+=['','满足固定历史可行性规则的全部非对照系数：'+str(feasible),'','后续分支：'+s['next_branch'],'','法向仅相对C132固定共享响应行空间定义；半径裁剪会同时缩放切向与法向成分，所有实际提交点单独验收。仅6个几何构造中心有归档局部导数；另外6态不套用平均导数。余项不构成曲率或前向误差证书。','','完整历史通过仍不是新态确认或排除输入泛化。这12态及既有结果均属于已知开发数据；任何后续候选选择必须另行说明并冻结，再用未用于设计的新态评估。没有LC2或单折通过声明。']
        (out/'REPORT_ZH.md').write_text('\n'.join(report)+'\n',encoding='utf-8');stage='complete'
    except BaseException:errors.append(traceback.format_exc());s.update(complete=False,decision='C135_INCOMPLETE_NO_CANDIDATE');print(errors[-1],flush=True)
    finally:
        checks={}
        for k,p in paths.items():
            try:checks[k]=sha(p)==c['pins'][k]['sha256']
            except Exception:checks[k]=False
        s['original_inputs_unchanged']=checks
        if not all(checks.values()):s.update(complete=False,decision='C135_INCOMPLETE_NO_CANDIDATE');errors.append('Original input changed')
        checkpoint();print(json.dumps(dict(s,errors=errors),ensure_ascii=False,indent=2),flush=True)
    return dict(s,errors=errors)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in INPUT_KEYS:p.add_argument('--'+k.lower(),required=True)
    p.add_argument('--out',required=True);a=p.parse_args();s=run({k:Path(getattr(a,k.lower())) for k in INPUT_KEYS},a.out);sys.exit(0 if s['complete'] else 1)
