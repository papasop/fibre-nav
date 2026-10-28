"""One read-only finite-step decomposition, fixed registry and fixed formulas."""
import argparse,json,traceback
from core133 import *
from reader133 import load

def run(path,out,c=None):
    c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);started=time.monotonic();records=[];contrasts=[];arrays={};errors=[]
    s=dict(protocol=PROTOCOL,complete=False,decision='C133_INCOMPLETE_NO_LC2_CLAIM',synthetic_fixture=not c['production'],parent_protocol=PARENT,
        parent_decision_reclassified=False,local_decomposition_records_requested=36,local_decomposition_records_completed=0,arm_vs_raw_contrasts_requested=24,arm_vs_raw_contrasts_completed=0,
        states_with_local_derivatives=6,states_without_local_derivatives=6,new_model_forwards=0,new_backward_calls=0,new_training_steps=0,new_head_fits=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,
        archived_source_executed=False,candidate_selected=False,writer_deployed=False,thresholds_changed=False,LC2_closed=False,LC2_single_fold_passed=False,
        excluded_input_states_evaluated=0,confirmation_states_evaluated=0,physical_curvature_cause_established=False,AD_accuracy_independently_verified=False,forward_error_bound_established=False,
        scope='Sealed C132 arrays, archived model observables and independent arithmetic only. Remainders include all departures from archived differential prediction; they do not isolate curvature, AD error or forward rounding. No independent model replay, new writer or generalisation evidence.')
    def checkpoint():
        s.update(local_decomposition_records_completed=len(records),arm_vs_raw_contrasts_completed=len(contrasts),elapsed_worker_seconds=time.monotonic()-started)
        dump(out/'SUMMARY.json',dict(s,errors=errors));dump(out/'DECOMPOSITION_PROGRESS.json',records);dump(out/'CONTRAST_PROGRESS.json',contrasts)
    try:
        need(not (out/'SUMMARY.json').exists(),'Fresh C133 output required');dump(out/'DESIGN_FROZEN_BEFORE_ANALYSIS.json',design());checkpoint()
        need(sha(path)==c['pin']['sha256'],'Wrong fixed original C132 ZIP');dump(out/'INPUT_FROZEN.json',dict(C132=dict(path=str(path),sha256=sha(path),known_before_C133=True)))
        with safe.no_solvers():
            data=load(path,c,out);s.update(parent_decision=data['parent']['decision'],parent_branch=data['parent']['next_branch'],parent_repeatable=data['repeatable'],parent_endpoint_summaries=data['summaries'])
            component_map={}
            for rec in data['records']:
                if rec['role']!='geometry_training':continue
                need(time.monotonic()-started<TIMEOUT,'C133 fixed20-minute budget exceeded')
                sid=rec['state_id'];arm=rec['arm'];rnd=rec['round'];base=dict(state_id=sid,case=rec['case'],origin=rec['origin'],role=rec['role'],arm=arm,round=rnd)
                if rec['status']!='EVALUATED':records.append(dict(**base,status='UNAVAILABLE_COMMON_READ_OPERATOR',statistics=None));checkpoint();continue
                p=data['points'][sid];mapping=data['maps'][(sid,arm)]
                vals,stats=decompose(data['jacobians'][sid],p['prepared'],mapping['point'],p['preparation']['observed'][:10],rec['metrics']['observed'][:10],mapping['precap'],mapping['postcap'])
                key=(sid,arm,rnd);component_map[key]=vals;prefix=f'record{len(records):02d}';arrays.update({prefix+'__'+k:v for k,v in vals.items()})
                records.append(dict(**base,status='DECOMPOSED',array_prefix=prefix,statistics=stats));checkpoint()
                print('C133_DECOMPOSE',rnd,sid,arm,'linear',stats['linear_actual']['response']['Linf'],'remainder',stats['finite_step_remainder']['response']['Linf'],'endpoint',stats['endpoint']['response']['Linf'],flush=True)
            for rnd in (0,1):
                for spec in [r for r in registry() if r['role']=='geometry_training']:
                    sid=spec['state_id']
                    for arm in ARMS[1:]:
                        key=(sid,arm,rnd);base=dict(**spec,arm=arm,round=rnd)
                        if key not in component_map:contrasts.append(dict(**base,status='UNAVAILABLE_COMMON_READ_OPERATOR',statistics=None));continue
                        vals,stats=contrast(component_map[(sid,ARMS[0],rnd)],component_map[key]);prefix=f'contrast{len(contrasts):02d}';arrays.update({prefix+'__'+k:v for k,v in vals.items()});contrasts.append(dict(**base,status='DECOMPOSED',array_prefix=prefix,statistics=stats))
            need(len(records)==36 and len(contrasts)==24,'Full fixed decomposition registry required');np.savez_compressed(out/'FINITE_STEP_COMPONENTS.npz',**arrays)
            dump(out/'ALL36_LOCAL_DECOMPOSITIONS.json',records);dump(out/'ALL24_ARM_VS_RAW_CONTRASTS.json',contrasts)
            unavailable=[dict(**r,reason='No local Jacobian collected in C132; no local linear/remainder decomposition claimed') for r in registry() if r['role']!='geometry_training'];dump(out/'SIX_PROBE_LOCAL_DERIVATIVES_NOT_AVAILABLE.json',unavailable)
            aggregate={}
            for rnd in (0,1):
                aggregate[str(rnd)]={}
                for arm in ARMS:
                    q=[r for r in records if r['arm']==arm and r['round']==rnd];ev=[r for r in q if r['status']=='DECOMPOSED']
                    aggregate[str(rnd)][arm]=dict(n_requested=6,n_decomposed=len(ev),n_unavailable=6-len(ev),first_order_response_within_budget=sum(r['statistics']['first_order_response_within_budget'] for r in ev),finite_response_within_budget=sum(r['statistics']['finite_response_within_budget'] for r in ev),
                        maximum_actual_linear_response_Linf=max((r['statistics']['linear_actual']['response']['Linf'] for r in ev),default=None),maximum_finite_remainder_response_Linf=max((r['statistics']['finite_step_remainder']['response']['Linf'] for r in ev),default=None),
                        maximum_linear_cap_effect_response_Linf=max((r['statistics']['linear_cap_effect']['response']['Linf'] for r in ev),default=None),maximum_linear_float32_effect_response_Linf=max((r['statistics']['linear_float32_commit_effect']['response']['Linf'] for r in ev),default=None))
            dump(out/'LOCAL_DECOMPOSITION_SUMMARIES.json',aggregate)
            s.update(complete=True,decision='C133_FINITE_STEP_DECOMPOSITION_COMPLETE_PARENT_DECISION_RETAINED',next_diagnostic_branch=choose_branch(records),local_decomposition_summaries=aggregate,
                evaluated_local_decompositions=sum(r['status']=='DECOMPOSED' for r in records),unavailable_local_decompositions=sum(r['status']!='DECOMPOSED' for r in records),historical_probe_local_decompositions_claimed=0)
            report=['# C133：有限步长响应分解','','全部12态、三臂、两轮端点门槛重新核对；仅对具有局部Jacobian的6个构造中心进行36条分解。另6态不套用平均Jacobian。','','分解：端点观测 = 写前观测 + 局部Jacobian × 实际提交增量 + 有限步长余项。余项按定义包含所有未被归档一阶预测解释的变化，不是纯曲率或数值误差证书。','','|轮次|臂|一阶预测响应达标/6|实测响应达标/6|最大一阶作用|最大余项|','|---|---|---:|---:|---:|---:|']
            for rnd in ('0','1'):
                for arm,v in aggregate[rnd].items():report.append(f"|{rnd}|{arm}|{v['first_order_response_within_budget']}|{v['finite_response_within_budget']}|{v['maximum_actual_linear_response_Linf']}|{v['maximum_finite_remainder_response_Linf']}|")
            report+=['','固定诊断分支：'+s['next_diagnostic_branch'],'','半径裁剪与float32提交的影响仅分解其一阶作用；没有测量这些阶段各自的有限端点，不能解释为有限因果贡献。坐标抵消量是描述统计。','','保留C132原判定。无模型加载、前向、反传、拟合、求解或新Jacobian/SVD；不生成、不选择或部署写入器。不独立验证归档导数或模型测量准确性，归档一致性也不能证明测量时序和非访问。LC2及单折均未通过。']
            (out/'REPORT_ZH.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    except BaseException:errors.append(traceback.format_exc());s.update(complete=False,decision='C133_INCOMPLETE_NO_LC2_CLAIM');print(errors[-1],flush=True)
    finally:
        unchanged=Path(path).is_file() and sha(path)==c['pin']['sha256'];s['original_input_unchanged']=unchanged
        if not unchanged:s.update(complete=False,decision='C133_INCOMPLETE_NO_LC2_CLAIM');errors.append('Original C132 changed or missing')
        checkpoint();print(json.dumps(dict(s,errors=errors),ensure_ascii=False,indent=2),flush=True)
    return dict(s,errors=errors)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--c132',required=True);p.add_argument('--out',required=True);a=p.parse_args();s=run(Path(a.c132),a.out);sys.exit(0 if s['complete'] else 1)
