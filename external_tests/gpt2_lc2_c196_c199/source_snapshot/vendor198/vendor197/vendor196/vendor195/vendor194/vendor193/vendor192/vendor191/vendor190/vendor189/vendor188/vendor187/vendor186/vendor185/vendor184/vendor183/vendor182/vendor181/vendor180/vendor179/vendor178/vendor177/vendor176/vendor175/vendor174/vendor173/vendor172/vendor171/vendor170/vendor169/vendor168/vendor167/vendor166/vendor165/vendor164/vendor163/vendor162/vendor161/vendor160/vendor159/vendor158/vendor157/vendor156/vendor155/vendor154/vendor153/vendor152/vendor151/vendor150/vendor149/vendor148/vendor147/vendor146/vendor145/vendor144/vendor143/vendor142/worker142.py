"""Bounded read-only CPU archive worker."""
import argparse,traceback
from core142 import *
from audit142 import audit

def run(paths,out,c=None):
    c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True)
    s=dict(protocol=PROTOCOL,complete=False,decision='C142_INCOMPLETE_AUDIT_NOT_ACCEPTED',synthetic_fixture=not c['production'],parent_decision_reclassified=False,new_model_forwards=0,new_backward_calls=0,new_training_steps=0,new_head_fits=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,archived_source_executed=False,thresholds_changed=False,candidate_selected=False,writer_deployed=False,new_states_prepared=0,excluded_input_states_evaluated=0,LC2_closed=False,LC2_single_fold_passed=False,statistical_causal_failure_explanation_established=False,measurement_accuracy_or_chronology_certified=False,external_C132_C139_original_ZIP_bytes_rehashed=False,scope='Only frozen tensors, archived observations and arithmetic. C141/C140 original bytes rehashed; no independent model replay. Finite response-effect differences are descriptive, not derivative accuracy, curvature bounds, operator-error isolation or guaranteed remedies. Norm subgroups do not replace all48 confirmation.')
    errors=[]
    try:
        dump(out/'DESIGN.json',design())
        with safe.no_solvers():s.update(audit(paths,out,c))
        s.update(complete=True,decision='C142_READONLY_CONFIRMATION_AUDIT_COMPLETE_PARENT_DECISION_RETAINED')
        lines=['# C142：C141冻结确认只读审计','','C141原判定：'+s['parent_decision'],'','|臂|首轮通过|重复轮通过|首轮最大响应漂移|','|---|---:|---:|---:|']
        for arm in ARMS:
            q=s['confirmation_summaries']['all48'];lines.append(f"|{arm}|{q['0'][arm]['passed']}/48|{q['1'][arm]['passed']}/48|{q['0'][arm]['maximum_response_drift']}|")
        lines+=['','两轮均有效制备：'+str(s['preparation_states_valid_both_rounds'])+'/48','', '两档范数分别统计，属于同一次确认内的描述；不得事后删除较大扰动态或改变原48态验收。', '', '|范数|C140首轮通过|C140重复轮通过|','|---|---:|---:|']
        for norm in (.125,.375):q=s['confirmation_summaries']['norm_'+str(norm)];lines.append(f"|{norm}|{q['0'][ARMS[2]]['passed']}/24|{q['1'][ARMS[2]]['passed']}/24|")
        lines+=['','原六个失败态（若重算一致）完整保留：']+['- '+sid for sid in s['C140_failures']]
        lines+=['','残差诊断逐坐标保留raw响应、候选响应、实测有限补偿作用(candidate−raw)、理想抵消向量(−raw)、实际权重补偿位移和剩余误差。该恒等分解不独立确定控制器误差、导数误差、曲率或硬件原因，也不提供新修复头。','','重新核对C140原包的实际第12步向量及原始writer/control/basis；不在C142重审全部12步训练或独立重算历史梯度。重算C141全部48个写前映射、96制备、288端点及144已知原生重放。对已知制备，额外直接比较C140原生制备测量；C141历史比较标量的原C132/C139全部参照未重新打开，因此不宣称独立重算这些历史比较标量。','','两轮为同48态的重复测量。C141原失败不改写，全部状态已暴露。无模型、训练、拟合、求解或新Jacobian/SVD；未评价排除输入，LC2未闭合。']
        (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    except BaseException:errors.append(traceback.format_exc());print(errors[-1],flush=True)
    finally:
        unchanged={}
        for k,p in paths.items():
            try:unchanged[k]=sha(p)==c['pins'][k]['sha256']
            except Exception:unchanged[k]=False
        s['original_inputs_unchanged']=unchanged
        if not all(unchanged.values()):s.update(complete=False,decision='C142_INCOMPLETE_AUDIT_NOT_ACCEPTED');errors.append('Original input changed')
        s['errors']=errors;dump(out/'SUMMARY.json',s)
    return s

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--c141',required=True);p.add_argument('--c140',required=True);p.add_argument('--out',required=True);a=p.parse_args();s=run(dict(C141=Path(a.c141),C140=Path(a.c140)),a.out);print(json.dumps(s,ensure_ascii=False,indent=2),flush=True)
    if not s['complete']:raise SystemExit(1)
