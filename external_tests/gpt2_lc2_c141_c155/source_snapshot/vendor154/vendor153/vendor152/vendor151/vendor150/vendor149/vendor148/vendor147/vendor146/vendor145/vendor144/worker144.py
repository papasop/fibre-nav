"""Bounded archival arithmetic audit; no training or model execution."""
import argparse,traceback
from core144 import *
from audit144 import audit

def run(paths,out,c=None):
    c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True)
    s=dict(protocol=PROTOCOL,complete=False,decision='C144_INCOMPLETE_AUDIT_NOT_ACCEPTED',synthetic_fixture=not c['production'],parent_decision_reclassified=False,new_model_forwards=0,new_backward_calls=0,new_training_steps=0,new_head_fits=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,new_states_prepared=0,new_model_features_acquired=0,confirmation_states_evaluated=0,excluded_input_states_evaluated=0,archived_source_executed=False,thresholds_changed=False,candidate_selected=False,writer_deployed=False,LC2_closed=False,LC2_single_fold_passed=False,measurement_accuracy_or_chronology_certified=False,causal_failure_explanation_established=False,AD_accuracy_independently_certified=False,convergence_or_stationarity_established=False,scope='CPU NumPy arithmetic on four fixed original archives. Delivered audit code only; no model replay, derivatives, fit, solving or new endpoint measurements. Archived gradients are checked as arithmetic inputs, not as true derivatives. All84 states are exposed training; repetitions do not create independent samples. Audit consistency does not establish measurement accuracy, chronology, non-access, causal explanation, feasibility, convergence or generalisation.')
    errors=[]
    try:
        need(not(out/'SUMMARY.json').exists(),'Fresh C144 output required');dump(out/'DESIGN.json',design())
        with safe.no_solvers():s.update(audit(paths,out,c))
        s.update(complete=True,decision='C144_READONLY_JOINT84_AUDIT_COMPLETE_PARENT_DECISION_RETAINED')
        lines=['# C144：C143全84训练只读审计','','C143原判定：'+s['parent_decision'],'','|臂|首轮通过|重复轮通过|首轮最大响应漂移|','|---|---:|---:|---:|']
        for arm in ARMS:
            q=s['final_training_summaries']['all84'];lines.append(f"|{arm}|{q['0'][arm]['passed']}/84|{q['1'][arm]['passed']}/84|{q['0'][arm]['maximum_response_drift']}|")
        lines+=['','全部84态保留，12步1008条训练记录及两轮504端点核对；最终控制必须等于实际第12步。原36态与原C141暴露48态都属于训练，不构成独立确认。','','相对C140新增失败态：']+['- '+sid for sid in s['newly_failed_vs_C140']]
        lines+=['','C143全部失败态：']+['- '+sid for sid in s['C143_failed_ids']]
        lines+=['','独立重算目标、84态聚合与权重、梯度贡献之和、梯度裁剪、Adam矩及更新、全部冻结点、原生门槛和配对得失。逐态逐坐标保留带符号响应、0.002预算余量及越界量。损失改善与通过数分别报告。','','梯度取自归档；未验证其为真实导数。零/暖功能与原生观测一致性只核对记录，不证明导数精度。两轮是同84态重复，不是168个独立状态。','','不挑中间头、不训练、不修改门槛、不加载GPT2/Torch/HF，不声称控制结构不可行或优化收敛。C143/C141原失败判定保留；LC2未闭合。']
        (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    except BaseException:errors.append(traceback.format_exc());print(errors[-1],flush=True)
    finally:
        unchanged={}
        for k,p in paths.items():
            try:unchanged[k]=sha(p)==c['pins'][k]['sha256']
            except Exception:unchanged[k]=False
        s['original_inputs_unchanged']=unchanged
        if not all(unchanged.values()):s.update(complete=False,decision='C144_INCOMPLETE_AUDIT_NOT_ACCEPTED');errors.append('Original input changed')
        s['errors']=errors;dump(out/'SUMMARY.json',s)
    return s

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for k in INPUT_KEYS:parser.add_argument('--'+k.lower(),required=True)
    parser.add_argument('--out',required=True);args=parser.parse_args();s=run({k:Path(getattr(args,k.lower())) for k in INPUT_KEYS},args.out)
    print(json.dumps(s,ensure_ascii=False,indent=2),flush=True)
    if not s['complete']:raise SystemExit(1)
