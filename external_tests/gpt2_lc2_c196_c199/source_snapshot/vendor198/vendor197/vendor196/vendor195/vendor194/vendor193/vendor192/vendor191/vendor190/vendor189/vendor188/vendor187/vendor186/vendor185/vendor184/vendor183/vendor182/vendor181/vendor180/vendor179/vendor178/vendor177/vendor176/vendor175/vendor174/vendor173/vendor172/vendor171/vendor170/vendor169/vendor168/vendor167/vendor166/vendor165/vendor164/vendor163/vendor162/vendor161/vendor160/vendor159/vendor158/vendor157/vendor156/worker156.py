"""Pure NumPy read-only audit, no model/HF/Torch or numerical solvers."""
import argparse,resource,traceback
from core156 import *
from audit156 import bind,baseline,recorded_gradients,analyse
def run(input_zip,out,c=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);input_zip=Path(input_zip);c=contract() if c is None else c;start=time.monotonic();stage='registered';errors=[]
    s=dict(protocol=PROTOCOL,complete=False,decision='C156_INCOMPLETE_NO_TRAJECTORY_CONCLUSION',synthetic_fixture=not c['production'],parent_decision_reclassified=False,all84_remain_exposed_training=True,training_states_requested=84,training_states_audited=0,records_requested=1530,records_audited=0,FP64_focus_state_ids=list(FOCUS),new_model_forwards=0,new_backward_calls=0,new_training_steps=0,new_head_fits=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,new_model_features_acquired=0,new_states_prepared=0,confirmation_states_evaluated=0,excluded_input_states_evaluated=0,archived_source_executed=False,gradients_independently_recomputed=False,endpoint_KL_independently_reconstructed_from_logits=False,ten_predecessor_original_ZIP_bytes_rehashed=False,measurement_accuracy_established=False,AD_accuracy_certified=False,systematic_forward_error_bound_established=False,curvature_bound_established=False,causal_failure_explanation_established=False,precision_superiority_established=False,FP64_full84_objective_available=False,thresholds_changed=False,candidate_selected=False,writer_deployed=False,LC2_closed=False,LC2_single_fold_passed=False,scope=design()['scope'])
    def checkpoint():s.update(stage=stage,elapsed_worker_seconds=time.monotonic()-start,process_peak_RSS_KiB=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss));dump(out/'SUMMARY.json',dict(s,errors=errors))
    def budget():need(time.monotonic()-start<TIMEOUT,'C156 fixed20-minute worker budget')
    try:
        need(not (out/'SUMMARY.json').exists(),'Fresh C156 audit');dump(out/'DESIGN_FROZEN_BEFORE_AUDIT.json',design());checkpoint();compare(sha(input_zip),c['pins']['C155']['sha256'],'Fixed C155 whole ZIP before payload');dump(out/'INPUT_FROZEN.json',dict(path=str(input_zip),sha256=sha(input_zip)));stage='source_and_full1530_record_binding'
        for n,h in c['source155'].items():compare(sha(ROOT/'vendor155'/n),h,'Delivered trusted C155 source/'+n)
        with parent.prior.no_models(),safe.no_solvers(),contextlib.closing(Archive(input_zip)) as z:
            data=bind(z,c,budget);budget();s.update(parent_protocol=data['parent']['protocol'],parent_decision=data['parent']['decision'],C155_members_hashed=data['members_hashed'],training_states_audited=84,records_audited=1530,actual_frozen_point_entries_verified=588,actual_frozen_coefficient_tensors_verified=7,delivered_and_archived_source_pins_passed=True);checkpoint();stage='baseline_and_archived_gradient_arithmetic';b=baseline(z,c,data);g=recorded_gradients(data);dump(out/'BASELINE_REFERENCE_RECONSTRUCTION.json',b);dump(out/'ARCHIVED_GRADIENT_ARITHMETIC_ONLY.json',g);stage='all_scales_coordinate_and_precision_trajectories';d=analyse(z,data,out);budget();dump(out/'PRECISION_TRAJECTORY_SUMMARY.json',d);s.update(baseline_reconstruction=b,archived_gradient_arithmetic=g,trajectory_diagnostics=d,FP32_maximum_native_functional_observed_difference=max(x['observed_Linf'] for x in data['parities']),FP32_maximum_native_functional_KL_difference=max(x['KL_absolute'] for x in data['parities']),FP32_pair_gate_disagreements=sum(not x['gates_equal'] for x in data['parities']),saved_parent_diagnostics_recomputed=True,all_requested_scales_retained=True)
        s.update(complete=True,decision='C156_READONLY_TRAJECTORY_AUDIT_COMPLETE_PARENT_DECISION_RETAINED');stage='complete';report(out,s)
        for r in d['focus']:print('C156_FOCUS',r['state_id'],'FP32_coordinate_slope_spread',r['FP32_response_central_slope_spread_Linf'],'FP64_coordinate_slope_spread',r['FP64_local_response_central_slope_spread_Linf'],'ratio',r['spread_ratio_FP64_local_over_FP32'],flush=True)
    except BaseException:errors.append(traceback.format_exc());print(errors[-1],flush=True)
    finally:
        try:s['original_C155_input_unchanged']=sha(input_zip)==c['pins']['C155']['sha256']
        except Exception:s['original_C155_input_unchanged']=False
        if not s['original_C155_input_unchanged']:s['complete']=False;errors.append('Original C155 input missing or changed')
        if not s['complete']:s['decision']='C156_INCOMPLETE_NO_TRAJECTORY_CONCLUSION'
        checkpoint()
    return dict(s,errors=errors)
def report(out,s):
    lines=['# C156：C155逐坐标精度轨迹只读审计','','判定：'+s['decision'],'','仅重开C155总包；全部84态FP32、两个固定态FP64，1530条原测量及588个封存点。没有新模型测量。','','|固定训练态|FP32响应坐标中心斜率跨度L∞|FP64自身基线跨度L∞|FP64/FP32|','|---|---:|---:|---:|']
    for r in s['trajectory_diagnostics']['focus']:lines.append(f"|{r['state_id']}|{r['FP32_response_central_slope_spread_Linf']}|{r['FP64_local_response_central_slope_spread_Linf']}|{r['spread_ratio_FP64_local_over_FP32']}|")
    lines+=['','跨度只比较已注册三个有限尺度，不是导数、精度优越性、噪声或曲率的认证。不同精度使用相同已舍入float32物理坐标。FP64不是数学真值，也不是未舍入写入映射。','响应坐标的恒定基线偏移在点间差分中抵消；绝对值、最大值及铰链响应损失则可能受基线选择影响。自身FP64基线与原缓存FP32基线的全部结果均保留。','原FP32全84态目标与归档梯度点积对应；两个FP64态的响应损失斜率不能充当全84态FP64目标或FP64 AD验证。','没有保存FP64逐端点完整logits，因此端点KL仅保留原测量，不能独立从logits重算。十个前代原ZIP未重开；C155封存时序、非访问和测量准确性不由本审计证明。','无模型、Torch、HF、前向、反传、训练、拟合、求解、Jacobian/SVD、选点、选头、部署、确认或排除输入；父判定和原FP32门槛不变，LC2与单折仍开放。'];(out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--c155',required=True);p.add_argument('--out',required=True);a=p.parse_args();s=run(a.c155,a.out);print(json.dumps(s,ensure_ascii=False,indent=2),flush=True)
    if not s['complete']:raise SystemExit(1)
