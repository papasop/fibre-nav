"""Fixed two-state real model AD/FD diagnostic. Completion is not derivative or LC2 acceptance."""
import argparse,resource,traceback
from core157 import *
from inputs157 import load
def run(paths,out,c=None,runtime_factory=None,input_loader=None):
    c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);start=time.monotonic();stage='registered';errors=[];calls=dict(model_forwards=0,functional_FP64_forwards=0,backward_attempts=0,backward_calls=0);runtime=None
    s=dict(protocol=PROTOCOL,complete=False,decision='C157_INCOMPLETE_NO_DIRECTIONAL_CONCLUSION',synthetic_fixture=not c['production'],real_model_execution_claim=False,all84_remain_exposed_training=True,focus_training_states_requested=2,focus_training_states_evaluated=0,requested_model_forwards=REQUESTED_FORWARDS,requested_scalar_backward_calls=REQUESTED_BACKWARDS,new_model_forwards=0,new_backward_calls=0,new_training_steps=0,new_head_fits=0,new_parameter_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,new_model_features_acquired=0,new_states_prepared=0,confirmation_states_evaluated=0,excluded_input_states_evaluated=0,candidate_selected=False,writer_deployed=False,thresholds_changed=False,parent_decision_reclassified=False,LC2_closed=False,LC2_single_fold_passed=False,FP64_full84_objective_available=False,AD_accuracy_certified=False,AD_error_established=False,systematic_forward_error_bound_established=False,curvature_bound_established=False,causal_failure_explanation_established=False,archived_source_executed=False,scope=design()['scope'])
    def checkpoint():
        s.update(delivery_revision='R2_BASELINE_RECORDING_ONLY',stage=stage,new_model_forwards=calls['model_forwards'],new_backward_calls=calls['backward_calls'],scalar_backward_attempts=calls['backward_attempts'],elapsed_worker_seconds=time.monotonic()-start,process_peak_RSS_KiB=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        diag=out/'model_runtime'
        s['baseline_replay_diagnostics']={p.stem:clean_json(p.read_bytes()) for p in diag.glob('*BASELINE_LOGITS.json')} if diag.is_dir() else {}
        s['model_initialization_forward_recorded']=calls['model_forwards']>0
        if (diag/'BASELINE_RUNTIME_PROVENANCE.json').is_file():
            env=clean_json((diag/'BASELINE_RUNTIME_PROVENANCE.json').read_bytes());s['baseline_torch_builds']=dict(current=env['torch'],archived_C155=env['archived_C155_MODEL_PINS'].get('torch'),difference_not_a_causal_explanation=True)
        dump(out/'SUMMARY.json',dict(s,errors=errors))
    def budget():need(time.monotonic()-start<TIMEOUT,'C157 fixed120-minute worker budget')
    try:
        need(not (out/'SUMMARY.json').exists(),'Fresh worker output required');dump(out/'DESIGN_FROZEN_BEFORE_MODEL.json',design());checkpoint();stage='original_archive_and_C156_reaudit';context=(load if input_loader is None else input_loader)(paths,out/'binding',c,budget);budget();data=context['data'];s['parent_decision']=data['parent']['decision'];stage='seal_all14_chord_points_before_model';points,directions,requests,seals=freeze(data,out);ids={k:point_hash(p) for k,p in points.items()};at=0;checkpoint()
        def unchanged():
            for n,h in seals.items():compare(sha(out/n),h,'Frozen artifact/'+n)
            need({k:point_hash(p) for k,p in points.items()}==ids,'Actual fixed points changed')
            for sid in FOCUS:exact(directions[sid],points[(sid,'plus_one')]-points[(sid,'zero')],'Fixed physical chord direction')
        def authorize(kind,sid,n):
            nonlocal at
            budget();unchanged();compare(requests[at],dict(stage=kind,state_id=sid,label=n),'Fixed request order');at+=1
        if runtime_factory is None:
            from runtime157 import factory
            runtime_factory=factory
        stage='pinned_model_and_baseline_replay';runtime=runtime_factory(out/'model_runtime',context,points,directions,calls);local=np.asarray(context['baseline']['FP64_response_reference'],float);shared=np.asarray(context['baseline']['FP32_cached_response_reference'],float);s['real_model_execution_claim']=c['production'];checkpoint();all_records=[]
        def save(kind,row):
            all_records.append(dict(stage=kind,**row));dump(out/'ALL_FORWARD_AND_AD_RECORDS.json',all_records);checkpoint()
        stage='four_endpoint_replays_before_AD'
        for sid in FOCUS:
            for n in ('zero','plus_one'):
                authorize('endpoint_replay',sid,n);r=runtime.measure(sid,n);want=context['replay_absolute'][(sid,n)];err=float(np.max(abs(np.asarray(r['absolute_response'])-want)));r['C155_endpoint_replay_Linf']=err;save('endpoint_replay',r);need(err<=REPLAY_TOL,'C155 exact endpoint replay differs; stop before AD')
        diagnostics=[]
        for sid in FOCUS:
            stage='scalar_alpha_AD';authorize('AD',sid,'zero');ad=runtime.ad(sid);save('AD',ad);need(np.shape(ad['coordinate_derivatives'])==(6,) and np.isfinite(ad['coordinate_derivatives']).all(),'Six finite AD scalars');panels={};repeats=[]
            stage='all_fixed_finite_panels'
            for n in LABELS:
                authorize('finite',sid,n);r=runtime.measure(sid,n);need(r['FP64_point_sha256']==ids[(sid,n)],'Measured point identity');panels[n]=r;save('finite',r)
            need(np.max(abs(np.array(ad['absolute_response'])-panels['zero']['absolute_response']))<=REPLAY_TOL,'AD/no-grad forward center differs')
            stage='fixed_zero_plus_one_repeats'
            for n in ('zero','plus_one'):
                authorize('repeat',sid,n);r=runtime.measure(sid,n);save('repeat',r);err=float(np.max(abs(np.asarray(r['absolute_response'])-panels[n]['absolute_response'])));repeats.append(dict(label=n,absolute_response_change_Linf=err));need(err<=REPLAY_TOL,'Fixed FP64 repeat differs')
            diag=dict(state_id=sid,**analyse(panels,ad,local,shared),repeat_comparisons=repeats,archived_FP32_coefficient_response_gradient_dot_update=context['archived_gradient_arithmetic']['archived_FP32_response_gradient_dot_actual_delta'][sid],archived_FP32_gradient_not_the_same_precision_or_path=True);diagnostics.append(diag);s['focus_training_states_evaluated']+=1;dump(out/'TWO_STATE_AD_FINITE_DIAGNOSTICS.json',diagnostics);checkpoint()
            print('C157_FOCUS',sid,'coordinate_AD',ad['coordinate_derivatives'],'local_loss_AD',ad['response_loss_derivatives']['local'],'FD_errors',[x['central_minus_AD_Linf'] for x in diag['scales']],flush=True)
        runtime.verify(True);unchanged();compare(at,len(requests),'Full fixed request denominator');compare(len(all_records),24,'All24 state-level model records');compare(calls['model_forwards'],26,'All26 model forwards');compare(calls['functional_FP64_forwards'],25,'All25 FP64 forwards');compare(calls['backward_calls'],16,'All16 scalar reverse calls');compare(calls['backward_attempts'],16)
        s.update(complete=True,decision='C157_TWO_STATE_FP64_AD_FD_DIAGNOSTIC_COMPLETE_NO_CANDIDATE_SELECTED',all14_points_frozen_before_model=True,all_scales_retained=True,endpoint_replays_consistent=True,temporary_functional_parameters_restored=True,diagnostics=diagnostics);stage='complete'
        (out/'REPORT_ZH.md').write_text('# C157：固定两态 FP64 同函数 AD／方向差分\n\n本轮完成状态仅指固定诊断流程完成，不是导数认证或 LC2 通过。\n\n以原 C155 零点和 +1 物理端点定义连续 FP64 直线路径；不是原系数更新／FP32 舍入写入映射，全部七点的路径差异均保留。仅标量 alpha 求导；不训练、不拟合、不选头。\n\n六个响应坐标和两种固定基线下的响应损失分别报告三个尺度的中心、正向、反向差分及与 AD 的差异。最大坐标、符号与铰链状态只描述采样点；不证明区间光滑性。\n\nFP64 非真值。无全84态 FP64 目标、KL／读出资格、导数误差界、曲率界或失败因果结论。已暴露态始终为训练态；没有确认或排除输入评估，LC2 仍开放。\n')
    except BaseException:errors.append(traceback.format_exc());print(errors[-1],flush=True)
    finally:
        checks={}
        for k in INPUT_KEYS:
            try:checks[k]=sha(paths[k])==c['pins'][k]['sha256']
            except Exception:checks[k]=False
        s['original_inputs_unchanged']=checks
        if not all(checks.values()):s['complete']=False;errors.append('Original archive missing or changed')
        if runtime is not None:
            try:runtime.verify(True)
            except BaseException:s['complete']=False;errors.append(traceback.format_exc())
        if not s['complete']:s['decision']='C157_INCOMPLETE_NO_DIRECTIONAL_CONCLUSION'
        checkpoint()
    return dict(s,errors=errors)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--c155',required=True);p.add_argument('--c156',required=True);p.add_argument('--out',required=True);a=p.parse_args();s=run(dict(C155=a.c155,C156=a.c156),a.out);print(json.dumps(s,ensure_ascii=False,indent=2),flush=True)
    if not s['complete']:raise SystemExit(1)
