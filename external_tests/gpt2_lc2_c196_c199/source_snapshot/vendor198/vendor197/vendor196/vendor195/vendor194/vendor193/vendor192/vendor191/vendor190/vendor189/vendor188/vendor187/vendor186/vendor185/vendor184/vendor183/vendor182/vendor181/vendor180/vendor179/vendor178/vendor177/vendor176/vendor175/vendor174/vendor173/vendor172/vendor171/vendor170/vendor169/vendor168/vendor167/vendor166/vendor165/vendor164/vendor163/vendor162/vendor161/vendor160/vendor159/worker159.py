"""Separate C159 diagnostic; cross-run drift is reported, same-run consistency is required."""
import argparse,resource,traceback
from core159 import *
from inputs159 import load
from map159 import chain_arithmetic
def run(paths,out,c=None,runtime_factory=None,input_loader=None):
    c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);start=time.monotonic();stage='registered';errors=[];calls=dict(model_forwards=0,functional_FP64_forwards=0,backward_attempts=0,backward_calls=0);runtime=None
    s=dict(protocol=PROTOCOL,complete=False,decision='C159_INCOMPLETE_NO_DIRECTIONAL_CONCLUSION',synthetic_fixture=not c['production'],real_model_execution_claim=False,all84_remain_exposed_training=True,focus_training_states_requested=2,focus_training_states_evaluated=0,requested_model_forwards=27,requested_scalar_backward_calls=18,new_training_steps=0,new_head_fits=0,new_parameter_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,new_model_features_acquired=0,new_states_prepared=0,confirmation_states_evaluated=0,excluded_input_states_evaluated=0,candidate_selected=False,writer_deployed=False,LC2_acceptance_thresholds_changed=False,diagnostic_protocol_changed_from_C158=True,exact_FP32_commit_derivative_claim=False,new_model_parameter_gradients=0,coefficient_gradient_scalar_losses_requested=6,C157_decision_reclassified=False,C158_decision_reclassified=False,LC2_closed=False,LC2_single_fold_passed=False,FP64_full84_objective_available=False,AD_accuracy_certified=False,AD_error_established=False,systematic_forward_error_bound_established=False,curvature_bound_established=False,causal_failure_explanation_established=False,archived_source_executed=False,scope=design()['scope'])
    def checkpoint():
        s.update(stage=stage,new_model_forwards=calls['model_forwards'],new_backward_calls=calls['backward_calls'],scalar_backward_attempts=calls['backward_attempts'],model_initialization_forward_recorded=calls['model_forwards']>0,elapsed_worker_seconds=time.monotonic()-start,process_peak_RSS_KiB=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss));dump(out/'SUMMARY.json',dict(s,errors=errors))
    def budget():need(time.monotonic()-start<TIMEOUT,'C159 fixed120-minute worker budget')
    try:
        need(not (out/'SUMMARY.json').exists(),'Fresh output required');dump(out/'DESIGN_FROZEN_BEFORE_MODEL.json',design());checkpoint();stage='original_archives_and_C157_failure_binding';context=(load if input_loader is None else input_loader)(paths,out/'binding',c,budget);s['C158_original_decision']=context['completed158']['decision'];compare(s['C158_original_decision'],'C158_SAME_RUN_TWO_STATE_FP64_DIAGNOSTIC_COMPLETE_NO_CANDIDATE_SELECTED');s['C157_original_decision']=context['failed157']['decision'];compare(s['C157_original_decision'],'C157_INCOMPLETE_NO_DIRECTIONAL_CONCLUSION');budget();stage='all14_points_sealed_before_model';points,directions,requests,seals=freeze(context['data'],out);chain_constants=constants(context['data']);ids={k:point_hash(p) for k,p in points.items()};tangent_ids={sid:point_hash(p) for sid,p in directions.items()};at=0;checkpoint()
        def unchanged():
            for n,h in seals.items():compare(sha(out/n),h,'Frozen artifact/'+n)
            need({k:point_hash(p) for k,p in points.items()}==ids,'Actual fixed point changed')
            for sid in FOCUS:compare(point_hash(directions[sid]),tangent_ids[sid],'Actual fixed chain tangent')
        def authorize(kind,sid,n):
            nonlocal at
            budget();unchanged();compare(requests[at],dict(stage=kind,state_id=sid,label=n),'Fixed request order');at+=1
        if runtime_factory is None:
            from runtime159 import factory
            runtime_factory=factory
        stage='model_identity_and_current_run_baseline_repeat';runtime=runtime_factory(out/'model_runtime',context,points,directions,calls);s.update(real_model_execution_claim=c['production'],cross_run_baseline_diagnostics=runtime.baseline_records,reference_conditions=list(REFERENCES));checkpoint();records=[]
        def save(kind,row):records.append(dict(stage=kind,**row));dump(out/'ALL_FORWARD_AND_AD_RECORDS.json',records);checkpoint()
        preflight={};cross=[];stage='fixed_endpoint_preflights'
        for sid in FOCUS:
            for n in ('zero','plus_one'):
                authorize('preflight',sid,n);r=runtime.measure(sid,n);need(r['FP64_point_sha256']==ids[(sid,n)],'Measured preflight point');preflight[(sid,n)]=r;old=context['replay_absolute'][(sid,n)];d=float(max(abs(np.asarray(r['absolute_response'])-old)));cross.append(dict(state_id=sid,label=n,absolute_response_Linf=d,historical_C157_tolerance=previous.previous.REPLAY_TOL,within_historical_tolerance=d<=previous.previous.REPLAY_TOL,descriptive_only_not_C159_gate=True));save('preflight',r)
        dump(out/'C155_ENDPOINT_CROSS_RUN_COMPARISONS.json',cross);s['cross_run_endpoint_comparisons']=cross;diagnostics=[]
        for sid in FOCUS:
            stage='same_run_scalar_alpha_AD';authorize('AD',sid,'zero');ad=runtime.ad(sid);save('AD',ad);farray(np.asarray(ad['coordinate_derivatives'],float),(6,));compare(set(ad['response_loss_derivatives']),set(REFERENCES));need(np.isfinite(list(ad['response_loss_derivatives'].values())).all(),'Finite three-reference derivatives');need(ad['FP64_point_sha256']==ids[(sid,'zero')],'AD zero point identity');need(max(abs(np.array(ad['absolute_response'])-preflight[(sid,'zero')]['absolute_response']))<=WITHIN_RUN_TOL,'Within-run AD/preflight center differs');panels={};repeats=[]
            stage='all_seven_finite_panels'
            for n in LABELS:
                authorize('finite',sid,n);r=runtime.measure(sid,n);need(r['FP64_point_sha256']==ids[(sid,n)],'Measured finite point');panels[n]=r;save('finite',r)
            need(max(abs(np.array(ad['absolute_response'])-panels['zero']['absolute_response']))<=WITHIN_RUN_TOL,'Within-run AD/no-grad center differs')
            for n in ('zero','plus_one'):need(max(abs(np.array(preflight[(sid,n)]['absolute_response'])-panels[n]['absolute_response']))<=WITHIN_RUN_TOL,'Within-run preflight/panel differs')
            stage='zero_plus_one_repeats'
            for n in ('zero','plus_one'):
                authorize('repeat',sid,n);r=runtime.measure(sid,n);save('repeat',r);d=float(max(abs(np.array(r['absolute_response'])-panels[n]['absolute_response'])));repeats.append(dict(label=n,absolute_response_change_Linf=d));need(d<=WITHIN_RUN_TOL,'Within-run final FP64 repeat differs')
            need(set(ad['chain_pullbacks'])==set(REFERENCES),'All three actual coefficient pullbacks');
            for name in REFERENCES:
                cc=ad['chain_pullbacks'][name];compare(cc,chain_arithmetic(ad['response_loss_derivatives'][name],cc['effective_gradient'],cc['direct_coefficient_gradient'],chain_constants,sid),'Independent saved chain arithmetic')
            diagnostics.append(dict(state_id=sid,**analyse(panels,ad,runtime.references),chain_pullbacks=ad['chain_pullbacks'],repeat_comparisons=repeats,archived_FP32_coefficient_response_gradient_dot_update=context['archived_gradient_arithmetic']['archived_FP32_response_gradient_dot_actual_delta'][sid],archived_gradient_not_same_precision_or_path=True));s['focus_training_states_evaluated']+=1;dump(out/'TWO_STATE_SAME_RUN_AD_FINITE_DIAGNOSTICS.json',diagnostics);checkpoint();print('C159_FOCUS',sid,'coordinate_AD',ad['coordinate_derivatives'],'loss_AD',ad['response_loss_derivatives'],'FD_coordinate_errors',[r['central_minus_AD_Linf'] for r in diagnostics[-1]['scales']],flush=True)
        runtime.verify(True);unchanged();compare(at,len(requests),'Full request denominator');compare(len(records),24,'All24 state records');compare(calls['model_forwards'],27,'All27 model forwards');compare(calls['functional_FP64_forwards'],26,'All26 FP64 forwards');compare(calls['backward_calls'],18,'All18 scalar reverse calls');compare(calls['backward_attempts'],18)
        s.update(complete=True,decision='C159_COEFFICIENT_CHAIN_FP64_DIAGNOSTIC_COMPLETE_NO_CANDIDATE_SELECTED',same_run_consistency_checks_passed=True,all_scales_retained=True,all_three_references_retained=True,diagnostics=diagnostics);stage='complete'
        (out/'REPORT_ZH.md').write_text('# C159：系数链路 FP64 AD／差分\n\n固定两训练态，冻结原路由权重与全部控制常量；对连续、最终 FP32 转换前的系数写入函数求导。保留14个点、三个步长与三种参考。每个损失同次反传取得 alpha、有效控制与完整头系数梯度，并独立重算权重拉回及与原更新的收缩。\n\n原 FP32 提交函数、C158 物理端点直线与新连续路径不混同；逐点坐标与插值舍入差异均保留。链式残差与 AD/FD 差异仅报告，不设导数通过阈值；不认证离散 FP32 提交的导数。\n\n所有模型内参数与常量冻结；不训练、不拟合、不选路由/步长/头，不部署，不进行新确认或排除类评估。C155-C158 原判定保留，LC2 开放。\n')
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
        if not s['complete']:s['decision']='C159_INCOMPLETE_NO_DIRECTIONAL_CONCLUSION'
        checkpoint()
    return dict(s,errors=errors)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--c155',required=True);p.add_argument('--c156',required=True);p.add_argument('--c157',required=True);p.add_argument('--c158',required=True);p.add_argument('--out',required=True);a=p.parse_args();s=run(dict(C155=a.c155,C156=a.c156,C157=a.c157,C158=a.c158),a.out);console={k:v for k,v in s.items() if k not in ('diagnostics','cross_run_baseline_diagnostics','cross_run_endpoint_comparisons')};console['diagnostic_summary']=[dict(state_id=d['state_id'],coordinate_AD=d['scales'][0]['AD'],scales=[dict(h=r['h'],coordinate_central_minus_AD_Linf=r['central_minus_AD_Linf'],losses={n:dict(AD=v['AD'],central_minus_AD=v['central_minus_AD']) for n,v in r['response_loss'].items()}) for r in d['scales']],chain_residuals={n:{k:v[k] for k in ('pullback_vs_direct_Linf','explicit_dot_minus_alpha','direct_dot_minus_alpha')} for n,v in d['chain_pullbacks'].items()},repeat_comparisons=d['repeat_comparisons']) for d in s.get('diagnostics',[])];print(json.dumps(console,ensure_ascii=False,indent=2),flush=True)
    if not s['complete']:raise SystemExit(1)
