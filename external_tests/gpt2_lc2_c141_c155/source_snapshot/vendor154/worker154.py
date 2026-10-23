"""No optimizer or backward: sealed all84 native stencil at two archived nonzero heads."""
import argparse,resource,traceback
from core154 import *
from inputs154 import load

def resolve_default_runtime_factory():
    import runtime154
    need(Path(runtime154.__file__).resolve()==ROOT/'runtime154.py','C154 runtime source identity');need(runtime154.factory.__module__=='runtime154','C154 factory identity');runtime154.resolve_pinned_factory();return runtime154.factory

def run(paths,out,c=None,runtime_factory=None,input_loader=None):
    c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);start=time.monotonic();stage='registered';calls=dict(model_forwards=0,initialisation_model_forwards=0,preflight_metric_attempts=0,preflight_metrics=0,path_metric_attempts=0,path_metrics=0);records=[];panels={};errors=[];seals={};d=None;phase=None
    s=dict(protocol=PROTOCOL,complete=False,decision='C154_INCOMPLETE_NO_DIRECTIONAL_CONCLUSION',synthetic_fixture=not c['production'],real_model_execution_claim=False,all84_remain_exposed_training=True,anchors_requested=2,training_states=84,requested_native_metrics=1512,requested_model_forwards=3025,new_model_forwards=0,new_backward_calls=0,new_training_steps=0,new_head_fits=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,new_states_prepared=0,new_model_features_acquired=0,confirmation_states_evaluated=0,excluded_input_states_evaluated=0,GPT2_optimizer_updates=0,LoRA_A_optimizer_updates=0,base_writer_updates=0,teacher_calls=0,backtracking_steps=0,online_response_repair_steps=0,archived_source_executed=False,gradients_independently_recomputed=False,AD_accuracy_certified=False,systematic_forward_error_bound_established=False,curvature_bound_established=False,thresholds_changed=False,candidate_selected=False,writer_deployed=False,parent_decision_reclassified=False,LC2_closed=False,LC2_single_fold_passed=False,scope=design()['scope'])
    def checkpoint():
        s.update(stage=stage,elapsed_worker_seconds=time.monotonic()-start,process_peak_RSS_KiB=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),runtime_calls=calls,new_model_forwards=calls['model_forwards'],native_metric_records=len(records),preflight_records=sum(r['phase']=='preflight' for r in records),probe_records=sum(r['phase']=='probe' for r in records),repeat_records=sum(r['phase']=='repeat' for r in records));dump(out/'SUMMARY.json',dict(s,errors=errors))
    def budget():need(time.monotonic()-start<TIMEOUT,'C154 fixed120-minute worker budget')
    def unchanged():
        for n,h in seals.items():compare(sha(out/n),h,'Sealed stencil/'+n)
        if d is not None:compare(prior.parent.fixed_digest(d),identity,'Frozen model inputs/router/writer')
        if phase is not None:phase.verify()
    try:
        need(not (out/'SUMMARY.json').exists(),'Fresh C154 run');dump(out/'DESIGN_FROZEN_BEFORE_MODEL.json',design());checkpoint();stage='default_runtime_binding'
        actual=runtime_factory is None and input_loader is None and c['production']
        if runtime_factory is None:runtime_factory=resolve_default_runtime_factory();s['default_runtime_contract_passed']=True
        stage='nine_original_binding_and_C153_reaudit';d,anchors=(load if input_loader is None else input_loader)(paths,out/'binding',c);identity=prior.parent.fixed_digest(d);budget();stage='all_stencil_sealed_before_model';params,points,bindings,seals=freeze_stencil(d,anchors,out);seals['STENCIL_FROZEN_BEFORE_MODEL.json']=sha(out/'STENCIL_FROZEN_BEFORE_MODEL.json');phase=Phase(points);unchanged();budget();checkpoint();stage='pinned_native_CPU_model_initialization';rt=runtime_factory(out/'model_runtime',d,phase,calls)
        if actual:
            compare({k:v for k,v in rt.pins.items() if k!='torch'},{k:v for k,v in d['modelpins'].items() if k!='torch'},'Actual model pins');need(calls['initialisation_model_forwards']==1,'One initialization forward');s['real_model_execution_claim']=True
        unchanged();replay=[];repeat_failures=[];by={sp['state_id']:sp for sp in registry()};groups={};preflight_done=False
        for a,label,sid,kind in request_order():
            budget()
            if kind!='preflight' and not preflight_done:
                need(len(records)==168,'All168 baseline checks before probes');unchanged();s['both_zero_panels_replayed_before_probes']=True;preflight_done=True
            stage='zero_preflight' if kind=='preflight' else 'fixed_signed_probes' if kind=='probe' else 'fixed_repeat_panels';sp=by[sid];q=d['points'][sid];h=anchors[a];p=params[(a,label)];point=points[(a,label,sid)];phase.authorize(a,label,sid,kind)
            with (out/'ALL_NATIVE_ATTEMPTS.jsonl').open('a') as f:f.write(json.dumps(dict(request_index=phase.at,anchor=a,label=label,state_id=sid,phase=kind,expected_point_sha256=phase.ids[(a,label,sid)]))+'\n')
            met=rt.metrics(point,sp['case'],True);g=endpoint_gates(met,sp['case'],point,q['prepared'],True);e=prior.effective(p,d['weights'][sid],ARM);r=dict(anchor=a,label=label,alpha=LABELS[label],phase=kind,**sp,metrics=met,gates=g,objective_parts=prior.loss(met['observed'],met['KL'],point,q['prepared'],e,sp['case']));records.append(r);key=(a,label,kind);groups.setdefault(key,[]).append(r)
            with (out/'ALL_NATIVE_MEASUREMENTS.jsonl').open('a') as f:f.write(json.dumps(r,allow_nan=False)+'\n')
            i=[x['state_id'] for x in registry()].index(sid)
            if kind=='preflight':
                ref=h['rows'][i];compare(safe.tensor_hash(point),ref['functional_point_sha256'],'Archived zero point');close(met['observed'],ref['observed'],'Native/archived-functional zero observations',2e-6);need(abs(met['KL']-ref['KL'])<=2e-7,'Native/archived-functional zero KL')
                if h['zero_native_references'] is not None:
                    rc=repeat_check(met,h['zero_native_references'][i]);need(rc['same_point_identity'] and rc['within_replay_tolerance'],'Initial C150 native replay before probes')
            if label=='plus_one' and kind=='probe':
                ref=h['plus_references'][i]
                if h['step']==12:
                    rc=repeat_check(met,ref)
                    if not(rc['same_point_identity'] and rc['within_replay_tolerance']):replay.append(dict(anchor=a,state_id=sid,comparison=rc))
                else:
                    compare(safe.tensor_hash(point),ref['functional_point_sha256'],'Original step02 point');err=close(met['observed'],ref['observed'],'Step01 plus-one/native archived-functional parity',2e-6);need(abs(met['KL']-ref['KL'])<=2e-7,'Step02 recorded-functional KL parity')
            if kind=='repeat':
                origin=groups[(a,label,'preflight' if label=='zero' else 'probe')][i];rc=repeat_check(met,origin['metrics'])
                if not(rc['same_point_identity'] and rc['within_replay_tolerance']):repeat_failures.append(dict(anchor=a,label=label,state_id=sid,comparison=rc))
            if len(groups[key])==84:
                panel=objective_panel(groups[key],p,d);panels[key]=panel;dump(out/'panels'/f'{a}__{label}__{kind}.json',dict(records=groups[key],summary=panel));unchanged();print('C154_PANEL',a,label,kind,'all84','objective',panel['objective'],flush=True)
            checkpoint()
        phase.finish();unchanged();need(calls['model_forwards']==3025 and calls['preflight_metrics']==168 and calls['path_metrics']==1344,'Full fixed native denominator');stage='fixed_direction_arithmetic';diagnostics=diagnose(anchors,panels,bindings,out);dump(out/'ORIGINAL_PLUS_ONE_REPLAY_FAILURES.json',replay);dump(out/'REPEATABILITY_FAILURES.json',repeat_failures);s.update(complete=True,decision='C154_FIXED_DIRECTION_DIAGNOSTIC_COMPLETE_NO_CANDIDATE_SELECTED' if not replay and not repeat_failures else 'C154_DIAGNOSTIC_COMPLETE_REPLAY_DRIFT_NO_DERIVATIVE_CONCLUSION',repeatable=not repeat_failures,original_plus_one_replay_consistent=not replay,diagnostics=diagnostics,all1176_points_frozen_before_model=True);stage='complete';write_report(out,s)
    except BaseException:errors.append(traceback.format_exc());s.update(complete=False,decision='C154_INCOMPLETE_NO_DIRECTIONAL_CONCLUSION');print(errors[-1],flush=True)
    finally:
        checks={}
        for k,p in paths.items():
            try:checks[k]=sha(p)==c['pins'][k]['sha256']
            except Exception:checks[k]=False
        s['original_inputs_unchanged']=checks
        if set(checks)!=set(INPUT_KEYS) or not all(checks.values()):s.update(complete=False,decision='C154_INCOMPLETE_NO_DIRECTIONAL_CONCLUSION');errors.append('Original input missing or changed')
        checkpoint()
    return dict(s,errors=errors)

def diagnose(anchors,panels,bindings,out):
    result={}
    for a,_ in ANCHORS:
        h=anchors[a];delta=h['after']-h['before'];g,agg=prior.aggregate(h['rows'],h['before'].shape);pred=float(np.sum(g*delta));response=np.sum(np.asarray(agg['response_weights'])[:,None]*np.array([r['response_gradient'] for r in h['rows']]).reshape(84,-1),axis=0).reshape(delta.shape);guard=np.mean(np.array([r['guard_gradient'] for r in h['rows']]),axis=0);pp=dict(objective=pred,response_objective=float(np.sum(response*delta)),mean_guard=float(np.sum(guard*delta)));zero=panels[(a,'zero','preflight')];rows=[]
        for size,minus,plus in [(.25,'minus_quarter','plus_quarter'),(.5,'minus_half','plus_half'),(1.,'minus_one','plus_one')]:
            m=panels[(a,minus,'probe')];p=panels[(a,plus,'probe')];rows.append(dict(h=size,scalar_diagnostics={name:finite_difference(zero[name],m[name],p[name],size,pp[name]) for name in pp},minus_maximizing_ids=m['maximizing_state_ids'],plus_maximizing_ids=p['maximizing_state_ids'],zero_maximizing_ids=zero['maximizing_state_ids'],active_sets_changed=bool(m['active_sets']!=zero['active_sets'] or p['active_sets']!=zero['active_sets']),changed_float32_points={label:sum(not r['same_float32_point_as_zero'] for r in bindings if r['anchor']==a and r['label']==label) for label in (minus,plus)}))
        diff=[x['scalar_diagnostics']['objective']['central_slope'] for x in rows];result[a]=dict(archived_step=h['step'],direction_L2=float(np.linalg.norm(delta)),archived_gradient_dot_actual_update=pred,archived_base_objective=agg['objective'],native_minus_archived_base_objective=zero['objective']-agg['objective'],archived_base_maximizing_state_ids=agg['maximizing_state_ids'],base_native_objective=zero['objective'],plus_one_native_objective=panels[(a,'plus_one','probe')]['objective'],observed_full_update_objective_change=panels[(a,'plus_one','probe')]['objective']-zero['objective'],central_slope_spread=float(max(diff)-min(diff)),FD_scales=rows,zero_repeat_objective_change=panels[(a,'zero','repeat')]['objective']-zero['objective'],plus_one_repeat_objective_change=panels[(a,'plus_one','repeat')]['objective']-panels[(a,'plus_one','probe')]['objective'],AD_accuracy_certified=False,derivative_agreement_rule=None)
    dump(out/'FIXED_DIRECTION_DIAGNOSTICS.json',result);return result

def write_report(out,s):
    lines=['# C154：真实GPT-2固定更新方向诊断','','判定：'+s['decision'],'','两个非零控制头位置、两条归档实际更新方向；固定±1/4、±1/2、±1及零点/+1复测。全部84态，无新训练或候选选择。','','|位置|归档梯度预测|实际完整更新目标变化|三步长中心斜率跨度|','|---|---:|---:|---:|']
    for a,r in s['diagnostics'].items():lines.append(f"|{a}|{r['archived_gradient_dot_actual_update']}|{r['observed_full_update_objective_change']}|{r['central_slope_spread']}|")
    lines+=['','斜率单位为对“归档实际更新倍数alpha”的导数，不是单位范数方向导数。alpha1保留归档after张量原字节；其余按已登记插值生成。','差分不一致不能单独证明AD错误。最大值/铰链切换、有限步长、float32物理提交和前向数值误差均可能参与；重复一致不界定系统误差。零点重放、原+1重放和复测结果分别保留。无导数一致性验收阈值，无曲率/收敛/架构可行性结论。','本轮全部为已暴露训练态，任何测点都未选为写入器或部署。LC2及单折仍未闭合。'];(out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in INPUT_KEYS:p.add_argument('--'+k.lower(),required=True)
    p.add_argument('--out',required=True);args=p.parse_args();s=run({k:Path(getattr(args,k.lower())) for k in INPUT_KEYS},args.out);print(json.dumps(s,ensure_ascii=False,indent=2),flush=True)
    if not s['complete']:raise SystemExit(1)
