"""All84 paired FP32 routes, fixed two-state FP64 diagnostic; no backwards."""
import argparse,resource,traceback
from core155 import *
from inputs155 import load

def resolve_default_runtime_factory():
    import runtime155
    need(Path(runtime155.__file__).resolve()==ROOT/'runtime155.py' and runtime155.factory.__module__=='runtime155','C155 actual default runtime identity');runtime155.resolve_pinned_factory();return runtime155.factory
def resolve_default_pair_factory():
    import paired155
    need(Path(paired155.__file__).resolve()==ROOT/'paired155.py' and paired155.factory.__module__=='paired155','C155 actual paired backend identity');return paired155.factory

def run(paths,out,c=None,runtime_factory=None,pair_factory=None,input_loader=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);c=contract() if c is None else c;start=time.monotonic();stage='registered';errors=[];records=[];groups={};panels={};checks=[];replays=[];repeats=[];seals={};d=None;phase=None
    calls=dict(model_forwards=0,initialisation_model_forwards=0,preflight_metric_attempts=0,preflight_metrics=0,path_metric_attempts=0,path_metrics=0,functional_FP32_forwards=0,functional_FP64_forwards=0,functional_FP32_metrics=0,FP64_focus_metrics=0,FP64_baseline_forwards=0)
    s=dict(protocol=PROTOCOL,complete=False,decision='C155_INCOMPLETE_NO_FORWARD_CAUSE_CONCLUSION',synthetic_fixture=not c['production'],real_model_execution_claim=False,all84_remain_exposed_training=True,training_states_requested=84,FP64_focus_state_ids=list(FOCUS),requested_metric_records=1530,requested_model_forwards=3062,new_model_forwards=0,new_backward_calls=0,new_training_steps=0,new_head_fits=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,new_states_prepared=0,new_model_features_acquired=0,confirmation_states_evaluated=0,excluded_input_states_evaluated=0,GPT2_optimizer_updates=0,LoRA_A_optimizer_updates=0,base_writer_updates=0,teacher_calls=0,backtracking_steps=0,online_response_repair_steps=0,archived_source_executed=False,gradients_independently_recomputed=False,AD_accuracy_certified=False,systematic_forward_error_bound_established=False,curvature_bound_established=False,thresholds_changed=False,candidate_selected=False,writer_deployed=False,parent_decision_reclassified=False,FP64_training_qualification_evaluated=False,LC2_closed=False,LC2_single_fold_passed=False,scope=design()['scope'])
    def checkpoint():
        s.update(stage=stage,elapsed_worker_seconds=time.monotonic()-start,process_peak_RSS_KiB=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),runtime_calls=calls,new_model_forwards=calls['model_forwards'],metric_records=len(records));dump(out/'SUMMARY.json',dict(s,errors=errors))
    def budget():need(time.monotonic()-start<TIMEOUT,'C155 fixed120-minute worker budget')
    def unchanged():
        for n,h in seals.items():compare(sha(out/n),h,'C155 sealed file/'+n)
        if d is not None:compare(prior.parent.fixed_digest(d),identity,'Fixed router/basis/writer/data')
        if phase is not None:phase.verify()
    def append(r):
        records.append(r)
        with (out/'ALL_METRIC_RECORDS.jsonl').open('a') as f:f.write(json.dumps(r,allow_nan=False)+'\n')
        groups.setdefault((r['label'],r['stage'],r['route']),[]).append(r)
    try:
        need(not (out/'SUMMARY.json').exists(),'Fresh C155 session');dump(out/'DESIGN_FROZEN_BEFORE_MODEL.json',design());checkpoint();actual=c['production'] and runtime_factory is None and pair_factory is None and input_loader is None
        stage='default_runtime_binding';runtime_factory=resolve_default_runtime_factory() if runtime_factory is None else runtime_factory
        stage='ten_original_binding_and_C154_reaudit';d,anchors,parent_points,refs,parent_params=(load if input_loader is None else input_loader)(paths,out/'binding',c);identity=prior.parent.fixed_digest(d);budget();stage='seal_all588_before_model';params,points,seals=freeze(d,anchors,out,parent_points,parent_params);phase=Phase(points);unchanged();checkpoint();stage='original_FP32_model_initialization';rt=runtime_factory(out/'model_runtime',d,phase,calls)
        if actual:compare({k:v for k,v in rt.pins.items() if k!='torch'},{k:v for k,v in d['modelpins'].items() if k!='torch'},'Actual original model pins');need(calls['initialisation_model_forwards']==1,'One original FP32 baseline forward');s['real_model_execution_claim']=True
        stage='paired_backend_and_FP64_reference';pair=(resolve_default_pair_factory() if pair_factory is None else pair_factory)(rt,calls);unchanged();by={sp['state_id']:sp for sp in registry()};preflight_done=False
        for label,sid,st in requests():
            budget();stage='all84_zero_preflight' if st=='preflight' else 'fixed_signed_pairs' if st=='probe' else 'repeat_zero_plus_one'
            if st!='preflight' and not preflight_done:need(len(groups[('zero','preflight','native_FP32')])==84,'All84 paired zero before probes');s['all84_zero_pairs_checked_before_probes']=True;preflight_done=True
            sp=by[sid];q=d['points'][sid];p=points[(label,sid)];phase.authorize(label,sid,st)
            with (out/'ALL_ATTEMPTS.jsonl').open('a') as f:f.write(json.dumps(dict(label=label,state_id=sid,stage=st,expected_point_sha256=safe.tensor_hash(p),routes=['native_FP32','functional_FP32']+(['functional_FP64'] if sid in FOCUS else [])))+'\n')
            native=rt.metrics(p,sp['case'],True);rr=[]
            for route in ('native_FP32','functional_FP32'):
                met=native if route=='native_FP32' else pair.functional32(p,sp['case'])
                e=prior.effective(params[label],d['weights'][sid],ARM);r=dict(label=label,alpha=LABELS[label],stage=st,route=route,**sp,metrics=met,gates=endpoint_gates(met,sp['case'],p,q['prepared'],True),objective_parts=prior.loss(met['observed'],met['KL'],p,q['prepared'],e,sp['case']));append(r);rr.append(r)
                if st=='repeat':
                    first=next(x for x in groups[(label,'preflight' if label=='zero' else 'probe',route)] if x['state_id']==sid);repeats.append(dict(route=route,label=label,state_id=sid,comparison=repeat_check(met,first['metrics'])))
            pc=parity(native,rr[1]['metrics']);checks.append(dict(label=label,state_id=sid,stage=st,**pc));rc=repeat_check(native,refs[(label,sid,st)]);replays.append(dict(label=label,state_id=sid,stage=st,comparison=rc))
            if sid in FOCUS:
                r=dict(label=label,alpha=LABELS[label],stage=st,route='functional_FP64',**sp,precision_measurement=pair.precision64(p,sp['case']));validate_precision(r['precision_measurement'],p);append(r)
            if st=='preflight':need(rc['same_point_identity'] and rc['within_replay_tolerance'],'Original C154 late zero replay before probes');need(pc['same_point'] and pc['observed_Linf']<=2e-6 and pc['KL_absolute']<=2e-7,'Native/functional zero parity before probes')
            for route in ('native_FP32','functional_FP32'):
                key=(label,st,route)
                if len(groups[key])==84 and key not in panels:panels[key]=prev.objective_panel(groups[key],params[label],d);dump(out/'panels'/f'{label}__{st}__{route}.json',dict(records=groups[key],summary=panels[key]));print('C155_PANEL',label,st,route,'all84','objective',panels[key]['objective'],flush=True)
            if sid==registry()[-1]['state_id']:pair.verify(rehash=True);unchanged()
            checkpoint()
        phase.finish();pair.verify(rehash=True);unchanged();need(len(records)==1530 and calls['model_forwards']==3062 and calls['preflight_metrics']==84 and calls['path_metrics']==672 and calls['functional_FP32_metrics']==756 and calls['FP64_focus_metrics']==18 and calls['functional_FP32_forwards']==1512 and calls['functional_FP64_forwards']==37 and calls['FP64_baseline_forwards']==1,'Complete C155 fixed denominators')
        stage='coordinate_and_baseline_diagnostics';diag=diagnose(records,panels,out);dump(out/'ALL756_FP32_PARITY_COMPARISONS.json',checks);dump(out/'ALL756_C154_NATIVE_REPLAYS.json',replays);dump(out/'ALL336_FP32_REPEAT_COMPARISONS.json',repeats);native_ok=all(r['comparison']['within_replay_tolerance'] and r['comparison']['same_point_identity'] for r in replays);repeat_ok=all(r['comparison']['within_replay_tolerance'] and r['comparison']['same_point_identity'] for r in repeats);parity_ok=all(r['same_point'] and r['observed_Linf']<=2e-6 and r['KL_absolute']<=2e-7 for r in checks)
        s.update(complete=True,decision='C155_PAIRED_FORWARD_PRECISION_DIAGNOSTIC_COMPLETE_NO_CANDIDATE_SELECTED' if native_ok and repeat_ok and parity_ok else 'C155_DIAGNOSTIC_COMPLETE_ROUTE_OR_REPLAY_DRIFT_NO_CAUSAL_CONCLUSION',C154_all756_native_replays_consistent=native_ok,FP32_repeats_consistent=repeat_ok,all756_FP32_pairs_within_original_parity_tolerance=parity_ok,maximum_native_functional_observed_difference=max(r['observed_Linf'] for r in checks),maximum_native_functional_KL_difference=max(r['KL_absolute'] for r in checks),diagnostics=diag,all588_point_entries_sealed_before_model=True);stage='complete';write_report(out,s)
    except BaseException:errors.append(traceback.format_exc());s.update(complete=False,decision='C155_INCOMPLETE_NO_FORWARD_CAUSE_CONCLUSION');print(errors[-1],flush=True)
    finally:
        original={}
        for k,p in paths.items():
            try:original[k]=sha(p)==c['pins'][k]['sha256']
            except Exception:original[k]=False
        s['original_inputs_unchanged']=original
        if set(original)!=set(INPUT_KEYS) or not all(original.values()):s['complete']=False;errors.append('Original inputs missing or changed')
        checkpoint()
    return dict(s,errors=errors)

def validate_precision(m,point):
    compare(m['point_sha256'],safe.tensor_hash(point),'FP64 exact FP32 point binding');a=farray(np.asarray(m['observed_local_reference'],float),(22,));b=farray(np.asarray(m['observed_FP32_response_reference'],float),(22,));offset=farray(np.asarray(m['response_reference_offset'],float),(6,));close(b[:6]-a[:6],offset,'Precision baseline decomposition',1e-12);exact(b[6:],a[6:],'Baseline-independent read margins');need(m['acceptance_evaluated'] is False and m['training_qualified'] is False,'FP64 is diagnostic only');need(np.isfinite([m['KL_local_reference'],m['KL_FP32_logits_reference_recomputed64']]).all(),'Finite precision KL')

def diagnose(records,panels,out):
    index={(r['route'],r['label'],r['state_id'],r['stage']):r for r in records};trajectories=[];precision=[]
    def r0(route,n,sid):return index[(route,n,sid,'preflight' if n=='zero' else 'probe')]
    for sp in registry():
        sid=sp['state_id']
        for route in ('native_FP32','functional_FP32'):
            o=lambda n:r0(route,n,sid)['metrics']['observed'];trajectories.append(dict(route=route,state_id=sid,scales=[coordinate_fd(o('zero'),o(m),o(p),h) for h,m,p in [( .25,'minus_quarter','plus_quarter'),(.5,'minus_half','plus_half'),(1.,'minus_one','plus_one')]]))
    for sid in FOCUS:
        observations={n:r0('functional_FP64',n,sid)['precision_measurement'] for n in LABELS};native={n:r0('native_FP32',n,sid)['metrics']['observed'] for n in LABELS};o=lambda n:observations[n]['observed_local_reference'];rep=[]
        for n in ('zero','plus_one'):
            a=observations[n];b=index[('functional_FP64',n,sid,'repeat')]['precision_measurement'];rep.append(dict(label=n,local_observed_change_Linf=float(np.max(abs(np.array(a['observed_local_reference'])-b['observed_local_reference']))),cross_observed_change_Linf=float(np.max(abs(np.array(a['observed_FP32_response_reference'])-b['observed_FP32_response_reference']))),KL_local_change=abs(a['KL_local_reference']-b['KL_local_reference'])))
        precision.append(dict(state_id=sid,panels={n:dict(precision_measurement=observations[n],FP64_local_minus_native_response=(np.array(o(n)[:6])-native[n][:6]).tolist(),FP64_shared_reference_minus_native_response=(np.array(observations[n]['observed_FP32_response_reference'][:6])-native[n][:6]).tolist(),FP64_minus_native_read_margin=(np.array(o(n)[6:])-native[n][6:]).tolist()) for n in LABELS},scales=[coordinate_fd(o('zero'),o(m),o(p),h) for h,m,p in [(.25,'minus_quarter','plus_quarter'),(.5,'minus_half','plus_half'),(1.,'minus_one','plus_one')]],repeats=rep))
    dump(out/'ALL84_FP32_COORDINATE_TRAJECTORIES.json',trajectories);dump(out/'TWO_FOCUS_PRECISION_DIAGNOSTICS.json',precision);z=lambda route:panels[('zero','preflight',route)];obj={}
    for route in ('native_FP32','functional_FP32'):
        obj[route]=dict(base_objective=z(route)['objective'],plus_one_change=panels[('plus_one','probe',route)]['objective']-z(route)['objective'],scales=[dict(h=h,central=(panels[(p,'probe',route)]['objective']-panels[(m,'probe',route)]['objective'])/(2*h),positive=(panels[(p,'probe',route)]['objective']-z(route)['objective'])/h,negative=(z(route)['objective']-panels[(m,'probe',route)]['objective'])/h) for h,m,p in [(.25,'minus_quarter','plus_quarter'),(.5,'minus_half','plus_half'),(1.,'minus_one','plus_one')]],panel_maximizing_state_ids={n:panels[(n,'preflight' if n=='zero' else 'probe',route)]['maximizing_state_ids'] for n in LABELS})
    return dict(all84_FP32_objectives=obj,FP64_focus_summaries=[dict(state_id=r['state_id'],repeats=r['repeats'],maximum_response_reference_offset=max(abs(x) for x in r['panels']['zero']['precision_measurement']['response_reference_offset'])) for r in precision],FP64_full84_objective_available=False,FP64_is_ground_truth=False,AD_error_established=False,systematic_error_bound_established=False,causal_failure_explanation_established=False)
def write_report(out,s):
    lines=['# C155：固定晚期方向的前向／精度诊断','','判定：'+s['decision'],'','全部84态保留原FP32口径；七点与零/+1复测均做原生/功能前向配对。FP64仅覆盖预登记两个已暴露训练态，使用同一封存float32坐标的精确double转换。','','FP64独立零点基线、原FP32响应基线和二者偏移分别归档；不把不同基线的差异记为写入改善。不计算不完整的全84态FP64目标，不选最优步长/精度/头。','','|路线|零点目标|+1目标变化|','|---|---:|---:|']
    for name,v in s['diagnostics']['all84_FP32_objectives'].items():lines.append(f"|{name}|{v['base_objective']}|{v['plus_one_change']}|")
    lines+=['','逐响应坐标、读裕度、极值坐标、铰链、两类基线和重复变化见配套JSON。功能前向、精度和有限差分均不是导数认证；FP64不是真值，重复一致不提供系统误差界。无训练、选头、确认或排除输入评估；LC2与单折仍未闭合。'];(out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in INPUT_KEYS:p.add_argument('--'+k.lower(),required=True)
    p.add_argument('--out',required=True);a=p.parse_args();s=run({k:Path(getattr(a,k.lower())) for k in INPUT_KEYS},a.out);print(json.dumps(s,ensure_ascii=False,indent=2),flush=True)
    if not s['complete']:raise SystemExit(1)
