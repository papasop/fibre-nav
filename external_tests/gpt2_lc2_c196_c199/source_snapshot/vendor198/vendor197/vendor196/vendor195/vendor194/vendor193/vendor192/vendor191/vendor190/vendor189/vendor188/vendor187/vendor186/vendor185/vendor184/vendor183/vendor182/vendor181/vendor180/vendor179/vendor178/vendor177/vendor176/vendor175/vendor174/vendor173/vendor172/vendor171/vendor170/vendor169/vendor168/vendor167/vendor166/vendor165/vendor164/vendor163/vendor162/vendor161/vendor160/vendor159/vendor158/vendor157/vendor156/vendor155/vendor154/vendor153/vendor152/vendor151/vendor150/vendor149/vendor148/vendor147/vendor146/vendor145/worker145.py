"""All84 fixed directional probes at frozen C143; every point is diagnostic only."""
import argparse,traceback
from core145 import *
from inputs145 import load

def run(paths,out,c=None,runtime_factory=None,backend_factory=None,input_loader=None):
    c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);started=time.monotonic();stage='registered';data=None;backend=None;preflight=[];parities=[];gradients=[];records=[];errors=[];seals={};control_records=[]
    calls=dict(model_forwards=0,initialisation_model_forwards=0,preflight_metric_attempts=0,preflight_metrics=0,path_metric_attempts=0,path_metrics=0)
    s=dict(protocol=PROTOCOL,complete=False,decision='C145_INCOMPLETE_DIRECTIONAL_DIAGNOSTIC_NOT_ACCEPTED',synthetic_fixture=not c['production'],real_model_execution_claim=False,parent_decision_reclassified=False,training_states=84,all84_remain_exposed_training=True,confirmation_states_evaluated=0,excluded_input_states_evaluated=0,new_states_prepared=0,new_model_features_acquired=0,new_training_steps=0,new_head_fits=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,full_response_Jacobian_constructed=False,new_backward_calls=0,diagnostic_control_vectors_generated=0,optimizer_updates=0,GPT2_optimizer_updates=0,LoRA_A_optimizer_updates=0,base_writer_updates=0,teacher_calls=0,backtracking_steps=0,online_response_repair_steps=0,candidate_selected=False,writer_deployed=False,thresholds_changed=False,archived_source_executed=False,LC2_closed=False,LC2_single_fold_passed=False,AD_accuracy_independently_certified=False,global_or_local_feasibility_infeasibility_established=False,requested_scalar_backward_calls=168,requested_preflight_records=336,requested_endpoint_records=1512,requested_probe_configurations=8,requested_configurations_including_baseline=9,scope=design()['scope'])
    def budget():need(time.monotonic()-started<TIMEOUT,'C145 fixed120-minute worker budget exceeded')
    def checkpoint(full=False):
        stats=backend.stats() if backend is not None else {};s.update(stage=stage,elapsed_worker_seconds=time.monotonic()-started,runtime_calls=calls,new_model_forwards=calls['model_forwards'],backend_stats=stats,new_backward_calls=stats.get('scalar_backward_calls',0),gradient_state_records=len(gradients),preflight_records=len(preflight),functional_parity_records=len(parities),endpoint_records=len(records));dump(out/'SUMMARY.json',dict(s,errors=errors))
        if full:
            for n,obj in [('PREFLIGHT_PROGRESS.json',preflight),('FUNCTIONAL_PARITY_PROGRESS.json',parities),('GRADIENT_PROGRESS.json',gradients),('ENDPOINT_PROGRESS.json',records)]:dump(out/n,obj)
    def seal(name):seals[name]=sha(out/name)
    def unchanged():
        for n,h in seals.items():need(sha(out/n)==h,'Frozen artifact changed: '+n)
        if data is not None:compare(frozen_digest(data),data['fixed_identities'],'Frozen inputs/control changed')
    try:
        need(not(out/'SUMMARY.json').exists(),'Fresh C145 output required');dump(out/'DESIGN_FROZEN_BEFORE_MODEL.json',design());seal('DESIGN_FROZEN_BEFORE_MODEL.json');checkpoint();stage='five_original_archive_bindings'
        data=(load if input_loader is None else input_loader)(paths,out,c);data['fixed_identities']=frozen_digest(data);budget()
        for n in ('INPUTS_FROZEN.json','ALL84_EXPOSED_TRAINING_REGISTRY.json','FROZEN_INPUT_IDENTITIES.json','MODEL_PINS_REFERENCE.json','BASELINE_C143_CONTROL.npz','ALL84_FROZEN_TRAINING_INPUTS.npz','C144_BOUND_RECEIPT.json'):
            need((out/n).is_file(),'Missing frozen input '+n);seal(n)
        rank=len(data['a']);need(rank==36 if c['production'] else 0<rank<=36,'Fixed shared dimension');need(set(data['points'])=={r['state_id'] for r in registry()},'Exactly84 training states');s.update(shared_dimension=rank,C143_parent_decision=data['C143_summary']['decision'],C144_parent_decision=data['C144_summary']['decision'])
        phase=Phase(data);stage='model_initialisation';checkpoint();actual=runtime_factory is None and backend_factory is None and input_loader is None
        if runtime_factory is None:
            from runtime145 import factory
            runtime_factory=factory
        rt=runtime_factory(out/'model_runtime',data,phase,calls);s['real_model_execution_claim']=actual;compare({k:v for k,v in rt.pins.items() if k!='torch'},{k:v for k,v in data['modelpins'].items() if k!='torch'},'Pinned model identity');stage='all84_four_native_preflights';checkpoint();native={}
        for sp in registry():
            budget();sid=sp['state_id'];p=data['points'][sid]
            for kind,point,write in [('prepared',p['prepared'],False),('raw',p['raw'],True),('C140',parent.proposal(p['prepared'],p['raw'],data['V'],data['C140_a'])[0],True),('C143',parent.proposal(p['prepared'],p['raw'],data['V'],data['a'])[0],True)]:
                phase.authorize((sid,kind),point,sp['case'],write);met=rt.metrics(point,sp['case'],write);g,_,_=native_gates(met,sp['case'],point,write)
                if not write:need(all(g.values()),'Invalid preparation blocks before gradients')
                else:endpoint_gates(met,sp['case'],point,p['prepared'],True)
                comparisons=[repeat_check(met,old) for old in data['refs'][(sid,kind)]];preflight.append(dict(**sp,kind=kind,metrics=met,comparisons=comparisons));checkpoint();need(all(q['same_point_identity'] and q['within_replay_tolerance'] for q in comparisons),'Native original replay drift blocks before gradients')
                if kind=='C143':native[sid]=met
            checkpoint(True);print('C145_PREFLIGHT',sid,'OK',flush=True)
        unchanged();phase.begin_parity();stage='all84_functional_native_parity';checkpoint()
        if backend_factory is None:
            from functional145 import factory
            backend_factory=factory
        backend=backend_factory(rt,data,phase,calls)
        for sp in registry():
            budget();sid=sp['state_id'];r=backend.parity(sp);compare(r['state_id'],sid);need(0<=r['point_parity_error_Linf']<=2e-6,'Invalid point parity');error=close(r['observed'],native[sid]['observed'],'Frozen functional/native observation parity',2e-6);kl=abs(r['KL']-native[sid]['KL']);need(kl<=2e-7,'Frozen functional/native KL parity');parities.append(dict(r,observed_parity_Linf=error,KL_parity_absolute_error=kl));checkpoint()
        need(backend.stats()['scalar_backward_calls']==0,'No backward before complete functional/native parity');s['all84_parity_complete_before_backwards']=True;backend.verify(rehash=True);unchanged();phase.begin_gradients();stage='two_scalars_at_one_frozen_control';checkpoint(True);parity_by={r['state_id']:r for r in parities}
        for index,sp in enumerate(registry()):
            budget();sid=sp['state_id'];p=data['points'][sid];r=backend.gradient(sp,parity_by[sid])
            for k,v in sp.items():compare(r[k],v,'Scalar-gradient state binding/'+k)
            compare(r['baseline_coefficient_digest'],row_digest(data['a']));need(r['forward_parity_checked_before_backwards'] is True,'Gradient graph forward parity not checked');need(0<=r['point_parity_error_Linf']<=2e-6,'Gradient point parity');close(r['observed'],parity_by[sid]['observed'],'Gradient/frozen functional observations',2e-6);need(abs(r['KL']-parity_by[sid]['KL'])<=2e-7,'Gradient/frozen KL parity')
            point=parent.proposal(p['prepared'],p['raw'],data['V'],data['a'])[0];expected=auditcore.loss(r['observed'],r['KL'],point,p['prepared'],data['a'],sp['case']);compare(r['original_objective_parts'],expected,'Original objective',atol=1e-10,rtol=1e-9);compare(r['original_objective'],sum(expected.values()),atol=1e-10,rtol=1e-9);compare(r['response_ratio'],qvalue(r['observed']));farray(np.asarray(r['original_gradient'],float),(rank,));farray(np.asarray(r['response_gradient'],float),(rank,));gradients.append(r);checkpoint(True)
            if (index+1)%12==0:print('C145_GRADIENT',index+1,'/84 scalar_backwards',backend.stats()['scalar_backward_calls'],flush=True)
        stats=backend.stats();need(stats['scalar_backward_calls']==168 and stats['gradient_state_records']==84 and stats['functional_endpoint_calls']==168,'Fixed scalar gradient budget');backend.verify(rehash=True);unchanged();stage='seal_all_directions_controls_and_points';checkpoint()
        directions,detail=directions_from_rows(gradients,rank);controls,control_records=probe_controls(data['a'],directions);maps={};outcomes=[]
        np.savez_compressed(out/'TWO_FROZEN_DIRECTIONS.npz',**directions);dump(out/'DIRECTION_RULES_AND_WEIGHTS.json',detail);np.savez_compressed(out/'ALL9_FROZEN_CONTROLS.npz',**controls);dump(out/'ALL9_CONTROL_REGISTRY.json',control_records)
        for sp in registry():
            sid=sp['state_id'];p=data['points'][sid]
            for cfg in configurations():
                name=cfg['config_id'];point,diag=parent.proposal(p['prepared'],p['raw'],data['V'],controls[name]);maps[(sid,name)]=point;outcomes.append(dict(**sp,**cfg,point_sha256=safe.tensor_hash(point),control_digest=row_digest(controls[name]),proposal_diagnostics=diag))
        np.savez_compressed(out/'ALL756_FROZEN_DIAGNOSTIC_POINTS.npz',**{sid+'__'+name:p for (sid,name),p in maps.items()});dump(out/'ALL756_FROZEN_POINT_REGISTRY.json',outcomes)
        for n in ('TWO_FROZEN_DIRECTIONS.npz','DIRECTION_RULES_AND_WEIGHTS.json','ALL9_FROZEN_CONTROLS.npz','ALL9_CONTROL_REGISTRY.json','ALL756_FROZEN_DIAGNOSTIC_POINTS.npz','ALL756_FROZEN_POINT_REGISTRY.json'):seal(n)
        dump(out/'ALL_PROBES_SEALED_BEFORE_MEASUREMENTS.json',dict(native_probe_endpoint_measurements_so_far=0,training_states=84,configurations=9,points=756,base_control_digest=row_digest(data['a']),artifact_seals=dict(seals),candidate_selected=False));seal('ALL_PROBES_SEALED_BEFORE_MEASUREMENTS.json');phase.freeze(maps);s.update(all_probe_points_frozen_before_measurements=True,diagnostic_control_vectors_generated=8,direction_availability={k:v['available'] for k,v in detail.items()});stage='fixed_two_round1512_native_endpoints';checkpoint(True)
        first={};drifts=[];repeats=[];gby={r['state_id']:r for r in gradients};control_by={r['config_id']:r for r in control_records}
        for rnd in (0,1):
            for index,sp in enumerate(registry()):
                sid=sp['state_id'];p=data['points'][sid]
                for cfg in configurations():
                    budget();name=cfg['config_id'];point=maps[(sid,name)];phase.authorize((rnd,sid,name),point,sp['case'],True);met=rt.metrics(point,sp['case'],True);g=endpoint_gates(met,sp['case'],point,p['prepared'],True);change=controls[name]-data['a'];predicted=gby[sid]['response_ratio']+float(np.asarray(gby[sid]['response_gradient'])@change);measured=qvalue(met['observed']);r=dict(**sp,**cfg,round=rnd,metrics=met,gates=g,control_digest=row_digest(controls[name]),control_projection_active=bool(control_by[name]['projected_coordinates']),AD_predicted_response_ratio=predicted,measured_response_ratio=measured,response_ratio_linearization_residual=measured-predicted,linearization_is_not_accuracy_or_causality_proof=True)
                    if name=='C143_baseline':
                        q=repeat_check(met,native[sid]);r['baseline_replay']=q
                        if not(q['same_point_identity'] and q['within_replay_tolerance']):drifts.append(dict(state_id=sid,round=rnd,comparison=q))
                    if rnd==0:first[(sid,name)]=met
                    else:
                        q=repeat_check(met,first[(sid,name)]);r['repeat_comparison']=q
                        if not(q['same_point_identity'] and q['within_replay_tolerance']):repeats.append(dict(state_id=sid,config_id=name,comparison=q))
                    records.append(r);dump(out/'endpoints'/f'record{len(records)-1:04d}.json',r);checkpoint()
                print('C145_ENDPOINT',rnd,index+1,'/84',sid,[(r['config_id'],r['gates']['complete_pass'],r['metrics']['response_drift']) for r in records[-9:]],flush=True)
                if (index+1)%12==0:unchanged();checkpoint(True)
        phase.finish();unchanged();backend.verify(rehash=True,check_chart=False);need(calls['preflight_metrics']==336 and calls['path_metrics']==1512,'Fixed native request denominator');need(calls['model_forwards']==4033,'Model forward accounting differs');sums=tables(records);consistent=not drifts and not repeats
        allpass=[cfg['config_id'] for cfg in configurations()[1:] if consistent and all(sums['all84'][str(j)][cfg['config_id']]['passed']==84 for j in (0,1))]
        nondestructive=[cfg['config_id'] for cfg in configurations()[1:] if consistent and all(sums['all84'][str(j)][cfg['config_id']]['rescued_vs_baseline'] and not sums['all84'][str(j)][cfg['config_id']]['destroyed_vs_baseline'] for j in (0,1))]
        decision='C145_COMPLETE_REPLAY_DRIFT_NO_DIRECTIONAL_INTERPRETATION' if not consistent else 'C145_FIXED_DIRECTIONAL_DIAGNOSTIC_COMPLETE_NO_CANDIDATE_SELECTED'
        interpretation='ALL84_TRAINING_PASS_OBSERVED_AT_REGISTERED_PROBE_NO_SELECTION' if allpass else ('NONDESTRUCTIVE_TRAINING_IMPROVEMENT_OBSERVED_IN_FIXED_GRID' if nondestructive else 'NO_NONDESTRUCTIVE_IMPROVEMENT_OBSERVED_IN_FIXED_GRID')
        for n,obj in [('ALL1512_DIAGNOSTIC_ENDPOINTS.json',records),('DIAGNOSTIC_SUMMARIES.json',sums),('BASELINE_REPLAY_FAILURES.json',drifts),('REPEATABILITY_FAILURES.json',repeats)]:dump(out/n,obj)
        s.update(complete=True,decision=decision,repeatable=not repeats,baseline_replay_consistent=not drifts,diagnostic_interpretation=(interpretation if consistent else None),diagnostic_configurations_with_all84_passed=allpass,diagnostic_configurations_with_rescue_and_no_destruction=nondestructive,diagnostic_summaries=sums,next_branch='PREREGISTER_SEPARATE_TRAINING_PROTOCOL_FROM_FIXED_DIAGNOSTIC' if consistent else 'RESOLVE_BASELINE_OR_REPEAT_DRIFT',no_probe_selected_as_writer=True)
        lines=['# C145：固定C143最终头的全84态方向诊断','','原训练目标下降方向与响应softmax32方向；全部正反号、两个固定步长均保留，无优化器或迭代训练。','','|配置|首轮通过|重复轮通过|首轮最大响应漂移|新增失败/首轮|获救/首轮|','|---|---:|---:|---:|---:|---:|']
        for cfg in configurations():
            name=cfg['config_id'];q=sums['all84']['0'][name];r=sums['all84']['1'][name];lines.append(f"|{name}|{q['passed']}/84|{r['passed']}/84|{q['maximum_response_drift']}|{len(q['destroyed_vs_baseline'])}|{len(q['rescued_vs_baseline'])}|")
        lines+=['','判定：'+decision,'诊断描述：'+str(s['diagnostic_interpretation']),'','两轮是同84态的重复测量。所有状态已暴露并属于训练。未选最优方向/步长/头，未部署；即使某诊断点84/84，也不构成独立确认或LC2闭合。','','本次在固定最终a上计算168个离线标量梯度向量，没有构造完整响应Jacobian，没有SVD或方程求解。AD线性预测与有限步差异仅为诊断，不证明导数错误、不可行、曲率界或收敛；在有限网格中没有改善也不排除其它方向、步长或控制结构。'];(out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');stage='complete'
    except BaseException:errors.append(traceback.format_exc());s.update(complete=False,decision='C145_INCOMPLETE_DIRECTIONAL_DIAGNOSTIC_NOT_ACCEPTED');print(errors[-1],flush=True)
    finally:
        checks={}
        for k,p in paths.items():
            try:checks[k]=sha(p)==c['pins'][k]['sha256']
            except Exception:checks[k]=False
        s['original_inputs_unchanged']=checks
        if not all(checks.values()):s.update(complete=False,decision='C145_INCOMPLETE_DIRECTIONAL_DIAGNOSTIC_NOT_ACCEPTED');errors.append('Original input changed')
        checkpoint(True)
    return dict(s,errors=errors)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for k in INPUT_KEYS:parser.add_argument('--'+k.lower(),required=True)
    parser.add_argument('--out',required=True);args=parser.parse_args();s=run({k:Path(getattr(args,k.lower())) for k in INPUT_KEYS},args.out);print(json.dumps(s,ensure_ascii=False,indent=2),flush=True)
    if not s['complete']:raise SystemExit(1)
