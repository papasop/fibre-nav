"""Prospective fixed48 confirmation with sealed prewrite-only frozen proposals."""
import argparse,traceback
from core141 import *
from inputs141 import load

def run(paths,out,c=None,runtime_factory=None,input_loader=None):
    c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);started=time.monotonic();stage='registered';data=None;seals={};errors=[];controls=[];preps=[];features=[];records=[];repeat_failures=[]
    calls=dict(model_forwards=0,initialisation_model_forwards=0,metric_attempts=0,metric_calls=0,feature_attempts=0,feature_calls=0)
    s=dict(protocol=PROTOCOL,complete=False,decision='C141_INCOMPLETE_NO_CONFIRMATION',synthetic_fixture=not c['production'],real_model_execution_claim=False,requested_confirmation_states=N,confirmation_states_evaluated=0,requested_preparation_records=96,requested_endpoint_records=288,
        new_training_steps=0,new_backward_calls=0,new_head_fits=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,teacher_calls=0,backtracking_steps=0,online_response_repair_steps=0,excluded_input_states_evaluated=0,
        candidate_selected=False,writer_deployed=False,thresholds_changed=False,parent_decision_reclassified=False,LC2_closed=False,LC2_single_fold_passed=False,archived_source_executed=False,exhaustive_programme_history_verified=False,scope=design()['scope'])
    def checkpoint():
        s.update(stage=stage,elapsed_worker_seconds=time.monotonic()-started,new_model_forwards=calls['model_forwards'],runtime_calls=calls,known_control_records=len(controls),preparation_records=len(preps),new_feature_records=len(features),final_endpoint_records=len(records),confirmation_states_evaluated=len({r['state_id'] for r in records if r['round']==0 and r['arm']==ARMS[2]}))
        dump(out/'SUMMARY.json',dict(s,errors=errors))
        for n,x in (('KNOWN_CONTROL_PROGRESS.json',controls),('PREPARATION_PROGRESS.json',preps),('PREWRITE_PROGRESS.json',features),('ENDPOINT_PROGRESS.json',records)):dump(out/n,x)
    def seal(n):seals[n]=sha(out/n)
    def budget():
        need(time.monotonic()-started<TIMEOUT,'C141 fixed50-minute worker budget exceeded')
        for n,h in seals.items():need(sha(out/n)==h,'Frozen artifact changed: '+n)
        if data is not None and 'frozen_heads' in data:compare(head_digests(data),data['frozen_heads'],'Frozen candidate tensors')
    def consistent(m,ref):
        q=safe.repeat_comparison(m,ref);need(q['same_point_identity'] and q['within_replay_tolerance'],'Known native replay drift blocks new-state confirmation');return q
    try:
        need(not (out/'SUMMARY.json').exists(),'Fresh output required');dump(out/'DESIGN_FROZEN_BEFORE_MODEL.json',design());seal('DESIGN_FROZEN_BEFORE_MODEL.json');checkpoint();stage='nine_original_bindings'
        data=(load if input_loader is None else input_loader)(paths,out,c);data['frozen_heads']=head_digests(data);budget()
        for n in ('INPUTS_FROZEN.json','BOUND_HISTORICAL_COORDINATE_HASHES.json','C140_BOUND_RECEIPT.json'):
            if (out/n).is_file():seal(n)
        for n,obj in (('FROZEN_RAW_WRITER.npz',data['writer']),('FROZEN_ORIGINAL_CONTROL.npz',data['control']),('FROZEN_SHARED_CONTROLS.npz',dict(V=data['V'],C137_a=data['initial_a'],C140_a=data['a140']))):np.savez_compressed(out/n,**obj);seal(n)
        dump(out/'CANDIDATE_FROZEN_BEFORE_NEW_STATES.json',dict(parent_protocol=parent.PROTOCOL,parent_decision=data['C140_summary']['decision'],parent_decision_reclassified=False,checkpoint='C140 fixed step12 only',heads=data['frozen_heads'],new_parameter_selection=False));seal('CANDIDATE_FROZEN_BEFORE_NEW_STATES.json')
        fresh,rows=fresh_coordinates(data);np.savez_compressed(out/'ALL48_PREPARED_COORDINATES_BEFORE_MODEL.npz',**fresh);dump(out/'ALL48_CONFIRMATION_REGISTRY.json',rows)
        for n in ('ALL48_PREPARED_COORDINATES_BEFORE_MODEL.npz','ALL48_CONFIRMATION_REGISTRY.json'):seal(n)
        s.update(candidate_frozen_before_new_state_generation=True,all48_coordinates_frozen_before_model=True,C140_parent_decision=data['C140_summary']['decision']);phase=Phase(data,fresh);stage='model_initialisation';checkpoint();actual=runtime_factory is None and input_loader is None
        if runtime_factory is None:
            from runtime141 import factory
            runtime_factory=factory
        rt=runtime_factory(out/'model_runtime',data,phase,calls);s['real_model_execution_claim']=actual;compare({k:v for k,v in rt.pins.items() if k!='torch'},{k:v for k,v in data['modelpins'].items() if k!='torch'},'Pinned model/chart identity');stage='known36_full_prewrite_and144_native_controls';checkpoint()
        for sp in parent.registry():
            budget();sid=sp['state_id'];p=data['points'][sid];case=sp['case'];desired=int(case[-2])^int(case[-1]);phase.feature(sid,p['prepared'],desired);x,o=rt.features(p['prepared'],desired);farray(x,(3147,));farray(o,(22,));fe=close(x,p['features'][:3147],'Known captured3147',2e-6);raw,full=safe.reconstruct_raw(data['writer'],data['control']['delta_sketch'],p['prepared'],x);re=close(raw,p['raw'],'Known original raw map',2e-6);ff=close(full,p['features'],'Known full3211',2e-6);rr=[]
            for kind,point,write in (('prepared',p['prepared'],False),('raw',p['raw'],True),('C137',data['C140_points'][(sid,parent.EVAL_ARMS[1])],True),('C140',data['C140_points'][(sid,parent.EVAL_ARMS[2])],True)):
                phase.metric((sid,kind),point,case,write);met=rt.metrics(point,case,write)
                if kind=='prepared':need(safe.metric_gates(met,case,point,False),'Known training preparation fails');refs=data['native_refs'][(sid,'prepared')];close(o,met['observed'],'Known feature/native parity',.0002)
                else:
                    parent.parent.gate_record(met,case,point,p['prepared']);arm=parent.EVAL_ARMS[0] if kind=='raw' else (parent.EVAL_ARMS[1] if kind=='C137' else parent.EVAL_ARMS[2]);refs=[data['C140_controls'][(sid,arm,j)]['metrics'] for j in (0,1)]
                rr.append(dict(kind=kind,metrics=met,comparisons=[consistent(met,ref) for ref in refs]))
            controls.append(dict(**sp,feature_error_Linf=fe,full_feature_error_Linf=ff,raw_map_error_Linf=re,measurements=rr));checkpoint();print('C141_KNOWN_CONTROL',sid,'OK',flush=True)
        phase.begin_preparations();stage='two_round96_new_preparations';checkpoint();prepby={}
        for rnd in (0,1):
            for sp in registry():
                budget();sid=sp['state_id'];theta=fresh[sid];phase.metric((rnd,sid),theta,sp['case'],False);m=rt.metrics(theta,sp['case'],False);valid=bool(safe.metric_gates(m,sp['case'],theta,False));r=dict(**sp,round=rnd,metrics=m,preparation_valid=valid)
                if rnd==1:
                    q=safe.repeat_comparison(m,prepby[(0,sid)]['metrics']);r['repeat_comparison']=q
                    if not(q['same_point_identity'] and q['within_replay_tolerance']):repeat_failures.append(dict(kind='preparation',state_id=sid,comparison=q))
                preps.append(r);prepby[(rnd,sid)]=r;checkpoint();print('C141_NEW_PREPARATION',rnd,sid,valid,'response',m['response_drift'],flush=True)
        valids={sp['state_id']:all(prepby[(j,sp['state_id'])]['preparation_valid'] for j in (0,1)) for sp in registry()};phase.begin_prewrite();stage='all48_prewrite_features_and144_proposals';checkpoint();points={};arrays={};outcomes=[];(out/'prewrite').mkdir()
        for sp in registry():
            budget();sid=sp['state_id'];theta=fresh[sid];desired=int(sp['case'][-2])^int(sp['case'][-1]);phase.feature(sid,theta,desired);x,o=rt.features(theta,desired);farray(x,(3147,));farray(o,(22,));compare(float(x[3076]),float(desired),'Desired bit feature');close(o,prepby[(0,sid)]['metrics']['observed'],'New feature/primary preparation parity',.0002)
            raw,full=safe.reconstruct_raw(data['writer'],data['control']['delta_sketch'],theta,x);old,old_diag=parent.proposal(theta,raw,data['V'],data['initial_a']);candidate,diag=parent.proposal(theta,raw,data['V'],data['a140']);exact(full[:3147],x,'Capture/full3211 prefix')
            for arm,point,dd in ((ARMS[0],raw,dict(original_raw_rule=True)),(ARMS[1],old,old_diag),(ARMS[2],candidate,diag)):
                points[(sid,arm)]=point;arrays[sid+'__'+arm]=point;outcomes.append(dict(**sp,arm=arm,point_sha256=safe.tensor_hash(point),preparation_valid_both_rounds=valids[sid],proposal_diagnostics=dd))
            np.savez_compressed(out/'prewrite'/f'{sid}.npz',prepared=theta,raw=raw,C137=old,C140=candidate,capture3147=x,write_features3211=full,observed=o);features.append(dict(**sp,prepared_sha256=safe.tensor_hash(theta),raw_sha256=safe.tensor_hash(raw),C137_sha256=safe.tensor_hash(old),C140_sha256=safe.tensor_hash(candidate),full_feature_digest=row_digest(full),preparation_valid_both_rounds=valids[sid]));checkpoint()
        np.savez_compressed(out/'ALL144_PROPOSALS_BEFORE_ENDPOINTS.npz',**arrays);dump(out/'ALL144_PROPOSAL_OUTCOMES.json',outcomes)
        for n in ('ALL144_PROPOSALS_BEFORE_ENDPOINTS.npz','ALL144_PROPOSAL_OUTCOMES.json'):seal(n)
        dump(out/'PROPOSALS_SEALED_BEFORE_NEW_ENDPOINTS.json',dict(artifact_seals=dict(seals),heads=data['frozen_heads'],new_states=N,proposal_points=144,new_writer_endpoint_measurements_so_far=0));seal('PROPOSALS_SEALED_BEFORE_NEW_ENDPOINTS.json');phase.freeze(points);s['all_proposals_frozen_before_new_endpoint_measurements']=True;stage='two_round288_fixed_native_writer_endpoints';checkpoint();first={}
        for rnd in (0,1):
            for sp in registry():
                sid=sp['state_id'];case=sp['case']
                for arm in ARMS:
                    budget();point=points[(sid,arm)];phase.metric((rnd,sid,arm),point,case,True);m=rt.metrics(point,case,True);g=gate_record(m,case,point,fresh[sid],valids[sid]);r=dict(**sp,round=rnd,arm=arm,status='EVALUATED',preparation_valid_both_rounds=valids[sid],metrics=m,recomputed_gates=g)
                    if rnd==0:first[(sid,arm)]=m
                    else:
                        q=safe.repeat_comparison(m,first[(sid,arm)]);r['repeat_comparison']=q
                        if not(q['same_point_identity'] and q['within_replay_tolerance']):repeat_failures.append(dict(kind='writer_endpoint',state_id=sid,arm=arm,comparison=q))
                    records.append(r);dump(out/'endpoints'/f'record{len(records)-1:03d}.json',r);checkpoint()
                print('C141_NEW_ENDPOINT',rnd,sid,[(r['arm'],r['recomputed_gates']['complete_pass'],r['metrics']['response_drift']) for r in records[-3:]],flush=True)
        phase.finish();budget();rt.invariant();need(calls['metric_calls']==528 and calls['feature_calls']==84 and calls['model_forwards']==calls['initialisation_model_forwards']+2*528+3*84,'Fixed forward/measurement coverage differs');sums,pairs=summaries(records);ps={str(j):preparation_summary([r for r in preps if r['round']==j]) for j in (0,1)}
        passed=not repeat_failures and all(valids.values()) and all(sums['all48'][str(j)][ARMS[2]]['passed']==N for j in (0,1));branch='PREREGISTER_SEPARATE_EXCLUDED_INPUT_PROVENANCE_AND_EVALUATION' if passed else 'STOP_FROZEN_CONFIRMATION_RETAIN_FAILURES_AND_EXPOSED_STATES'
        if repeat_failures:branch='RESOLVE_REPEAT_DRIFT_NO_CONFIRMATION'
        for n,x in (('ALL96_NEW_PREPARATIONS.json',preps),('ALL288_CONFIRMATION_ENDPOINTS.json',records),('PREPARATION_SUMMARIES.json',ps),('CONFIRMATION_SUMMARIES.json',sums),('PAIRED_VS_RAW_AND_C137.json',pairs),('REPEATABILITY_FAILURES.json',repeat_failures)):dump(out/n,x)
        s.update(complete=True,decision=('C141_COMPLETE_REPEAT_DRIFT_NO_CONFIRMATION' if repeat_failures else ('C141_FROZEN_KERNEL_LOCAL_CONFIRMATION_ALL48_PASSED' if passed else 'C141_FROZEN_KERNEL_LOCAL_CONFIRMATION_FAILURE')),local_confirmation_passed=bool(passed),repeatable=not repeat_failures,known_controls_replay_consistent=True,preparation_states_valid_both_rounds=sum(valids.values()),preparation_summaries=ps,confirmation_summaries=sums,next_branch=branch,all_C141_states_exposed_after_this_run=True)
        report=['# C141：冻结C140的48态共享核局部确认','','原12个已知中心，每中心2档范数、各2个新方向，共48态；三个已训练输入类各16态。使用全新固定种子；不重抽，不训练，不修复，不删失败。','','候选是固定C140第12步，raw和C137仅作平行对照，不替代C140判定。模型前封存坐标；所有144个提案在首次新写入端点前封存。','','|臂|首轮通过/请求|重复轮通过/请求|最大响应漂移|','|---|---:|---:|---:|']
        for arm in ARMS:report.append(f"|{arm}|{sums['all48']['0'][arm]['passed']}/48|{sums['all48']['1'][arm]['passed']}/48|{sums['all48']['0'][arm]['maximum_response_drift']}|")
        report+=['','两轮均有效的制备态：'+str(sum(valids.values()))+'/48','最终判定：'+s['decision'],'后续分支：'+branch,'','两轮为同48态的重复测量。完整通过仅支持同共享核制备族、已知中心周围的新方向确认，不是新中心或排除输入泛化。历史坐标比对仅覆盖九原包显式记录，非全项目非访问/时序证书。LC2和单折仍未闭合。所有48态已暴露；后续若据此适配，不能再作未见测试。'];(out/'REPORT_ZH.md').write_text('\n'.join(report)+'\n');stage='complete'
    except BaseException:errors.append(traceback.format_exc());s.update(complete=False,decision='C141_INCOMPLETE_NO_CONFIRMATION');print(errors[-1],flush=True)
    finally:
        checks={}
        for k,p in paths.items():
            try:checks[k]=sha(p)==c['pins'][k]['sha256']
            except Exception:checks[k]=False
        s['original_inputs_unchanged']=checks
        if not all(checks.values()):s.update(complete=False,decision='C141_INCOMPLETE_NO_CONFIRMATION');errors.append('Original input changed')
        checkpoint()
    return dict(s,errors=errors)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in INPUT_KEYS:p.add_argument('--'+k.lower(),required=True)
    p.add_argument('--out',required=True);args=p.parse_args();s=run({k:Path(getattr(args,k.lower())) for k in INPUT_KEYS},args.out);print(json.dumps(s,ensure_ascii=False,indent=2),flush=True)
    if not s['complete']:raise SystemExit(1)
