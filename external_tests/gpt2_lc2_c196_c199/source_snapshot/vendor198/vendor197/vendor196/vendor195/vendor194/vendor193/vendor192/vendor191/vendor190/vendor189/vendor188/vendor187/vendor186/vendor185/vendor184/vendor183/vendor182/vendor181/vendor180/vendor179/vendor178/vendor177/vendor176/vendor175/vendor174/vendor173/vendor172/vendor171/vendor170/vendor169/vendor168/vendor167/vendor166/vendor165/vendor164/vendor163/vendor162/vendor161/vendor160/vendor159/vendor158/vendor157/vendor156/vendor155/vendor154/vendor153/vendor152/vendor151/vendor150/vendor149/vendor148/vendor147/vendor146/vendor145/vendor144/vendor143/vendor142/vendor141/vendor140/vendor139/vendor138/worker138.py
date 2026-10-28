"""Prospective local confirmation with immutable prewrite-only proposals."""
import argparse,traceback
from core138 import *
from inputs138 import load

def run(paths,out,c=None,runtime_factory=None,input_loader=None):
    c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);started=time.monotonic();errors=[];stage='registered';controls=[];prewrite=[];records=[];seals={};data=None
    calls=dict(model_forwards=0,initialisation_model_forwards=0,metric_attempts=0,metric_calls=0,feature_attempts=0,feature_calls=0)
    s=dict(protocol=PROTOCOL,complete=False,decision='C138_INCOMPLETE_NO_CONFIRMATION',synthetic_fixture=not c['production'],real_model_execution_claim=False,
        requested_confirmation_states=24,confirmation_states_evaluated=0,requested_endpoint_records=96,final_endpoint_records=0,known_training_centers=12,
        new_training_steps=0,new_backward_calls=0,new_head_fits=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,
        teacher_calls=0,backtracking_steps=0,online_response_repair_steps=0,excluded_input_states_evaluated=0,candidate_selected=False,writer_deployed=False,thresholds_changed=False,
        parent_decision_reclassified=False,LC2_closed=False,LC2_single_fold_passed=False,archived_source_executed=False,exhaustive_historical_catalog_verified=False,
        scope=design()['scope'])
    def checkpoint():
        s.update(stage=stage,elapsed_worker_seconds=time.monotonic()-started,new_model_forwards=calls['model_forwards'],runtime_calls=calls,prewrite_records=len(prewrite),final_endpoint_records=len(records),confirmation_states_evaluated=len({r['state_id'] for r in records if r['arm']==ARMS[1] and r['round']==0}))
        dump(out/'SUMMARY.json',dict(s,errors=errors));dump(out/'KNOWN_CONTROL_PROGRESS.json',controls);dump(out/'PREWRITE_PROGRESS.json',prewrite);dump(out/'ENDPOINT_PROGRESS.json',records)
    def seal(n):seals[n]=sha(out/n)
    def budget():
        need(time.monotonic()-started<TIMEOUT,'C138 fixed50-minute budget reached')
        for n,h in seals.items():need(sha(out/n)==h,'Frozen artifact changed: '+n)
        if data is not None and 'frozen_heads' in data:compare(head_digests(data),data['frozen_heads'],'Frozen candidate tensors')
    def consistent(m,old):
        q=safe.repeat_comparison(m,old);need(q['within_replay_tolerance'] and q['same_point_identity'],'Known native replay drift blocks confirmation');return q
    try:
        need(not (out/'SUMMARY.json').exists(),'Fresh output required');dump(out/'DESIGN_FROZEN_BEFORE_MODEL.json',design());seal('DESIGN_FROZEN_BEFORE_MODEL.json');checkpoint();stage='six_original_bindings'
        data=(load if input_loader is None else input_loader)(paths,out,c);data['frozen_heads']=head_digests(data);budget()
        for n in ('INPUTS_FROZEN.json','BOUND_HISTORICAL_COORDINATE_HASHES.json','C137_BOUND_RECEIPT.json','FROZEN_WRITER_SCHEMA_VALIDATION.json'):
            if (out/n).is_file():seal(n)
        for n,obj in (('FROZEN_RAW_WRITER.npz',data['writer']),('FROZEN_CONTROL.npz',data['control']),('FROZEN_C137_CONTROL.npz',dict(a=data['a'],V=data['V']))):np.savez_compressed(out/n,**obj);seal(n)
        dump(out/'CANDIDATE_FROZEN_BEFORE_NEW_STATES.json',dict(parent_protocol=parent.PROTOCOL,parent_decision=data['C137_summary']['decision'],parent_decision_reclassified=False,heads=data['frozen_heads'],candidate_checkpoint='C137 fixed step24',new_parameter_selection=False));seal('CANDIDATE_FROZEN_BEFORE_NEW_STATES.json')
        fresh,rows=fresh_coordinates(data);np.savez_compressed(out/'ALL24_PREPARED_COORDINATES_BEFORE_MODEL.npz',**fresh);dump(out/'ALL24_NEW_STATE_REGISTRY.json',rows)
        for n in ('ALL24_PREPARED_COORDINATES_BEFORE_MODEL.npz','ALL24_NEW_STATE_REGISTRY.json'):seal(n)
        s.update(candidate_frozen_before_new_state_generation=True,all24_coordinates_frozen_before_model=True,parent_decision=data['C137_summary']['decision']);phase=Phase(data,fresh);stage='model_initialisation';checkpoint()
        actual=runtime_factory is None and input_loader is None
        if runtime_factory is None:
            from runtime138 import factory
            runtime_factory=factory
        rt=runtime_factory(out/'model_runtime',data,phase,calls);s['real_model_execution_claim']=actual
        compare({k:v for k,v in rt.pins.items() if k!='torch'},{k:v for k,v in data['modelpins'].items() if k!='torch'},'Pinned model/chart identity');stage='known12_full_prewrite_and_endpoint_controls';checkpoint()
        for sp in parent.registry():
            budget();sid=sp['state_id'];p=data['points'][sid];case=sp['case'];desired=int(case[-2])^int(case[-1]);phase.feature(sid,p['prepared'],desired);x,o=rt.features(p['prepared'],desired);farray(x,(3147,));farray(o,(22,));fe=close(x,p['features'][:3147],'Known captured features',2e-6)
            rebuilt,full=safe.reconstruct_raw(data['writer'],data['control']['delta_sketch'],p['prepared'],x);re=close(rebuilt,p['raw'],'Known raw map',2e-6);ff=close(full,p['features'],'Known full3211 features',2e-6);entries=[]
            for kind,point,write in (('prepared',p['prepared'],False),('raw',p['raw'],True),('candidate',data['C137_points'][sid],True)):
                phase.metric((sid,kind),point,case,write);met=rt.metrics(point,case,write)
                if kind=='prepared':need(safe.metric_gates(met,case,point,False),'Known preparation fails');refs=[p['preparation']];close(o,met['observed'],'Known feature/native observation parity',.0002)
                else:
                    parent.gate_record(met,case,point,p['prepared']);arm=parent.EVAL_ARMS[0] if kind=='raw' else parent.EVAL_ARMS[1];refs=[data['C137_controls'][(sid,arm,rnd)]['metrics'] for rnd in (0,1)]
                entries.append(dict(kind=kind,metrics=met,comparisons=[consistent(met,r) for r in refs]))
            controls.append(dict(**sp,feature_error_Linf=fe,full_feature_error_Linf=ff,raw_map_error_Linf=re,measurements=entries));checkpoint();print('C138_KNOWN_CONTROL',sid,'OK',flush=True)
        phase.begin_prewrite();s['known_controls_complete_before_new_measurements']=True;stage='all24_new_prewrite_measurements_and_proposals';checkpoint();points={};arrays={};outcomes=[];valids={};specs=registry();(out/'prewrite').mkdir()
        for sp in specs:
            budget();sid=sp['state_id'];case=sp['case'];theta=fresh[sid];phase.metric(sid,theta,case,False);prep=rt.metrics(theta,case,False);valid=bool(safe.metric_gates(prep,case,theta,False));valids[sid]=valid
            desired=int(case[-2])^int(case[-1]);phase.feature(sid,theta,desired);x,o=rt.features(theta,desired);farray(x,(3147,));farray(o,(22,));compare(float(x[3076]),float(desired),'Desired bit feature');close(o,prep['observed'],'New prewrite/native observation parity',.0002)
            raw,full=safe.reconstruct_raw(data['writer'],data['control']['delta_sketch'],theta,x);exact(full[:3147],x,'Captured3147/full3211 binding');candidate,diag=parent.proposal(theta,raw,data['V'],data['a'])
            for arm,point in ((ARMS[0],raw),(ARMS[1],candidate)):
                points[(sid,arm)]=point;arrays[sid+'__'+arm]=point;outcomes.append(dict(**sp,arm=arm,point_sha256=safe.tensor_hash(point),preparation_valid=valid,proposal_diagnostics=diag if arm==ARMS[1] else dict(original_raw_rule=True)))
            np.savez_compressed(out/'prewrite'/f'{sid}.npz',prepared=theta,raw=raw,candidate=candidate,capture3147=x,write_features3211=full,observed=o)
            prewrite.append(dict(**sp,preparation=prep,preparation_valid=valid,prepared_sha256=safe.tensor_hash(theta),raw_sha256=safe.tensor_hash(raw),candidate_sha256=safe.tensor_hash(candidate),full_feature_digest=row_digest(full)))
            checkpoint();print('C138_NEW_PREWRITE',sid,'preparation_valid',valid,flush=True)
        np.savez_compressed(out/'ALL48_PROPOSALS_BEFORE_ENDPOINTS.npz',**arrays);dump(out/'ALL48_PROPOSAL_OUTCOMES_BEFORE_ENDPOINTS.json',outcomes)
        for n in ('ALL48_PROPOSALS_BEFORE_ENDPOINTS.npz','ALL48_PROPOSAL_OUTCOMES_BEFORE_ENDPOINTS.json'):seal(n)
        dump(out/'ALL_PROPOSALS_SEALED_BEFORE_NEW_ENDPOINTS.json',dict(artifact_seals=dict(seals),heads=data['frozen_heads'],new_states=24,proposal_points=48,new_endpoint_measurements_so_far=0));seal('ALL_PROPOSALS_SEALED_BEFORE_NEW_ENDPOINTS.json')
        phase.freeze(points);s['all_proposals_frozen_before_new_endpoint_measurements']=True;stage='two_round96_fixed_native_endpoints';checkpoint();first={};repeat_failures=[]
        for rnd in (0,1):
            for sp in specs:
                sid=sp['state_id'];case=sp['case']
                for arm in ARMS:
                    budget();point=points[(sid,arm)];phase.metric((rnd,sid,arm),point,case,True);met=rt.metrics(point,case,True);g=gate_record(met,case,point,fresh[sid],valids[sid]);r=dict(**sp,arm=arm,round=rnd,status='EVALUATED',preparation_valid=valids[sid],metrics=met,recomputed_gates=g)
                    if rnd==0:first[(sid,arm)]=met
                    else:
                        q=safe.repeat_comparison(met,first[(sid,arm)]);r['repeat_comparison']=q
                        if not(q['within_replay_tolerance'] and q['same_point_identity']):repeat_failures.append(dict(state_id=sid,arm=arm,comparison=q))
                    records.append(r);dump(out/'endpoints'/f'record{len(records)-1:02d}.json',r);checkpoint()
                print('C138_NEW_FINAL',rnd,sid,[(r['arm'],r['recomputed_gates']['complete_pass'],r['metrics']['response_drift']) for r in records[-2:]],flush=True)
        phase.finish();budget();rt.invariant();need(calls['feature_calls']==36 and calls['metric_calls']==156 and calls['model_forwards']==calls['initialisation_model_forwards']+3*36+2*156,'Fixed forward/measurement coverage differs')
        sums,pairs=summaries(records);passed=not repeat_failures and all(valids.values()) and all(sums['all24'][str(rnd)][ARMS[1]]['passed']==24 for rnd in (0,1))
        dump(out/'ALL96_CONFIRMATION_ENDPOINTS.json',records);dump(out/'CONFIRMATION_SUMMARIES.json',sums);dump(out/'PAIRED_VS_RAW.json',pairs);dump(out/'REPEATABILITY_FAILURES.json',repeat_failures)
        s.update(complete=True,decision=('C138_COMPLETE_REPEAT_DRIFT_NO_CONFIRMATION' if repeat_failures else ('C138_FROZEN_LOCAL_CONFIRMATION_ALL24_PASSED' if passed else 'C138_FROZEN_LOCAL_CONFIRMATION_FAILURE')),
            local_confirmation_passed=bool(passed),preparation_states_passed=sum(valids.values()),preparation_states_failed=24-sum(valids.values()),repeatable=not repeat_failures,known_controls_replay_consistent=True,confirmation_summaries=sums,
            next_branch=('PREREGISTER_EXCLUDED_INPUT_PROVENANCE_AND_EVALUATION' if passed else 'STOP_FROZEN_CANDIDATE_CONFIRMATION_RETAIN_ALL_FAILURES'))
        report=['# C138：冻结C137的局部新态确认','','固定C137第24步控制、原响应基和原写入器；无训练、反传、新Jacobian/SVD、教师、回溯或写后修复。','','24态是已知12中心周围的预登记新扰动，属于已训练输入类。不是新中心或排除输入泛化。新坐标在模型初始化前封存，全部提案在首次新端点测量前封存。','','历史坐标哈希仅覆盖六个绑定原包内可获得的状态与端点；未证明全项目历史目录穷尽、时序或非访问。','','|臂|首轮通过/请求|重复轮通过/请求|最大响应漂移|','|---|---:|---:|---:|']
        for arm in ARMS:report.append(f"|{arm}|{sums['all24']['0'][arm]['passed']}/24|{sums['all24']['1'][arm]['passed']}/24|{sums['all24']['0'][arm]['maximum_response_drift']}|")
        report+=['','制备失败数：'+str(24-sum(valids.values())),'最终判定：'+s['decision'],'后续分支：'+s['next_branch'],'','两轮是相同24态的重复测量，不是48个独立样本。所有请求、失败和中断分母保留；完整通过也只支持此局部新态范围。LC2及单折仍未通过。']
        (out/'REPORT_ZH.md').write_text('\n'.join(report)+'\n');stage='complete'
    except BaseException:errors.append(traceback.format_exc());s.update(complete=False,decision='C138_INCOMPLETE_NO_CONFIRMATION');print(errors[-1],flush=True)
    finally:
        checks={}
        for k,p in paths.items():
            try:checks[k]=sha(p)==c['pins'][k]['sha256']
            except Exception:checks[k]=False
        s['original_inputs_unchanged']=checks
        if not all(checks.values()):s.update(complete=False,decision='C138_INCOMPLETE_NO_CONFIRMATION');errors.append('Original input changed')
        checkpoint();print(json.dumps(dict(s,errors=errors),ensure_ascii=False,indent=2),flush=True)
    return dict(s,errors=errors)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in INPUT_KEYS:p.add_argument('--'+k.lower(),required=True)
    p.add_argument('--out',required=True);args=p.parse_args();s=run({k:Path(getattr(args,k.lower())) for k in INPUT_KEYS},args.out);sys.exit(0 if s['complete'] else 1)
