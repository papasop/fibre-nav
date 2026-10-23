"""Fixed exposed-direction intervention; all requested preparations/endpoints retained."""
import argparse,traceback
from core139 import *
from inputs139 import load

def run(paths,out,c=None,runtime_factory=None,input_loader=None):
    c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);started=time.monotonic();stage='registered';data=None;seals={};errors=[];known=[];fullreplay=[];preps=[];features=[];records=[];repeat_failures=[]
    calls=dict(model_forwards=0,initialisation_model_forwards=0,metric_attempts=0,metric_calls=0,feature_attempts=0,feature_calls=0)
    s=dict(protocol=PROTOCOL,complete=False,decision='C139_INCOMPLETE_NO_DIAGNOSTIC_CONCLUSION',synthetic_fixture=not c['production'],real_model_execution_claim=False,requested_development_states=24,requested_kernel_preparation_records=48,requested_endpoint_records=96,
        independent_confirmation_states_evaluated=0,excluded_input_states_evaluated=0,new_training_steps=0,new_backward_calls=0,new_head_fits=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,teacher_calls=0,backtracking_steps=0,online_response_repair_steps=0,
        candidate_selected=False,writer_deployed=False,thresholds_changed=False,parent_decision_reclassified=False,LC2_closed=False,LC2_single_fold_passed=False,archived_source_executed=False,scope=design()['scope'])
    def checkpoint():
        s.update(stage=stage,elapsed_worker_seconds=time.monotonic()-started,new_model_forwards=calls['model_forwards'],runtime_calls=calls,known_parent_records=len(known),C138_preparation_replays=len(fullreplay),kernel_preparation_records=len(preps),kernel_feature_records=len(features),final_endpoint_records=len(records))
        dump(out/'SUMMARY.json',dict(s,errors=errors))
        for n,x in (('KNOWN_CONTROL_PROGRESS.json',known),('FULL_REPLAY_PROGRESS.json',fullreplay),('KERNEL_PREPARATION_PROGRESS.json',preps),('PREWRITE_PROGRESS.json',features),('ENDPOINT_PROGRESS.json',records)):dump(out/n,x)
    def seal(n):seals[n]=sha(out/n)
    def budget():
        need(time.monotonic()-started<TIMEOUT,'C139 fixed50-minute budget exceeded')
        for n,h in seals.items():need(sha(out/n)==h,'Frozen artifact changed: '+n)
        if data is not None and 'frozen_heads' in data:compare(head_digests(data),data['frozen_heads'],'Frozen candidate tensors')
    def consistent(m,old):
        q=safe.repeat_comparison(m,old);need(q['within_replay_tolerance'] and q['same_point_identity'],'Known replay drift blocks kernel diagnostic');return q
    try:
        need(not (out/'SUMMARY.json').exists(),'Fresh output required');dump(out/'DESIGN_FROZEN_BEFORE_MODEL.json',design());seal('DESIGN_FROZEN_BEFORE_MODEL.json');checkpoint();stage='seven_original_bindings'
        data=(load if input_loader is None else input_loader)(paths,out,c);data['frozen_heads']=head_digests(data);budget()
        for n in ('INPUTS_FROZEN.json','C138_PREPARATION_AND_ENDPOINT_AUDIT.json'):
            if (out/n).is_file():seal(n)
        for n,obj in (('FROZEN_RAW_WRITER.npz',data['writer']),('FROZEN_CONTROL.npz',data['control']),('FROZEN_C137_CONTROL.npz',dict(a=data['a'],V=data['V']))):np.savez_compressed(out/n,**obj);seal(n)
        dump(out/'CANDIDATE_FROZEN_BEFORE_COORDINATES.json',dict(heads=data['frozen_heads'],checkpoint='C137 fixed step24',new_parameter_selection=False));seal('CANDIDATE_FROZEN_BEFORE_COORDINATES.json')
        kernel,rows=freeze_kernel_coordinates(data);np.savez_compressed(out/'ALL24_KERNEL_COORDINATES_BEFORE_MODEL.npz',**kernel);dump(out/'ALL24_KERNEL_REGISTRY.json',rows)
        for n in ('ALL24_KERNEL_COORDINATES_BEFORE_MODEL.npz','ALL24_KERNEL_REGISTRY.json'):seal(n)
        phase=Phase(data,kernel);stage='model_initialisation';checkpoint();actual=runtime_factory is None and input_loader is None
        if runtime_factory is None:
            from runtime139 import factory
            runtime_factory=factory
        rt=runtime_factory(out/'model_runtime',data,phase,calls);s['real_model_execution_claim']=actual
        compare({k:v for k,v in rt.pins.items() if k!='torch'},{k:v for k,v in data['modelpins'].items() if k!='torch'},'Pinned model identity');stage='known12_controls';checkpoint()
        for sp in parent.registry():
            budget();sid=sp['state_id'];p=data['points'][sid];case=sp['case'];desired=int(case[-2])^int(case[-1]);phase.feature(sid,p['prepared'],desired);x,o=rt.features(p['prepared'],desired);farray(x,(3147,));farray(o,(22,));fe=close(x,p['features'][:3147],'Known capture',2e-6)
            rebuilt,full=safe.reconstruct_raw(data['writer'],data['control']['delta_sketch'],p['prepared'],x);re=close(rebuilt,p['raw'],'Known raw',2e-6);ff=close(full,p['features'],'Known full3211',2e-6);entries=[]
            for kind,point,write in (('prepared',p['prepared'],False),('raw',p['raw'],True),('candidate',data['C137_points'][sid],True)):
                phase.metric((sid,kind),point,case,write);met=rt.metrics(point,case,write)
                if kind=='prepared':need(safe.metric_gates(met,case,point,False),'Known preparation fails');refs=[p['preparation']];close(o,met['observed'],'Known feature/native parity',.0002)
                else:
                    parent.gate_record(met,case,point,p['prepared']);arm=parent.EVAL_ARMS[0] if kind=='raw' else parent.EVAL_ARMS[1];refs=[data['C137_controls'][(sid,arm,rnd)]['metrics'] for rnd in (0,1)]
                entries.append(dict(kind=kind,metrics=met,comparisons=[consistent(met,r) for r in refs]))
            known.append(dict(**sp,feature_error_Linf=fe,full_feature_error_Linf=ff,raw_map_error_Linf=re,measurements=entries));checkpoint();print('C139_KNOWN',sid,'OK',flush=True)
        phase.advance('full_replay');stage='all24_C138_preparation_replay';checkpoint()
        for sp in registry():
            budget();sid=sp['state_id'];old=sp['source_state_id'];point=data['C138_fresh'][old];phase.metric(sid,point,sp['case'],False);m=rt.metrics(point,sp['case'],False);g=preparation_gates(m,sp['case'],point);q=consistent(m,data['C138_prewrite'][old]['preparation']);fullreplay.append(dict(**sp,metrics=m,gates=g,comparison=q));checkpoint()
        phase.advance('kernel_preparation');stage='two_round_kernel_preparations';checkpoint();prepby={}
        for rnd in (0,1):
            for sp in registry():
                budget();sid=sp['state_id'];point=kernel[sid];phase.metric((rnd,sid),point,sp['case'],False);m=rt.metrics(point,sp['case'],False);g=preparation_gates(m,sp['case'],point);r=dict(**sp,round=rnd,metrics=m,gates=g)
                if rnd==1:
                    q=safe.repeat_comparison(m,prepby[(0,sid)]['metrics']);r['repeat_comparison']=q
                    if not(q['within_replay_tolerance'] and q['same_point_identity']):repeat_failures.append(dict(kind='preparation',state_id=sid,comparison=q))
                preps.append(r);prepby[(rnd,sid)]=r;checkpoint();print('C139_KERNEL_PREPARATION',rnd,sid,g['complete_pass'],g['failed_gates'],'response',m['response_drift'],flush=True)
        valids={sp['state_id']:all(prepby[(rnd,sp['state_id'])]['gates']['complete_pass'] for rnd in (0,1)) for sp in registry()}
        phase.advance('prewrite');stage='all24_prewrite_features_and48_proposals';checkpoint();points={};arrays={};outcomes=[];(out/'prewrite').mkdir()
        for sp in registry():
            budget();sid=sp['state_id'];theta=kernel[sid];desired=int(sp['case'][-2])^int(sp['case'][-1]);phase.feature(sid,theta,desired);x,o=rt.features(theta,desired);farray(x,(3147,));farray(o,(22,));compare(float(x[3076]),float(desired),'Desired bit')
            close(o,prepby[(0,sid)]['metrics']['observed'],'Kernel feature/primary native observation parity',.0002)
            raw,full=safe.reconstruct_raw(data['writer'],data['control']['delta_sketch'],theta,x);candidate,diag=parent.proposal(theta,raw,data['V'],data['a']);exact(full[:3147],x,'Full3211 capture prefix')
            for arm,point in ((ARMS[0],raw),(ARMS[1],candidate)):
                points[(sid,arm)]=point;arrays[sid+'__'+arm]=point;outcomes.append(dict(**sp,arm=arm,point_sha256=safe.tensor_hash(point),preparation_valid_both_rounds=valids[sid],proposal_diagnostics=diag if arm==ARMS[1] else dict(original_raw_rule=True)))
            np.savez_compressed(out/'prewrite'/f'{sid}.npz',prepared=theta,raw=raw,candidate=candidate,capture3147=x,write_features3211=full,observed=o)
            features.append(dict(**sp,prepared_sha256=safe.tensor_hash(theta),raw_sha256=safe.tensor_hash(raw),candidate_sha256=safe.tensor_hash(candidate),full_feature_digest=row_digest(full),preparation_valid_both_rounds=valids[sid]));checkpoint()
        np.savez_compressed(out/'ALL48_PROPOSALS_BEFORE_ENDPOINTS.npz',**arrays);dump(out/'ALL48_PROPOSAL_OUTCOMES.json',outcomes)
        for n in ('ALL48_PROPOSALS_BEFORE_ENDPOINTS.npz','ALL48_PROPOSAL_OUTCOMES.json'):seal(n)
        dump(out/'PROPOSALS_SEALED_BEFORE_ENDPOINTS.json',dict(artifact_seals=dict(seals),heads=data['frozen_heads'],proposal_points=48,endpoint_measurements_so_far=0));seal('PROPOSALS_SEALED_BEFORE_ENDPOINTS.json');phase.freeze(points);stage='two_round96_kernel_writer_endpoints';checkpoint();first={}
        for rnd in (0,1):
            for sp in registry():
                sid=sp['state_id'];case=sp['case']
                for arm in ARMS:
                    budget();point=points[(sid,arm)];phase.metric((rnd,sid,arm),point,case,True);m=rt.metrics(point,case,True);g=previous.gate_record(m,case,point,kernel[sid],valids[sid]);r=dict(**sp,round=rnd,arm=arm,status='EVALUATED',preparation_valid_both_rounds=valids[sid],metrics=m,recomputed_gates=g)
                    if rnd==0:first[(sid,arm)]=m
                    else:
                        q=safe.repeat_comparison(m,first[(sid,arm)]);r['repeat_comparison']=q
                        if not(q['within_replay_tolerance'] and q['same_point_identity']):repeat_failures.append(dict(kind='writer_endpoint',state_id=sid,arm=arm,comparison=q))
                    records.append(r);dump(out/'endpoints'/f'record{len(records)-1:02d}.json',r);checkpoint()
                print('C139_KERNEL_ENDPOINT',rnd,sid,[(r['arm'],r['recomputed_gates']['complete_pass'],r['metrics']['response_drift']) for r in records[-2:]],flush=True)
        phase.finish();budget();rt.invariant();need(calls['metric_calls']==204 and calls['feature_calls']==36 and calls['model_forwards']==calls['initialisation_model_forwards']+2*204+3*36,'Fixed request coverage differs')
        sums,pairs=endpoint_summaries(records);pt={str(rnd):gate_table([r for r in preps if r['round']==rnd]) for rnd in (0,1)}
        paired={str(rnd):dict(full_preparation_passed=sum(r['gates']['complete_pass'] for r in fullreplay),kernel_preparation_passed=pt[str(rnd)]['passed'],rescued_preparation_ids=[sp['state_id'] for sp in registry() if not data['C138_prewrite'][sp['source_state_id']]['preparation_valid'] and prepby[(rnd,sp['state_id'])]['gates']['complete_pass']],destroyed_preparation_ids=[sp['state_id'] for sp in registry() if data['C138_prewrite'][sp['source_state_id']]['preparation_valid'] and not prepby[(rnd,sp['state_id'])]['gates']['complete_pass']]) for rnd in (0,1)}
        prep_ok=not repeat_failures and all(valids.values());writer_ok=prep_ok and all(sums['all24'][str(rnd)][ARMS[1]]['passed']==24 for rnd in (0,1))
        branch='PREREGISTER_SEPARATE_UNSEEN_LOCAL_CONFIRMATION' if writer_ok else ('RESOLVE_SEEN_CLASS_WRITER_FAILURES_BEFORE_CONFIRMATION' if prep_ok else 'STOP_FIXED_KERNEL_PREPARATION_FAMILY_RETAIN_FAILURES')
        if repeat_failures:branch='RESOLVE_REPEAT_DRIFT_NO_POSITIVE_DIAGNOSTIC'
        for n,x in (('ALL48_KERNEL_PREPARATIONS.json',preps),('ALL96_KERNEL_ENDPOINTS.json',records),('KERNEL_PREPARATION_SUMMARIES.json',pt),('PAIRED_PREPARATIONS.json',paired),('ENDPOINT_SUMMARIES.json',sums),('PAIRED_VS_RAW.json',pairs),('REPEATABILITY_FAILURES.json',repeat_failures)):dump(out/n,x)
        s.update(complete=True,decision='C139_EXPOSED_DIRECTION_DIAGNOSTIC_COMPLETE_NO_CANDIDATE_SELECTED',repeatable=not repeat_failures,known_controls_replay_consistent=True,kernel_preparations_all24_passed=prep_ok,frozen_writer_all24_development_gates_passed=writer_ok,preparation_valid_both_rounds=sum(valids.values()),preparation_summaries=pt,paired_preparations=paired,endpoint_summaries=sums,next_branch=branch,C138_parent_decision=data['C138_summary']['decision'])
        report=['# C139：已暴露方向的共享核制备诊断','','先独立核对C138归档，再实测相同24方向投影前后的制备门槛。保留24态全部分母；两个重复轮不构成48个独立样本。冻结C137和原raw写入器不变。','','共享核是固定C132响应行空间的正交补；不能把一阶抑制当作有限制备保证。所有状态属于开发诊断，不是新的确认态或排除输入。','','|制备|首轮通过/请求|重复轮通过/请求|','|---|---:|---:|',f"|C138原扰动重放|{sum(r['gates']['complete_pass'] for r in fullreplay)}/24|原C138记录核对|",f"|固定共享核扰动|{pt['0']['passed']}/24|{pt['1']['passed']}/24|",'','|核制备后的写入器|首轮通过/请求|重复轮通过/请求|','|---|---:|---:|']
        for arm in ARMS:report.append(f"|{arm}|{sums['all24']['0'][arm]['passed']}/24|{sums['all24']['1'][arm]['passed']}/24|")
        report+=['','下一分支：'+branch,'','无训练/反传/新Jacobian/SVD/方程求解/教师/回溯/写后修复。未选择候选或部署；LC2与单折均不闭合。'];(out/'REPORT_ZH.md').write_text('\n'.join(report)+'\n');stage='complete'
    except BaseException:errors.append(traceback.format_exc());s.update(complete=False,decision='C139_INCOMPLETE_NO_DIAGNOSTIC_CONCLUSION');print(errors[-1],flush=True)
    finally:
        checks={}
        for k,p in paths.items():
            try:checks[k]=sha(p)==c['pins'][k]['sha256']
            except Exception:checks[k]=False
        s['original_inputs_unchanged']=checks
        if not all(checks.values()):s.update(complete=False,decision='C139_INCOMPLETE_NO_DIAGNOSTIC_CONCLUSION');errors.append('Original input changed')
        checkpoint()
    return clean_json((out/'SUMMARY.json').read_bytes())

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in INPUT_KEYS:p.add_argument('--'+k.lower(),required=True)
    p.add_argument('--out',required=True);args=p.parse_args();s=run({k:Path(getattr(args,k.lower())) for k in INPUT_KEYS},Path(args.out));print(json.dumps(s,ensure_ascii=False,indent=2),flush=True)
    if not s['complete']:raise SystemExit(1)
