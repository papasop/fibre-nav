"""Read fixed C143 tensors and observations; execute only delivered audit code."""
from core144 import *
from inputs143 import load as load_bound_inputs

def same_arrays(actual,expected,label):
    need(set(actual)==set(expected),label+' tensor schema')
    for k in expected:exact(actual[k],expected[k],label+'/'+k)

def audit(paths,out,c=None):
    c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    need(set(paths)==set(INPUT_KEYS),'Exactly four original archives required')
    for k,p in paths.items():compare(sha(p),c['pins'][k]['sha256'],'Fixed original SHA256/'+k)
    dump(out/'INPUTS_FROZEN.json',{k:dict(path=str(p),sha256=sha(p),known_before_C144=True) for k,p in paths.items()})
    def budget():need(time.monotonic()-started<TIMEOUT,'C144 fixed20-minute audit budget exceeded')
    binding=dict(c['binding143'],production=c['production'],pins={k:c['pins'][k] for k in parent.INPUT_KEYS})
    data=load_bound_inputs({k:paths[k] for k in parent.INPUT_KEYS},out/'ORIGINAL_INPUT_RECONSTRUCTION',binding);budget()
    rows=parent.registry();rank=len(data['V']);need(rank==36 if c['production'] else 0<rank<=36,'Shared dimension differs')
    with contextlib.closing(Archive(paths['C143'])) as z:
        members=z.verify_manifest();names={n[7:] for n in z.z.namelist() if n.startswith('source/')}
        need(names==set(c['source143']),'Trusted C143 source inventory differs')
        for n,h in c['source143'].items():compare(hashlib.sha256(z.raw('source/'+n)).hexdigest(),h,'Trusted C143 source/'+n)
        s=z.json('SUMMARY.json');compare(s['protocol'],parent.PROTOCOL);compare(s['complete'],True);compare(s['stage'],'complete');compare(s['errors'],[])
        if c['production']:compare(s['synthetic_fixture'],False);compare(s['real_model_execution_claim'],True)
        compare(z.json('DESIGN_FROZEN_BEFORE_MODEL.json'),parent.design(),'Fixed C143 design/budgets')
        for k in parent.INPUT_KEYS:compare(z.json('INPUTS_FROZEN.json')[k]['sha256'],c['pins'][k]['sha256'],'Original linked archive/'+k)
        expected_registry=[dict(sp,prepared_sha256=safe.tensor_hash(data['points'][sp['state_id']]['prepared']),raw_sha256=safe.tensor_hash(data['points'][sp['state_id']]['raw'])) for sp in rows]
        compare(z.json('ALL84_TRAINING_REGISTRY_FROZEN.json'),expected_registry,'Exact84 ordered training registry')
        need(all(sp['dataset_role']=='training' and sp['used_for_training'] and not sp['independent_validation_eligible'] for sp in rows),'All84 exposed membership')
        inputs={sid+'__'+k:p[k] for sid,p in data['points'].items() for k in ('prepared','raw','features')}
        same_arrays(z.arrays('ALL84_TRAINING_INPUTS.npz'),inputs,'Original84 coordinates and full3211 features')
        same_arrays(z.arrays('FROZEN_ORIGINAL_WRITER.npz'),data['writer'],'Raw writer original');same_arrays(z.arrays('FROZEN_ORIGINAL_CONTROL.npz'),data['control'],'Raw control original')
        same_arrays(z.arrays('FROZEN_SHARED_ROW_BASIS.npz'),{'V':data['V']},'Original shared row basis')
        initial=dict(a=data['initial_a'],m=np.zeros(rank),v=np.zeros(rank));same_arrays(z.arrays('INITIAL_C140_STEP12_CONTROL.npz'),initial,'Exact C140 step12/reset moments')
        compare(z.json('FROZEN_INPUT_IDENTITIES.json'),parent.frozen_digest(data));compare(z.json('MODEL_PINS_REFERENCE.json'),data['modelpins']);compare(z.json('C142_BOUND_RECEIPT.json'),clean_json((out/'ORIGINAL_INPUT_RECONSTRUCTION/C142_BOUND_RECEIPT.json').read_bytes()))
        # All252 native preflight records are independently gated and rebound to original observations.
        preflight=z.json('PREFLIGHT_PROGRESS.json');need(len(preflight)==252,'All252 preflights required');pre={}
        for r,(sp,kind) in zip(preflight,[(sp,kind) for sp in rows for kind in ('prepared','raw','warm')]):
            budget();sid=sp['state_id'];p=data['points'][sid]
            for k,v in dict(**sp,kind=kind).items():compare(r[k],v,'Preflight identity/'+k)
            point=p[kind] if kind in ('prepared','raw') else proposal(p['prepared'],p['raw'],data['V'],initial['a'])[0]
            gates,_,_=native_gates(r['metrics'],sp['case'],point,kind!='prepared')
            if kind=='prepared':need(all(gates.values()),'Invalid preparation retained; cannot qualify')
            comparisons=[repeat_check(r['metrics'],old) for old in data['native_refs'][(sid,kind)]];compare(r['comparisons'],comparisons,'Original preflight replay')
            need(all(q['same_point_identity'] and q['within_replay_tolerance'] for q in comparisons),'Original replay drift')
            pre[(sid,kind)]=r['metrics']
        functional_records={};maximum_point_parity=0.
        for member,kind in [('ZERO_FUNCTIONAL_PROGRESS.json','raw'),('WARM_FUNCTIONAL_PROGRESS.json','warm')]:
            rr=z.json(member);need(len(rr)==84,'All84 functional/native parity records required')
            for r,sp in zip(rr,rows):
                sid=sp['state_id'];p=data['points'][sid];point=p['raw'] if kind=='raw' else proposal(p['prepared'],p['raw'],data['V'],initial['a'])[0]
                compare(r['state_id'],sid);compare(r['point_sha256'],safe.tensor_hash(point),'Functional point identity')
                err=close(r['observed'],pre[(sid,kind)]['observed'],'Functional/native observations',2e-6);kl=abs(r['KL']-pre[(sid,kind)]['KL']);need(kl<=2e-7,'Functional/native KL parity')
                compare(r['observed_parity_Linf'],err);compare(r['KL_parity_absolute_error'],kl)
                need(type(r['point_parity_error_Linf']) in (int,float) and np.isfinite(r['point_parity_error_Linf']) and 0<=r['point_parity_error_Linf']<=2e-6,'Invalid functional parity scalar')
                functional_records[(sid,kind)]=r;maximum_point_parity=max(maximum_point_parity,r['point_parity_error_Linf'])
                if kind=='warm':compare(r['native_point_sha256'],safe.tensor_hash(point))
        # The archived state gradients are treated as inputs, never certified as true derivatives.
        progress=z.json('TRAINING_PROGRESS.json');need(len(progress)==12,'All12 fixed updates required')
        training_members={n for n in z.z.namelist() if n.startswith('training/')};expected_training={f'training/step{step:02d}/{name}' for step in range(1,13) for name in ('OBSERVATIONS.json','GRADIENT_PROGRESS.json','AGGREGATION.json','UPDATE.npz')}
        need(training_members==expected_training,'Training member coverage or fixed12 schedule differs')
        a=initial['a'].copy();m=initial['m'].copy();v=initial['v'].copy();trend=[];perstate=[];max_update_error=0.;max_gradient_error=0.
        for step in range(1,13):
            budget();folder=f'training/step{step:02d}/';u=z.arrays(folder+'UPDATE.npz');need(set(u)=={'before_a','before_m','before_v','full_weighted_gradient','after_a','after_m','after_v'},'Update schema differs')
            for k,expected in [('before_a',a),('before_m',m),('before_v',v)]:exact(u[k],expected,'Sequential update chain/'+k)
            obs=z.json(folder+'OBSERVATIONS.json');compare(z.json(folder+'GRADIENT_PROGRESS.json'),obs,'Gradient progress duplicate');need(len(obs)==84,'Missing gradient state')
            for r,sp in zip(obs,rows):
                sid=sp['state_id'];p=data['points'][sid]
                for k,value in dict(state_id=sid,case=sp['case'],C143_membership='training').items():compare(r[k],value,'Gradient row binding/'+k)
                compare(r['coefficient_digest'],row_digest(a),'Gradient shared vector identity');farray(np.asarray(r['state_gradient'],float),(rank,))
                point=proposal(p['prepared'],p['raw'],data['V'],a)[0];compare(r['functional_point_sha256'],safe.tensor_hash(point),'Archived gradient point reconstructed')
                need(type(r['point_parity_error_Linf']) in (float,int) and np.isfinite(r['point_parity_error_Linf']) and 0<=r['point_parity_error_Linf']<=2e-6,'Invalid gradient point parity')
                maximum_point_parity=max(maximum_point_parity,r['point_parity_error_Linf'])
                if step==1:
                    close(r['observed'],functional_records[(sid,'warm')]['observed'],'Initial gradient/warm functional observations',2e-6)
                    need(abs(r['KL']-functional_records[(sid,'warm')]['KL'])<=2e-7,'Initial gradient/warm KL parity')
                parts=loss(r['observed'],r['KL'],point,p['prepared'],a,sp['case']);compare(r['objective_parts'],parts,'Independent state objective',atol=1e-10,rtol=1e-9);compare(r['objective'],sum(parts.values()),'State objective sum',atol=1e-10,rtol=1e-9)
            g,agg=aggregate([r['objective'] for r in obs],[r['state_gradient'] for r in obs]);compare(z.json(folder+'AGGREGATION.json'),agg,'Independent84 aggregation')
            max_gradient_error=max(max_gradient_error,close(u['full_weighted_gradient'],g,'Independent weighted gradient',1e-10))
            for r,weight in zip(obs,agg['weights']):
                compare(r['aggregate_weight'],weight);close(r['gradient_contribution'],weight*np.asarray(r['state_gradient']), 'Weighted state contribution',1e-10)
                perstate.append(dict(step=step,state_id=r['state_id'],case=r['case'],objective=r['objective'],aggregate_weight=weight,archived_gradient_L2=float(np.linalg.norm(r['state_gradient'])),weighted_gradient_L2=float(np.linalg.norm(r['gradient_contribution'])),signed_response=r['observed'][:6],response_Linf=float(np.max(abs(np.asarray(r['observed'][:6]))))))
            # Recompute updates using the archived aggregate after independently checking its sum.
            na,nm,nv,info=update(a,u['full_weighted_gradient'],m,v,step)
            for k,expected in [('after_a',na),('after_m',nm),('after_v',nv)]:max_update_error=max(max_update_error,close(u[k],expected,'Independent Adam/'+k,1e-14))
            expected_info=dict(**info,aggregate_objective_before_update=agg['objective'],state_weights=agg['weights'],before_digest=row_digest(a),after_digest=row_digest(u['after_a']),mean_objective_before_update=agg['mean_state_loss'],maximum_objective_before_update=agg['maximum_state_loss'],update_file=folder+'UPDATE.npz',update_sha256=hashlib.sha256(z.raw(folder+'UPDATE.npz')).hexdigest(),n_training_records=84)
            compare(progress[step-1],expected_info,'Fixed update progress')
            trend.append(dict(step=step,aggregate_objective_before_update=agg['objective'],mean_objective_before_update=agg['mean_state_loss'],maximum_objective_before_update=agg['maximum_state_loss'],gradient_L2=info['gradient_L2'],gradient_clip_factor=info['gradient_clip_factor'],projected_coordinates=info['projected_coordinates']))
            a=u['after_a'];m=u['after_m'];v=u['after_v'];print('C144_UPDATE_AUDIT',step,'/12',flush=True)
        same_arrays(z.arrays('FINAL_SHARED_CONTROL_STEP12.npz'),{'a':a},'Only actual fixed step12 final head')
        same_arrays(z.arrays('LATEST_PARTIAL_CONTROL.npz'),dict(a=a,m=m,v=v,completed_steps=np.array([12],np.int64)),'Latest partial copy is final12')
        freeze=z.json('STEP12_FREEZE_BEFORE_EVALUATION.json');compare(freeze['completed_steps'],12);compare(freeze['checkpoint_choice'],'Fixed C143 step12 only');compare(freeze['no_best_or_recovery_selection'],True);compare(freeze['control_digest'],row_digest(a));compare(freeze['control_sha256'],hashlib.sha256(z.raw('FINAL_SHARED_CONTROL_STEP12.npz')).hexdigest())
        sealed={'DESIGN_FROZEN_BEFORE_MODEL.json','INPUTS_FROZEN.json','ALL84_TRAINING_REGISTRY_FROZEN.json','ALL84_TRAINING_INPUTS.npz','MODEL_PINS_REFERENCE.json','FROZEN_SHARED_ROW_BASIS.npz','INITIAL_C140_STEP12_CONTROL.npz','FROZEN_ORIGINAL_WRITER.npz','FROZEN_ORIGINAL_CONTROL.npz','FROZEN_INPUT_IDENTITIES.json','C142_BOUND_RECEIPT.json','FINAL_SHARED_CONTROL_STEP12.npz','ALL252_FROZEN_FINAL_POINTS.npz','ALL252_FROZEN_FINAL_OUTCOMES.json'}
        need(set(freeze['proposal_seals'])==sealed,'Final freeze seal coverage differs')
        for n,h in freeze['proposal_seals'].items():compare(hashlib.sha256(z.raw(n)).hexdigest(),h,'Frozen actual bytes/'+n)
        points={};outcomes=[]
        for sp in rows:
            sid=sp['state_id'];p=data['points'][sid];warm,wd=proposal(p['prepared'],p['raw'],data['V'],initial['a']);final,fd=proposal(p['prepared'],p['raw'],data['V'],a)
            for arm,point,diag in [(ARMS[0],p['raw'],dict(original_raw_bytes_retained=True)),(ARMS[1],warm,wd),(ARMS[2],final,fd)]:points[(sid,arm)]=point;outcomes.append(dict(**sp,arm=arm,point_sha256=safe.tensor_hash(point),proposal_diagnostics=diag))
        same_arrays(z.arrays('ALL252_FROZEN_FINAL_POINTS.npz'),{sid+'__'+arm:point for (sid,arm),point in points.items()},'All252 actual final proposals');compare(z.json('ALL252_FROZEN_FINAL_OUTCOMES.json'),outcomes,'Independent frozen finite maps')
        records=z.json('ALL504_FINAL_TRAINING_ENDPOINTS.json');compare(z.json('FINAL_ENDPOINT_PROGRESS.json'),records);need(len(records)==504,'All504 endpoint records required')
        measured={n for n in z.z.namelist() if n.startswith('final_measurements/')};need(measured=={f'final_measurements/record{i:03d}.json' for i in range(504)},'Endpoint member inventory differs')
        first={};drifts=[];repeats=[]
        for i,(r,(rnd,sp,arm)) in enumerate(zip(records,[(rnd,sp,arm) for rnd in (0,1) for sp in rows for arm in ARMS])):
            budget();sid=sp['state_id'];p=data['points'][sid];point=points[(sid,arm)]
            for k,value in dict(**sp,round=rnd,arm=arm).items():compare(r[k],value,'Endpoint state/arm/round/'+k)
            compare(z.json(f'final_measurements/record{i:03d}.json'),r,'Individual endpoint copy')
            gates=endpoint_gates(r['metrics'],sp['case'],point,p['prepared'],True);compare(r['recomputed_gates'],gates,'Independent native endpoint gates')
            control=np.zeros(rank) if arm==ARMS[0] else (initial['a'] if arm==ARMS[1] else a);parts=loss(r['metrics']['observed'],r['metrics']['KL'],point,p['prepared'],control,sp['case'])
            compare(r['training_objective_parts'],parts,'Final native objective parts',atol=1e-10,rtol=1e-9);compare(r['training_objective'],sum(parts.values()),'Final native objective sum',atol=1e-10,rtol=1e-9)
            if arm in ARMS[:2]:
                q=repeat_check(r['metrics'],pre[(sid,'raw' if arm==ARMS[0] else 'warm')]);compare(r['native_control_replay'],q)
                if not(q['same_point_identity'] and q['within_replay_tolerance']):drifts.append(dict(state_id=sid,arm=arm,round=rnd,comparison=q))
            if rnd==0:first[(sid,arm)]=r['metrics']
            else:
                q=repeat_check(r['metrics'],first[(sid,arm)]);compare(r['repeat_comparison'],q)
                if not(q['same_point_identity'] and q['within_replay_tolerance']):repeats.append(dict(state_id=sid,arm=arm,comparison=q))
        sums,pairs=tables(records)
        compare(z.json('FINAL_TRAINING_SUMMARIES.json'),sums);compare(s['final_training_summaries'],sums);compare(z.json('PAIRED_VS_RAW_AND_C140.json'),pairs);compare(s['paired_vs_raw_and_C140'],pairs)
        compare(z.json('CONTROL_REPLAY_FAILURES.json'),drifts);compare(z.json('REPEATABILITY_FAILURES.json'),repeats)
        consistent=not drifts and not repeats;passed=consistent and all(sums['all84'][str(j)][ARMS[2]]['passed']==84 for j in (0,1))
        decision='C143_FIXED_FINAL_COMPLETE_REPLAY_DRIFT_NO_QUALIFICATION' if not consistent else ('C143_FIXED_FINAL_ALL84_TRAINING_GATES_PASSED' if passed else 'C143_FIXED_FINAL_TRAINING_CONSTRAINT_FAILURE')
        branch='RESOLVE_REPLAY_DRIFT' if not consistent else ('PREREGISTER_SEPARATE_UNSEEN_LOCAL_CONFIRMATION' if passed else 'STOP_FIXED12_SCHEDULE_NO_CHECKPOINT_SUBSTITUTION')
        compare(s['decision'],decision);compare(s['fixed_final_training_qualification_passed'],bool(passed));compare(s['repeatable'],not repeats);compare(s['controls_replay_consistent'],not drifts);compare(s['next_branch'],branch)
        scalar_fields=dict(training_states=84,independent_confirmation_states=0,confirmation_states_evaluated=0,excluded_input_states_evaluated=0,requested_training_steps=12,requested_scalar_backward_calls=1008,requested_final_endpoint_records=504,new_model_forwards=3865,new_backward_calls=1008,new_training_steps=12,completed_training_steps=12,new_head_fits=1,preflight_records=252,zero_parity_records=84,warm_parity_records=84,final_endpoint_records=504,trainable_shared_coordinates=rank,training_response_safety=.00125,acceptance_response_budget=.002)
        for k,val in scalar_fields.items():compare(s[k],val,'Parent fixed accounting/'+k)
        compare(s['runtime_calls'],dict(model_forwards=3865,initialisation_model_forwards=1,preflight_metric_attempts=252,preflight_metrics=252,path_metric_attempts=504,path_metrics=504))
        stats=s['gradient_stats']
        compare(stats['maximum_point_parity_error_Linf'],maximum_point_parity,'Maximum recorded functional point parity')
        for k,val in dict(functional_endpoint_calls=1176,functional_zero_calls=84,functional_warm_calls=84,scalar_backward_attempts=1008,scalar_backward_calls=1008,full84_gradient_batches=12,torch_optimizer_objects=0,GPT2_optimizer_updates=0,LoRA_A_optimizer_updates=0,base_writer_updates=0).items():compare(stats[k],val)
        for k in ('all84_used_for_training','functional_zero_parity_complete_before_gradients','functional_warm_parity_complete_before_gradients','step12_frozen_before_final_evaluation'):compare(s[k],True)
        for k in ('candidate_selected','writer_deployed','best_checkpoint_selected','recovery_checkpoint_selected','thresholds_changed','parent_decision_reclassified','LC2_closed','LC2_single_fold_passed','archived_source_executed','AD_accuracy_independently_certified','forward_error_bound_established'):compare(s[k],False)
        for k in ('new_states_prepared','new_model_features_acquired','new_Jacobians','new_SVD_calls','linear_systems_solved','teacher_calls','backtracking_steps','online_response_repair_steps','GPT2_optimizer_updates','LoRA_A_optimizer_updates','base_writer_updates'):compare(s[k],0)
        compare(s['C141_parent_decision_retained'],'C141_FROZEN_KERNEL_LOCAL_CONFIRMATION_FAILURE');compare(s['C142_parent_decision'],data['C142_summary']['decision']);compare(s['initialization'],'Exact C140 step12; zero Adam moments');compare(s['original_inputs_unchanged'],{k:True for k in parent.INPUT_KEYS})
        by={(r['round'],r['state_id'],r['arm']):r for r in records};diagnostics=[]
        for rnd in (0,1):
            for sp in rows:
                sid=sp['state_id'];old=by[(rnd,sid,ARMS[1])];new=by[(rnd,sid,ARMS[2])];og=old['recomputed_gates']['complete_pass'];ng=new['recomputed_gates']['complete_pass']
                status='BOTH_PASSED' if og and ng else ('NEWLY_FAILED' if og else ('RESCUED' if ng else 'BOTH_FAILED'))
                diagnostics.append(dict(**sp,round=rnd,C140_passed=og,C143_passed=ng,status=status,failed_gates=new['recomputed_gates']['failed_gates'],C140_objective=old['training_objective'],C143_objective=new['training_objective'],**response_diagnostic(old['metrics']['observed'][:6],new['metrics']['observed'][:6])))
        affected=[r for r in diagnostics if not r['C143_passed']];affected_ids=set(r['state_id'] for r in affected)
        training_focus=[r for r in perstate if r['state_id'] in affected_ids]
        old=sums['all84']['0'][ARMS[1]];new=sums['all84']['0'][ARMS[2]]
        relation=dict(C140_aggregate_objective=old['aggregate_training_objective'],C143_aggregate_objective=new['aggregate_training_objective'],aggregate_reduction_fraction=1-new['aggregate_training_objective']/old['aggregate_training_objective'] if old['aggregate_training_objective'] else None,C140_passed=old['passed'],C143_passed=new['passed'],acceptance_count_change=new['passed']-old['passed'],maximum_response_drift_change=new['maximum_response_drift']-old['maximum_response_drift'],C140_response_safety_count=old['within_training_response_safety'],C143_response_safety_count=new['within_training_response_safety'],loss_improvement_does_not_imply_all_gate_acceptance=True,convergence_or_feature_insufficiency_established=False)
        for name,obj in [('RECOMPUTED_FINAL_SUMMARIES.json',sums),('RECOMPUTED_PAIRED_EFFECTS.json',pairs),('ALL168_RESPONSE_EFFECT_DIAGNOSTICS.json',diagnostics),('C143_FAILED_STATE_DIAGNOSTICS.json',affected),('RECOMPUTED_12STEP_TREND.json',trend),('ALL1008_TRAINING_ROW_DIAGNOSTICS.json',perstate),('FAILED_STATE_12STEP_TRAJECTORIES.json',training_focus),('LOSS_VS_ACCEPTANCE_DIAGNOSTIC.json',relation)]:dump(out/name,obj)
        budget()
        return dict(parent_protocol=parent.PROTOCOL,parent_decision=decision,parent_decision_reclassified=False,parent_training_qualification_passed=bool(passed),C143_members_hashed=members,C143_source_members_checked=len(names),original_archives_rehashed=list(INPUT_KEYS),training_states_audited=84,training_steps_audited=12,training_observations_audited=1008,preflight_records_audited=252,functional_parity_records_audited=168,final_proposals_reconstructed=252,endpoint_records_audited=504,frozen_artifact_seals_checked=len(sealed),full3211_features_rebound_to_originals=True,actual_final_head_is_update12=True,maximum_weighted_gradient_arithmetic_error_Linf=max_gradient_error,maximum_update_arithmetic_error_Linf=max_update_error,gradients_independently_recomputed=False,repeatable=not repeats,controls_replay_consistent=not drifts,final_training_summaries=sums,paired_effects=pairs,C143_failed_ids=sums['all84']['0'][ARMS[2]]['failed_ids'],newly_failed_vs_C140=pairs['all84']['0'][ARMS[2]][ARMS[1]]['destroyed_ids'],rescued_vs_C140=pairs['all84']['0'][ARMS[2]][ARMS[1]]['rescued_ids'],loss_vs_acceptance=relation,all84_remain_training=True,C141_confirmation_failure_retained=True,external_C132_C139_original_ZIPs_reopened=False,elapsed_audit_seconds=time.monotonic()-started)
