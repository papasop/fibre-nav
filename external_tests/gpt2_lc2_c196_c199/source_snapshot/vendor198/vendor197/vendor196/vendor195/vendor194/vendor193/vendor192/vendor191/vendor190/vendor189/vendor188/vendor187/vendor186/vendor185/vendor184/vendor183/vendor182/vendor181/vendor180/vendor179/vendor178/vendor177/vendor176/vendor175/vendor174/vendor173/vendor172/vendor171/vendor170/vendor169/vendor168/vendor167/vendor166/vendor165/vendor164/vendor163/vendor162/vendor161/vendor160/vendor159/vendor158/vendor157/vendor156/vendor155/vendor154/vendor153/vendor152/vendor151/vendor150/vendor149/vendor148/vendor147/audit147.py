"""Reconstruct all recorded C146 training updates without loading a neural model."""
from core147 import *
from inputs146 import audit145

def source_check(z,pins,label):
    names={n[7:] for n in z.z.namelist() if n.startswith('source/')};need(names==set(pins),label+' source inventory')
    for n,h in pins.items():compare(hashlib.sha256(z.raw('source/'+n)).hexdigest(),h,label+' trusted source/'+n)

def bind_originals(paths,out,c):
    need(set(paths)==set(INPUT_KEYS),'Exactly C146/C145/C143 original archives')
    for k,p in paths.items():compare(sha(p),c['pins'][k]['sha256'],'Fixed original/'+k)
    dump(out/'INPUTS_FROZEN.json',{k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()})
    with contextlib.closing(Archive(paths['C143'])) as z:
        count143=z.verify_manifest();source_check(z,parent.prior.auditcore.contract()['source143'],'C143');s143=z.json('SUMMARY.json');compare(s143['protocol'],parent.base.PROTOCOL);compare(s143['complete'],True)
        ar=z.arrays('ALL84_TRAINING_INPUTS.npz');need(set(ar)=={r['state_id']+'__'+k for r in registry() for k in ('prepared','raw','features')},'Original C143 full84 inputs')
        points={r['state_id']:{k:ar[r['state_id']+'__'+k] for k in ('prepared','raw','features')} for r in registry()};V=z.arrays('FROZEN_SHARED_ROW_BASIS.npz')['V'];a=z.arrays('FINAL_SHARED_CONTROL_STEP12.npz')['a'];warm=z.arrays('INITIAL_C140_STEP12_CONTROL.npz')['a'];writer=z.arrays('FROZEN_ORIGINAL_WRITER.npz');control=z.arrays('FROZEN_ORIGINAL_CONTROL.npz');modelpins=z.json('MODEL_PINS_REFERENCE.json')
        pre={(r['state_id'],r['kind']):r['metrics'] for r in z.json('PREFLIGHT_PROGRESS.json')};ends={(r['round'],r['state_id'],r['arm']):r['metrics'] for r in z.json('ALL504_FINAL_TRAINING_ENDPOINTS.json')};refs={}
        for sp in registry():
            sid=sp['state_id'];refs[(sid,'prepared')]=[pre[(sid,'prepared')]]
            for kind,arm in [('raw','raw_control'),('C140','C140_frozen_step12'),('C143','C143_shared_step12')]:refs[(sid,kind)]=[ends[(rnd,sid,arm)] for rnd in (0,1)]
    rank=len(a);need(rank==36 if c['production'] else 0<rank<=36,'Declared shared control rank');farray(V,(rank,DIM));farray(a,(rank,));d=dict(points=points,V=V,a=a,C140_a=warm,writer=writer,control=control,modelpins=modelpins,refs=refs)
    with contextlib.closing(Archive(paths['C145'])) as z:
        cc=dict(c['binding146'],production=c['production']);cc['pins']=dict(cc['pins'],C143=c['pins']['C143'],C145=c['pins']['C145']);s145,receipt145=audit145(z,d,cc)
    dump(out/'C145_ORIGINAL_BOUND_RECEIPT.json',receipt145)
    return d,dict(C143=count143,C145=receipt145['members_hashed'])

def audit(paths,out,c=None):
    c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    def budget():need(time.monotonic()-started<TIMEOUT,'C147 fixed20-minute audit budget')
    d,counts=bind_originals(paths,out,c);budget()
    with contextlib.closing(Archive(paths['C146'])) as z:
        counts['C146']=z.verify_manifest();source_check(z,c['source146'],'C146');s=z.json('SUMMARY.json');compare(s['protocol'],parent.PROTOCOL);compare(s['complete'],True);compare(s['decision'],c['expected_parent_decision']);compare(z.json('DESIGN_FROZEN_BEFORE_MODEL.json'),parent.design(),'Original fixed C146 protocol')
        if c['production']:compare(s['synthetic_fixture'],False);compare(s['real_model_execution_claim'],True)
        for k,v in dict(training_states=84,requested_training_steps=12,new_training_steps=12,completed_training_steps=12,new_head_fits=1,new_backward_calls=2016,new_model_forwards=4033,preflight_records=336,zero_parity_records=84,warm_parity_records=84,final_endpoint_records=504,confirmation_states_evaluated=0,excluded_input_states_evaluated=0,new_states_prepared=0,new_model_features_acquired=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,GPT2_optimizer_updates=0,LoRA_A_optimizer_updates=0,base_writer_updates=0,teacher_calls=0,backtracking_steps=0,online_response_repair_steps=0).items():compare(s[k],v,'Original counts/'+k)
        for k in ('candidate_selected','writer_deployed','C145_probe_selected','best_checkpoint_selected','recovery_checkpoint_selected','thresholds_changed','parent_decision_reclassified','archived_source_executed','LC2_closed','LC2_single_fold_passed'):compare(s[k],False)
        for k in ('all84_remain_exposed_training','functional_zero_parity_complete_before_gradients','functional_warm_parity_complete_before_gradients','step12_frozen_before_final_evaluation'):compare(s[k],True)
        compare(s['training_response_safety'],.0018);compare(s['acceptance_response_budget'],.002)
        compare(z.json('ALL84_TRAINING_REGISTRY_FROZEN.json'),registry(),'All84 exposed training membership')
        linked=z.json('INPUTS_FROZEN.json');need(set(linked)==set(parent.INPUT_KEYS),'All six original links')
        for k in parent.INPUT_KEYS:compare(linked[k]['sha256'],c['binding146']['pins'][k]['sha256'],'C146 fixed original link/'+k)
        compare(z.json('FROZEN_INPUT_IDENTITIES.json'),parent.frozen_digest(d));compare(z.json('MODEL_PINS_REFERENCE.json'),d['modelpins'])
        compare(s['original_inputs_unchanged'],{k:True for k in parent.INPUT_KEYS});compare(z.json('ORIGINAL_INPUTS_UNCHANGED.json'),{k:True for k in parent.INPUT_KEYS})
        if c['production']:
            runtime_pins=z.json('model_runtime/MODEL_PINS.json');compare({k:v for k,v in runtime_pins.items() if k!='torch'},{k:v for k,v in d['modelpins'].items() if k!='torch'},'Recorded actual model pin identity')
        ar=z.arrays('ALL84_TRAINING_INPUTS.npz');need(set(ar)=={sid+'__'+k for sid in d['points'] for k in ('prepared','raw','features')},'All full3211 features and coordinates')
        for sid,p in d['points'].items():
            for k in ('prepared','raw','features'):exact(ar[sid+'__'+k],p[k],'C143/C146 actual input/'+sid+'/'+k)
        exact(z.arrays('FROZEN_SHARED_ROW_BASIS.npz')['V'],d['V'],'Actual basis')
        for name,values in [('FROZEN_ORIGINAL_WRITER.npz',d['writer']),('FROZEN_ORIGINAL_CONTROL.npz',d['control'])]:
            found=z.arrays(name);need(set(found)==set(values),'Head tensor inventory')
            for k,v in values.items():exact(found[k],v,'Actual frozen head/'+k)
        initial=z.arrays('INITIAL_C143_STEP12_CONTROL.npz');need(set(initial)=={'a','m','v'},'Initialization tensor inventory');exact(initial['a'],d['a'],'Exact C143 final initialization');exact(initial['m'],np.zeros_like(d['a']),'Zero first moment');exact(initial['v'],np.zeros_like(d['a']),'Zero second moment');compare(s['initial_C143_control_digest'],row_digest(d['a']))
        pre=z.json('PREFLIGHT_PROGRESS.json');compare([(r['state_id'],r['kind']) for r in pre],[(r['state_id'],kind) for r in registry() for kind in ('prepared','raw','C140','C143')]);native={}
        byspec={sp['state_id']:sp for sp in registry()}
        for r in pre:
            for k,val in byspec[r['state_id']].items():compare(r[k],val,'Native preflight state binding/'+k)
            budget();sid=r['state_id'];kind=r['kind'];p=d['points'][sid];point=p[kind] if kind in ('prepared','raw') else proposal(p['prepared'],p['raw'],d['V'],d['C140_a'] if kind=='C140' else d['a'])[0];g,_,_=native_gates(r['metrics'],r['case'],point,kind!='prepared')
            if kind=='prepared':need(all(g.values()),'Invalid preparation retained')
            q=[repeat_check(r['metrics'],old) for old in d['refs'][(sid,kind)]];compare(r['comparisons'],q,'Original replay comparisons');need(all(x['same_point_identity'] and x['within_replay_tolerance'] for x in q),'Original native replay drift');native[(sid,kind)]=r['metrics']
        for filename,kind,aa in [('ZERO_FUNCTIONAL_PROGRESS.json','raw',np.zeros_like(d['a'])),('WARM_FUNCTIONAL_PROGRESS.json','C143',d['a'])]:
            rows=z.json(filename);compare([r['state_id'] for r in rows],[r['state_id'] for r in registry()])
            for r in rows:
                sid=r['state_id'];p=d['points'][sid];point=proposal(p['prepared'],p['raw'],d['V'],aa)[0];compare(r['point_sha256'],safe.tensor_hash(point));need(0<=r['point_parity_error_Linf']<=2e-6,'Recorded functional point parity');e=close(r['observed'],native[(sid,kind)]['observed'],'Recorded functional/native observations',2e-6);k=abs(r['KL']-native[(sid,kind)]['KL']);need(k<=2e-7,'Recorded functional/native KL');compare(r['observed_parity_Linf'],e);compare(r['KL_parity_absolute_error'],k)
        history=z.json('TRAINING_PROGRESS.json');need(len(history)==12,'All12 updates retained');a=initial['a'];m=initial['m'];v=initial['v'];rank=len(a);steps=[];state_rows=[];batches=[];deltas=[];max_update_error=0.;max_gradient_error=0.
        for step in range(1,13):
            budget();folder=f'training/step{step:02d}/';rows=z.json(folder+'OBSERVATIONS.json');compare(rows,z.json(folder+'GRADIENT_PROGRESS.json'),'Gradient duplicate');compare([r['state_id'] for r in rows],[r['state_id'] for r in registry()]);u=z.arrays(folder+'UPDATE.npz');need(set(u)=={'before_a','before_m','before_v','full_weighted_gradient','after_a','after_m','after_v'},'Exact update tensor inventory')
            for k,want in [('before_a',a),('before_m',m),('before_v',v)]:exact(u[k],want,'Chained update/'+k)
            for sp,r in zip(registry(),rows):
                for k,val in sp.items():compare(r[k],val,'Training row identity/'+k)
                sid=sp['state_id'];p=d['points'][sid];point=proposal(p['prepared'],p['raw'],d['V'],a)[0];compare(r['coefficient_digest'],row_digest(a));compare(r['functional_point_sha256'],safe.tensor_hash(point));need(0<=r['point_parity_error_Linf']<=2e-6,'Recorded training point parity');compare(r['objective_parts'],loss(r['observed'],r['KL'],point,p['prepared'],a,sp['case']),'Independent loss',atol=1e-10,rtol=1e-9)
            gradient,agg=aggregate(rows,rank);compare(z.json(folder+'AGGREGATION.json'),agg,'Independent max/mean aggregation');max_gradient_error=max(max_gradient_error,close(u['full_weighted_gradient'],gradient,'Full weighted gradient arithmetic',1e-12))
            for r,w,part in zip(rows,agg['response_weights'],agg['state_gradient_contributions']):compare(r['response_weight'],w);compare(r['guard_weight'],1/84);close(r['gradient_contribution'],part,'Per-state gradient contribution',1e-12)
            new,mm,vv,info=update(a,gradient,m,v,step)
            for k,want in [('after_a',new),('after_m',mm),('after_v',vv)]:max_update_error=max(max_update_error,close(u[k],want,'Independent Adam '+k,1e-12))
            h=history[step-1];expected=dict(**info,aggregate_objective_before_update=agg['objective'],maximum_response_loss=agg['maximum_response_loss'],mean_response_loss=agg['mean_response_loss'],mean_guard_loss=agg['mean_guard_loss'],maximizing_state_ids=agg['maximizing_state_ids'],response_weights=agg['response_weights'],before_digest=row_digest(a),after_digest=row_digest(u['after_a']),update_file=folder+'UPDATE.npz',update_sha256=hashlib.sha256(z.raw(folder+'UPDATE.npz')).hexdigest(),n_training_records=84);compare(h,expected,'Training progress arithmetic',atol=1e-12,rtol=1e-9)
            delta=u['after_a']-a;deltas.append(delta.copy());batches.append(rows);obs=np.asarray([r['observed'][:6] for r in rows]);state_max=np.max(abs(obs),axis=1);rg=np.asarray([r['response_gradient'] for r in rows]);gg=np.asarray([r['guard_gradient'] for r in rows]);weights=np.asarray(agg['response_weights']);response_total=np.sum(weights[:,None]*rg,axis=0);guard_total=np.mean(gg,axis=0)
            steps.append(dict(step=step,aggregate_objective=agg['objective'],maximum_response_loss=agg['maximum_response_loss'],mean_response_loss=agg['mean_response_loss'],mean_guard_loss=agg['mean_guard_loss'],maximum_archived_functional_response=float(state_max.max()),functional_response_within_acceptance_budget=int(np.sum(state_max<=.002)),functional_response_within_training_safety=int(np.sum(state_max<=.0018)),maximizing_loss_state_ids=agg['maximizing_state_ids'],maximizing_response_state_ids=[rows[i]['state_id'] for i in np.flatnonzero(state_max==state_max.max())],response_gradient_L2=float(np.linalg.norm(response_total)),guard_gradient_L2=float(np.linalg.norm(guard_total)),total_gradient_L2=float(np.linalg.norm(gradient)),response_guard_gradient_dot=float(response_total@guard_total),total_archived_gradient_dot_actual_update=float(gradient@delta),actual_update_Linf=float(np.max(abs(delta))),actual_update_L2=float(np.linalg.norm(delta)),projected_coordinates=info['projected_coordinates'],gradient_clip_factor=info['gradient_clip_factor'],functional_counts_are_not_native_acceptance=True))
            for sp,r,w,contribution in zip(registry(),rows,weights,agg['state_gradient_contributions']):
                signed=np.asarray(r['observed'][:6],float);normc=float(np.linalg.norm(contribution));normg=float(np.linalg.norm(gradient));state_rows.append(dict(state_id=sp['state_id'],case=sp['case'],training_origin=sp['training_origin'],step=step,signed_response=signed.tolist(),active_response_coordinates=np.flatnonzero(abs(signed)==max(abs(signed))).tolist(),response_Linf=float(max(abs(signed))),response_budget_coordinate_slack=(.002-abs(signed)).tolist(),response_safety_coordinate_slack=(.0018-abs(signed)).tolist(),objective_parts=r['objective_parts'],response_weight=float(w),guard_weight=1/84,response_gradient_L2=float(np.linalg.norm(r['response_gradient'])),guard_gradient_L2=float(np.linalg.norm(r['guard_gradient'])),weighted_contribution_L2=normc,contribution_total_dot=float(np.asarray(contribution)@gradient),contribution_total_cosine=(float(np.asarray(contribution)@gradient)/(normc*normg) if normc>0 and normg>0 else None),response_loss_predicted_change=float(np.asarray(r['response_gradient'])@delta),guard_loss_predicted_change=float(np.asarray(r['guard_gradient'])@delta),predictions_use_unverified_archived_gradient_vectors=True))
            a,m,v=u['after_a'],u['after_m'],u['after_v'];print('C147_UPDATE_AUDIT',step,'/12',flush=True)
        exact(z.arrays('FINAL_SHARED_CONTROL_STEP12.npz')['a'],a,'Actual final tensor must be step12');latest=z.arrays('LATEST_PARTIAL_CONTROL.npz')
        for k,want in [('a',a),('m',m),('v',v)]:exact(latest[k],want,'Last partial duplicate/'+k)
        exact(latest['completed_steps'],np.array([12],np.int64),'Last partial step count')
        seal=z.json('STEP12_FREEZE_BEFORE_EVALUATION.json');compare(seal['completed_steps'],12);compare(seal['control_digest'],row_digest(a));compare(seal['final_endpoint_measurements_so_far'],0);compare(seal['checkpoint_choice'],'Fixed step12 only');compare(seal['no_best_or_recovery_selection'],True)
        expected_seals={'DESIGN_FROZEN_BEFORE_MODEL.json','INPUTS_FROZEN.json','ALL84_TRAINING_REGISTRY_FROZEN.json','FROZEN_INPUT_IDENTITIES.json','MODEL_PINS_REFERENCE.json','C145_BOUND_RECEIPT.json','ALL84_TRAINING_INPUTS.npz','INITIAL_C143_STEP12_CONTROL.npz','FROZEN_SHARED_ROW_BASIS.npz','FROZEN_ORIGINAL_WRITER.npz','FROZEN_ORIGINAL_CONTROL.npz','FINAL_SHARED_CONTROL_STEP12.npz','ALL252_FROZEN_FINAL_POINTS.npz','ALL252_FROZEN_FINAL_OUTCOMES.json'};need(set(seal['artifact_seals'])==expected_seals,'Exact frozen artifact inventory')
        for n,h in seal['artifact_seals'].items():compare(hashlib.sha256(z.raw(n)).hexdigest(),h,'Frozen artifact seal/'+n)
        frozen=z.arrays('ALL252_FROZEN_FINAL_POINTS.npz');need(set(frozen)=={sid+'__'+arm for sid in d['points'] for arm in ARMS},'All252 points');points={};outcomes=[]
        for sp in registry():
            sid=sp['state_id'];p=d['points'][sid];warm,wd=proposal(p['prepared'],p['raw'],d['V'],d['a']);trained,td=proposal(p['prepared'],p['raw'],d['V'],a)
            for arm,point,diag in [(ARMS[0],p['raw'].copy(),dict(original_raw_bytes_retained=True)),(ARMS[1],warm,wd),(ARMS[2],trained,td)]:exact(frozen[sid+'__'+arm],point,'Recomputed final physical point');points[(sid,arm)]=point;outcomes.append(dict(**sp,arm=arm,point_sha256=safe.tensor_hash(point),proposal_diagnostics=diag))
        compare(z.json('ALL252_FROZEN_FINAL_OUTCOMES.json'),outcomes,'All252 proposal identities')
        records=z.json('ALL504_FINAL_TRAINING_ENDPOINTS.json');compare(records,z.json('FINAL_ENDPOINT_PROGRESS.json'),'Final endpoint duplicate');compare([(r['round'],r['state_id'],r['arm']) for r in records],[(rnd,r['state_id'],arm) for rnd in (0,1) for r in registry() for arm in ARMS]);first={};drifts=[];repeats=[]
        for i,r in enumerate(records):
            for k,val in byspec[r['state_id']].items():compare(r[k],val,'Native endpoint state binding/'+k)
            budget();sid=r['state_id'];arm=r['arm'];rnd=r['round'];point=points[(sid,arm)];p=d['points'][sid];g=endpoint_gates(r['metrics'],r['case'],point,p['prepared'],True);compare(r['gates'],g,'Recomputed native gate');aa=np.zeros_like(a) if arm==ARMS[0] else (d['a'] if arm==ARMS[1] else a);compare(r['training_objective_parts'],loss(r['metrics']['observed'],r['metrics']['KL'],point,p['prepared'],aa,r['case']))
            compare(z.json(f'final_measurements/record{i:03d}.json'),r,'Native measurement duplicate')
            if arm in ARMS[:2]:
                q=repeat_check(r['metrics'],native[(sid,'raw' if arm==ARMS[0] else 'C143')]);compare(r['native_control_replay'],q)
                if not(q['same_point_identity'] and q['within_replay_tolerance']):drifts.append(dict(state_id=sid,round=rnd,arm=arm,comparison=q))
            if rnd==0:first[(sid,arm)]=r['metrics']
            else:
                q=repeat_check(r['metrics'],first[(sid,arm)]);compare(r['repeat_comparison'],q)
                if not(q['same_point_identity'] and q['within_replay_tolerance']):repeats.append(dict(state_id=sid,arm=arm,comparison=q))
        sums,pairs=final_tables(records);compare(z.json('FINAL_TRAINING_SUMMARIES.json'),sums);compare(z.json('PAIRED_VS_RAW_AND_C143.json'),pairs);compare(s['final_training_summaries'],sums);compare(s['paired_vs_raw_and_C143'],pairs);compare(z.json('CONTROL_REPLAY_FAILURES.json'),drifts);compare(z.json('REPEATABILITY_FAILURES.json'),repeats)
        consistent=not drifts and not repeats;passed=consistent and all(sums['all84'][str(rnd)][ARMS[2]]['passed']==84 for rnd in (0,1));decision='C146_FIXED_FINAL_COMPLETE_REPLAY_DRIFT_NO_QUALIFICATION' if not consistent else ('C146_FIXED_FINAL_ALL84_TRAINING_GATES_PASSED' if passed else 'C146_FIXED_FINAL_TRAINING_CONSTRAINT_FAILURE');compare(s['decision'],decision);compare(s['fixed_final_training_qualification_passed'],bool(passed));compare(s['repeatable'],not repeats);compare(s['controls_replay_consistent'],not drifts)
        next_branch='RESOLVE_REPLAY_DRIFT' if not consistent else ('PREREGISTER_SEPARATE_UNSEEN_LOCAL_CONFIRMATION' if passed else 'STOP_FIXED12_SCHEDULE_NO_CHECKPOINT_SUBSTITUTION');compare(s['next_branch'],next_branch)
        for k,vv in dict(model_forwards=4033,initialisation_model_forwards=1,preflight_metric_attempts=336,preflight_metrics=336,path_metric_attempts=504,path_metrics=504).items():compare(s['runtime_calls'][k],vv)
        for k,vv in dict(functional_endpoint_calls=1176,functional_zero_calls=84,functional_warm_calls=84,full84_gradient_batches=12,scalar_backward_attempts=2016,scalar_backward_calls=2016,torch_optimizer_objects=0,GPT2_optimizer_updates=0,LoRA_A_optimizer_updates=0,base_writer_updates=0).items():compare(s['gradient_stats'][k],vv)
        e=s['gradient_stats']['maximum_point_parity_error_Linf'];need(type(e) in (float,int) and np.isfinite(e) and 0<=e<=2e-6,'Recorded total point-parity error')
        final_by={r['state_id']:r for r in records if r['round']==0 and r['arm']==ARMS[2]};final_rows=[dict(state_id=sp['state_id'],objective_parts=final_by[sp['state_id']]['training_objective_parts'],response_gradient=[0.]*rank,guard_gradient=[0.]*rank) for sp in registry()];_,final_agg=aggregate(final_rows,rank)
        transitions=[]
        for i,rows in enumerate(batches):
            nxt=batches[i+1] if i<11 else final_rows;_,nextagg=aggregate(nxt,rank);before=steps[i]['aggregate_objective'];pred=steps[i]['total_archived_gradient_dot_actual_update'];nextresponses=np.asarray([r['observed'][:6] for r in nxt]) if i<11 else np.asarray([final_by[r['state_id']]['metrics']['observed'][:6] for r in registry()]);maxstates=[registry()[j]['state_id'] for j in np.flatnonzero(np.max(abs(nextresponses),axis=1)==np.max(abs(nextresponses)))]
            transitions.append(dict(from_step=i+1,to_observation=('training_step'+str(i+2) if i<11 else 'final_native_round0'),next_recorded_objective=nextagg['objective'],observed_objective_change=nextagg['objective']-before,archived_gradient_predicted_objective_change=pred,objective_linearization_residual=nextagg['objective']-before-pred,loss_maximizer_set_changed=set(steps[i]['maximizing_loss_state_ids'])!=set(nextagg['maximizing_state_ids']),next_response_maximizer_state_ids=maxstates,native_functional_sources_differ=bool(i==11),not_an_AD_accuracy_or_step_size_causality_proof=True))
            for j,r in enumerate(rows):
                target=nxt[j]['objective_parts'];current=r['objective_parts'];q=state_rows[i*84+j];q.update(next_observation_source=transitions[-1]['to_observation'],next_response_loss=target['response_loss'],observed_response_loss_change=target['response_loss']-current['response_loss'],response_loss_linearization_residual=target['response_loss']-current['response_loss']-q['response_loss_predicted_change'])
        per_state=[]
        for sp in registry():
            sid=sp['state_id'];rr=[r for r in state_rows if r['state_id']==sid];native_final=final_by[sid];per_state.append(dict(**sp,training_history=rr,final_native_gate=native_final['gates'],final_signed_response=native_final['metrics']['observed'][:6],final_coordinate_budget_slack=(.002-abs(np.asarray(native_final['metrics']['observed'][:6]))).tolist(),max_loss_appearances=sum(sid in x['maximizing_loss_state_ids'] for x in steps),max_response_appearances=sum(sid in x['maximizing_response_state_ids'] for x in steps)))
    dump(out/'ALL1008_STATE_TRAINING_DIAGNOSTICS.json',state_rows);dump(out/'ALL84_STATE_HISTORIES_AND_FINAL_GATES.json',per_state);dump(out/'ALL12_UPDATE_AND_MAXIMIZER_DIAGNOSTICS.json',steps);dump(out/'ALL12_RECORDED_TRANSITIONS.json',transitions);dump(out/'RECOMPUTED_NATIVE_SUMMARIES.json',sums);dump(out/'RECOMPUTED_PAIRED_RESULTS.json',pairs)
    result=dict(protocol=PROTOCOL,complete=True,decision='C147_READONLY_WORST_RESPONSE_AUDIT_COMPLETE_PARENT_DECISION_RETAINED',synthetic_fixture=not c['production'],parent_protocol=parent.PROTOCOL,parent_decision=decision,parent_decision_reclassified=False,original_archives_rehashed=list(INPUT_KEYS),original_C144_C140_C141_C142_ZIP_bytes_rehashed=False,members_hashed=counts,delivered_source_pins_passed=True,actual_frozen_head_tensor_identities_passed=True,training_states_audited=84,training_steps_audited=12,training_observations_checked=1008,sealed_final_proposals_recomputed=252,native_gate_records_recomputed=504,parent_fixed_final_training_qualification_passed=bool(passed),repeatable=not repeats,controls_replay_consistent=not drifts,final_training_summaries=sums,paired_vs_raw_and_C143=pairs,maximum_aggregate_gradient_arithmetic_error_Linf=max_gradient_error,maximum_update_arithmetic_error_Linf=max_update_error,first_recorded_training_objective=steps[0]['aggregate_objective'],final_native_training_objective=final_agg['objective'],observed_objective_increase_transitions=sum(x['observed_objective_change']>0 for x in transitions),max_loss_set_switches=sum(x['loss_maximizer_set_changed'] for x in transitions),total_projected_coordinates=sum(x['projected_coordinates'] for x in steps),new_model_forwards=0,new_backward_calls=0,new_training_steps=0,new_head_fits=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,gradients_independently_recomputed=False,archived_source_executed=False,thresholds_changed=False,candidate_selected=False,writer_deployed=False,new_schedule_or_checkpoint_selected=False,causal_failure_explanation_established=False,LC2_closed=False,LC2_single_fold_passed=False,scope=design()['scope'])
    lines=['# C147：C146只读训练审计','','原判定保留：'+decision,'','|臂|首轮通过|重复轮通过|首轮最大响应漂移|','|---|---:|---:|---:|']
    for arm in ARMS:lines.append(f"|{arm}|{sums['all84']['0'][arm]['passed']}/84|{sums['all84']['1'][arm]['passed']}/84|{sums['all84']['0'][arm]['maximum_response_drift']}|")
    lines+=['','核对12步、1008条训练记录、252封存点和504条原生端点；全部84态、失败、获救及破坏均保留。','','训练阶段的响应计数来自归档功能前向，不是新增原生验收；第12步之后的比较使用原生端点，单独标记。梯度向量只作归档算术输入。最大值切换、线性预测残差和梯度夹角仅为描述，不证明导数错误、步长过大、结构不足、收敛或不可行。','','未加载模型或重测端点；没有新训练、求解、Jacobian、SVD、选头或放宽门槛；不自动选择新训练日程。C144/C140/C141/C142原ZIP未重哈希。封存一致性不独立证明时序、非访问或测量准确性。LC2与单折均未闭合。'];(out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    result['original_inputs_unchanged']={k:sha(p)==c['pins'][k]['sha256'] for k,p in paths.items()};need(all(result['original_inputs_unchanged'].values()),'Original input changed');result['elapsed_audit_seconds']=time.monotonic()-started;budget();return result
