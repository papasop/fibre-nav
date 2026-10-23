"""Inspect C152 archived scalar gradients as data; independently check the arithmetic."""
from core153 import *
from inputs152 import load as load152
from audit147 import source_check
INITIAL_ARTIFACTS={'DESIGN_FROZEN_BEFORE_MODEL.json','INPUTS_FROZEN.json','ALL84_TRAINING_REGISTRY_FROZEN.json','FROZEN_TRAINING_ONLY_ROUTER.npz','ALL84_PREWRITE_ROUTING.npz','ALL84_TRAINING_INPUTS.npz','FROZEN_SHARED_ROW_BASIS.npz','INITIAL_C150_COUPLED_CONTROL.npz','INITIAL_C148_ROUTED_REFERENCE.npz','FROZEN_ORIGINAL_WRITER.npz','FROZEN_ORIGINAL_CONTROL.npz','MODEL_PINS_REFERENCE.json','FROZEN_INPUT_IDENTITIES.json','PARENT_AUDIT_RECEIPT.json'}

def inspect_archive(z,d,out,c,budget=lambda:None):
    count=z.verify_manifest();source_check(z,c['source152'],'C152');s=z.json('SUMMARY.json');compare(s['protocol'],parent.PROTOCOL);compare(s['complete'],True);compare(s['errors'],[]);compare(s['stage'],'complete');compare(z.json('DESIGN_FROZEN_BEFORE_MODEL.json'),parent.design(),'Original fixed C152 design');compare(s['scope'],parent.design()['scope']);specs=registry();spby={sp['state_id']:sp for sp in specs};rank=d['V'].shape[0];compare(z.json('ALL84_TRAINING_REGISTRY_FROZEN.json'),specs)
    if c['production']:
        compare(s['production_factory_contract_passed'],True);compare(s['synthetic_fixture'],False);compare(s['real_model_execution_claim'],True);farray(d['V'],(36,DIM));pins=z.json('model_runtime/MODEL_PINS.json');compare({k:v for k,v in pins.items() if k!='torch'},{k:v for k,v in d['modelpins'].items() if k!='torch'})
    for k,v in dict(training_states=84,requested_training_steps_per_arm=12,requested_model_forwards=6553,requested_scalar_backwards=4032,requested_native_preflight=336,requested_native_endpoints=672,new_states_prepared=0,confirmation_states_evaluated=0,excluded_input_states_evaluated=0,new_model_features_acquired=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,GPT2_optimizer_updates=0,LoRA_A_optimizer_updates=0,base_writer_updates=0,teacher_calls=0,backtracking_steps=0,online_response_repair_steps=0,new_model_forwards=6553,new_backward_calls=4032,new_training_steps=24,new_head_fits=2,preflight_records=336,zero_parity_records=84,warm_parity_records=168,final_endpoint_records=672,fixed_coupling_offset=1).items():compare(s[k],v,'Original count/setting/'+k)
    for k in ('archived_source_executed','thresholds_changed','parent_decision_reclassified','candidate_selected','writer_deployed','best_checkpoint_selected','AD_accuracy_independently_certified','router_label_access','router_parameters_trained','geometric_atlas_established','coupling_pattern_selected','LC2_closed','LC2_single_fold_passed'):compare(s[k],False)
    for k in ('all84_remain_exposed_training','learning_rate_only_arm_difference','Adam_moments_reset_for_both','parameter_counts_matched','both_warm_parities_complete_before_backward','both_step12_heads_frozen_before_evaluation'):compare(s[k],True)
    compare(s['trainable_parameters'],{a:6*rank for a in TRAIN_ARMS});compare(s['learning_rates'],RATES);compare(s['delivery_revision'],'R1');linked=z.json('INPUTS_FROZEN.json');need(set(linked)==set(parent.INPUT_KEYS),'Seven original links')
    for k in parent.INPUT_KEYS:compare(linked[k]['sha256'],c['pins'][k]['sha256'],'Original input pin/'+k)
    compare(s['original_inputs_unchanged'],{k:True for k in parent.INPUT_KEYS});compare(z.json('ORIGINAL_INPUTS_UNCHANGED.json'),{k:True for k in parent.INPUT_KEYS});arrays=z.arrays('ALL84_TRAINING_INPUTS.npz');need(set(arrays)=={sid+'__'+k for sid in d['points'] for k in ('prepared','raw','features')},'Full84 coordinate/feature inventory')
    for sid,p in d['points'].items():
        for k in ('prepared','raw','features'):exact(arrays[sid+'__'+k],p[k],'Actual coordinate/feature/'+sid+'/'+k)
    for filename,want in [('FROZEN_SHARED_ROW_BASIS.npz',dict(V=d['V'])),('INITIAL_C148_ROUTED_REFERENCE.npz',dict(A=d['initial_A'])),('INITIAL_C150_COUPLED_CONTROL.npz',dict(parameters=d['initial_P'])),('FROZEN_TRAINING_ONLY_ROUTER.npz',d['router']),('FROZEN_ORIGINAL_WRITER.npz',d['writer']),('FROZEN_ORIGINAL_CONTROL.npz',d['control']),('ALL84_PREWRITE_ROUTING.npz',dict(weights=np.array([d['weights'][sp['state_id']] for sp in specs])))]:
        got=z.arrays(filename);need(set(got)==set(want),'Actual frozen tensor inventory/'+filename)
        for k,v in want.items():exact(got[k],v,'Actual frozen tensor/'+filename+'/'+k)
    compare(z.json('FROZEN_INPUT_IDENTITIES.json'),parent.fixed_digest(d));compare(z.json('MODEL_PINS_REFERENCE.json'),d['modelpins']);compare(z.json('PARENT_AUDIT_RECEIPT.json'),d['C153_binding_receipt'],'Original C152 input audit receipt')
    pre=z.json('PREFLIGHT_PROGRESS.json');compare([(r['state_id'],r['kind']) for r in pre],[(sp['state_id'],kind) for sp in specs for kind in ('prepared','raw','C148','C150')]);native={}
    for r in pre:
        sid=r['state_id'];kind=r['kind'];p=d['points'][sid]
        for k,v in spby[sid].items():compare(r[k],v,'Preflight row binding/'+k)
        point=parent.preflight_point(d,sid,kind);g,_,_=native_gates(r['metrics'],r['case'],point,kind!='prepared')
        if kind=='prepared':need(all(g.values()),'Invalid preparation retained')
        else:endpoint_gates(r['metrics'],r['case'],point,p['prepared'],True)
        q=[repeat_check(r['metrics'],ref) for ref in d['refs'][(sid,kind)]];compare(r['comparisons'],q);need(len(q)>0 and all(v['same_point_identity'] and v['within_replay_tolerance'] for v in q),'Original preflight replay drift');native[(sid,kind)]=r['metrics'];budget()
    params={a:d['initial_P'].copy() for a in TRAIN_ARMS};mom={a:(np.zeros_like(p),np.zeros_like(p)) for a,p in params.items()};zero=np.zeros((3,2,rank));zero_e=np.zeros((2,rank))
    for name,expected,kind in [('ZERO_FUNCTIONAL_PROGRESS.json',[(sp,TRAIN_ARMS[0],zero) for sp in specs],'raw'),('WARM_FUNCTIONAL_PROGRESS.json',[(sp,a,params[a]) for a in TRAIN_ARMS for sp in specs],'C150')]:
        rows=z.json(name);compare([(r['state_id'],r['arm']) for r in rows],[(sp['state_id'],a) for sp,a,_ in expected],'All functional parities')
        for r,(sp,a,p) in zip(rows,expected):
            sid=sp['state_id'];q=d['points'][sid];point=proposal(q['prepared'],q['raw'],d['V'],effective(p,d['weights'][sid],a));compare(r['point_sha256'],safe.tensor_hash(point));need(0<=r['point_parity_error_Linf']<=2e-6,'Recorded point parity');err=close(r['observed'],native[(sid,kind)]['observed'],'Recorded functional/native observation',2e-6);ke=abs(r['KL']-native[(sid,kind)]['KL']);need(ke<=2e-7,'Recorded KL parity');compare(r['observed_parity_Linf'],err);compare(r['KL_parity_absolute_error'],ke)
    progress=z.json('TRAINING_PROGRESS.json');compare([(r['step'],r['arm']) for r in progress],[(step,a) for step in range(1,13) for a in TRAIN_ARMS],'Fixed interleaved24 update order');step_reports=[];state_reports=[];histories={a:[] for a in TRAIN_ARMS};grad_error=0.;update_error=0.;mix_orders=Counter()
    for step in range(1,13):
        paired=dict(step=step,arms={})
        for ai,arm in enumerate(TRAIN_ARMS):
            budget();p=params[arm];m,v=mom[arm];folder=f'training/{arm}/step{step:02d}/';rows=z.json(folder+'OBSERVATIONS.json');compare(rows,z.json(folder+'GRADIENT_PROGRESS.json'),'Training duplicate');compare([r['state_id'] for r in rows],[sp['state_id'] for sp in specs]);u=z.arrays(folder+'UPDATE.npz');need(set(u)=={'before','before_m','before_v','gradient','after','after_m','after_v'},'Update inventory')
            for k,want in [('before',p),('before_m',m),('before_v',v)]:exact(u[k],want,'Sequential actual Adam tensor/'+k)
            for sp,r in zip(specs,rows):
                sid=sp['state_id'];w=d['weights'][sid];e,order=archived_effective(p,w,arm,r['effective_control_digest']);mix_orders[order]+=1;q=d['points'][sid];point=proposal(q['prepared'],q['raw'],d['V'],e)
                for k,val in sp.items():compare(r[k],val,'Training identity/'+k)
                compare(r['arm'],arm);compare(r['parameters_digest'],row_digest(p));compare(r['effective_control_digest'],row_digest(e));compare(r['routing_weights'],w.tolist());compare(r['functional_point_sha256'],safe.tensor_hash(point));need(0<=r['point_parity_error_Linf']<=2e-6,'Recorded training point parity');compare(r['objective_parts'],loss(r['observed'],r['KL'],point,q['prepared'],e,sp['case']),'Independent scalar losses',atol=1e-10,rtol=1e-9)
                for key in ('response_gradient','guard_gradient'):
                    eg=farray(np.asarray(r['effective_'+key],float),e.shape);actual=farray(np.asarray(r[key],float),p.shape);grad_error=max(grad_error,close(actual,pullback(eg,w,arm),'Independent routing pullback',1e-12))
            g,agg=aggregate(rows,p.shape);compare(z.json(folder+'AGGREGATION.json'),agg,'Independent exact-max objective/aggregation');grad_error=max(grad_error,close(u['gradient'],g,'Independent aggregated gradient',1e-12));new,mm,vv,info=update(p,g,m,v,step,arm)
            for key,want in [('after',new),('after_m',mm),('after_v',vv)]:update_error=max(update_error,close(u[key],want,'Independent Adam arithmetic/'+key,1e-12))
            expected=dict(arm=arm,**info,objective_before_update=agg['objective'],maximum_response_loss=agg['maximum_response_loss'],maximizing_state_ids=agg['maximizing_state_ids'],before_digest=row_digest(p),after_digest=row_digest(u['after']),update_sha256=hashlib.sha256(z.raw(folder+'UPDATE.npz')).hexdigest(),n_training_records=84);compare(progress[(step-1)*2+ai],expected,'Original update progress',atol=1e-12,rtol=1e-9)
            response=np.sum(np.asarray(agg['response_weights'])[:,None]*np.array([r['response_gradient'] for r in rows]).reshape(84,-1),axis=0).reshape(p.shape);guard=np.mean(np.array([r['guard_gradient'] for r in rows]),axis=0);delta=u['after']-p;part=dict(**info,objective_before_update=agg['objective'],maximum_response_loss=agg['maximum_response_loss'],maximizing_state_ids=agg['maximizing_state_ids'],response_aggregate_L2=norm(response),guard_aggregate_L2=norm(guard),response_guard_cosine=cosine(response,guard),recorded_response_aggregate_dot_actual_update=float(np.sum(response*delta)),recorded_guard_aggregate_dot_actual_update=float(np.sum(guard*delta)),actual_parameter_update_L2=norm(delta),derivatives_unverified=True)
            if arm in TRAIN_ARMS:part.update(diagonal_gradient_L2=norm(g[:,0,:]),cross_gradient_L2=norm(g[:,1,:]),diagonal_update_L2=norm(delta[:,0,:]),cross_update_L2=norm(delta[:,1,:]),cross_coefficients_Linf=linf(u['after'][:,1,:]))
            paired['arms'][arm]=part
            for sp,r in zip(specs,rows):
                sid=sp['state_id'];de=effective(u['after'],d['weights'][sid],arm)-effective(p,d['weights'][sid],arm);rg=np.asarray(r['effective_response_gradient']);gg=np.asarray(r['effective_guard_gradient']);o=np.asarray(r['observed'][:6]);state_reports.append(dict(**sp,arm=arm,step=step,signed_response=o.tolist(),response_slack=(.002-abs(o)).tolist(),response_loss=r['objective_parts']['response_loss'],response_weight=agg['response_weights'][len(state_reports)%84],active_response_coordinates=np.flatnonzero(abs(o)==max(abs(o))).tolist(),recorded_response_gradient_dot_effective_update=float(np.sum(rg*de)),recorded_guard_gradient_dot_effective_update=float(np.sum(gg*de)),effective_update_Linf=linf(de),diagonal_recorded_response_dot=float(np.sum(rg[0]*de[0])),cross_recorded_response_dot=float(np.sum(rg[1]*de[1])),derivatives_unverified=True,functional_observation_not_native_acceptance=True))
            histories[arm].append(dict(rows=rows,after=u['after'].copy(),before=p.copy()));params[arm]=u['after'];mom[arm]=(u['after_m'],u['after_v'])
        step_reports.append(paired);print('C153_UPDATE_PAIR_AUDIT',step,'/12',flush=True)
    for arm in TRAIN_ARMS:
        got=z.arrays('FINAL_'+arm+'_STEP12.npz');need(set(got)=={'parameters'},'Final tensor inventory');exact(got['parameters'],params[arm],'Actual final head is fixed step12');compare(s['per_arm_training'][arm]['steps'],12);elapsed=s['per_arm_training'][arm]['elapsed_training_seconds'];need(type(elapsed) in (int,float) and np.isfinite(elapsed) and elapsed>=0,'Recorded training time')
    seal=z.json('BOTH_STEP12_FREEZE_BEFORE_EVALUATION.json');compare(seal['completed_steps'],{a:12 for a in TRAIN_ARMS});compare(seal['parameter_digests'],{a:row_digest(p) for a,p in params.items()});compare(seal['native_final_measurements_so_far'],0);compare(seal['no_arm_or_checkpoint_selection'],True);required=INITIAL_ARTIFACTS|{'FINAL_'+a+'_STEP12.npz' for a in TRAIN_ARMS}|{'ALL336_FROZEN_FINAL_POINTS.npz','ALL336_FROZEN_FINAL_OUTCOMES.json'};need(set(seal['artifact_seals'])==required,'Exact frozen artifact inventory')
    for name,h in seal['artifact_seals'].items():compare(hashlib.sha256(z.raw(name)).hexdigest(),h,'Original artifact seal/'+name)
    frozen=z.arrays('ALL336_FROZEN_FINAL_POINTS.npz');need(set(frozen)=={sid+'__'+a for sid in d['points'] for a in ARMS},'Complete336 point inventory');points={};eff={};outcomes=[]
    for sp in specs:
        sid=sp['state_id'];q=d['points'][sid]
        for a in ARMS:
            e=zero_e if a==ARMS[0] else effective(d['initial_P'],d['weights'][sid],TRAIN_ARMS[0]) if a==ARMS[1] else effective(params[a],d['weights'][sid],a);point=q['raw'].copy() if a==ARMS[0] else proposal(q['prepared'],q['raw'],d['V'],e);exact(frozen[sid+'__'+a],point,'Independent final physical point/'+sid+'/'+a);points[(sid,a)]=point;eff[(sid,a)]=e;outcomes.append(dict(**sp,arm=a,routing_weights=d['weights'][sid].tolist(),effective_control=e.tolist(),point_sha256=safe.tensor_hash(point)))
    compare(z.json('ALL336_FROZEN_FINAL_OUTCOMES.json'),outcomes,'Frozen coefficients and point bindings');records=z.json('ALL672_FINAL_TRAINING_ENDPOINTS.json');compare(records,z.json('FINAL_ENDPOINT_PROGRESS.json'));compare([(r['round'],r['state_id'],r['arm']) for r in records],[(rnd,sp['state_id'],a) for rnd in (0,1) for sp in specs for a in ARMS]);first={};drifts=[];repeats=[]
    for i,r in enumerate(records):
        budget();sid=r['state_id'];a=r['arm'];rnd=r['round'];q=d['points'][sid];point=points[(sid,a)]
        for k,v in spby[sid].items():compare(r[k],v,'Final state binding')
        compare(r['gates'],endpoint_gates(r['metrics'],r['case'],point,q['prepared'],True),'Independent final gate');compare(r['training_objective_parts'],loss(r['metrics']['observed'],r['metrics']['KL'],point,q['prepared'],eff[(sid,a)],r['case']));compare(z.json(f'final_measurements/record{i:03d}.json'),r,'Individual measurement duplicate')
        if a in ARMS[:2]:
            rc=repeat_check(r['metrics'],native[(sid,'raw' if a==ARMS[0] else 'C150')]);compare(r['control_replay'],rc)
            if not(rc['same_point_identity'] and rc['within_replay_tolerance']):drifts.append(dict(state_id=sid,arm=a,round=rnd,comparison=rc))
        if rnd==0:first[(sid,a)]=r['metrics']
        else:
            rc=repeat_check(r['metrics'],first[(sid,a)]);compare(r['repeat_comparison'],rc)
            if not(rc['same_point_identity'] and rc['within_replay_tolerance']):repeats.append(dict(state_id=sid,arm=a,comparison=rc))
    sums,pairs=final_tables(records);compare(z.json('FINAL_TRAINING_SUMMARIES.json'),sums);compare(s['final_training_summaries'],sums);compare(z.json('PAIRED_ALL_ARMS.json'),pairs);compare(s['paired_all_arms'],pairs);compare(z.json('CONTROL_REPLAY_FAILURES.json'),drifts);compare(z.json('REPEATABILITY_FAILURES.json'),repeats);consistent=not drifts and not repeats;qualified={a:consistent and all(sums['all84'][str(r)][a]['passed']==84 for r in (0,1)) for a in TRAIN_ARMS};decision='C152_FIXED_FINAL_RATE_COMPARISON_COMPLETE_NO_CANDIDATE_SELECTED' if consistent else 'C152_COMPARISON_COMPLETE_REPLAY_DRIFT_NO_QUALIFICATION';compare(s['decision'],decision);compare(s['training_qualified'],qualified);compare(s['repeatable'],not repeats);compare(s['controls_replay_consistent'],not drifts);compare(s['next_branch'],'PREREGISTER_NEW_CONFIRMATION_FOR_QUALIFIED_ARMS' if any(qualified.values()) else 'RETAIN_TRAINING_FAILURE_NO_UNSEEN_EVALUATION')
    compare(s['runtime_calls'],dict(model_forwards=6553,initialisation_model_forwards=1,preflight_metric_attempts=336,preflight_metrics=336,path_metric_attempts=672,path_metrics=672))
    for k,v in dict(functional_endpoint_calls=2268,functional_zero_calls=84,functional_warm_calls=168,full84_gradient_batches=24,scalar_backward_attempts=4032,scalar_backward_calls=4032,torch_optimizer_objects=0,GPT2_optimizer_updates=0).items():compare(s['gradient_stats'][k],v,'Archived gradient counts/'+k)
    if c['production']:compare(s['gradient_stats']['per_arm'],{a:dict(graph_forwards=1008,backwards=2016) for a in TRAIN_ARMS})
    need(0<=s['gradient_stats']['maximum_point_parity_error_Linf']<=2e-6,'Recorded aggregate parity');final_by={(r['state_id'],r['arm']):r for r in records if r['round']==0};final_reports=[];transitions=[]
    for sp in specs:
        sid=sp['state_id'];q=d['points'][sid];u=d['V']@(q['raw'].astype(float)-q['prepared'].astype(float));e=eff[(sid,TRAIN_ARMS[1])];sd=d['V'].T@(e[0]*u);sx=d['V'].T@(e[1]*np.concatenate([u[-1:],u[:-1]]));total=d['V'].T@(e[0]*u+e[1]*np.concatenate([u[-1:],u[:-1]]));perarm={}
        for a in ARMS:
            rr=final_by[(sid,a)];o=np.asarray(rr['metrics']['observed'][:6]);perarm[a]=dict(complete_pass=rr['gates']['complete_pass'],failed_gates=rr['gates']['failed_gates'],signed_response=o.tolist(),response_coordinate_slack=(.002-abs(o)).tolist(),response_Linf=rr['metrics']['response_drift'],KL=rr['metrics']['KL'],effective_control=eff[(sid,a)].tolist())
        final_reports.append(dict(**sp,routing_weights=d['weights'][sid].tolist(),arms=perarm,coupled_diagonal_parameter_shift_Linf=linf(sd),coupled_cross_parameter_shift_Linf=linf(sx),coupled_total_parameter_shift_Linf=linf(total),unrounded_decomposition_residual_Linf=linf(sd+sx-total),coupled_float32_commit_residual_Linf=linf(points[(sid,TRAIN_ARMS[1])].astype(float)-q['raw'].astype(float)-total),small_vs_standard_rate_proposal_Linf=linf(points[(sid,TRAIN_ARMS[1])].astype(float)-points[(sid,TRAIN_ARMS[0])].astype(float)),counterfactual_endpoints_measured=False))
    for a,h in histories.items():
        for si,item in enumerate(h):
            for sp,r in zip(specs,item['rows']):
                sid=sp['state_id'];next_obs=h[si+1]['rows'][[sp2['state_id'] for sp2 in specs].index(sid)] if si<11 else final_by[(sid,a)];nloss=next_obs['objective_parts']['response_loss'] if si<11 else next_obs['training_objective_parts']['response_loss'];de=effective(item['after'],d['weights'][sid],a)-effective(item['before'],d['weights'][sid],a);pred=float(np.sum(np.asarray(r['effective_response_gradient'])*de));observed=nloss-r['objective_parts']['response_loss'];transitions.append(dict(**sp,arm=a,step=si+1,endpoint_kind='next_functional_training_record' if si<11 else 'final_native_endpoint',recorded_gradient_linear_response_loss_change=pred,observed_response_loss_change=observed,linearization_residual=observed-pred,derivative_accuracy_or_curvature_bound=False))
    summary=change_summary(sums,pairs,final_reports);rate_diagnostics=write_rate_diagnostics(out,d,histories,records,step_reports,sums,pairs)
    for name,obj in [('ALL24_UPDATE_DIAGNOSTICS.json',step_reports),('ALL2016_STATE_UPDATE_DIAGNOSTICS.json',state_reports),('ALL2016_RECORDED_TRANSITIONS.json',transitions),('ALL84_FINAL_RESPONSE_AND_BLOCK_DIAGNOSTICS.json',final_reports),('RECOMPUTED_NATIVE_SUMMARIES.json',sums),('RECOMPUTED_PAIRED_RESULTS.json',pairs),('RESCUE_AND_WORST_RESPONSE_COMPARISON.json',summary)]:dump(out/name,obj)
    return dict(protocol=PROTOCOL,complete=True,decision='C153_READONLY_RATE_AUDIT_COMPLETE_PARENT_DECISION_RETAINED',synthetic_fixture=not c['production'],parent_protocol=parent.PROTOCOL,parent_decision=decision,parent_decision_reclassified=False,C152_members_hashed=count,delivered_source_pins_passed=True,actual_frozen_tensor_identities_passed=True,training_states_audited=84,training_updates_audited=24,training_records_audited=2016,frozen_proposals_recomputed=336,native_endpoint_records_recomputed=672,training_qualified=qualified,repeatable=not repeats,controls_replay_consistent=not drifts,final_training_summaries=sums,paired_all_arms=pairs,change_summary=summary,rate_diagnostics=rate_diagnostics,archived_effective_digest_matches_by_registered_order=dict(mix_orders),maximum_gradient_arithmetic_error_Linf=grad_error,maximum_update_arithmetic_error_Linf=update_error,total_projected_coordinates=sum(x['projected_coordinates'] for st in step_reports for x in st['arms'].values()),maximum_unrounded_block_decomposition_residual_Linf=max(r['unrounded_decomposition_residual_Linf'] for r in final_reports),new_model_forwards=0,new_backward_calls=0,new_training_steps=0,new_head_fits=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,archived_source_executed=False,gradients_independently_recomputed=False,prior_C149_audit_reexecuted=False,thresholds_changed=False,candidate_selected=False,writer_deployed=False,counterfactual_endpoints_measured=False,causal_failure_explanation_established=False,learning_rate_superiority_established=False,confirmation_states_evaluated=0,excluded_input_states_evaluated=0,new_states_prepared=0,all84_remain_exposed_training=True,C152_zip_sha256_known_before_runtime=True,LC2_closed=False,LC2_single_fold_passed=False,scope=design()['scope'])

def change_summary(sums,pairs,states):
    result={}
    for rnd in ('0','1'):
        ss=sums['all84'][rnd];r={}
        for a in TRAIN_ARMS:
            per=dict(passed=ss[a]['passed'],failed_ids=ss[a]['failed_ids'],maximum_response_drift=ss[a]['maximum_response_drift'],within_training_response_safety=ss[a]['within_training_response_safety'],worst_response_state_ids=[x['state_id'] for x in states if x['arms'][a]['response_Linf']==ss[a]['maximum_response_drift']])
            # State diagnostic values are round0; repeated-round differences are explicitly kept in the full tables.
            if rnd=='1':per.pop('worst_response_state_ids')
            per['versus_frozen_C150']=dict(**pairs['all84'][rnd][a][ARMS[1]],passed_change=ss[a]['passed']-ss[ARMS[1]]['passed'],maximum_response_drift_change=ss[a]['maximum_response_drift']-ss[ARMS[1]]['maximum_response_drift'],training_safety_count_change=ss[a]['within_training_response_safety']-ss[ARMS[1]]['within_training_response_safety']);r[a]=per
        r['small_rate_versus_standard_rate']=dict(**pairs['all84'][rnd][TRAIN_ARMS[1]][TRAIN_ARMS[0]],passed_change=ss[TRAIN_ARMS[1]]['passed']-ss[TRAIN_ARMS[0]]['passed'],maximum_response_drift_change=ss[TRAIN_ARMS[1]]['maximum_response_drift']-ss[TRAIN_ARMS[0]]['maximum_response_drift'],training_safety_count_change=ss[TRAIN_ARMS[1]]['within_training_response_safety']-ss[TRAIN_ARMS[0]]['within_training_response_safety']);result[rnd]=r
    return result

def audit(paths,out,c=None,input_loader=None):
    c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);start=time.monotonic();need(set(paths)==set(INPUT_KEYS),'Exactly eight pinned originals')
    def budget():need(time.monotonic()-start<TIMEOUT,'C153 fixed20-minute audit budget')
    for k,p in paths.items():compare(sha(p),c['pins'][k]['sha256'],'Fixed original/'+k)
    dump(out/'INPUTS_FROZEN.json',{k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()});d=(load152 if input_loader is None else input_loader)({k:paths[k] for k in parent.INPUT_KEYS},out/'BOUND_C152_INPUTS',c['binding152']);d['C153_binding_receipt']=clean_json((out/'BOUND_C152_INPUTS/PARENT_AUDIT_RECEIPT.json').read_bytes());budget()
    with contextlib.closing(Archive(paths['C152'])) as z:s=inspect_archive(z,d,out,c,budget)
    s.update(original_archives_rehashed=list(INPUT_KEYS),original_inputs_unchanged={k:sha(p)==c['pins'][k]['sha256'] for k,p in paths.items()},elapsed_audit_seconds=time.monotonic()-start);need(all(s['original_inputs_unchanged'].values()),'Original changed');budget();write_report(out,s);return s

def write_report(out,s):
    lines=['# C153：C152固定学习率对照只读审计','','保留原C152判定：'+s['parent_decision'],'','|臂|首轮通过|重复轮通过|首轮最坏响应漂移|','|---|---:|---:|---:|']
    for a in ARMS:lines.append(f"|{a}|{s['final_training_summaries']['all84']['0'][a]['passed']}/84|{s['final_training_summaries']['all84']['1'][a]['passed']}/84|{s['final_training_summaries']['all84']['0'][a]['maximum_response_drift']}|")
    lines+=['','本轮新增/保留失败及获救：'+json.dumps(s['rate_diagnostics']['baseline_comparisons'],ensure_ascii=False),'','全部84态保留；逐坐标裕度、12个写前训练观测和两轮原生端点分别标记。归档梯度与更新的点积仅为代数诊断，不能证明真实导数、因果冲突或曲率误差界。最后一条转移跨功能路径与原生路径，不作为独立优化效应。','','重哈希八个原包；可信C152读取器重审C150并重绑输入。C149保留旧审计凭据，不重跑更早递归链。无模型/前向/反传/训练/拟合/求解/Jacobian/SVD、新测量、候选选择或部署。LC2与单折仍未闭合。'];(out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')

def write_rate_diagnostics(out,d,histories,records,steps,sums,pairs):
    final={(r['round'],r['state_id'],r['arm']):r for r in records};specs=registry();trajectories=[];blocks=[];baseline={};focus=set();all_crossings=[]
    for a in TRAIN_ARMS:
        rr=sums['all84']['0'];oldfail=set(rr[ARMS[1]]['failed_ids']);newfail=set(rr[a]['failed_ids']);focus.update(oldfail|newfail)
        baseline[a]=dict(reference=ARMS[1],passed_change=rr[a]['passed']-rr[ARMS[1]]['passed'],maximum_response_drift_change=rr[a]['maximum_response_drift']-rr[ARMS[1]]['maximum_response_drift'],retained_failure_ids=[sp['state_id'] for sp in specs if sp['state_id'] in oldfail&newfail],new_failure_ids=pairs['all84']['0'][a][ARMS[1]]['destroyed_ids'],rescued_ids=pairs['all84']['0'][a][ARMS[1]]['rescued_ids'])
    for ri,sp in enumerate(specs):
        sid=sp['state_id'];q=d['points'][sid];u=d['V']@(q['raw'].astype(float)-q['prepared'].astype(float));arms={}
        for a in TRAIN_ARMS:
            observations=[]
            for j,h in enumerate(histories[a]):
                r=h['rows'][ri];o=np.asarray(r['observed'][:6]);observations.append(dict(before_step=j+1,measurement_kind='archived_functional_training',signed_response=o.tolist(),coordinate_slack=(.002-abs(o)).tolist(),response_Linf=float(np.max(abs(o))),within_response_budget=bool(np.max(abs(o))<=.002),within_training_safety=bool(np.max(abs(o))<=.0018),native_endpoint_acceptance_claim=False))
            native=[]
            for rnd in (0,1):
                r=final[(rnd,sid,a)];o=np.asarray(r['metrics']['observed'][:6]);native.append(dict(round=rnd,measurement_kind='archived_native_fixed_step12',signed_response=o.tolist(),coordinate_slack=(.002-abs(o)).tolist(),response_Linf=float(np.max(abs(o))),native_gates=r['gates']))
            crossings=[]
            for i in range(11):
                x,y=observations[i:i+2]
                if x['within_response_budget']!=y['within_response_budget']:crossings.append(dict(from_before_step=i+1,to_before_step=i+2,from_within=x['within_response_budget'],to_within=y['within_response_budget'],same_functional_measurement_path=True))
            if observations[-1]['within_response_budget']!=(native[0]['response_Linf']<=.002):crossings.append(dict(from_before_step=12,to_fixed_step12_native=True,from_within=observations[-1]['within_response_budget'],to_within=bool(native[0]['response_Linf']<=.002),same_functional_measurement_path=False))
            arms[a]=dict(training_observations=observations,native_final_rounds=native,recorded_response_crossings=crossings)
            all_crossings.extend(dict(state_id=sid,arm=a,**r) for r in crossings)
            e=effective(histories[a][-1]['after'],d['weights'][sid],a);sd=d['V'].T@(e[0]*u);sx=d['V'].T@(e[1]*np.concatenate([u[-1:],u[:-1]]));total=d['V'].T@(e[0]*u+e[1]*np.concatenate([u[-1:],u[:-1]]));point=proposal(q['prepared'],q['raw'],d['V'],e)
            blocks.append(dict(**sp,arm=a,diagonal_parameter_shift_Linf=linf(sd),cross_parameter_shift_Linf=linf(sx),total_parameter_shift_Linf=linf(total),unrounded_decomposition_residual_Linf=linf(sd+sx-total),float32_commit_residual_Linf=linf(point.astype(float)-q['raw'].astype(float)-total),no_ablation_endpoint_measured=True))
        trajectories.append(dict(**sp,routing_weights=d['weights'][sid].tolist(),baseline_native={a:dict(metrics=final[(0,sid,a)]['metrics'],gates=final[(0,sid,a)]['gates']) for a in ARMS[:2]},arms=arms))
    training_summary={}
    for a,h in histories.items():
        objs=[x['arms'][a]['objective_before_update'] for x in steps];mx=[x['arms'][a]['maximum_response_loss'] for x in steps]
        training_summary[a]=dict(observed_objectives_before_steps=objs,observed_max_response_losses_before_steps=mx,observed_objective_increases=sum(y>x for x,y in zip(objs,objs[1:])),response_guard_positive_inner_product_steps=[x['arms'][a]['step'] for x in steps if x['arms'][a]['response_guard_cosine'] is not None and x['arms'][a]['response_guard_cosine']>0],response_guard_negative_inner_product_steps=[x['arms'][a]['step'] for x in steps if x['arms'][a]['response_guard_cosine'] is not None and x['arms'][a]['response_guard_cosine']<0],recorded_response_uphill_update_steps=[x['arms'][a]['step'] for x in steps if x['arms'][a]['recorded_response_aggregate_dot_actual_update']>0],projected_coordinates=sum(x['arms'][a]['projected_coordinates'] for x in steps),no_convergence_or_causality_claim=True)
    initial_difference=linf(histories[TRAIN_ARMS[0]][0]['before']-histories[TRAIN_ARMS[1]][0]['before'])
    focusrows=[r for r in trajectories if r['state_id'] in focus]
    summary=dict(baseline_comparisons=baseline,focus_failure_union_ids=[sp['state_id'] for sp in specs if sp['state_id'] in focus],all84_trajectories_retained=len(trajectories),all2016_training_observations_retained=sum(len(v['training_observations']) for r in trajectories for v in r['arms'].values()),first_initial_parameter_difference_Linf=initial_difference,per_arm_training_diagnostics=training_summary,recorded_response_crossings=len(all_crossings),maximum_both_arm_unrounded_decomposition_residual_Linf=max(x['unrounded_decomposition_residual_Linf'] for x in blocks),functional_records_are_not_native_acceptance=True,last_transition_crosses_functional_and_native_paths=True,true_gradient_or_failure_cause_established=False,candidate_selected=False)
    for n,o in [('ALL84_TWO_RATE_RESPONSE_TRAJECTORIES.json',trajectories),('FAILURE_UNION_TRAJECTORIES.json',focusrows),('ALL168_BOTH_ARM_BLOCK_DIAGNOSTICS.json',blocks),('ALL_RECORDED_RESPONSE_CROSSINGS.json',all_crossings),('RATE_AND_FAILURE_DIAGNOSTIC_SUMMARY.json',summary)]:dump(out/n,o)
    return summary
