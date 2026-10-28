"""Six fixed originals; C145 vectors are diagnostic, never selected as initialization."""
from core146 import *
from inputs145 import load as load145

def audit145(z,d,c):
    members=z.verify_manifest();s=z.json('SUMMARY.json');compare(s['protocol'],prior.PROTOCOL);compare(s['complete'],True);compare(s['decision'],'C145_FIXED_DIRECTIONAL_DIAGNOSTIC_COMPLETE_NO_CANDIDATE_SELECTED')
    if c['production']:
        compare(s['synthetic_fixture'],False);compare(s['real_model_execution_claim'],True)
    for k in ('candidate_selected','writer_deployed','thresholds_changed','archived_source_executed','LC2_closed','LC2_single_fold_passed','parent_decision_reclassified'):compare(s[k],False)
    for k,v in dict(training_states=84,new_training_steps=0,new_head_fits=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,excluded_input_states_evaluated=0,confirmation_states_evaluated=0,new_backward_calls=168,new_model_forwards=4033,preflight_records=336,functional_parity_records=84,gradient_state_records=84,endpoint_records=1512,diagnostic_control_vectors_generated=8).items():compare(s[k],v,'C145 fixed counts/'+k)
    compare(s['repeatable'],True);compare(s['baseline_replay_consistent'],True);compare(s['all_probe_points_frozen_before_measurements'],True);compare(s['all84_parity_complete_before_backwards'],True)
    names={n[7:] for n in z.z.namelist() if n.startswith('source/')};need(names==set(c['source145']),'Trusted C145 source inventory')
    for n,h in c['source145'].items():compare(hashlib.sha256(z.raw('source/'+n)).hexdigest(),h,'Trusted C145 source/'+n)
    linked=z.json('INPUTS_FROZEN.json')
    for k in prior.INPUT_KEYS:compare(linked[k]['sha256'],c['pins'][k]['sha256'],'C145 original link/'+k)
    baseline=z.arrays('BASELINE_C143_CONTROL.npz')
    for name,x in [('a',d['a']),('V',d['V']),('C140_a',d['C140_a'])]:exact(baseline[name],x,'Original baseline/'+name)
    compare(z.json('FROZEN_INPUT_IDENTITIES.json'),prior.frozen_digest(d),'Original frozen identity')
    arrays=z.arrays('ALL84_FROZEN_TRAINING_INPUTS.npz');need(set(arrays)=={sid+'__'+k for sid in d['points'] for k in ('prepared','raw','features')},'Full84 input inventory')
    for sid,p in d['points'].items():
        for k in ('prepared','raw','features'):exact(arrays[sid+'__'+k],p[k],'C145 duplicate training input/'+sid+'/'+k)
    grads=z.json('GRADIENT_PROGRESS.json');need([r['state_id'] for r in grads]==[r['state_id'] for r in prior.registry()],'All84 C145 gradient rows ordered');parities=z.json('FUNCTIONAL_PARITY_PROGRESS.json');need([r['state_id'] for r in parities]==[r['state_id'] for r in prior.registry()],'All84 C145 parity rows')
    for sp,r,pr in zip(prior.registry(),grads,parities):
        sid=sp['state_id'];p=d['points'][sid];point=proposal(p['prepared'],p['raw'],d['V'],d['a'])[0]
        for k,v in sp.items():compare(r[k],v,'C145 gradient state binding')
        compare(r['baseline_coefficient_digest'],row_digest(d['a']));compare(r['functional_point_sha256'],safe.tensor_hash(point));compare(r['forward_parity_checked_before_backwards'],True)
        close(r['observed'],pr['observed'],'C145 gradient forward',2e-6);need(abs(r['KL']-pr['KL'])<=2e-7,'C145 gradient KL drift')
        oldparts=prior.auditcore.loss(r['observed'],r['KL'],point,p['prepared'],d['a'],sp['case']);compare(r['original_objective_parts'],oldparts);compare(r['original_objective'],sum(oldparts.values()));compare(r['response_ratio'],prior.qvalue(r['observed']))
    directions,details=prior.directions_from_rows(grads,len(d['a']));compare(z.json('DIRECTION_RULES_AND_WEIGHTS.json'),details,'C145 direction arithmetic')
    stored=z.arrays('TWO_FROZEN_DIRECTIONS.npz');need(set(stored)==set(directions),'Both diagnostic directions retained')
    for k,v in directions.items():exact(stored[k],v,'C145 diagnostic direction/'+k)
    controls,control_records=prior.probe_controls(d['a'],directions);compare(z.json('ALL9_CONTROL_REGISTRY.json'),control_records,'C145 fixed probe registry');stored=z.arrays('ALL9_FROZEN_CONTROLS.npz');need(set(stored)==set(controls),'All nine controls')
    for k,v in controls.items():exact(stored[k],v,'C145 fixed control/'+k)
    frozen=z.arrays('ALL756_FROZEN_DIAGNOSTIC_POINTS.npz');need(set(frozen)=={sid+'__'+cfg['config_id'] for sid in d['points'] for cfg in prior.configurations()},'All756 point inventory');points={};outcomes=[]
    for sp in prior.registry():
        sid=sp['state_id'];p=d['points'][sid]
        for cfg in prior.configurations():
            name=cfg['config_id'];point,diag=proposal(p['prepared'],p['raw'],d['V'],controls[name]);exact(frozen[sid+'__'+name],point,'C145 sealed point/'+sid+'/'+name);points[(sid,name)]=point;outcomes.append(dict(**sp,**cfg,point_sha256=safe.tensor_hash(point),control_digest=row_digest(controls[name]),proposal_diagnostics=diag))
    compare(z.json('ALL756_FROZEN_POINT_REGISTRY.json'),outcomes,'C145 point registry')
    seal=z.json('ALL_PROBES_SEALED_BEFORE_MEASUREMENTS.json');compare(seal['native_probe_endpoint_measurements_so_far'],0);compare(seal['points'],756);compare(seal['base_control_digest'],row_digest(d['a']))
    required={'INPUTS_FROZEN.json','ALL84_EXPOSED_TRAINING_REGISTRY.json','FROZEN_INPUT_IDENTITIES.json','MODEL_PINS_REFERENCE.json','BASELINE_C143_CONTROL.npz','ALL84_FROZEN_TRAINING_INPUTS.npz','C144_BOUND_RECEIPT.json','DESIGN_FROZEN_BEFORE_MODEL.json','TWO_FROZEN_DIRECTIONS.npz','DIRECTION_RULES_AND_WEIGHTS.json','ALL9_FROZEN_CONTROLS.npz','ALL9_CONTROL_REGISTRY.json','ALL756_FROZEN_DIAGNOSTIC_POINTS.npz','ALL756_FROZEN_POINT_REGISTRY.json'}
    need(set(seal['artifact_seals'])==required,'Exact C145 frozen artifact registry')
    for n,h in seal['artifact_seals'].items():compare(hashlib.sha256(z.raw(n)).hexdigest(),h,'C145 frozen artifact seal/'+n)
    pre=z.json('PREFLIGHT_PROGRESS.json');need(len(pre)==336,'All336 C145 original replays');expected_pre=[(sp['state_id'],kind) for sp in prior.registry() for kind in ('prepared','raw','C140','C143')];compare([(r['state_id'],r['kind']) for r in pre],expected_pre)
    for r in pre:
        sid=r['state_id'];kind=r['kind'];p=d['points'][sid];point=p[kind] if kind in ('prepared','raw') else proposal(p['prepared'],p['raw'],d['V'],d['C140_a'] if kind=='C140' else d['a'])[0]
        native_gates(r['metrics'],r['case'],point,kind!='prepared')
        for old in d['refs'][(sid,kind)]:
            q=repeat_check(r['metrics'],old);need(q['within_replay_tolerance'] and q['same_point_identity'],'C145 native replay')
    records=z.json('ALL1512_DIAGNOSTIC_ENDPOINTS.json');need(len(records)==1512,'All1512 C145 endpoints');compare([(r['round'],r['state_id'],r['config_id']) for r in records],[(rnd,sp['state_id'],cfg['config_id']) for rnd in (0,1) for sp in prior.registry() for cfg in prior.configurations()])
    gb={r['state_id']:r for r in grads};first={}
    for r in records:
        sid=r['state_id'];name=r['config_id'];point=points[(sid,name)];gate=endpoint_gates(r['metrics'],r['case'],point,d['points'][sid]['prepared'],True);compare(r['gates'],gate,'C145 original gates');compare(r['control_digest'],row_digest(controls[name]))
        predicted=gb[sid]['response_ratio']+float(np.asarray(gb[sid]['response_gradient'])@(controls[name]-d['a']));measured=prior.qvalue(r['metrics']['observed']);compare(r['AD_predicted_response_ratio'],predicted);compare(r['measured_response_ratio'],measured);compare(r['response_ratio_linearization_residual'],measured-predicted)
        if r['round']==0:first[(sid,name)]=r['metrics']
        else:
            q=repeat_check(r['metrics'],first[(sid,name)]);need(q['within_replay_tolerance'] and q['same_point_identity'],'C145 repeated endpoint drift')
        if name=='C143_baseline':
            for old in d['refs'][(sid,'C143')]:
                q=repeat_check(r['metrics'],old);need(q['within_replay_tolerance'] and q['same_point_identity'],'C145 baseline replay drift')
    summaries=prior.tables(records);compare(z.json('DIAGNOSTIC_SUMMARIES.json'),summaries);compare(s['diagnostic_summaries'],summaries)
    allpass=[cfg['config_id'] for cfg in prior.configurations()[1:] if all(summaries['all84'][str(j)][cfg['config_id']]['passed']==84 for j in (0,1))]
    rescue=[cfg['config_id'] for cfg in prior.configurations()[1:] if all(summaries['all84'][str(j)][cfg['config_id']]['rescued_vs_baseline'] and not summaries['all84'][str(j)][cfg['config_id']]['destroyed_vs_baseline'] for j in (0,1))]
    compare(s['diagnostic_configurations_with_all84_passed'],allpass);compare(s['diagnostic_configurations_with_rescue_and_no_destruction'],rescue)
    interpretation='ALL84_TRAINING_PASS_OBSERVED_AT_REGISTERED_PROBE_NO_SELECTION' if allpass else ('NONDESTRUCTIVE_TRAINING_IMPROVEMENT_OBSERVED_IN_FIXED_GRID' if rescue else 'NO_NONDESTRUCTIVE_IMPROVEMENT_OBSERVED_IN_FIXED_GRID');compare(s['diagnostic_interpretation'],interpretation)
    return s,dict(protocol=prior.PROTOCOL,decision=s['decision'],members_hashed=members,all84_inputs_rebound=True,all756_proposals_recomputed=True,all1512_gates_recomputed=True,archived_gradient_vectors_used_only_for_arithmetic=True,gradients_independently_recomputed=False,source_executed=False,no_probe_selected=True,chronology_or_nonaccess_independently_proved=False)

def emit(out,d,paths,receipt):
    dump(out/'INPUTS_FROZEN.json',{k:dict(path=str(p),sha256=sha(p),known_before_C146=True) for k,p in paths.items()});dump(out/'ALL84_TRAINING_REGISTRY_FROZEN.json',registry());dump(out/'FROZEN_INPUT_IDENTITIES.json',frozen_digest(d));dump(out/'MODEL_PINS_REFERENCE.json',d['modelpins']);dump(out/'C145_BOUND_RECEIPT.json',receipt)
    np.savez_compressed(out/'ALL84_TRAINING_INPUTS.npz',**{sid+'__'+k:p[k] for sid,p in d['points'].items() for k in ('prepared','raw','features')});np.savez_compressed(out/'INITIAL_C143_STEP12_CONTROL.npz',a=d['a'],m=np.zeros_like(d['a']),v=np.zeros_like(d['a']));np.savez_compressed(out/'FROZEN_SHARED_ROW_BASIS.npz',V=d['V']);np.savez_compressed(out/'FROZEN_ORIGINAL_WRITER.npz',**d['writer']);np.savez_compressed(out/'FROZEN_ORIGINAL_CONTROL.npz',**d['control'])

def load(paths,out,c=None):
    c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);need(set(paths)==set(INPUT_KEYS),'Exactly six fixed original archives')
    for k,p in paths.items():compare(sha(p),c['pins'][k]['sha256'],'Fixed input/'+k)
    cc=dict(c['binding145'],production=c['production'],pins={k:c['pins'][k] for k in prior.INPUT_KEYS})
    with safe.no_solvers():
        d=load145({k:paths[k] for k in prior.INPUT_KEYS},out/'ORIGINAL_C145_PARENT_BINDINGS',cc)
        with contextlib.closing(Archive(paths['C145'])) as z:s,receipt=audit145(z,d,c)
    d['C145_summary']=s;emit(out,d,paths,receipt);print('C146_INPUT_BINDINGS_OK six originals; original C143 initialization; no C145 probe selected.',flush=True);return d
