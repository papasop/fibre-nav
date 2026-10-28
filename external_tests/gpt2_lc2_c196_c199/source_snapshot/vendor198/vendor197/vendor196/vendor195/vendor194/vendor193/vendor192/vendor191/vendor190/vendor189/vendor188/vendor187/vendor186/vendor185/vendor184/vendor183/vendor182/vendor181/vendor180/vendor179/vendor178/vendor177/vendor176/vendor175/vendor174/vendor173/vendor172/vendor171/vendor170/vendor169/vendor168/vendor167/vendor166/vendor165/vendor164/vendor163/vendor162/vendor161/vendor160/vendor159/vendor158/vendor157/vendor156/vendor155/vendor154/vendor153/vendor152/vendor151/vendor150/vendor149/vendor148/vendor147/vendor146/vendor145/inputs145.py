"""Rebind the five originals; reuse frozen coordinates/features, never create states."""
from core145 import *
from audit144 import audit as audit144

def load(paths,out,c=None):
    c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);need(set(paths)==set(INPUT_KEYS),'Exactly five fixed scientific ZIPs required')
    for k,p in paths.items():compare(sha(p),c['pins'][k]['sha256'],'Fixed original/'+k)
    dump(out/'INPUTS_FROZEN.json',{k:dict(path=str(p),sha256=sha(p),known_before_C145=True) for k,p in paths.items()})
    with contextlib.closing(Archive(paths['C144'])) as z:
        count=z.verify_manifest();s=z.json('SUMMARY.json');compare(s['protocol'],auditcore.PROTOCOL);compare(s['complete'],True);compare(s['decision'],'C144_READONLY_JOINT84_AUDIT_COMPLETE_PARENT_DECISION_RETAINED');compare(s['parent_decision'],'C143_FIXED_FINAL_TRAINING_CONSTRAINT_FAILURE');compare(s['parent_decision_reclassified'],False)
        if c['production']:compare(s['synthetic_fixture'],False)
        names={n[7:] for n in z.z.namelist() if n.startswith('source/')};need(names==set(c['source144']),'Trusted C144 source inventory')
        for n,h in c['source144'].items():compare(hashlib.sha256(z.raw('source/'+n)).hexdigest(),h,'Trusted C144 source/'+n)
        for k in auditcore.INPUT_KEYS:compare(z.json('INPUTS_FROZEN.json')[k]['sha256'],c['pins'][k]['sha256'],'C144 linked original/'+k)
        for k in ('new_model_forwards','new_backward_calls','new_training_steps','new_head_fits','new_Jacobians','new_SVD_calls','linear_systems_solved','excluded_input_states_evaluated'):compare(s[k],0)
        for k in ('LC2_closed','LC2_single_fold_passed','candidate_selected','writer_deployed','thresholds_changed','archived_source_executed'):compare(s[k],False)
    cc=dict(c['audit144'],production=c['production'],pins={k:c['pins'][k] for k in auditcore.INPUT_KEYS})
    with safe.no_solvers():recomputed=audit144({k:paths[k] for k in auditcore.INPUT_KEYS},out/'C143_ORIGINAL_READONLY_AUDIT',cc)
    for k,value in recomputed.items():
        if k!='elapsed_audit_seconds':compare(s[k],value,'Independently rebound C144/'+k)
    with contextlib.closing(Archive(paths['C143'])) as z:
        a=z.arrays('FINAL_SHARED_CONTROL_STEP12.npz')['a'];warm=z.arrays('INITIAL_C140_STEP12_CONTROL.npz')['a'];V=z.arrays('FROZEN_SHARED_ROW_BASIS.npz')['V'];arrays=z.arrays('ALL84_TRAINING_INPUTS.npz')
        points={sp['state_id']:{k:arrays[sp['state_id']+'__'+k] for k in ('prepared','raw','features')} for sp in registry()}
        writer=z.arrays('FROZEN_ORIGINAL_WRITER.npz');control=z.arrays('FROZEN_ORIGINAL_CONTROL.npz');modelpins=z.json('MODEL_PINS_REFERENCE.json');summary=z.json('SUMMARY.json')
        pre={(r['state_id'],r['kind']):r['metrics'] for r in z.json('PREFLIGHT_PROGRESS.json')};ends={(r['round'],r['state_id'],r['arm']):r['metrics'] for r in z.json('ALL504_FINAL_TRAINING_ENDPOINTS.json')};refs={}
        for sp in registry():
            sid=sp['state_id'];refs[(sid,'prepared')]=[pre[(sid,'prepared')]]
            for kind,arm in [('raw','raw_control'),('C140','C140_frozen_step12'),('C143','C143_shared_step12')]:refs[(sid,kind)]=[ends[(j,sid,arm)] for j in (0,1)]
    data=dict(points=points,V=V,a=a.copy(),C140_a=warm.copy(),writer=writer,control=control,modelpins=modelpins,refs=refs,C143_summary=summary,C144_summary=s)
    rank=len(a);need(rank==36 if c['production'] else 0<rank<=36,'Fixed shared dimension');farray(V,(rank,DIM));farray(a,(rank,));need(np.max(abs(a),initial=0)<=.125,'Baseline gain bound')
    dump(out/'ALL84_EXPOSED_TRAINING_REGISTRY.json',registry());dump(out/'FROZEN_INPUT_IDENTITIES.json',frozen_digest(data));dump(out/'MODEL_PINS_REFERENCE.json',modelpins)
    np.savez_compressed(out/'BASELINE_C143_CONTROL.npz',a=a,V=V,C140_a=warm);np.savez_compressed(out/'ALL84_FROZEN_TRAINING_INPUTS.npz',**arrays)
    dump(out/'C144_BOUND_RECEIPT.json',dict(protocol=auditcore.PROTOCOL,decision=s['decision'],parent_decision=s['parent_decision'],C144_members_hashed=count,C143_original_audited=True,source_executed=False,all84_remain_training=True))
    print('C145_INPUT_BINDINGS_OK five original ZIPs; C143 fixed final a; all84 exposed training.',flush=True);return data
