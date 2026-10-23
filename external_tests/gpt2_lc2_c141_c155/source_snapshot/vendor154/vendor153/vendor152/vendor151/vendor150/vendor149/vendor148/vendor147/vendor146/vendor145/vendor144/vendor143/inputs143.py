"""Original C140/C141/C142 bindings; all exposed states explicitly become training."""
import contextlib
from core143 import *
from audit142 import audit as audit142

def load(paths,out,c=None):
    c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);need(set(paths)==set(INPUT_KEYS),'Exactly C140/C141/C142 scientific archives required')
    for k,p in paths.items():compare(sha(p),c['pins'][k]['sha256'],'Fixed original '+k)
    dump(out/'INPUTS_FROZEN.json',{k:dict(path=str(p),sha256=sha(p),known_before_C143=True) for k,p in paths.items()})
    with contextlib.closing(Archive(paths['C142'])) as z:
        count=z.verify_manifest();s=z.json('SUMMARY.json');compare(s['protocol'],auditcore.PROTOCOL);compare(s['complete'],True);compare(s['decision'],'C142_READONLY_CONFIRMATION_AUDIT_COMPLETE_PARENT_DECISION_RETAINED');compare(s['parent_decision'],'C141_FROZEN_KERNEL_LOCAL_CONFIRMATION_FAILURE');compare(s['parent_decision_reclassified'],False)
        if c['production']:compare(s['synthetic_fixture'],False)
        names={n[7:] for n in z.z.namelist() if n.startswith('source/')};need(names==set(c['source142']),'C142 delivered source inventory')
        for n,h in c['source142'].items():compare(hashlib.sha256(z.raw('source/'+n)).hexdigest(),h,'C142 trusted source/'+n)
        for k in ('new_model_forwards','new_backward_calls','new_training_steps','new_head_fits','new_Jacobians','new_SVD_calls','linear_systems_solved','excluded_input_states_evaluated'):compare(s[k],0)
        for k in ('LC2_closed','LC2_single_fold_passed','candidate_selected','thresholds_changed','archived_source_executed'):compare(s[k],False)
        for k in ('C140','C141'):compare(z.json('INPUTS_FROZEN.json')[k]['sha256'],c['pins'][k]['sha256'],'C142 linked original/'+k)
    cc=dict(auditcore.contract(),production=c['production'],pins={k:c['pins'][k] for k in ('C140','C141')})
    recomputed=audit142({k:paths[k] for k in ('C140','C141')},out/'C141_C140_READONLY_BINDING',cc)
    for k,v in recomputed.items():
        if k!='elapsed_audit_seconds':compare(s[k],v,'C142 independently reconstructed/'+k)
    receipt=dict(parent_protocol=auditcore.PROTOCOL,parent_decision=s['decision'],C141_decision=s['parent_decision'],parent_decision_reclassified=False,C142_members_hashed=count,C140_C141_original_bytes_rehashed=True,C141_states_reconstructed=48,C141_preparations_checked=96,C141_endpoints_checked=288,source_executed=False,original_C132_C139_ZIPs_reopened=False)
    dump(out/'C142_BOUND_RECEIPT.json',receipt)
    with contextlib.closing(Archive(paths['C140'])) as old,contextlib.closing(Archive(paths['C141'])) as fresh:
        arrays=old.arrays('ALL36_TRAINING_INPUTS.npz');points={}
        for sp in base.registry():points[sp['state_id']]={k:arrays[sp['state_id']+'__'+k] for k in ('prepared','raw','features')}
        for sp in prior.registry():
            sid=sp['state_id'];f=fresh.arrays('prewrite/'+sid+'.npz');points[sid]=dict(prepared=f['prepared'],raw=f['raw'],features=f['write_features3211'])
        controls=old.arrays('FROZEN_ORIGINAL_CONTROL.npz');writer=old.arrays('FROZEN_ORIGINAL_WRITER.npz');V=old.arrays('FROZEN_SHARED_ROW_BASIS.npz')['V'];a=old.arrays('FINAL_SHARED_CONTROL_STEP12.npz')['a'];modelpins=old.json('MODEL_PINS_REFERENCE.json')
        old_pre={r['state_id']:r['metrics'] for r in old.json('PREFLIGHT_PROGRESS.json') if r['kind']=='prepared'};old_end={(r['state_id'],r['arm'],r['round']):r['metrics'] for r in old.json('ALL216_FINAL_TRAINING_ENDPOINTS.json')}
        new_pre={(r['state_id'],r['round']):r['metrics'] for r in fresh.json('ALL96_NEW_PREPARATIONS.json')};new_end={(r['state_id'],r['arm'],r['round']):r['metrics'] for r in fresh.json('ALL288_CONFIRMATION_ENDPOINTS.json')}
        refs={}
        for sp in registry():
            sid=sp['state_id'];p=points[sid]
            if sp['C143_training_origin']=='C140_training36':
                refs[(sid,'prepared')]=[old_pre[sid]]
                refs[(sid,'raw')]=[old_end[(sid,base.EVAL_ARMS[0],j)] for j in (0,1)];refs[(sid,'warm')]=[old_end[(sid,base.EVAL_ARMS[2],j)] for j in (0,1)]
            else:
                refs[(sid,'prepared')]=[new_pre[(sid,j)] for j in (0,1)];refs[(sid,'raw')]=[new_end[(sid,prior.ARMS[0],j)] for j in (0,1)];refs[(sid,'warm')]=[new_end[(sid,prior.ARMS[2],j)] for j in (0,1)]
            farray(p['prepared'],(DIM,));farray(p['raw'],(DIM,));farray(p['features'],(3211,));raw,full=safe.reconstruct_raw(writer,controls['delta_sketch'],p['prepared'],p['features'][:3147]);exact(raw,p['raw'],'Frozen84 raw map');close(full,p['features'],'Frozen84 full3211 including64 tail',1e-10)
            need(all(safe.metric_gates(m,sp['case'],p['prepared'],False) for m in refs[(sid,'prepared')]),'All84 valid preparation references required')
    d=dict(points=points,writer=writer,control=controls,V=V,initial_a=a.copy(),modelpins=modelpins,native_refs=refs,C142_summary=s,C141_parent_decision=s['parent_decision'])
    need(set(points)=={r['state_id'] for r in registry()},'All84 recovered training states')
    dump(out/'ALL84_TRAINING_REGISTRY_FROZEN.json',[dict(r,prepared_sha256=safe.tensor_hash(points[r['state_id']]['prepared']),raw_sha256=safe.tensor_hash(points[r['state_id']]['raw'])) for r in registry()]);dump(out/'MODEL_PINS_REFERENCE.json',modelpins);dump(out/'FROZEN_INPUT_IDENTITIES.json',frozen_digest(d))
    np.savez_compressed(out/'ALL84_TRAINING_INPUTS.npz',**{sid+'__'+k:p[k] for sid,p in points.items() for k in ('prepared','raw','features')});np.savez_compressed(out/'FROZEN_SHARED_ROW_BASIS.npz',V=V);np.savez_compressed(out/'INITIAL_C140_STEP12_CONTROL.npz',a=a,m=np.zeros_like(a),v=np.zeros_like(a));np.savez_compressed(out/'FROZEN_ORIGINAL_WRITER.npz',**writer);np.savez_compressed(out/'FROZEN_ORIGINAL_CONTROL.npz',**controls)
    print('C143_INPUT_BINDINGS_OK C14036 + exposed C14148; all84 training, original C141 failure retained.',flush=True);return d
