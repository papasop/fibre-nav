"""Seven pinned originals; re-audit C150 with trusted C151 code, bind common cyclic head."""
from core152 import *
from inputs150 import load as load150
from audit151 import inspect_archive,write_report

def emit(out,d,paths,receipt):
    dump(out/'INPUTS_FROZEN.json',{k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()});dump(out/'ALL84_TRAINING_REGISTRY_FROZEN.json',registry())
    W=np.array([d['weights'][sp['state_id']] for sp in registry()])
    for name,arrays in [('ALL84_TRAINING_INPUTS.npz',{sid+'__'+k:v for sid,p in d['points'].items() for k,v in p.items()}),('FROZEN_SHARED_ROW_BASIS.npz',dict(V=d['V'])),('INITIAL_C150_COUPLED_CONTROL.npz',dict(parameters=d['initial_P'])),('INITIAL_C148_ROUTED_REFERENCE.npz',dict(A=d['initial_A'])),('FROZEN_TRAINING_ONLY_ROUTER.npz',d['router']),('ALL84_PREWRITE_ROUTING.npz',dict(weights=W)),('FROZEN_ORIGINAL_WRITER.npz',d['writer']),('FROZEN_ORIGINAL_CONTROL.npz',d['control'])]:np.savez_compressed(out/name,**arrays)
    dump(out/'MODEL_PINS_REFERENCE.json',d['modelpins']);dump(out/'FROZEN_INPUT_IDENTITIES.json',fixed_digest(d));dump(out/'PARENT_AUDIT_RECEIPT.json',receipt)

def load(paths,out,c=None):
    c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);need(set(paths)==set(INPUT_KEYS),'Exactly seven pinned originals')
    for k,p in paths.items():compare(sha(p),c['pins'][k]['sha256'],'Fixed original/'+k)
    for n,h in c['source151'].items():compare(sha(ROOT/'vendor151'/n),h,'Delivered trusted C151 source/'+n)
    pc=c['audit151'];pp={k:paths[k] for k in previous.INPUT_KEYS};audit_out=out/'RECOMPUTED_C151_C150_AUDIT';audit_out.mkdir(exist_ok=True)
    with auditcore.no_models(),safe.no_solvers():
        d=load150(pp,out/'BOUND_C150_INPUTS',pc['binding150'])
        with contextlib.closing(Archive(paths['C150'])) as z:
            result=inspect_archive(z,d,audit_out,pc);result.update(original_archives_rehashed=list(INPUT_KEYS),original_inputs_unchanged={k:sha(p)==c['pins'][k]['sha256'] for k,p in paths.items()});dump(audit_out/'SUMMARY.json',result);write_report(audit_out,result);d['initial_P']=z.arrays('FINAL_routed_cyclic_coupling_STEP12.npz')['parameters'];rows=z.json('ALL672_FINAL_TRAINING_ENDPOINTS.json')
            for sp in registry():d['refs'][(sp['state_id'],'C150')]=[r['metrics'] for r in rows if r['state_id']==sp['state_id'] and r['arm']=='routed_cyclic_coupling']
    rank=d['V'].shape[0];need(rank==36 if c['production'] else 1<rank<=36,'Fixed coupled rank');farray(d['initial_P'],(3,2,rank))
    for sp in registry():
        sid=sp['state_id'];q=d['points'][sid];point=preflight_point(d,sid,'C150');refs=d['refs'][(sid,'C150')];need(len(refs)==2,'Both original coupled endpoint rounds')
        for r in refs:native_gates(r,sp['case'],point,True)
        for arm in TRAIN_ARMS:exact(proposal(q['prepared'],q['raw'],d['V'],effective(initial_parameters(d,arm),d['weights'][sid],arm))[0],point,'Both initial rate heads reproduce C150 point')
    emit(out,d,paths,dict(C151_code_audit_complete=result['complete'],C150_parent_decision=result['parent_decision'],C150_training_qualified=result['training_qualified'],C151_original_result_ZIP_required=False,same_full_nonzero_cyclic_initial_head_for_both_arms=True,original_C150_step12_digest=row_digest(d['initial_P']),all84_retained_as_exposed_training=True));return d
