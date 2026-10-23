"""Reopen original C147 and rerun its trusted arithmetic before accessing model code."""
from core148 import *
from audit147 import audit,bind_originals,source_check

def load(paths,out,c=None):
    c=contract() if c is None else c;out=Path(out);need(set(paths)==set(INPUT_KEYS),'Exactly four original ZIPs')
    for k,p in paths.items():compare(sha(p),c['pins'][k]['sha256'],'Fixed input/'+k)
    dump(out/'INPUTS_FROZEN.json',{k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()})
    pc=c['binding147'];pp={k:paths[k] for k in previous.INPUT_KEYS}
    with previous.no_models(),safe.no_solvers():recomputed=audit(pp,out/'C147_RECOMPUTED_AUDIT',pc)
    with contextlib.closing(Archive(paths['C147'])) as z:
        count=z.verify_manifest();source_check(z,c['source147'],'C147');original=z.json('SUMMARY.json');compare(original.get('errors'),[])
        for k,v in recomputed.items():
            if k!='elapsed_audit_seconds':compare(original[k],v,'C147 original/recomputed/'+k)
    with previous.no_models(),safe.no_solvers():d,_=bind_originals(pp,out/'BOUND_INPUT_READER',pc)
    with contextlib.closing(Archive(paths['C146'])) as z:
        d['initial_a']=z.arrays('FINAL_SHARED_CONTROL_STEP12.npz')['a'];rows=z.json('ALL504_FINAL_TRAINING_ENDPOINTS.json')
        for sp in registry():d['refs'][(sp['state_id'],'C146')]=[r['metrics'] for r in rows if r['state_id']==sp['state_id'] and r['arm']=='C146_fixed_step12']
    rank=len(d['initial_a']);need(rank==36 if c['production'] else 0<rank<=36,'Frozen basis/control rank');farray(d['V'],(rank,DIM));farray(d['initial_a'],(rank,))
    U=np.array([coordinates(d['points'][sp['state_id']]['prepared'],d['points'][sp['state_id']]['raw'],d['V']) for sp in registry()]);router,W,meta=fit_router(U);d['router']=router;d['weights']={sp['state_id']:w.copy() for sp,w in zip(registry(),W)}
    dump(out/'ALL84_TRAINING_REGISTRY_FROZEN.json',registry());dump(out/'ROUTER_DESIGN_AND_COVERAGE.json',meta);np.savez_compressed(out/'FROZEN_TRAINING_ONLY_ROUTER.npz',**router);np.savez_compressed(out/'ALL84_PREWRITE_ROUTING.npz',u=U,weights=W)
    for filename,arrays in [('ALL84_TRAINING_INPUTS.npz',{sid+'__'+k:v for sid,p in d['points'].items() for k,v in p.items()}),('FROZEN_SHARED_ROW_BASIS.npz',dict(V=d['V'])),('INITIAL_C146_CONTROL.npz',dict(a=d['initial_a'])),('FROZEN_ORIGINAL_WRITER.npz',d['writer']),('FROZEN_ORIGINAL_CONTROL.npz',d['control'])]:np.savez_compressed(out/filename,**arrays)
    dump(out/'MODEL_PINS_REFERENCE.json',d['modelpins']);dump(out/'FROZEN_INPUT_IDENTITIES.json',fixed_digest(d));dump(out/'PARENT_AUDIT_RECEIPT.json',dict(C147_original_members_hashed=count,C147_decision=recomputed['decision'],C146_decision=recomputed['parent_decision'],other_linked_originals_rehashed=False))
    return d
