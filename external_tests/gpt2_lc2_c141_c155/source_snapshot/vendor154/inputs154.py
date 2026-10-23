"""Nine pinned originals; rerun trusted C153 arithmetic and retain actual bound data."""
from core154 import *
from audit153 import audit as audit153
from inputs152 import load as load152
from audit147 import source_check

def load(paths,out,c=None):
    c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);need(set(paths)==set(INPUT_KEYS),'Exactly nine pinned originals')
    for k,p in paths.items():compare(sha(p),c['pins'][k]['sha256'],'Fixed original/'+k)
    for n,h in c['source153'].items():compare(sha(ROOT/'vendor153'/n),h,'Delivered trusted C153 source/'+n)
    captured=[]
    def capture(pp,oo,cc):
        d=load152(pp,oo,cc);captured.append(d);return d
    with prior.no_models(),safe.no_solvers():
        result=audit153({k:paths[k] for k in prior.INPUT_KEYS},out/'RECOMPUTED_C153',c['audit153'],input_loader=capture)
        need(result['complete'] and len(captured)==1,'Complete C153 re-audit before model');d=captured[0]
        with contextlib.closing(Archive(paths['C153'])) as z:
            count=z.verify_manifest();source_check(z,c['source153'],'C153');s=z.json('SUMMARY.json');compare(s['protocol'],prior.PROTOCOL);compare(s['errors'],[])
            compare({k:v for k,v in s.items() if k not in ('elapsed_audit_seconds','errors')},{k:v for k,v in result.items() if k!='elapsed_audit_seconds'},'Original and re-audited C153 scientific summary')
    anchors={}
    with contextlib.closing(Archive(paths['C152'])) as z:
        pre={(r['state_id'],r['kind']):r['metrics'] for r in z.json('PREFLIGHT_PROGRESS.json')};ends=z.json('ALL672_FINAL_TRAINING_ENDPOINTS.json')
        for a,step in ANCHORS:
            folder=f'training/{ARM}/step{step:02d}/';u=z.arrays(folder+'UPDATE.npz');rows=z.json(folder+'OBSERVATIONS.json');g,agg=prior.aggregate(rows,u['before'].shape);close(g,u['gradient'],'Audited archived aggregate',1e-12)
            plus_refs=([r['metrics'] for r in ends if r['arm']==ARM and r['round']==0] if step==12 else z.json(f'training/{ARM}/step{step+1:02d}/OBSERVATIONS.json'))
            anchors[a]=dict(step=step,before=u['before'].copy(),after=u['after'].copy(),gradient=g.copy(),aggregate=agg,rows=rows,plus_references=plus_refs,zero_native_references=[pre[(sp['state_id'],'C150')] for sp in registry()] if step==1 else None)
    dump(out/'ORIGINAL_INPUTS_FROZEN.json',{k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()});dump(out/'C153_AUDIT_RECEIPT.json',dict(C153_original_members_hashed=count,C153_reaudit_complete=True,parent_decision=result['parent_decision'],parent_training_qualified=result['training_qualified'],prior_C149_audit_reexecuted=False))
    np.savez_compressed(out/'ARCHIVED_ANCHORS_AND_DIRECTIONS.npz',**{a+'__'+k:h[k] for a,h in anchors.items() for k in ('before','after','gradient')});dump(out/'ARCHIVED_ANCHOR_RECORDS.json',{a:{k:h[k] for k in ('step','aggregate','rows','plus_references','zero_native_references')} for a,h in anchors.items()});return d,anchors
