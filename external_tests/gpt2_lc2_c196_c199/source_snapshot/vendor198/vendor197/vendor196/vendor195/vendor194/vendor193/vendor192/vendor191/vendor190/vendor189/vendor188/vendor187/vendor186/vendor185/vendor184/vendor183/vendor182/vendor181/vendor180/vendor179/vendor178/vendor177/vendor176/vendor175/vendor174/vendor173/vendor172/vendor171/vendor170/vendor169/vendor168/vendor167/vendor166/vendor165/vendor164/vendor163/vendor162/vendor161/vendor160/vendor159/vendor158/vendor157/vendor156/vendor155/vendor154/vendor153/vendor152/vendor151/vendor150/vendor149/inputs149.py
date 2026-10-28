"""Five pinned originals; C147 arithmetic and C146 actual final tensor binding."""
from core149 import *
from audit147 import audit as audit147,bind_originals,source_check

def bind(paths,out,c):
    need(set(paths)==set(INPUT_KEYS),'Exactly C148/C147/C146/C145/C143 original ZIPs')
    for k,p in paths.items():compare(sha(p),c['pins'][k]['sha256'],'Fixed original/'+k)
    dump(out/'INPUTS_FROZEN.json',{k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()});pc=c['binding148']['binding147'];pp={k:paths[k] for k in parent.previous.INPUT_KEYS};result=audit147(pp,out/'RECOMPUTED_C147_AUDIT',pc)
    with contextlib.closing(Archive(paths['C147'])) as z:
        n147=z.verify_manifest();source_check(z,c['binding148']['source147'],'C147');original=z.json('SUMMARY.json');compare(original['errors'],[])
        for k,v in result.items():
            if k!='elapsed_audit_seconds':compare(original[k],v,'Original C147 summary/'+k)
    d,counts=bind_originals(pp,out/'BOUND_PARENT_READER',pc);counts['C147']=n147
    with contextlib.closing(Archive(paths['C146'])) as z:
        d['initial_a']=z.arrays('FINAL_SHARED_CONTROL_STEP12.npz')['a'];ends=z.json('ALL504_FINAL_TRAINING_ENDPOINTS.json')
        for sp in registry():d['refs'][(sp['state_id'],'C146')]=[r['metrics'] for r in ends if r['state_id']==sp['state_id'] and r['arm']=='C146_fixed_step12']
    d['_counts']=counts;d['_C147_recomputed']=result;return d
