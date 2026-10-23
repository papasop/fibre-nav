"""Rebind all ten originals; validate C154 full archive/stencil/measurements."""
from core155 import *
from inputs154 import load as load154
from audit147 import source_check
from worker154 import diagnose as diagnose154

def load(paths,out,c=None,parent_loader=None):
    c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);compare(set(paths),set(INPUT_KEYS),'Exactly ten original inputs')
    for k,p in paths.items():compare(sha(p),c['pins'][k]['sha256'],'Fixed original/'+k)
    for n,h in c['source154'].items():compare(sha(ROOT/'vendor154'/n),h,'Delivered parent source/'+n)
    with prior.no_models(),safe.no_solvers():
        d,anchors=(load154 if parent_loader is None else parent_loader)({k:paths[k] for k in prev.INPUT_KEYS},out/'PARENT_BINDING',c['parent']);reconstruct=out/'RECONSTRUCTED_C154';reconstruct.mkdir();dump(reconstruct/'DESIGN_FROZEN_BEFORE_MODEL.json',prev.design());params,points,bindings,_=prev.freeze_stencil(d,anchors,reconstruct)
        with contextlib.closing(Archive(paths['C154'])) as z:
            count=z.verify_manifest();source_check(z,c['source154'],'C154');summary=z.json('SUMMARY.json');need(summary['complete'] and summary['real_model_execution_claim'] is c['production'] and summary['repeatable'] and summary['original_plus_one_replay_consistent'],'C154 complete consistent replay required');compare(summary['protocol'],prev.PROTOCOL);compare(summary['errors'],[]);compare(z.json('DESIGN_FROZEN_BEFORE_MODEL.json'),prev.design(),'C154 fixed design')
            parent_points=z.arrays('ALL1176_STENCIL_POINTS.npz');pcoef=z.arrays('STENCIL_PARAMETERS.npz');compare(set(parent_points),set(a+'__'+n+'__'+sid for a,n,sid in points),'C154 full point inventory')
            for (a,n,sid),p in points.items():exact(parent_points[a+'__'+n+'__'+sid],p,'Reconstructed C154 point')
            compare(set(pcoef),set(a+'__'+n for a,n in params),'C154 coefficient inventory')
            for (a,n),p in params.items():exact(pcoef[a+'__'+n],p,'Reconstructed C154 coefficient')
            compare(z.json('ALL1176_STENCIL_POINT_BINDINGS.json'),bindings,'Full parent bindings');compare(z.json('ALL1512_REQUESTS_FROZEN.json'),[dict(anchor=a,label=n,state_id=sid,phase=st) for a,n,sid,st in prev.request_order()],'Parent request registry')
            seal=z.json('STENCIL_FROZEN_BEFORE_MODEL.json')
            for n,h in seal['artifact_seals'].items():compare(hashlib.sha256(z.raw(n)).hexdigest(),h,'Parent before-model seal/'+n)
            rows=[clean_json(line) for line in z.raw('ALL_NATIVE_MEASUREMENTS.jsonl').splitlines()];compare([(r['anchor'],r['label'],r['state_id'],r['phase']) for r in rows],prev.request_order(),'Parent all1512 measurement order');groups={};panels={}
            for r in rows:
                a,n,sid,st=r['anchor'],r['label'],r['state_id'],r['phase'];q=d['points'][sid];p=points[(a,n,sid)];sp=next(sp for sp in registry() if sp['state_id']==sid);compare(r['case'],sp['case'],'Parent case binding');compare(r['gates'],endpoint_gates(r['metrics'],sp['case'],p,q['prepared'],True),'Parent actual gates');key=(a,n,st);groups.setdefault(key,[]).append(r)
            for key,rr in groups.items():panels[key]=prev.objective_panel(rr,params[key[:2]],d)
            diag=diagnose154(anchors,panels,bindings,reconstruct);compare(summary['diagnostics'],diag,'Recomputed full C154 directional arithmetic');compare(summary['native_metric_records'],1512);compare(summary['new_model_forwards'],3025)
    refs={(r['label'],r['state_id'],r['phase']):r['metrics'] for r in rows if r['anchor']==ANCHOR};need(len(refs)==756,'All late native reference measurements');dump(out/'C154_REAUDIT_RECEIPT.json',dict(parent_members_hashed=count,all1512_measurements_rebound=True,all1176_points_reconstructed=True,full_directional_summary_recomputed=True,measurements_independently_recomputed=False,legacy_seal_chronology_proved=False,all_inputs_original_bytes_rehashed=True));return d,anchors,parent_points,refs,pcoef
