"""Bind the original C132 and C133 scientific capsules before model access."""
import contextlib,io
from core134 import *
from reader133 import load as read132
from worker133 import run as audit133

def load(paths,out,c):
    need(set(paths)==set(INPUT_KEYS),'Exactly C132 and C133 originals required')
    for k in INPUT_KEYS:need(sha(paths[k])==c['pins'][k]['sha256'],'Wrong fixed original '+k)
    dump(out/'INPUTS_FROZEN.json',{k:dict(path=str(paths[k]),sha256=sha(paths[k]),known_before_C134=True) for k in INPUT_KEYS})
    c32=dict(production=c['production'],pin=c['pins']['C132'],source=c['source132'])
    with (out/'C133_AUDIT_CONSOLE.log').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log):result=audit133(paths['C132'],out/'C133_RECOMPUTED_AUDIT',c32)
    need(result['complete'],'Independent C133 arithmetic replay incomplete: '+str(result['errors']))
    with contextlib.closing(Archive(paths['C133'])) as z:
        members=z.verify_manifest();summary=z.json('SUMMARY.json');compare(summary['protocol'],'GPT2_LC2_C133_FINITE_STEP_DECOMPOSITION_CPU_V1');compare(summary['complete'],True)
        compare(summary['decision'],'C133_FINITE_STEP_DECOMPOSITION_COMPLETE_PARENT_DECISION_RETAINED')
        if c['production']:compare(summary['synthetic_fixture'],False)
        source={n[7:] for n in z.z.namelist() if n.startswith('source/')};need(source==set(c['source133']),'C133 delivered source inventory changed')
        for n,h in c['source133'].items():need(hashlib.sha256(z.raw('source/'+n)).hexdigest()==h,'C133 delivered source changed: '+n)
        compare(z.json('INPUT_FROZEN.json')['C132']['sha256'],c['pins']['C132']['sha256'],'C133 linked C132 original')
        for k in ('local_decomposition_summaries','next_diagnostic_branch','parent_endpoint_summaries','local_decomposition_records_completed','arm_vs_raw_contrasts_completed','parent_decision','parent_branch','parent_repeatable','LC2_closed','LC2_single_fold_passed','candidate_selected','thresholds_changed','physical_curvature_cause_established'):compare(summary[k],result[k],'C133 recomputed/'+k)
        for name in ('ALL36_LOCAL_DECOMPOSITIONS.json','ALL24_ARM_VS_RAW_CONTRASTS.json','RECOMPUTED_ENDPOINT_RECORDS.json','RECOMPUTED_ENDPOINT_SUMMARIES.json','LOCAL_DECOMPOSITION_SUMMARIES.json'):
            compare(z.json(name),clean_json((out/'C133_RECOMPUTED_AUDIT'/name).read_bytes()),'C133 recomputed '+name)
        archived=z.arrays('FINITE_STEP_COMPONENTS.npz')
        with np.load(out/'C133_RECOMPUTED_AUDIT/FINITE_STEP_COMPONENTS.npz',allow_pickle=False) as f:
            need(set(archived)==set(f.files),'C133 component coverage changed')
            for k in f.files:close(archived[k],f[k],'C133 component '+k,1e-9)
        dump(out/'C133_ORIGINAL_BINDING_RECEIPT.json',dict(members_hashed=members,source_members_checked=len(source),all36_local_decompositions_matched=True,all24_contrasts_matched=True,all_component_arrays_matched=True,source_executed=False))
    with safe.no_solvers():data=read132(paths['C132'],c32,out/'C132_BOUND_INPUT_AUDIT')
    with contextlib.closing(Archive(paths['C132'])) as z:
        data['modelpins']=z.json('model_runtime/MODEL_PINS.json')
        # Verify that the runtime we ship is byte-identical to C132's pinned copy.
        for p in sorted((ROOT/'runtime_vendor').glob('*.py')):
            need(hashlib.sha256(z.raw('source/runtime_vendor/'+p.name)).hexdigest()==sha(p),'Original runtime source differs: '+p.name)
    dump(out/'FROZEN_MODEL_PINS_REFERENCE.json',data['modelpins']);print('C134_INPUT_BINDINGS_OK C132+C133; archived arithmetic reconstructed; no model yet.',flush=True)
    return data
