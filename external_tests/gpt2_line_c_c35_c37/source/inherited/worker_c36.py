"""C35 weights stay fixed; physical replay then prospectively seeded local replication."""
import argparse,fcntl
import numpy as np
import torch
from pathlib import Path
from c36_paths import configure
ROOT=Path(__file__).resolve().parent;configure(ROOT)
from c36_protocol import *
from c36_parent import validate
from c36_audit import audit,writer_identity,parent_specs,reproduce_all,compare_replay
from c35_fit import check_frozen,records
from c35_writer import ComposedWriter
from c31_runtime import Runtime,load,tensor_hash
from worker_c35 import prepare_new,evaluate,counts

def freeze(out):
    out=Path(out);data=out/'c35';check_frozen(data)
    expected={'protocol':PROTOCOL,'settings':SETTINGS,'source_hashes':sources(ROOT),
      'parent_provenance_sha256':sha(out/'PARENT_PROVENANCE.json'),
      'parent_audit_sha256':sha(out/'PARENT_AUDIT.json'),
      'c35_coefficients_frozen_sha256':sha(data/'COEFFICIENTS_FROZEN.json'),
      'head_files':{c:sha(data/'fit'/c/'HEAD.pt') for c in CASES},
      'state_specs':specs(),'replay_ids':replay_ids(),'new_state_generation_started':False,
      'no_new_training':True,'no_branch_selection':True}
    p=out/'C36_FROZEN.json'
    if p.exists():require(read(p)==expected,'C36 frozen protocol/weights/specifications changed')
    else:
        require(not any((data/'new_states').glob('*__c36_new*')) and not any((data/'results').glob('*__c36_new*')),'Cannot freeze after C36 new generation')
        dump(p,expected)
    return expected

def runtime_dir(out):
    out=Path(out);data=out/'c35';r=out/'runtime';r.mkdir(exist_ok=True)
    names=['WRITER_FROZEN.json','RUNTIME.json']+['parent/'+n for n in ('MODEL_PINS.json','C25_FINAL_CHECKPOINT.pt','C9_CHECKPOINT.pt','C27_BRANCH.pt','C27_FEATURE_MAP.pt')]
    for n in names:
        if not (data/n).exists():continue
        p=r/n;p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():require(sha(p)==sha(data/n),'Runtime copy changed: '+n)
        else:shutil.copyfile(data/n,p)
    return r

def deployed_identity(head):
    return {kind+'/'+n:{'sha256':tensor_hash(t),'shape':list(t.shape),'dtype':str(t.dtype)}
            for kind,items in (('buffer',head.named_buffers()),('parameter',head.named_parameters())) for n,t in items}

def replay(rt,data,out):
    data=Path(data);out=Path(out);rows=[]
    for sid in replay_ids():
        saved=load(data/'proposals'/sid/'POINTS.pt');old=read(data/'results'/sid/'COMPLETE.json');spec=old['spec']
        path=out/'replay'/(sid+'.json');binding={'c35_ready_sha256':sha(data/'proposals'/sid/'READY.json'),'c36_freeze_sha256':sha(out/'C36_FROZEN.json')}
        if path.exists():
            rec=read(path);require(rec['binding']==binding,'Replay checkpoint binding changed')
            for a in ARMS:compare_replay(rec['arms'][a]['metrics'],old['arms'][a]['metrics'],spec['case'])
        else:
            start=STARTS[spec['case']];target=start[:3]+[start[0]^start[1]];arms={}
            for a in ARMS:
                m=rt.evaluate(saved['points'][a],target,start)
                arms[a]={'metrics':m,'comparison':compare_replay(m,old['arms'][a]['metrics'],spec['case'])}
            rec={'id':sid,'binding':binding,'arms':arms};dump(path,rec)
        rows.append(rec);print('C36 PHYSICAL REPLAY',sid,'OK',flush=True)
    return rows

def summary_value(out,rs):
    out=Path(out);co=counts(rs);require(len(rs)==32 and {r['spec']['id'] for r in rs}=={s['id'] for s in specs()},'New32 requested denominator differs')
    branch={a:co[a]['complete_passed']==32 for a in ('geometry6','selected6')}
    decision='C36_BOTH_FROZEN_WRITERS_LOCAL_REPLICATION_PASS' if all(branch.values()) else ('C36_ONE_BRANCH_LOCAL_REPLICATION_PASS' if any(branch.values()) else 'C36_LOCAL_REPLICATION_FAILURE')
    return {'protocol':PROTOCOL,'complete':True,'decision':decision,
      'parent_archival_states_audited':80,'parent_proposals_exactly_reproduced':80,'parent_physical_replay_states':8,
      'new_local':co,'new_local_by_case':{c:counts([r for r in rs if r['spec']['case']==c]) for c in CASES},
      'branch_complete_acceptance':branch,'all32_requested_states_retained':True,
      'invalid_preparations':sum(not r['preparation_valid'] for r in rs),'state_generation_retries':0,
      'C35_coefficients_frozen_sha256':sha(out/'c35/COEFFICIENTS_FROZEN.json'),'C36_frozen_sha256':sha(out/'C36_FROZEN.json'),
      'deployed_writer_identity_sha256':sha(out/'DEPLOYED_WRITER.json'),
      'new_training_steps':0,'test_based_branch_or_checkpoint_selection':False,
      'online_write_Jacobians':0,'online_write_SVD_calls':0,'online_write_correction_steps':0,'online_write_backtracking':0,
      'postwrite_response_used_for_proposal':False,'teacher_fallback_used':False,
      'offline_preparation_uses_geometry':True,'requested_output_bit_supplied':True,
      'fresh_state_seeds':True,'fresh_evaluation':False,'same_model_chart_access_prompts':True,
      'same_pipeline_local_replication_only':True,'independent_model_or_chart_replication':False,
      'LC1_same_model_local_replication_passed':any(branch.values()),'LC1_independently_confirmed':False,
      'internal_parameter_update':False,'controller_dispatch_tested':False,'full_Line_C_complete':False,
      'scope':SETTINGS['scope'],'interpretation':'Both frozen branches are reported separately; no test-selected switch or refitting. This is an additional predeclared local replication using the original implementation and same model/chart, not independent replication. A passing affine geometry branch alone does not establish a need for learned calibration.'}

def collected(out):
    out=Path(out);data=out/'c35';rs=[]
    for spec in specs():
        rec=read(data/'results'/spec['id']/'COMPLETE.json')
        require(rec['spec']==spec,'New result specification changed')
        for n,h in rec['files'].items():require(sha(data/n)==h,'Saved new result/seal changed')
        # Cached inherited evaluation validates all gates, seals and preparation validity below.
        folder=data/'new_states'/spec['id'];pm=read(folder/'COMPLETE.json')
        require(pm['spec']==spec,'Saved preparation specification changed')
        for n,h in pm['files'].items():require(sha(folder/n)==h,'Saved preparation changed')
        prep=load(folder/'prepared.pt')['theta'];inp=load(folder/'INPUT.pt')
        checked=evaluate(None,None,data,spec,prep,inp,pm['preparation_valid'],'new_local')
        for a in ARMS:
            require(checked['arms'][a]['complete_accepted']==bool(pm['preparation_valid'] and checked['arms'][a]['metrics']['pass']),'Acceptance flag differs')
        rs.append(rec)
    return rs

def main(out,backup='',runtime_factory=Runtime,identity_checker=writer_identity):
    out=Path(out);check_sources(ROOT);require(read(out/'identity.json')==identity(ROOT),'C36 identity/source/settings differ')
    data=validate(out,ROOT);require(read(out/'NEW_STATE_SPECS.json')==specs(),'New32 specifications changed')
    if not (out/'PARENT_AUDIT.json').exists():
        dump(out/'PARENT_AUDIT.json',audit(data,ROOT,identity_checker));snapshot(out,backup)
    else:
        a=read(out/'PARENT_AUDIT.json');require(a['complete'] and a['archived_states_audited']==80,'Incomplete archive audit')
    freeze(out)
    if (out/'SUMMARY.json').exists():
        require(read(out/'SUMMARY.json')==summary_value(out,collected(out)),'Completed summary recount differs')
        print((out/'SUMMARY.json').read_text(),flush=True);return read(out/'SUMMARY.json')
    rt=runtime_factory(runtime_dir(out),ROOT/'inherited/inherited/inherited')
    head=ComposedWriter(rt.writer,records(data),rt.device)
    versions={n:t._version for n,t in head.named_buffers()};pversions={n:p._version for n,p in head.named_parameters()}
    deployed={'c36_frozen_sha256':sha(out/'C36_FROZEN.json'),'tensors':deployed_identity(head)}
    if (out/'DEPLOYED_WRITER.json').exists():require(read(out/'DEPLOYED_WRITER.json')==deployed,'Loaded frozen writer tensor identity changed')
    else:dump(out/'DEPLOYED_WRITER.json',deployed)
    # No test results can fit/reselect a head; deterministic parent reproduction is cheap.
    reproduction=reproduce_all(rt,head,data);dump(out/'PROPOSAL_REPRODUCTION.json',reproduction)
    replay(rt,data,out);snapshot(out,backup)
    rs=[]
    for i,spec in enumerate(specs(),1):
        freeze(out);prep,inp,valid=prepare_new(rt,data,spec)
        rs.append(evaluate(rt,head,data,spec,prep,inp,valid,'new_local'))
        if i%4==0:snapshot(out,backup)
    require(all(t._version==versions[n] for n,t in head.named_buffers()) and all(p._version==pversions[n] for n,p in head.named_parameters()),'Frozen composed writer changed')
    require(deployed_identity(head)==deployed['tensors'],'Actual deployed writer tensor bytes changed')
    rt.invariant();freeze(out);validate(out,ROOT)
    s=summary_value(out,rs);dump(out/'SUMMARY.json',s);snapshot(out,backup);print(json.dumps(s,indent=2),flush=True);return s

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--drive-dir',default='');a=p.parse_args()
    lock=open(Path(a.out).parent.parent/'worker.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    main(a.out,a.drive_dir)
