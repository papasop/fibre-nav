"""Prospective, same-model local replication. No training or test-based selection."""
from pathlib import Path
import hashlib,json,os,shutil,time,zipfile
from c30_core import require,read,dump,sha,zread,verify,sources,CASES,GATES
from c35_protocol import ARMS,STARTS
PROTOCOL='GPT2_C36_FROZEN_LOCAL_REPLICATION_A100_V1'
PARENT_PROTOCOL='GPT2_C35_PREWRITE_COEFFICIENT_A100_V1'
RESULT='gpt2_c36_frozen_local_replication_results'
DRIVE_NAME=PROTOCOL+'_RECOVERY'
SETTINGS={
 'expected_parent_freeze_sha256':'d95a6d84f312261d355bf2d584c878eff2caeb2b7d19d38d92e0bc8f1c1faec8',
 'parent_protocol':PARENT_PROTOCOL,'parent_expected_decision':'C35_FROZEN_SINGLE_PROPOSAL_LOCAL_DEVELOPMENT_PASS',
 'parent_archival_audit_n':80,'parent_proposal_reproduction_n':80,'parent_physical_replay_n':8,
 'new_local_n':32,'new_per_case':8,'new_seed_base':61536001,'new_norms':[.375,.75],
 'new_bases':'C31 train00/train01/train02 preparation states, i modulo3, same centers as C35',
 'response_budget':.002,'kl_budget':.005,'margin_gate':.05,
 'proposal_reproduction_tolerance':0.,'physical_replay_response_tolerance':2e-5,
 'physical_replay_kl_tolerance':2e-5,'physical_replay_margin_tolerance':2e-4,
 'arms':list(ARMS),'primary_arms':['geometry6','selected6'],
 'no_new_training':True,'no_branch_selection':True,'no_threshold_changes':True,
 'new_state_resampling':False,'invalid_preparations_retained':True,
 'online_write_Jacobians':0,'online_write_SVD_calls':0,'online_write_correction_steps':0,
 'online_write_backtracking':0,'teacher_fallback_used':False,'postwrite_response_used_for_proposal':False,
 'offline_state_preparation_uses_external_geometry':True,
 'same_model_chart_access_prompts':True,'fresh_state_seeds':True,'fresh_evaluation':False,
 'independent_model_or_chart_seed':False,'requested_output_bit_supplied':True,
 'internal_parameter_update':False,'controller_dispatch_tested':False,
 'scope':'Prospectively fixed-seed same-model/chart local replication of both C35 frozen composed writers; no refitting or test-selected replacement. Offline geometry and external commitment remain. Not independent cross-model replication or full Line C.'}

def specs():
    return [{'id':c+'__c36_new%02d'%i,'case':c,'base_id':c+'__c31_train%02d'%(i%3),
             'cohort':'new_local','seed':SETTINGS['new_seed_base']+100*r+i,'norm':SETTINGS['new_norms'][i%2]}
            for r,c in enumerate(CASES) for i in range(SETTINGS['new_per_case'])]

def replay_ids():return [c+'__c35_new%02d'%i for c in CASES for i in (0,1)]

def check_sources(root):
    for n,h in read(Path(root)/'C35_SOURCE_PINS.json').items():
        require(sha(Path(root)/'inherited'/n)==h,'Pinned C35 source differs: '+n)

def identity(root):return {'protocol':PROTOCOL,'settings':SETTINGS,'source_hashes':sources(root)}

def snapshot(out,backup=''):
    out=Path(out);path=out.parent/(RESULT+'.zip');tmp=path.with_suffix('.zip.tmp')
    m={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*'))
       if p.is_file() and p.name!='MANIFEST.json' and not p.name.endswith('.tmp') and '__pycache__' not in p.parts}
    # A nested parent MANIFEST is evidence too; only this result's root MANIFEST is excluded.
    for p in out.rglob('MANIFEST.json'):
        if p!=out/'MANIFEST.json':m[p.relative_to(out).as_posix()]=sha(p)
    dump(out/'MANIFEST.json',m)
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for n in m:z.write(out/n,n)
        z.write(out/'MANIFEST.json','MANIFEST.json')
    with zipfile.ZipFile(tmp) as z:verify(z,True)
    os.replace(tmp,path)
    if backup:
        d=Path(backup);d.mkdir(parents=True,exist_ok=True);dest=d/(RESULT+'_'+str(time.time_ns())+'.zip');t=dest.with_suffix('.zip.tmp')
        shutil.copyfile(path,t);require(sha(path)==sha(t),'Drive copy digest differs');os.replace(t,dest)
        good=[]
        for p in sorted(d.glob(RESULT+'_*.zip'),key=lambda p:p.stat().st_mtime_ns,reverse=True):
            try:
                with zipfile.ZipFile(p) as z:verify(z,True);require(zread(z,'identity.json')['protocol']==PROTOCOL,'Other backup protocol')
                good.append(p)
            except Exception:continue
        for p in good[3:]:p.unlink()
        print('DRIVE_BACKUP_OK',dest,flush=True)
    print('RESULT_ARCHIVE='+str(path),flush=True);return path
