"""Frozen C35 compositions around fixed, previously inspected C36 preparation centres."""
from pathlib import Path
import hashlib,json,os,shutil,time,zipfile
from c30_core import require,read,dump,sha,zread,verify,sources,CASES,GATES
from c35_protocol import ARMS,STARTS
PROTOCOL='GPT2_C37_CENTRE_TRANSFER_A100_V1'
PARENT_PROTOCOL='GPT2_C36_FROZEN_LOCAL_REPLICATION_A100_V1'
RESULT='gpt2_c37_centre_transfer_results'
DRIVE_NAME=PROTOCOL+'_RECOVERY'
SETTINGS={
 'expected_parent_freeze_sha256':'6a694cf429590e007420a642cfed951a62c3253336f41df3faa9815ab4da8131',
 'expected_writer_freeze_sha256':'d95a6d84f312261d355bf2d584c878eff2caeb2b7d19d38d92e0bc8f1c1faec8',
 'parent_expected_decision':'C36_BOTH_FROZEN_WRITERS_LOCAL_REPLICATION_PASS',
 'new_n':32,'centres_per_case':4,'offspring_per_centre':2,'seed_base':71537001,'norms':[.375,.75],
 'centres':'C36 new00,new01,new02,new03 prepared.pt per XOR class, fixed by index, not written endpoints; inspected but never used to refit C35',
 'minimum_prepared_distance_fraction':.5,'response_budget':.002,'KL_budget':.005,'margin_gate':.05,
 'preparation_null_relative_cutoff':1e-8,'preparation_null_absolute_cutoff':1e-10,
 'offline_preparation_geometry':True,'online_write_Jacobians':0,'online_write_SVD_calls':0,
 'online_write_correction_steps':0,'online_write_backtracking':0,'teacher_fallback_used':False,
 'new_training_steps':0,'branch_or_checkpoint_selection':False,'resampling':False,'invalid_states_retained':True,
 'same_model_chart_access_prompts':True,'requested_output_bit_supplied':True,
 'fresh_state_seeds':True,'fresh_evaluation':False,'centres_previously_inspected':True,
 'independent_model_or_chart_replication':False,'internal_parameter_update':False,'controller_dispatch_tested':False,
 'scope':'Development-only transfer of frozen composed writers to 32 fixed-seed perturbations around16 inspected C36 preparation centres, with no refitting or branch selection. Same model/chart and four trained input classes; external preparation geometry and runtime commitment remain.'}

def centres():
 return [{'id':c+'__c36_new%02d'%i,'case':c} for c in CASES for i in range(4)]
def specs():
 return [{'id':c+'__c37_new%02d'%i,'case':c,'centre_id':c+'__c36_new%02d'%(i//2),
          'cohort':'new_local','seed':SETTINGS['seed_base']+100*r+i,'norm':SETTINGS['norms'][i%2]}
         for r,c in enumerate(CASES) for i in range(8)]
def check_sources(root):
 for n,h in read(Path(root)/'C36_SOURCE_PINS.json').items():require(sha(Path(root)/'inherited'/n)==h,'Pinned C36 FIX1 source differs: '+n)
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
