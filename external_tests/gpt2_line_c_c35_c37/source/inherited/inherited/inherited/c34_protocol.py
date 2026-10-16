"""Fixed training-derived parameter bases, with explicitly privileged coefficients."""
import hashlib,json,os,time,zipfile,shutil
from pathlib import Path
from c30_core import require,read,dump,sha,zread,verify,sources,CASES,GATES

PROTOCOL='GPT2_C34_LOW_DIMENSIONAL_CONTROL_A100_V1'
PARENT_PROTOCOL='GPT2_C31_RESIDUAL_COVERAGE_A100_V1'
RESULT='gpt2_c34_low_dimensional_control_results'
DRIVE_NAME=PROTOCOL+'_RECOVERY'
ARMS=('raw','common1_oracle','response6_oracle')
STARTS={'xor_00':[0,0,1,1],'xor_01':[0,1,1,0],'xor_10':[1,0,1,0],'xor_11':[1,1,1,1]}
SETTINGS={
 'basis_training_per_route':3,'basis_training_ids':[c+'__c31_train%02d'%i for c in CASES for i in range(3)],
 'basis_training_n':12,'unused_for_basis_training_n':36,'inspected_development_n':16,'total_states':64,
 'chart_dimension':36864,'response_dimension':6,'response_budget':.002,'kl_budget':.005,'margin_gate':.05,
 'relative_singular_cutoff':1e-8,'absolute_singular_cutoff':1e-10,
 'maximum_parameter_increment_norm':.05,'raw_replay_response_tolerance':2e-6,
 'basis_recipe':'For each route stack three raw-written response Jacobians A (18x36864); T=tile(I6,3); B6=A^+ T; b1=A^+ ones(18), algebraically B6 ones(6), evaluated directly to avoid cancellation. Truncated float64 SVD, fixed cutoffs.',
 'basis_contains_readout_derivative_constraints':False,
 'oracle_common_coefficients':'-mean(measured_raw_response_residual)',
 'oracle_response6_coefficients':'-measured_raw_response_residual',
 'increment':'One multiplication by frozen basis; requested float64 increment radial L2 clip to0.05; float32 addition to saved raw endpoint, actual displacement separately reported; no line search or iterative repair.',
 'basis_selection':'Fixed first3 C31 new-training states per route, including failures and invalid preparations; no replacement or success filtering.',
 'evaluated_sets':'12 basis-training +36 other C31 training +16 already inspected C31 development; no fresh state generation.',
 'arms':list(ARMS),'no_selection_among_arms_or_states':True,'no_target_based_checkpoint_selection':True,
 'offline_response_geometry':True,'oracle_uses_measured_raw_endpoint_response':True,
 'correction_free_writer_tested':False,'native_update_tested':False,'LC1_closed':False,
 'fresh_evaluation':False,'internal_parameter_update':False,'controller_dispatch_tested':False,
 'scope':'Offline low-dimensional oracle response-correction mechanism comparison on64 archived C31 states. Fixed bases, statewise privileged coefficients, actual finite-endpoint gates. Not a learned writer or online correction-free acceptance result.'}

def check_sources(root):
    root=Path(root);pins=read(root/'C31_SOURCE_PINS.json')
    for n,h in pins.items():require(sha(root/'inherited'/n)==h,'Original C31 source changed: '+n)

def group(spec):
    if spec['cohort']=='prospective':return 'inspected_development'
    require(spec['cohort']=='new_training','Unexpected state cohort')
    return 'basis_training' if spec['id'] in SETTINGS['basis_training_ids'] else 'other_training'

def snapshot(out,backup=''):
    out=Path(out);path=out.parent/(RESULT+'.zip');tmp=path.with_suffix('.zip.tmp')
    m={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*'))
       if p.is_file() and p.name!='MANIFEST.json' and not p.name.endswith('.tmp') and '__pycache__' not in p.parts}
    dump(out/'MANIFEST.json',m)
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for n in m:z.write(out/n,n)
        z.write(out/'MANIFEST.json','MANIFEST.json')
    with zipfile.ZipFile(tmp) as z:verify(z,True)
    os.replace(tmp,path)
    if backup:
        d=Path(backup);d.mkdir(parents=True,exist_ok=True);dest=d/(RESULT+'_'+str(time.time_ns())+'.zip');t=dest.with_suffix('.zip.tmp')
        shutil.copyfile(path,t);require(sha(t)==sha(path),'Backup copy differs');os.replace(t,dest)
        good=[]
        for p in sorted(d.glob(RESULT+'_*.zip'),key=lambda p:p.stat().st_mtime_ns,reverse=True):
            try:
                with zipfile.ZipFile(p) as z:
                    verify(z,True);require(zread(z,'identity.json')['protocol']==PROTOCOL,'Other protocol')
                good.append(p)
            except Exception:continue
        for p in good[3:]:p.unlink()
        print('DRIVE_BACKUP_OK',dest,flush=True)
    print('RESULT_ARCHIVE='+str(path),flush=True);return path
