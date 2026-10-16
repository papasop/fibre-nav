"""Frozen affine coefficient heads with offline geometric features."""
from c30_core import require,read,dump,sha,zread,verify,sources,CASES,GATES,folds
from pathlib import Path
import hashlib,json,os,time,zipfile,shutil
PROTOCOL='GPT2_C35_PREWRITE_COEFFICIENT_A100_V1'
PARENT_PROTOCOL='GPT2_C34_LOW_DIMENSIONAL_CONTROL_A100_V1'
RESULT='gpt2_c35_prewrite_coefficient_results'
DRIVE_NAME=PROTOCOL+'_RECOVERY'
ARMS=('raw','geometry6','selected6')
STARTS={'xor_00':[0,0,1,1],'xor_01':[0,1,1,0],'xor_10':[1,0,1,0],'xor_11':[1,1,1,1]}
SETTINGS={
 'prior_basis_states':[c+'__c31_train%02d'%i for c in CASES for i in range(3)],
 'prior_basis_n':12,'calibration_n':36,'calibration_per_route':9,'training_report_n':48,
 'inspected_n':16,'new_local_n':16,'new_seed_base':51535001,'new_norms':[.375,.75],
 'new_bases':'C31 train00/train01/train02 preparation states per route, i modulo3; not original C31 v0/v1/v2 centers',
 'response_budget':.002,'kl_budget':.005,'margin_gate':.05,'output_scale':.07025910168886185,
 'maximum_compensation_increment_norm':.05,
 'geometry_feature':'g=mean_training_basis_response + mean_training_basis_J @ (uncommitted_raw_point - mean_training_basis_raw_point)',
 'predictor_features':'concat(g/0.002, prewrite_response_residual/0.002),12 coordinates; no measured postwrite residual',
 'residual_calibration':'predict_e=g+dmean+standardise(features)@beta; ridge on measured training targets minus g',
 'deployment_compilation':'Offline absorb geometry features, standardisation and calibration into fixed affine coefficient weights on uncommitted raw coordinates and prewrite response; no runtime derivative object or solve.',
 'ridge_grid':[1e-6,1e-4,.01,1.,100.],'cv_folds':3,
 'CV_unit':'individual calibration preparation states; known base preparations can span folds',
 'CV_scope':'C34 prior12 geometric references fixed; only other36 C31 training states enter calibration CV; prior states never appear as validation targets',
 'selection_loss':'six-coordinate prediction MSE/response_budget squared; training calibration folds only',
 'tie_rule':'primary selected6 chooses minimum calibration-CV loss among5 ridge heads, ridge ascending; geometry is a separately reported control, never a test-selected replacement',
 'std_active_floor':1e-8,'std_scale_floor':1e-6,
 'preparation_null_relative_cutoff':1e-8,'preparation_null_absolute_cutoff':1e-10,
 'preparation_correction_max_steps':4,'preparation_correction_backtracking':6,
 'one_proposal_per_arm':True,'intermediate_raw_point_is_arithmetic_only':True,
 'postwrite_response_for_proposal':False,'online_write_Jacobians':0,'online_write_SVD_calls':0,
 'online_write_correction_steps':0,'online_write_backtracking':0,'teacher_fallback_used':False,
 'offline_geometry_inherited_and_preparation_uses_geometry':True,
 'fresh_state_seeds':True,'fresh_evaluation':False,'independent_model_or_chart_seed':False,
 'requested_output_bit_supplied':True,'internal_parameter_update':False,'controller_dispatch_tested':False,
 'scope':'Same-model/chart development of a frozen composed writer using prewrite geometric features and learned affine residual calibration. C34 bases fixed; final proposal committed once per arm. New16 seeded states generated after freeze, no resampling or test checkpoint selection. External runtime and offline geometry remain.'}

def specs():
    return [{'id':c+'__c35_new%02d'%i,'case':c,'base_id':c+'__c31_train%02d'%(i%3),
             'cohort':'new_local','seed':SETTINGS['new_seed_base']+100*r+i,'norm':SETTINGS['new_norms'][i%2]}
            for r,c in enumerate(CASES) for i in range(4)]

def check_sources(root):
    for n,h in read(Path(root)/'C34_SOURCE_PINS.json').items():require(sha(Path(root)/'inherited'/n)==h,'Pinned patched C34 source differs: '+n)

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
                with zipfile.ZipFile(p) as z:verify(z,True);require(zread(z,'identity.json')['protocol']==PROTOCOL,'Other protocol')
                good.append(p)
            except Exception:continue
        for p in good[3:]:p.unlink()
        print('DRIVE_BACKUP_OK',dest,flush=True)
    print('RESULT_ARCHIVE='+str(path),flush=True);return path
