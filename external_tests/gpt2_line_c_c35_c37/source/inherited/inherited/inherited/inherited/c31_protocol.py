"""C31: train-data coverage comparison; no new memory writer."""
import shutil
import c30_core as predictor
from c30_core import require, dump, read, zread, sha, verify, sources, CASES, GATES
from pathlib import Path
import hashlib, json, os, time, zipfile
PROTOCOL='GPT2_C31_RESIDUAL_COVERAGE_A100_V1'
PARENT_PROTOCOL='GPT2_C29_FROZEN_DIRECTION_DIAGNOSIS_A100_V1'
RESULT='gpt2_c31_residual_coverage_results'
DRIVE_NAME=PROTOCOL+'_RECOVERY'
STARTS={'xor_00':[0,0,1,1],'xor_01':[0,1,1,0],'xor_10':[1,0,1,0],'xor_11':[1,1,1,1]}
SETTINGS={
 'historical_training_states':76,'new_training_states':48,'prospective_local_development_states':16,
 'historical_bases':12,'model_seed':82931,'output_scale':.07025910168886185,
 'response_budget':.002,'kl_budget':.005,'margin_gate':.05,
 'train_seed_base':31531001,'development_seed_base':41531001,'perturbation_norms':[.375,.75],
 'perturbation':'fixed PCG64 Gaussian, numerical null projection at the original preparation, norm-normalised',
 'null_relative_cutoff':1e-8,'null_absolute_cutoff':1e-10,
 'correction_ridge':1e-6,'correction_max_steps':4,'correction_backtracking':6,
 'preparation_correction_only':True,'post_write_correction_steps':0,
 'sketch_seed':615031,'sketch_tensor_sha256':'c3ba53dfb89515b4f16e4554d0417d3c4c55b85f413f3bbb335f7eeeceaf156c',
 'historical_input_replay_tolerance':1e-4,'historical_endpoint_replay_tolerance':2e-6,
 'predictor_methods':predictor.SETTINGS,'predictor_arms':['historical76','expanded124'],
 'fit_uses_all_finite_requested_training_states_even_if_preparation_invalid':True,
 'no_generation_retries_or_resampling':True,'teacher_reference_for_new_states':False,
 'new_writer_training_steps':0,'new_memory_writer_selected':False,
 'requested_output_bit_supplied':True,'fresh_state_seeds':True,'fresh_evaluation':False,
 'internal_parameter_update':False,'controller_dispatch_tested':False,'LC1_closed':False,
 'archive_every_states':4,
 'scope':'Prospectively seeded same-model/chart development comparison of residual predictors, using fixed C27 one-proposal writes without post-write repair. Offline geometry prepares states. No generated correction or new writing method.'}


def specs(cohort):
 require(cohort in ('new_training','prospective'),'Unknown C31 cohort')
 count=12 if cohort=='new_training' else 4
 seed=SETTINGS['train_seed_base'] if cohort=='new_training' else SETTINGS['development_seed_base']
 return [{'id':case+('__c31_train%02d' if cohort=='new_training' else '__c31_new%02d')%i,
          'case':case,'base_id':case+'__v%d'%(i%3),'cohort':cohort,
          'seed':seed+100*r+i,'norm':SETTINGS['perturbation_norms'][i%2]}
         for r,case in enumerate(CASES) for i in range(count)]


def check_sources(root):
 pins=read(Path(root)/'C29_SOURCE_PINS.json')
 for n,h in pins.items():require(sha(Path(root)/'inherited'/n)==h,'Inherited C29 source changed: '+n)
 recipe=read(Path(root)/'RECIPE_PROVENANCE.json')
 require(sha(Path(root)/'C15_SKETCH_RECIPE_SOURCE.py')==recipe['source_sha256'],'C15 sketch recipe source differs')
 return pins


def snapshot(out, backup=''):
    import shutil
    out = Path(out)
    path = out.parent / (RESULT + '.zip')
    tmp = path.with_suffix('.zip.tmp')
    m = {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob('*'))
         if p.is_file() and p.name != 'MANIFEST.json' and not p.name.endswith('.tmp') and '__pycache__' not in p.parts}
    dump(out / 'MANIFEST.json', m)
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED, compresslevel=3) as z:
        for n in m:
            z.write(out / n, n)
        z.write(out / 'MANIFEST.json', 'MANIFEST.json')
    with zipfile.ZipFile(tmp) as z:
        verify(z, True)
    os.replace(tmp, path)
    if backup:
        d = Path(backup)
        d.mkdir(parents=True, exist_ok=True)
        dest = d / (RESULT + '_' + str(time.time_ns()) + '.zip')
        t = dest.with_suffix('.zip.tmp')
        shutil.copyfile(path, t)
        require(sha(t) == sha(path), 'Backup copy mismatch')
        os.replace(t, dest)
        good = []
        for p in sorted(d.glob(RESULT + '_*.zip'), key=lambda p: p.stat().st_mtime_ns, reverse=True):
            try:
                with zipfile.ZipFile(p) as z:
                    verify(z, True)
                    require(zread(z, 'identity.json')['protocol'] == PROTOCOL, 'Other protocol')
                good.append(p)
            except Exception:
                continue
        for p in good[3:]:
            p.unlink()
        print('DRIVE_BACKUP_OK', dest, flush=True)
    print('RESULT_ARCHIVE=' + str(path), flush=True)
    return path
