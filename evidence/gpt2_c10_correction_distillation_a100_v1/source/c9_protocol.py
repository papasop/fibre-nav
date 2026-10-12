"""C9 four-input training, then frozen-writer local state generalisation."""
import hashlib,io,json,os,shutil,time,zipfile
from pathlib import Path
from c7_protocol import sha,verify_open_zip,verified_zip,CASES,STATES,target_from_reads,endpoint_gate,episode_gate,REVISION,C6_SHA
PROTOCOL='GPT2_C9_STATE_GENERALISATION_A100_V1'
PARENT_PROTOCOL='GPT2_C8_EXTRAPOLATION_DIAGNOSIS_A100_V1'
RESULT='gpt2_c9_state_generalisation_results'
C8_SHA='92ce08de294a53a591920f8f52f15d2a1146c6d97b04e57e689221145fe40997'
SETTINGS={'response_budget':.002,'kl_budget':.005,'margin_gate':.05,
 'warm_steps':2000,'warm_lr':.001,'functional_steps':240,'functional_lr':.0001,
 'head_width':32,'head_seed':99109,'head_weight_decay':0.,'gradient_clip':5.,'checkpoint_every':25,
 'warm_archive_every':500,'functional_archive_every':120,
 'loss_weights':{'parameter_mse':1.,'read':.1,'response':.01,'KL':.01},
 'correction_ridge':1e-6,'correction_max_steps':8,'correction_backtracking':6,'correction_checkpoint_every':2,
 'test_norms':[.375,.75],'test_seed_base':109031,'feature_identity_tolerance':1e-4,
 'training_states':'all twelve previously inspected C5 states; all four XOR cases',
 'test_generation':'fixed random directions projected using original preparation response Jacobian saved in C8; no resampling',
 'teacher_reference':'same fixed perturbation applied to original teacher, then response correction; diagnostic positive control only',
 'selection':'fixed final functional checkpoint, no test-based checkpoint selection',
 'normalisation':'training twelve states only','requested_output_bit_in_features':True,
 'fresh_state_seeds':True,'fresh_evaluation':False,'independent_model_seed':False,
 'internal_parameter_update':False,'controller_dispatch_tested':False}

def test_specs():
    return [{'id':case+'__c9_new%d'%i,'case':case,'seed':SETTINGS['test_seed_base']+100*c+i,
             'norm':norm,'cohort':'new_state_development'}
            for c,case in enumerate(CASES) for i,norm in enumerate(SETTINGS['test_norms'])]

def decision(training,new_rows):
    if len(training)!=12 or len(new_rows)!=8:raise ValueError('Fixed C9 denominators required')
    if not all(r.get('state_usable',False) for r in new_rows):return 'C9_NEW_STATE_REFERENCES_INCONCLUSIVE'
    if not all(r.get('corrected',{}).get('episode_pass',False) for r in training):return 'C9_TRAINING_FUNCTION_FAILURE'
    return 'C9_LOCAL_STATE_GENERALISATION_'+('PASS' if all(r.get('corrected',{}).get('episode_pass',False) for r in new_rows) else 'FAIL')

def source_hashes(root):
    return {p.name:sha(p) for p in Path(root).iterdir() if p.is_file() and p.name!='build_package.py'}

def validate_c8(path):
    if sha(path)!=C8_SHA:raise ValueError('C9 V1 requires the exact inspected complete C8 result ZIP')
    with zipfile.ZipFile(path) as z:
        ident=verify_open_zip(z,PARENT_PROTOCOL);s=json.loads(z.read('SUMMARY.json'))
        if not s.get('complete') or s.get('n_prepared_states')!=12 or s.get('n_saved_candidates')!=120 or not s.get('positive_references_all_pass'):
            raise ValueError('Complete C8 diagnosis with original tensors required')
        pins=json.loads((Path(__file__).parent/'PARENT_IDENTITY_PINS.json').read_text())
        if ident['source_hashes']!=pins['C8_source_hashes']:raise ValueError('Original C8 source identity changed')
        b=z.read('parent_c7.zip')
        if hashlib.sha256(b).hexdigest()!=pins['c7_sha256']:raise ValueError('Original C7 identity mismatch')
        for sid in STATES:
            r=json.loads(z.read('states/'+sid+'/COMPLETE.json'))
            if not r.get('usable') or hashlib.sha256(z.read('states/'+sid+'/jacobian.pt')).hexdigest()!=r['jacobian_sha256']:
                raise ValueError('Original C8 reference/Jacobian mismatch')
        return ident

def checkpoint_pointer(z,base,p,max_steps):
    name=p['checkpoint']
    if Path(name).name!=name or '\\' in name or not name.endswith('.pt'):raise ValueError('Unsafe checkpoint pointer')
    if hashlib.sha256(z.read(base+name)).hexdigest()!=p['checkpoint_sha256']:raise ValueError('Actual checkpoint digest mismatch')
    if not isinstance(p['steps'],int) or not 0<=p['steps']<=max_steps:raise ValueError('Invalid checkpoint progress')
    return p['steps']

def resume_score(path):
    ident=verified_zip(path,PROTOCOL)
    if ident.get('settings')!=SETTINGS or ident.get('parent_c8_sha256')!=C8_SHA or ident.get('source_hashes')!=source_hashes(Path(__file__).parent):
        raise ValueError('Different C9 parent/source/settings')
    with zipfile.ZipFile(path) as z:
        if hashlib.sha256(z.read('parent_c8.zip')).hexdigest()!=C8_SHA:raise ValueError('Parent C8 corrupt')
        progress=[]
        for stage,max_steps in [('warm',2000),('functional',240)]:
            base='heads/'+stage+'/';name=base+'PROGRESS.json'
            progress.append(checkpoint_pointer(z,base,json.loads(z.read(name)),max_steps) if name in z.namelist() else 0)
        new=sum(n.startswith('evaluation/new/') and n.endswith('/COMPLETE.json') for n in z.namelist())
        train=sum(n.startswith('evaluation/training/') and n.endswith('/COMPLETE.json') for n in z.namelist())
        return (int('SUMMARY.json' in z.namelist()),new,train,*progress)

def snapshot(out,drive=''):
    out=Path(out)
    if not (out/'identity.json').exists():return None
    m={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='MANIFEST.json' and not p.name.endswith('.tmp')}
    t=out/'MANIFEST.json.tmp';t.write_text(json.dumps(m,indent=2));os.replace(t,out/'MANIFEST.json')
    dest=out.parent/(RESULT+'.zip');temp=dest.with_suffix('.partial')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
        for n in list(m)+['MANIFEST.json']:z.write(out/n,n,compress_type=zipfile.ZIP_STORED if n.endswith('.zip') else zipfile.ZIP_DEFLATED)
    verified_zip(temp,PROTOCOL);os.replace(temp,dest)
    if drive:
        d=Path(drive);d.mkdir(parents=True,exist_ok=True);q=d/(RESULT+'_'+str(time.time_ns())+'.zip');tmp=q.with_suffix('.partial')
        shutil.copyfile(dest,tmp)
        if sha(tmp)!=sha(dest):raise RuntimeError('Drive copy digest mismatch')
        verified_zip(tmp,PROTOCOL);os.replace(tmp,q);print('DRIVE_BACKUP_OK',q,flush=True)
        valid=[]
        for old in sorted(d.glob(RESULT+'_*.zip')):
            try:verified_zip(old,PROTOCOL);valid.append(old)
            except Exception:continue
        for old in valid[:-3]:old.unlink()
    print('RESULT_ARCHIVE='+str(dest),flush=True);return dest
