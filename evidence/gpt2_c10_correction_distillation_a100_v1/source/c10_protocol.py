"""C10: training-only correction distillation; raw write acceptance with no repair."""
import hashlib,io,json,os,shutil,time,zipfile
from pathlib import Path
from c9_protocol import sha,verify_open_zip,verified_zip,CASES,STATES,target_from_reads,endpoint_gate,episode_gate,REVISION
PROTOCOL='GPT2_C10_CORRECTION_DISTILLATION_A100_V1'
PARENT_PROTOCOL='GPT2_C9_STATE_GENERALISATION_A100_V1'
RESULT='gpt2_c10_correction_distillation_results'
C9_SHA='f0f5abd2197311438bf23588dbcb1bedef1649380873b04f86b5e7b1340fce3b'
HEAD_SHA='835793a66ec435a8cae6c879b65351e808152919af1a000607426404f6c955ab'
ARMS=('frozen_c9','distilled')
SETTINGS={'response_budget':.002,'kl_budget':.005,'margin_gate':.05,
 'distill_steps':1000,'distill_lr':.0001,'functional_steps':360,'functional_lr':.00001,
 'head_width':32,'head_weight_decay':0.,'gradient_clip':1.,'checkpoint_every':25,
 'distill_archive_every':500,'functional_archive_every':120,
 'loss_weights':{'parameter_mse':1.,'read':.1,'response':1.,'KL':.1},
 'response_loss_scale':.002,'response_loss':'mean smooth-L1 of response displacement divided by response budget, beta=1',
 'functional_schedule':'four training states per optimiser step, one from each input class; variants cycle step modulo three',
 'normalisation':'reuse C9 training-only normalisation unchanged',
 'supervision':'C9 corrected endpoints of twelve training states only; no C9 new-state endpoints',
 'selection':'fixed final functional checkpoint; no test-based checkpoint selection',
 'test_norms':[.375,.75],'test_seed_base':120031,
 'correction_ridge':1e-6,'correction_max_steps':8,'correction_backtracking':6,'correction_checkpoint_every':2,
 'dataset_preparation_geometry':'original saved C8 preparation J, SVD perturbation, bounded response correction before writing',
 'writer_evaluation_online_Jacobian':False,'writer_evaluation_online_SVD':False,'writer_evaluation_response_correction':False,
 'feature_identity_tolerance':1e-4,'requested_output_bit_in_features':True,
 'fresh_evaluation':False,'fresh_state_seeds':True,'independent_model_seed':False,
 'internal_parameter_update':False,'controller_dispatch_tested':False}

def test_specs():
    return [{'id':c+'__c10_new%d'%i,'case':c,'seed':SETTINGS['test_seed_base']+100*k+i,'norm':n}
            for k,c in enumerate(CASES) for i,n in enumerate(SETTINGS['test_norms'])]

def functional_indices(step):return [3*c+step%3 for c in range(4)]

def decision(training,new_rows):
    if len(training)!=12 or len(new_rows)!=8:raise ValueError('Fixed C10 denominators required')
    if not all(r.get('state_usable',False) for r in new_rows):return 'C10_NEW_STATE_REFERENCES_INCONCLUSIVE'
    if not all(r.get('distilled',{}).get('episode_pass',False) for r in training):return 'C10_RAW_TRAINING_FUNCTION_FAILURE'
    return 'C10_RAW_LOCAL_STATE_GENERALISATION_'+('PASS' if all(r.get('distilled',{}).get('episode_pass',False) for r in new_rows) else 'FAIL')

def source_hashes(root):
    return {p.name:sha(p) for p in Path(root).iterdir() if p.is_file() and p.name!='build_package.py'}

def validate_c9(path):
    if sha(path)!=C9_SHA:raise ValueError('C10 V1 requires the exact inspected complete C9 result ZIP')
    with zipfile.ZipFile(path) as z:
        ident=verify_open_zip(z,PARENT_PROTOCOL);s=json.loads(z.read('SUMMARY.json'));fr=json.loads(z.read('WRITER_FROZEN.json'))
        pins=json.loads((Path(__file__).parent/'C9_IDENTITY_PINS.json').read_text())
        if ident['source_hashes']!=pins['C9_source_hashes'] or ident['parent_c8_sha256']!=pins['parent_c8_sha256']:raise ValueError('Original C9 source/parent mismatch')
        if not s.get('complete') or s.get('decision')!='C9_LOCAL_STATE_GENERALISATION_PASS' or s.get('persisted_committed_head_steps')!=2240:
            raise ValueError('Complete passing C9 with actual head checkpoints required')
        if s['writer_head_sha256']!=HEAD_SHA or fr['head_sha256']!=HEAD_SHA:raise ValueError('Original frozen C9 head identity mismatch')
        p=json.loads(z.read('heads/functional/PROGRESS.json'));name=p['checkpoint']
        if Path(name).name!=name or '\\' in name or p['steps']!=240:raise ValueError('Invalid parent checkpoint pointer')
        if fr['selected_checkpoint']!='heads/functional/'+name or hashlib.sha256(z.read(fr['selected_checkpoint'])).hexdigest()!=p['checkpoint_sha256']:
            raise ValueError('Original final C9 checkpoint missing or corrupt')
        for sid in STATES:
            base='evaluation/training/'+sid+'/'
            row=json.loads(z.read(base+'COMPLETE.json'))
            if not row.get('state_usable') or not row['corrected']['episode_pass']:raise ValueError('Training supervision endpoint failed')
            for n in ['raw.pt','endpoint.pt']:
                if base+n not in z.namelist():raise ValueError('Actual C9 training state missing')
        if hashlib.sha256(z.read('parent_c8.zip')).hexdigest()!=pins['parent_c8_sha256']:raise ValueError('Original C8 corrupt')
        return ident

def checkpoint_pointer(z,base,p,max_steps):
    name=p['checkpoint']
    if Path(name).name!=name or '\\' in name or not name.endswith('.pt'):raise ValueError('Unsafe checkpoint pointer')
    if hashlib.sha256(z.read(base+name)).hexdigest()!=p['checkpoint_sha256']:raise ValueError('Checkpoint digest mismatch')
    if not isinstance(p['steps'],int) or not 0<=p['steps']<=max_steps:raise ValueError('Invalid training progress')
    return p['steps']

def resume_score(path):
    ident=verified_zip(path,PROTOCOL)
    if ident.get('settings')!=SETTINGS or ident.get('parent_c9_sha256')!=C9_SHA or ident.get('source_hashes')!=source_hashes(Path(__file__).parent):raise ValueError('Different C10 source/parent/settings')
    with zipfile.ZipFile(path) as z:
        if hashlib.sha256(z.read('parent_c9.zip')).hexdigest()!=C9_SHA:raise ValueError('Parent C9 corrupt')
        steps=[]
        for stage,maximum in [('distill',1000),('functional',360)]:
            base='heads/'+stage+'/';n=base+'PROGRESS.json'
            steps.append(checkpoint_pointer(z,base,json.loads(z.read(n)),maximum) if n in z.namelist() else 0)
        names=z.namelist();new=sum(n.startswith('evaluation/new/') and n.endswith('/COMPLETE.json') for n in names)
        train=sum(n.startswith('evaluation/training/') and n.endswith('/COMPLETE.json') for n in names)
        return (int('SUMMARY.json' in names),new,train,*steps)

def snapshot(out,drive=''):
    out=Path(out)
    if not (out/'identity.json').exists():return None
    m={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='MANIFEST.json' and not p.name.endswith('.tmp')}
    tmp=out/'MANIFEST.json.tmp';tmp.write_text(json.dumps(m,indent=2));os.replace(tmp,out/'MANIFEST.json')
    dest=out.parent/(RESULT+'.zip');temp=dest.with_suffix('.partial')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
        for n in list(m)+['MANIFEST.json']:z.write(out/n,n,compress_type=zipfile.ZIP_STORED if n.endswith('.zip') else zipfile.ZIP_DEFLATED)
    verified_zip(temp,PROTOCOL);os.replace(temp,dest)
    if drive:
        d=Path(drive);d.mkdir(parents=True,exist_ok=True);q=d/(RESULT+'_'+str(time.time_ns())+'.zip');t=q.with_suffix('.partial')
        shutil.copyfile(dest,t)
        if sha(t)!=sha(dest):raise RuntimeError('Drive digest mismatch')
        verified_zip(t,PROTOCOL);os.replace(t,q);print('DRIVE_BACKUP_OK',q,flush=True)
        valid=[]
        for old in sorted(d.glob(RESULT+'_*.zip')):
            try:verified_zip(old,PROTOCOL);valid.append(old)
            except Exception:continue
        for old in valid[:-3]:old.unlink()
    print('RESULT_ARCHIVE='+str(dest),flush=True);return dest
