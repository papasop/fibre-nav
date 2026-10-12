"""C8 diagnostics of already inspected C7 development states, no new training."""
import hashlib,io,json,os,shutil,time,zipfile
from pathlib import Path
import c7_protocol as parent
from c7_protocol import sha,verified_zip,verify_open_zip,CASES,STATES,fold_states,target_from_reads,endpoint_gate,episode_gate,REVISION,C6_SHA,ARMS
PROTOCOL='GPT2_C8_EXTRAPOLATION_DIAGNOSIS_A100_V1'
RESULT='gpt2_c8_extrapolation_diagnosis_results'
SETTINGS={'response_budget':.002,'kl_budget':.005,'margin_gate':.05,
 'span_rtol':1e-8,'span_atol':1e-10,'jacobian_states':12,'response_rows':6,'exact_readout_rows':4,
 'jacobian_dtype':'autograd float32 then diagnostic accumulation float64',
 'feature_identity_tolerance':1e-4,'snapshot_every_candidates':24,
 'new_training_steps':0,'new_correction_steps':0,'candidate_regeneration':False,
 'scope':'all archived C7 frozen-baseline and eligible paired candidates; training and excluded development only',
 'teacher_use':'diagnosis and positive reference only; never proposal replacement',
 'fresh_evaluation':False,'internal_parameter_update':False,'controller_dispatch_tested':False}

def candidate_jobs(eligible):
    jobs=[]
    for fold in CASES:
        for sid in fold_states(fold)+fold_states(fold,False):
            jobs.append({'family':'frozen_baseline','fold':fold,'state':sid,'arm':'baseline',
                         'member':'baseline/'+fold+'/'+sid})
    for fold in eligible:
        for sid in fold_states(fold)+fold_states(fold,False):
            for arm in ARMS:
                jobs.append({'family':'paired','fold':fold,'state':sid,'arm':arm,
                             'member':'evaluation/'+fold+'/'+sid+'/'+arm})
    for j in jobs:
        j['cohort']='training' if j['state'] in fold_states(j['fold']) else 'excluded_development'
        j['key']=j['family']+'__'+j['fold']+'__'+j['state']+'__'+j['arm']
    return jobs

def validate_c7(path):
    with zipfile.ZipFile(path) as z:
        ident=verify_open_zip(z,parent.PROTOCOL);s=json.loads(z.read('SUMMARY.json'))
        if ident.get('settings')!=parent.SETTINGS or ident.get('parent_c6_sha256')!=C6_SHA:
            raise ValueError('Different C7 settings or C6 parent')
        if not s.get('complete') or s.get('decision')!='C7_DEVELOPMENT_FUNCTIONAL_GATE_COMPARISON_COMPLETE' or not s.get('positive_references_all_pass'):
            raise ValueError('Complete functional-gate C7 result required, not source/launcher ZIP')
        pins=json.loads((Path(__file__).parent/'PARENT_SOURCE_PINS.json').read_text())
        for n,h in pins.items():
            if hashlib.sha256(z.read('source/'+n)).hexdigest()!=h:raise ValueError('Different original C7 source: '+n)
        b=z.read('parent_c6.zip')
        if hashlib.sha256(b).hexdigest()!=C6_SHA:raise ValueError('Original C6 identity mismatch')
        with zipfile.ZipFile(io.BytesIO(b)) as c6:
            c6id=verify_open_zip(c6,parent.C6_PROTOCOL)
            cb=c6.read('parent_c5_capsule.zip')
            if hashlib.sha256(cb).hexdigest()!=c6id.get('parent_capsule_sha256'):raise ValueError('Capsule hash mismatch')
            with zipfile.ZipFile(io.BytesIO(cb)) as cap:parent.validate_capsule_open(cap)
        eligible=[];base=[];paired=[]
        for fold in CASES:
            g=json.loads(z.read('training_gate/'+fold+'.json'))
            episodes=[json.loads(z.read('baseline/'+fold+'/'+sid+'/endpoint.json')) for sid in fold_states(fold)]
            if bool(g['eligible'])!=parent.functional_fit_gate(episodes):raise ValueError('Training eligibility inconsistency')
            if g['eligible']:eligible.append(fold)
        for j in candidate_jobs(eligible):
            for n in ['raw.pt','endpoint.pt','raw.json','endpoint.json']:
                if j['member']+'/'+n not in z.namelist():raise ValueError('Saved candidate missing: '+j['key'])
            ep=json.loads(z.read(j['member']+'/endpoint.json'))
            if j['family']=='frozen_baseline':base.append({'cohort':j['cohort'],'state':j['state'],'endpoint':ep})
        for fold in eligible:
            for sid in fold_states(fold)+fold_states(fold,False):
                r=json.loads(z.read('evaluation/'+fold+'/'+sid+'/COMPLETE.json'))
                cohort='training' if sid in fold_states(fold) else 'excluded_development'
                if r['fold']!=fold or r['state']!=sid or r['cohort']!=cohort:raise ValueError('Paired row identity mismatch')
                for arm in ARMS:
                    if r[arm]!=json.loads(z.read('evaluation/'+fold+'/'+sid+'/'+arm+'/endpoint.json')):
                        raise ValueError('Paired endpoint record mismatch')
                    basepath='heads/'+fold+'/'+arm+'/'
                    p=json.loads(z.read(basepath+'PROGRESS.json'))
                    name=p['checkpoint']
                    if Path(name).name!=name or '\\' in name or p['steps']!=200 or hashlib.sha256(z.read(basepath+name)).hexdigest()!=p['checkpoint_sha256']:
                        raise ValueError('Actual final writer checkpoint mismatch')
                paired.append(r)
        if eligible!=s.get('eligible_folds') or s.get('n_completed')!=len(paired):raise ValueError('C7 summary denominator mismatch')
        for kind in ['training','excluded_development']:
            if s['frozen_baseline'][kind]!=parent.counts_single([r['endpoint'] for r in base if r['cohort']==kind]):raise ValueError('Baseline count mismatch')
            if s['paired_cohorts'][kind]!=parent.matched_counts([r for r in paired if r['cohort']==kind]):raise ValueError('Paired count mismatch')
        if s['persisted_committed_new_head_steps']!=len(eligible)*400:raise ValueError('Head step denominator mismatch')
        return {'identity':ident,'summary':s,'eligible':eligible,'jobs':candidate_jobs(eligible),
                'sha256':sha(path),'C7_whole_file_digest_recorded_at_C8_start':True,
                'preknown_C7_whole_file_digest':False,'original_C6_digest_pinned':True}

def source_hashes(root):
    return {p.name:sha(p) for p in Path(root).iterdir() if p.is_file() and p.name!='build_package.py'}

def resume_score(path):
    ident=verified_zip(path,PROTOCOL)
    if ident.get('settings')!=SETTINGS or ident.get('source_hashes')!=source_hashes(Path(__file__).parent):raise ValueError('Different C8 source/settings')
    with zipfile.ZipFile(path) as z:
        if hashlib.sha256(z.read('parent_c7.zip')).hexdigest()!=ident['parent_c7_sha256']:raise ValueError('Parent changed')
        return (int('SUMMARY.json' in z.namelist()),sum(n.startswith('candidates/') and n.endswith('.json') for n in z.namelist()),
                sum(n.startswith('states/') and n.endswith('/COMPLETE.json') for n in z.namelist()))

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
