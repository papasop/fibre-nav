"""C7: development revision of the training gate; C6 is not relabelled."""
import hashlib,io,json,math,zipfile
from pathlib import Path
from c6_protocol import sha,verify_open_zip,verified_zip,CASES,STATES,fold_states,target_from_reads,endpoint_gate,episode_gate,REVISION,C5_SHA,validate_capsule_open
PROTOCOL='GPT2_C7_FUNCTIONAL_GATE_A100_V1'
C6_PROTOCOL='GPT2_C6_STAGED_WRITER_A100_V1'
C6_SHA='98b1429a5a33e5dabf59cd23862773b6ab71f39b0fb7611df9fb3aeb9c6b5ebd'
ARMS=('parameter_mse','functional')
SETTINGS={'margin_gate':.05,'response_budget':.002,'kl_budget':.005,
 'correction_ridge':1e-6,'correction_max_steps':8,'correction_backtracking':6,'correction_checkpoint_every':2,
 'head_steps':200,'head_width':32,'head_lr':.0001,'head_weight_decay':0.,'gradient_clip':5.,
 'head_checkpoint_every':25,'archive_training_every':200,
 'loss_weights':{'parameter_mse':1.,'read':.1,'response':.01,'KL':.01},
 'response_surrogate_scale':.1,'read_surrogate_margin':1.,
 'training_eligibility':'all nine actually evaluated training episodes pass unchanged endpoint gates; MSE diagnostic only',
 'eligibility_decided_before_excluded_input_evaluation':True,
 'baseline':'all four saved C6 1200-step heads, training and excluded inputs, no additional training',
 'initial_state':'actual fixed C6 warm head checkpoint at step 1200; no warm retraining',
 'paired_start':'same saved head and normalisation; fresh identical AdamW state',
 'sample_schedule':'full-batch nine-state MSE every step; functional state cycles by step modulo nine',
 'development_gate_revision_after_C6_observation':True,'C6_decision_relabelled':False,
 'fresh_evaluation':False,'internal_parameter_update':False,'controller_dispatch_tested':False}

def arm_order(i):return ARMS if i%2==0 else tuple(reversed(ARMS))
def functional_fit_gate(episodes):return bool(len(episodes)==9 and all(v.get('episode_pass',False) for v in episodes))
def counts_single(rows):return {'n':len(rows),'passed':sum(bool(r.get('episode_pass',False)) for r in rows)}
def matched_counts(rows):
 a=[bool(r['parameter_mse'].get('episode_pass',False)) for r in rows];b=[bool(r['functional'].get('episode_pass',False)) for r in rows]
 return {'n':len(rows),'parameter_mse_passed':sum(a),'functional_passed':sum(b),
  'both_passed':sum(x and y for x,y in zip(a,b)),'mse_only':sum(x and not y for x,y in zip(a,b)),
  'functional_only':sum(not x and y for x,y in zip(a,b)),'both_failed':sum(not x and not y for x,y in zip(a,b))}

def validate_c6(path):
 if sha(path)!=C6_SHA:raise ValueError('C7 V1 requires the exact inspected complete C6 result ZIP')
 with zipfile.ZipFile(path) as z:
  ident=verify_open_zip(z,C6_PROTOCOL);s=json.loads(z.read('SUMMARY.json'))
  if not s.get('complete') or s.get('persisted_committed_head_steps')!=4800 or not s.get('positive_references_all_pass'):
   raise ValueError('Complete C6 with four saved 1200-step heads required')
  b=z.read('parent_c5_capsule.zip')
  if hashlib.sha256(b).hexdigest()!=ident.get('parent_capsule_sha256'):raise ValueError('Parent capsule digest mismatch')
  with zipfile.ZipFile(io.BytesIO(b)) as cap:validate_capsule_open(cap)
  for fold in CASES:
   base='heads/'+fold+'/warm/';p=json.loads(z.read(base+'PROGRESS.json'))
   if p['steps']!=1200 or hashlib.sha256(z.read(base+p['checkpoint'])).hexdigest()!=p['checkpoint_sha256']:
    raise ValueError('Actual C6 warm head identity mismatch')
   spec=json.loads(z.read(base+'FIT_RESULT.json'))
   if spec['steps']!=1200:raise ValueError('Parent fit record is not the saved final head')
 return ident

def resume_score(path):
 ident=verified_zip(path,PROTOCOL)
 if ident.get('parent_c6_sha256')!=C6_SHA or ident.get('settings')!=SETTINGS:raise ValueError('Different C7 protocol/settings')
 with zipfile.ZipFile(path) as z:
  if hashlib.sha256(z.read('parent_c6.zip')).hexdigest()!=C6_SHA:raise ValueError('Original C6 missing/corrupt')
  names=set(z.namelist());steps=0
  for n in names:
   if n.startswith('heads/') and n.endswith('/PROGRESS.json'):
    p=json.loads(z.read(n));name=p['checkpoint']
    if Path(name).name!=name or '\\' in name or not name.endswith('.pt'):raise ValueError('Unsafe checkpoint pointer')
    if hashlib.sha256(z.read(n.rsplit('/',1)[0]+'/'+name)).hexdigest()!=p['checkpoint_sha256']:raise ValueError('Checkpoint digest mismatch')
    if not isinstance(p['steps'],int) or not 0<=p['steps']<=SETTINGS['head_steps']:raise ValueError('Invalid head progress')
    steps+=p['steps']
  baseline=sum(n.startswith('baseline/') and n.endswith('/endpoint.json') for n in names)
  done=sum(n.startswith('evaluation/') and n.endswith('/COMPLETE.json') for n in names)
  gates=sum('training_gate/'+f+'.json' in names for f in CASES)
  return (int('SUMMARY.json' in names),gates,done,steps,baseline)

def snapshot(out,drive=''):
 import os,shutil,time
 out=Path(out)
 if not (out/'identity.json').exists():return None
 m={str(p.relative_to(out)):sha(p) for p in out.rglob('*')
    if p.is_file() and p.name!='MANIFEST.json' and not p.name.endswith('.tmp')}
 temp=out/'MANIFEST.json.tmp';temp.write_text(json.dumps(m,indent=2));os.replace(temp,out/'MANIFEST.json')
 dest=out.parent/'gpt2_c7_functional_gate_results.zip';temp=dest.with_suffix('.partial')
 with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
  for n in list(m)+['MANIFEST.json']:z.write(out/n,n,compress_type=zipfile.ZIP_STORED if n.endswith('.zip') else zipfile.ZIP_DEFLATED)
 verified_zip(temp,PROTOCOL);os.replace(temp,dest)
 if drive:
  d=Path(drive);d.mkdir(parents=True,exist_ok=True)
  q=d/('gpt2_c7_functional_gate_results_'+str(time.time_ns())+'.zip');tmp=q.with_suffix('.partial')
  shutil.copyfile(dest,tmp)
  if sha(tmp)!=sha(dest):raise RuntimeError('Drive digest mismatch')
  verified_zip(tmp,PROTOCOL);os.replace(tmp,q);print('DRIVE_BACKUP_OK',q,flush=True)
  valid=[]
  for old in sorted(d.glob('gpt2_c7_functional_gate_results_*.zip')):
   try:verified_zip(old,PROTOCOL);valid.append(old)
   except Exception:continue
  for old in valid[:-3]:old.unlink()
 print('RESULT_ARCHIVE='+str(dest),flush=True);return dest
