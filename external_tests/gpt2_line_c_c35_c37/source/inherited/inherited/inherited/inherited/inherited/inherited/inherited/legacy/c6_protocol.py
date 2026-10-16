"""C6: full-batch warm fit followed by a matched, gently regularised objective ablation."""
import hashlib,io,json,math,zipfile
from pathlib import Path
from c5_protocol import sha,verify_open_zip,verified_zip,CASES,STATES,fold_states,state_id,target_from_reads,endpoint_gate,episode_gate,REVISION,C4_SHA,C3_SHA,C2_SHA,C1_SHA,V5_SHA
PROTOCOL='GPT2_C6_STAGED_WRITER_A100_V1'
C5_PROTOCOL='GPT2_C5_FUNCTIONAL_WRITER_A100_V1'
C5_SHA='bd75ee8f00d520f618c73a4630ddfb4817eb90afa041d751022d86004a4736c6'
ARMS=('parameter_mse','functional')
SETTINGS={'margin_gate':.05,'response_budget':.002,'kl_budget':.005,
 'correction_ridge':1e-6,'correction_max_steps':8,'correction_backtracking':6,'correction_checkpoint_every':2,
 'warm_max_steps':1200,'warm_check_every':200,'warm_lr':.001,'warm_mse_gate':.02,
 'warm_acceptance':'normalised full-batch MSE <= 0.02 AND all nine training episodes pass after identical direct correction',
 'head_steps':200,'head_width':32,'head_lr':.0001,'head_weight_decay':0.,'gradient_clip':5.,
 'head_checkpoint_every':25,'archive_training_every':200,
 'loss_weights':{'parameter_mse':1.,'read':.1,'response':.01,'KL':.01},
 'response_surrogate_scale':.1,'read_surrogate_margin':1.,
 'functional_response_loss':'mean smooth-L1(mean-anchor response displacement / 0.1, 0), beta=1',
 'functional_KL_loss':'relu(mean-anchor KL / 0.005 - 1), linear excess',
 'feature':'unchanged C5 hidden states, binary logit margins and requested output bit',
 'warm_start':'actual C5 parameter_mse head at step 300; new warm optimiser',
 'paired_start':'identical accepted warm head and normalisation; identical fresh AdamW state in both arms',
 'sample_schedule':'every step full-batch MSE on nine states; functional arm additionally evaluates state step modulo nine',
 'fresh_evaluation':False,'internal_parameter_update':False,'controller_dispatch_tested':False}

def arm_order(index):return ARMS if index%2==0 else tuple(reversed(ARMS))

def fit_gate(mse,episodes):
 return bool(math.isfinite(mse) and mse<=SETTINGS['warm_mse_gate'] and len(episodes)==9
             and all(v.get('episode_pass',False) for v in episodes))

def matched_counts(rows):
 a=[bool(r['parameter_mse'].get('episode_pass',False)) for r in rows]
 b=[bool(r['functional'].get('episode_pass',False)) for r in rows]
 return {'n':len(rows),'parameter_mse_passed':sum(a),'functional_passed':sum(b),
  'both_passed':sum(x and y for x,y in zip(a,b)),'mse_only':sum(x and not y for x,y in zip(a,b)),
  'functional_only':sum(not x and y for x,y in zip(a,b)),'both_failed':sum(not x and not y for x,y in zip(a,b))}

def validate_c5(path):
 if sha(path)!=C5_SHA:raise ValueError('C6 V1 requires the inspected complete C5 result, not a launcher ZIP')
 with zipfile.ZipFile(path) as z:
  ident=verify_open_zip(z,C5_PROTOCOL);s=json.loads(z.read('SUMMARY.json'))
  if not s.get('complete') or s.get('n_completed')!=48 or s.get('persisted_committed_head_steps')!=2400 or not s.get('positive_references_all_pass'):
   raise ValueError('Complete C5 with saved fitted heads and valid references required')
  for state in STATES:
   r=json.loads(z.read('pool/'+state+'/COMPLETE.json'))
   if not r.get('usable'):raise ValueError('Parent state rejected')
   for name in ['prepared.pt','teacher.pt','features.pt']:
    if 'pool/'+state+'/'+name not in z.namelist():raise ValueError('Parent tensor missing')
  for fold in CASES:
   base='heads/'+fold+'/parameter_mse/';p=json.loads(z.read(base+'PROGRESS.json'))
   if p['steps']!=300 or hashlib.sha256(z.read(base+p['checkpoint'])).hexdigest()!=p['checkpoint_sha256']:
    raise ValueError('Actual C5 MSE head identity/progress mismatch')
  b=z.read('parent_c4.zip')
  for protocol,member,digest in [('GPT2_C4_PROJECTION_ABLATION_A100_V1','parent_c3.zip',C3_SHA),
    ('GPT2_C3_TRAINING_GEOMETRY_A100_V1','parent_c2.zip',C2_SHA),
    ('GPT2_C2_C1_TRAIN_RECHECK_GEOMETRY_A100_V1','parent_c1.zip',C1_SHA),
    ('GPT2_C1_ONE_FORWARD_WRITER_LOO_A100_V1','parent_v5.zip',V5_SHA)]:
   if protocol.startswith('GPT2_C4') and hashlib.sha256(b).hexdigest()!=C4_SHA:raise ValueError('C4 identity mismatch')
   with zipfile.ZipFile(io.BytesIO(b)) as inner:
    verify_open_zip(inner,protocol);b=inner.read(member)
    if hashlib.sha256(b).hexdigest()!=digest:raise ValueError('Nested parent identity mismatch')
  with zipfile.ZipFile(io.BytesIO(b)) as v5:verify_open_zip(v5,'GPT2_B5_XOR_CONTINUATION_A100_V5')
 return ident

def checkpoint_pointer(z,base,p):
 name=p['checkpoint']
 if Path(name).name!=name or '\\' in name or not name.endswith('.pt'):raise ValueError('Unsafe checkpoint pointer')
 if hashlib.sha256(z.read(base+name)).hexdigest()!=p['checkpoint_sha256']:raise ValueError('Actual checkpoint digest mismatch')
 if not isinstance(p['steps'],int) or not 0<=p['steps']<=SETTINGS['warm_max_steps']:raise ValueError('Invalid progress')
 return p['steps']

def resume_score(path):
 ident=verified_zip(path,PROTOCOL)
 if ident.get('parent_c5_sha256')!=C5_SHA or ident.get('settings')!=SETTINGS:raise ValueError('Different C6 settings/parent')
 with zipfile.ZipFile(path) as z:
  b=z.read('parent_c5_capsule.zip')
  if hashlib.sha256(b).hexdigest()!=ident.get('parent_capsule_sha256'):raise ValueError('Missing/corrupt parent capsule')
  with zipfile.ZipFile(io.BytesIO(b)) as cap:validate_capsule_open(cap)
  names=set(z.namelist());steps=0
  for n in names:
   if n.startswith('heads/') and n.endswith('/PROGRESS.json'):
    steps+=checkpoint_pointer(z,n.rsplit('/',1)[0]+'/',json.loads(z.read(n)))
  folds=sum('heads/'+f+'/warm/FIT_RESULT.json' in names for f in CASES)
  done=sum(n.startswith('evaluation/') and n.endswith('/COMPLETE.json') for n in names)
  return (int('SUMMARY.json' in names),folds,done,steps)

def snapshot(out,drive=''):
 import os,shutil,time
 out=Path(out)
 if not (out/'identity.json').exists():return None
 m={str(p.relative_to(out)):sha(p) for p in out.rglob('*')
    if p.is_file() and p.name!='MANIFEST.json' and not p.name.endswith('.tmp')}
 temp=out/'MANIFEST.json.tmp';temp.write_text(json.dumps(m,indent=2));os.replace(temp,out/'MANIFEST.json')
 dest=out.parent/'gpt2_c6_staged_writer_results.zip';temp=dest.with_suffix('.partial')
 with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
  for n in list(m)+['MANIFEST.json']:z.write(out/n,n,compress_type=zipfile.ZIP_STORED if n.endswith('.zip') else zipfile.ZIP_DEFLATED)
 verified_zip(temp,PROTOCOL);os.replace(temp,dest)
 if drive:
  d=Path(drive);d.mkdir(parents=True,exist_ok=True)
  q=d/('gpt2_c6_staged_writer_results_'+str(time.time_ns())+'.zip');tmp=q.with_suffix('.partial')
  shutil.copyfile(dest,tmp)
  if sha(tmp)!=sha(dest):raise RuntimeError('Drive digest mismatch')
  verified_zip(tmp,PROTOCOL);os.replace(tmp,q);print('DRIVE_BACKUP_OK',q,flush=True)
  valid=[]
  for old in sorted(d.glob('gpt2_c6_staged_writer_results_*.zip')):
   try:verified_zip(old,PROTOCOL);valid.append(old)
   except Exception:continue
  for old in valid[:-3]:old.unlink()
 print('RESULT_ARCHIVE='+str(dest),flush=True);return dest

CAPSULE_PROTOCOL='GPT2_C6_C5_PARENT_CAPSULE_V1'
def validate_capsule_open(z):
 ident=verify_open_zip(z,CAPSULE_PROTOCOL)
 if ident.get('original_c5_sha256')!=C5_SHA:raise ValueError('Different original C5 input')
 original=json.loads(z.read('C5_MANIFEST.json'))
 included=json.loads(z.read('INCLUDED_ORIGINAL_MEMBERS.json'))
 for n,h in included.items():
  if original.get(n)!=h or hashlib.sha256(z.read(n)).hexdigest()!=h:raise ValueError('Original member identity mismatch')
 if hashlib.sha256(z.read('parent_v5.zip')).hexdigest()!=V5_SHA:raise ValueError('V5 identity mismatch')
 for sid in STATES:
  if not json.loads(z.read('pool/'+sid+'/COMPLETE.json')).get('usable'):raise ValueError('Parent pool unusable')
  for n in ['prepared.pt','teacher.pt','features.pt']:
   if 'pool/'+sid+'/'+n not in included:raise ValueError('Original pool tensor missing')
 for fold in CASES:
  base='heads/'+fold+'/parameter_mse/';p=json.loads(z.read(base+'PROGRESS.json'))
  if p['steps']!=300 or hashlib.sha256(z.read(base+p['checkpoint'])).hexdigest()!=p['checkpoint_sha256']:
   raise ValueError('Original MSE head checkpoint mismatch')
 return ident

def validate_capsule(path):
 with zipfile.ZipFile(path) as z:return validate_capsule_open(z)

def make_capsule(path,destination):
 import os
 validate_c5(path)
 with zipfile.ZipFile(path) as z:
  m=json.loads(z.read('MANIFEST.json'));wanted=[]
  for sid in STATES:
   wanted+=['pool/'+sid+'/'+n for n in ['prepared.pt','teacher.pt','features.pt','COMPLETE.json']]
  for fold in CASES:
   base='heads/'+fold+'/parameter_mse/';p=json.loads(z.read(base+'PROGRESS.json'))
   wanted+=[base+'PROGRESS.json',base+'TRAINING_SPEC.json',base+p['checkpoint']]
  entries={n:z.read(n) for n in wanted}
  entries.update({'C5_IDENTITY.json':z.read('identity.json'),'C5_SUMMARY.json':z.read('SUMMARY.json'),
                  'C5_MANIFEST.json':z.read('MANIFEST.json')})
  b=z.read('parent_c4.zip')
  for member in ['parent_c3.zip','parent_c2.zip','parent_c1.zip','parent_v5.zip']:
   with zipfile.ZipFile(io.BytesIO(b)) as inner:b=inner.read(member)
  entries['parent_v5.zip']=b
  entries['INCLUDED_ORIGINAL_MEMBERS.json']=json.dumps({n:m[n] for n in wanted},indent=2).encode()
  entries['identity.json']=json.dumps({'protocol':CAPSULE_PROTOCOL,'original_c5_sha256':C5_SHA,
    'complete_original_c5_verified_when_created':True,'is_subset':True,
    'resume_can_reverify_included_members_not_complete_original':True,
    'weights_regenerated':False},indent=2).encode()
  entries['MANIFEST.json']=json.dumps({n:hashlib.sha256(b).hexdigest() for n,b in entries.items()},indent=2).encode()
 dest=Path(destination);temp=dest.with_suffix('.partial')
 with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as cap:
  for n,b in sorted(entries.items()):
   info=zipfile.ZipInfo(n,date_time=(2026,10,12,0,0,0));info.compress_type=zipfile.ZIP_STORED if n.endswith('.zip') else zipfile.ZIP_DEFLATED
   cap.writestr(info,b)
 validate_capsule(temp);os.replace(temp,dest);return dest
