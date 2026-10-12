"""Frozen C5 matched functional-writer development experiment."""
import hashlib,io,json,zipfile
from parent_c4_protocol import sha,verify_open_zip,verified_zip,endpoint_gate,episode_gate,target_from_reads,CASES,REVISION,C3_SHA,C2_SHA,C1_SHA,V5_SHA
PROTOCOL='GPT2_C5_FUNCTIONAL_WRITER_A100_V1'
C4_PROTOCOL='GPT2_C4_PROJECTION_ABLATION_A100_V1'
C4_SHA='189e1cf58a0d54ee7941ff947f2998811d021ac34fb70ca66cdc4398af232614'
ARMS=('parameter_mse','functional')
SETTINGS={'margin_gate':.05,'response_budget':.002,'kl_budget':.005,
 'correction_ridge':1e-6,'correction_max_steps':8,'correction_backtracking':6,'correction_checkpoint_every':2,
 'head_steps':300,'head_width':32,'head_lr':.001,'head_weight_decay':0.,'gradient_clip':5.,
 'head_checkpoint_every':25,'variant_norms':[0.,.25,.5], 'variant_seed':94031,'head_seed':95031,
 'feature':'four exact-prompt final hidden states, four signed raw binary margins, requested output bit',
 'loss_weights':{'parameter_mse':1.,'read':.25,'response':.1,'KL':.1},
 'read_surrogate_margin':1.,'dataset_acceptance':'all 12 preparation and teacher endpoints must pass unchanged gates',
 'test_scope':'previously inspected leave-one-XOR-input-out development; not new holdout confirmation',
 'correction_acceptance':'strict measured infinity-response reduction only; final all gates mandatory'}

def state_id(case,variant):
 if case not in CASES or variant not in range(3):raise ValueError('Undeclared state')
 return case+'__v'+str(variant)

STATES=tuple(state_id(c,v) for c in CASES for v in range(3))
def fold_states(fold,training=True):
 if fold not in CASES:raise ValueError('Unknown fold')
 return tuple(s for s in STATES if (s.split('__')[0]!=fold)==training)

def arm_order(index):return ARMS if index%2==0 else tuple(reversed(ARMS))

def matched_counts(rows):
 a=[bool(r['parameter_mse'].get('episode_pass',False)) for r in rows]
 b=[bool(r['functional'].get('episode_pass',False)) for r in rows]
 return {'n':len(rows),'parameter_mse_passed':sum(a),'functional_passed':sum(b),
         'both_passed':sum(x and y for x,y in zip(a,b)),
         'mse_only':sum(x and not y for x,y in zip(a,b)),
         'functional_only':sum(not x and y for x,y in zip(a,b)),
         'both_failed':sum(not x and not y for x,y in zip(a,b))}

def validate_c4(path):
 if sha(path)!=C4_SHA:raise ValueError('Requires the exact inspected complete C4 result ZIP')
 with zipfile.ZipFile(path) as z:
  ident=verify_open_zip(z,C4_PROTOCOL);s=json.loads(z.read('SUMMARY.json'))
  if not s.get('complete') or s.get('n_completed')!=16 or not s.get('positive_references_all_pass'):
   raise ValueError('Complete C4 with positive controls required')
  b=z.read('parent_c3.zip')
  if hashlib.sha256(b).hexdigest()!=C3_SHA:raise ValueError('C3 identity mismatch')
  for protocol,member,digest in [('GPT2_C3_TRAINING_GEOMETRY_A100_V1','parent_c2.zip',C2_SHA),
    ('GPT2_C2_C1_TRAIN_RECHECK_GEOMETRY_A100_V1','parent_c1.zip',C1_SHA),
    ('GPT2_C1_ONE_FORWARD_WRITER_LOO_A100_V1','parent_v5.zip',V5_SHA)]:
   with zipfile.ZipFile(io.BytesIO(b)) as inner:
    verify_open_zip(inner,protocol);b=inner.read(member)
    if hashlib.sha256(b).hexdigest()!=digest:raise ValueError('Nested parent digest mismatch')
  with zipfile.ZipFile(io.BytesIO(b)) as v5:verify_open_zip(v5,'GPT2_B5_XOR_CONTINUATION_A100_V5')
 return ident

def resume_score(path):
 ident=verified_zip(path,PROTOCOL)
 if ident.get('parent_c4_sha256')!=C4_SHA or ident.get('settings')!=SETTINGS:raise ValueError('Different C5 settings/parent')
 with zipfile.ZipFile(path) as z:
  if hashlib.sha256(z.read('parent_c4.zip')).hexdigest()!=C4_SHA:raise ValueError('Missing C4 parent')
  names=set(z.namelist())
  done=sum(n.endswith('/COMPLETE.json') and n.startswith('evaluation/') for n in names)
  steps=0
  for n in names:
   if not (n.startswith('heads/') and n.endswith('/PROGRESS.json')):continue
   p=json.loads(z.read(n));name=p['checkpoint']
   if name!=name.split('/')[-1] or name in ('.','..') or not name.endswith('.pt'):raise ValueError('Unsafe checkpoint pointer')
   member=n.rsplit('/',1)[0]+'/'+name
   if hashlib.sha256(z.read(member)).hexdigest()!=p['checkpoint_sha256']:raise ValueError('Actual head checkpoint digest mismatch')
   if not isinstance(p['steps'],int) or not 0<=p['steps']<=SETTINGS['head_steps']:raise ValueError('Invalid progress')
   steps+=p['steps']
  pool=sum('pool/'+s+'/COMPLETE.json' in names for s in STATES)
  return (int('SUMMARY.json' in names),done,steps,pool)

def snapshot(out,drive=''):
 """Final launcher snapshot uses saved checkpoints only; no inferred weights or Torch load."""
 import os,shutil,time
 from pathlib import Path
 out=Path(out)
 if not (out/'identity.json').exists():return None
 m={str(p.relative_to(out)):sha(p) for p in out.rglob('*')
    if p.is_file() and p.name!='MANIFEST.json' and not p.name.endswith('.tmp')}
 temp=out/'MANIFEST.json.tmp';temp.write_text(json.dumps(m,indent=2));os.replace(temp,out/'MANIFEST.json')
 dest=out.parent/'gpt2_c5_functional_writer_results.zip';temp=dest.with_suffix('.partial')
 with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
  for n in list(m)+['MANIFEST.json']:z.write(out/n,n,compress_type=zipfile.ZIP_STORED if n.endswith('.zip') else zipfile.ZIP_DEFLATED)
 verified_zip(temp,PROTOCOL);os.replace(temp,dest)
 if drive:
  d=Path(drive);d.mkdir(parents=True,exist_ok=True)
  q=d/('gpt2_c5_functional_writer_results_'+str(time.time_ns())+'.zip');tmp=q.with_suffix('.partial')
  shutil.copyfile(dest,tmp)
  if sha(tmp)!=sha(dest):raise RuntimeError('Snapshot Drive digest mismatch')
  verified_zip(tmp,PROTOCOL);os.replace(tmp,q);print('DRIVE_BACKUP_OK',q,flush=True)
  valid=[]
  for old in sorted(d.glob('gpt2_c5_functional_writer_results_*.zip')):
   try:verified_zip(old,PROTOCOL);valid.append(old)
   except Exception:continue
  for old in valid[:-3]:old.unlink()
 print('RESULT_ARCHIVE='+str(dest),flush=True);return dest
