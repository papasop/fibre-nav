"""C38: clean, prospective SINGLE-FOLD LC2 pilot. No old writer is accepted."""
from pathlib import Path
import hashlib,json,os,shutil,time,zipfile
PROTOCOL='GPT2_LC2_C38_EXCLUDED_INPUT_A100_V1'
RESULT='gpt2_lc2_c38_excluded_input_results'
DRIVE_NAME=PROTOCOL+'_RECOVERY'
CASES=['xor_00','xor_01','xor_10','xor_11']
SETTINGS={
 'heldout_case':'xor_10','model_seed':82931,
 'model_name':'openai-community/gpt2','revision':'607a30d783dfa663caf39e06633721c8d4cfcd7e',
 'chart_dim':36864,'lora_rank':8,'lora_alpha':16,'anchor_count':12,'response_dim':6,
 'train_per_class':4,'heldout_n':8,'train_seed_base':81538001,'heldout_seed_base':91538001,
 'train_norms':[0.0,0.25,0.5,0.75],
 'heldout_norms':[0.125,0.25,0.375,0.5,0.625,0.75,0.875,1.0],
 'teacher_max_steps':96,'teacher_step_radius':2.0,'teacher_backtracks':8,
 'teacher_target_margin':0.35,'teacher_safety_margin':0.20,'teacher_safety_response':0.0003,
 'teacher_ridge':1e-10,'svd_relative_cutoff':1e-10,'svd_absolute_cutoff':1e-12,
 'writer_ridge':0.001,'feature_scale_floor':0.05,'state_sketch_seed':81538999,
 'proposal_radius':20.0,'compensation_radius':2.0,
 'response_budget':0.002,'KL_budget':0.005,'read_margin':0.05,
 'minimum_preparation_distance_fraction':0.5,
 'scope':'Single predeclared excluded-input fold, clean shared training-only writer; same model/chart and previously studied XOR classes at programme level. No independent confirmation or native update.'}
ARMS=['learned_raw','geometry_shared','learned_composite']
GATES=['bits_match','output_flip','unselected_preserved','response','KL','exact_read_margin','development_read_margin']

def require(ok,message):
 if not ok:raise ValueError(message)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp')
 q.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');os.replace(q,p)
def training_cases():return [x for x in CASES if x!=SETTINGS['heldout_case']]
def bits(case):
 require(case in CASES,'Unknown input class');x,y=map(int,case[-2:]);return [x,y,1,1-(x^y)]
def target(case):
 b=bits(case);return b[:3]+[1-b[3]]
def specs(cohort):
 require(cohort in ('training','heldout'),'Unknown cohort')
 cases=training_cases() if cohort=='training' else [SETTINGS['heldout_case']]
 norms=SETTINGS['train_norms'] if cohort=='training' else SETTINGS['heldout_norms']
 seed=SETTINGS['train_seed_base'] if cohort=='training' else SETTINGS['heldout_seed_base']
 return [{'id':f'{c}__lc2_{cohort}{i:02d}','case':c,'cohort':cohort,'index':i,
          'seed':seed+100*CASES.index(c)+i,'norm':n} for c in cases for i,n in enumerate(norms)]
def validate_training(records):
 require(len(records)==len(specs('training')),'Training denominator differs')
 expected={s['id']:s for s in specs('training')}
 require({x['spec']['id'] for x in records}==set(expected),'Training identity coverage differs')
 for r in records:
  s=r['spec'];require(s==expected[s['id']] and s['case'] in training_cases(),'Held-out leakage into fitting')
  require(r['preparation_valid'] and r['teacher_valid'],'Invalid training references cannot be silently dropped')
 return [s['id'] for s in specs('training')]
def sources(root):
 return {p.name:sha(p) for p in sorted(Path(root).glob('*')) if p.is_file() and p.name!='SOURCE_MANIFEST.json'}
def identity(root):return {'protocol':PROTOCOL,'settings':SETTINGS,'source_hashes':sources(root)}
def checked_zip(path):
 with zipfile.ZipFile(path) as z:
  names=z.namelist();require(len(names)==len(set(names)),'Duplicate archive members')
  require(all(not Path(n).is_absolute() and '..' not in Path(n).parts and '\\' not in n for n in names),'Unsafe archive path')
  m=json.loads(z.read('MANIFEST.json'));require(set(names)==set(m)|{'MANIFEST.json'},'Manifest coverage differs')
  for n,h in m.items():require(hashlib.sha256(z.read(n)).hexdigest()==h,'Archive member differs: '+n)
  require(json.loads(z.read('identity.json'))['protocol']==PROTOCOL,'Only matching C38 result ZIP is accepted; no old writer inputs')
 return True
def progress(path,root):
 checked_zip(path)
 with zipfile.ZipFile(path) as z:
  require(json.loads(z.read('identity.json'))==identity(root),'Resume source/settings differ')
  names=z.namelist();complete=int('SUMMARY.json' in names and json.loads(z.read('SUMMARY.json')).get('complete',False))
  steps=sum(json.loads(z.read(n)).get('steps',0) for n in names if n.endswith('/PROGRESS.json'))
  return (complete,sum(n.endswith('/EVALUATED.json') for n in names),sum(n.endswith('/REFERENCE.json') for n in names),steps,len(names))
def snapshot(out,backup=''):
 out=Path(out);result=out.parent/(RESULT+'.zip');tmp=result.with_suffix('.zip.tmp')
 files={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.relative_to(out).as_posix()!='MANIFEST.json' and not p.name.endswith('.tmp') and '__pycache__' not in p.parts}
 with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
  for n in files:z.write(out/n,n)
  z.writestr('MANIFEST.json',json.dumps(files,sort_keys=True))
 checked_zip(tmp);os.replace(tmp,result)
 if backup:
  b=Path(backup);b.mkdir(parents=True,exist_ok=True);dst=b/(RESULT+'_'+str(time.time_ns())+'.zip');shutil.copyfile(result,dst)
  require(sha(dst)==sha(result),'Backup transfer differs');print('DRIVE_BACKUP_OK',dst,flush=True)
  good=[]
  for p in b.glob(RESULT+'_*.zip'):
   try:checked_zip(p);good.append(p)
   except Exception:pass
  for p in sorted(good,key=lambda p:p.name,reverse=True)[3:]:p.unlink()
 print('RESULT_ARCHIVE='+str(result),flush=True);return result
