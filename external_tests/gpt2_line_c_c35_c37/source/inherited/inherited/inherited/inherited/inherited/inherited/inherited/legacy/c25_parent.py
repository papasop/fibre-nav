"""Import exact final C24 checkpoint and immutable training/evaluation capsule."""
import hashlib,json,os,tempfile,zipfile
from pathlib import Path
import c24_protocol as previous
from c24_parent import validate_capsule as validate_training_capsule
from c17_protocol import sha,verified_zip,verify_open_zip
CAPSULE_PROTOCOL='GPT2_C25_C24_PARENT_CAPSULE_V1'
START_MEMBER='heads/calibrated/checkpoint_000008_final.pt'
START_CP_SHA='b804fb325e7bfda784aa35249338520942928f2705a6410c78f3d13aa6a0b5b1'
START_HEAD_SHA='390170a4a99d1f8728de3f01b5165838a0bc19f8c1f895c1e0d93a0d4eb254b0'
RECORDS=('identity.json','SUMMARY.json','WRITER_FROZEN.json','heads/calibrated/PROGRESS.json','FINAL_TRAINING.json','SPLIT.json')

def pins_at_root():return json.loads((Path(__file__).parent/'C24_SOURCE_PINS.json').read_text())
def check_records(i,s,f,p,read,pins):
 if i.get('protocol')!=previous.PROTOCOL or i.get('settings')!=previous.SETTINGS or i.get('source_hashes')!=pins or i.get('start_head_sha256')!=previous.START_HEAD_SHA:raise ValueError('Different C24 source/settings/start identity')
 for n,h in pins.items():
  if hashlib.sha256(read('source/'+n)).hexdigest()!=h:raise ValueError('Archived C24 source changed: '+n)
 if not s.get('complete') or s.get('decision')!='C24_RAW_INSPECTED_LOCAL_STATE_FAIL' or s.get('writer_head_sha256')!=START_HEAD_SHA:raise ValueError('Exact complete C24 final head required')
 if s['training']['calibrated']['n_requested']!=60 or s['training']['calibrated']['passed']!=60 or s['inspected_local_state_development']['calibrated']['n_requested']!=16 or s['inspected_local_state_development']['calibrated']['passed']!=9:raise ValueError('Wrong C24 cohorts/results')
 if p.get('checkpoint')!='checkpoint_000008_final.pt' or p.get('checkpoint_sha256')!=START_CP_SHA or p.get('rounds')!=8 or p.get('accepted_updates')!=8 or not p.get('done'):raise ValueError('Wrong true C24 checkpoint pointer')
 if f.get('selected_checkpoint')!=START_MEMBER or f.get('head_sha256')!=START_HEAD_SHA:raise ValueError('Wrong frozen C24 identity')
 rows=json.loads(read('FINAL_TRAINING.json'))
 if len(previous.quality(rows)['passed_ids'])!=60:raise ValueError('C24 actual60 must all pass')

def validate_parent(path,pins=None,verified_identity=None):
 with zipfile.ZipFile(path) as z:
  i=verified_identity or verify_open_zip(z,previous.PROTOCOL);s=json.loads(z.read('SUMMARY.json'));f=json.loads(z.read('WRITER_FROZEN.json'));p=json.loads(z.read('heads/calibrated/PROGRESS.json'))
  check_records(i,s,f,p,z.read,pins or pins_at_root())
  if hashlib.sha256(z.read(START_MEMBER)).hexdigest()!=START_CP_SHA or hashlib.sha256(z.read('parent_capsule.zip')).hexdigest()!=i['parent_capsule_sha256']:raise ValueError('Actual C24 checkpoint/capsule digest mismatch')
  for t in previous.test_specs():z.getinfo('evaluation/inspected/'+t['id']+'/COMPLETE.json')
  return {'identity':i,'summary':s,'freeze':f}

def required_members(pins):
 n={'identity.json','MANIFEST.json','PROVENANCE.json','start_checkpoint.pt','c24_capsule.zip'}
 n.update('C24_RECORDS/'+m for m in RECORDS);n.update('C24_RECORDS/source/'+m for m in pins)
 n.update('C24_RECORDS/evaluation/inspected/'+t['id']+'/COMPLETE.json' for t in previous.test_specs());return n

def validate_capsule(path,pins=None):
 pins=pins or pins_at_root();ci=verified_zip(path,CAPSULE_PROTOCOL)
 with zipfile.ZipFile(path) as z:
  if set(z.namelist())!=required_members(pins):raise ValueError('C25 capsule member coverage mismatch')
  read=lambda n:z.read('C24_RECORDS/'+n);i,s,f,p=[json.loads(read(n)) for n in RECORDS[:4]];check_records(i,s,f,p,read,pins)
  if hashlib.sha256(z.read('start_checkpoint.pt')).hexdigest()!=START_CP_SHA or ci['start_head_sha256']!=START_HEAD_SHA:raise ValueError('C25 actual start checkpoint identity mismatch')
  provenance=json.loads(z.read('PROVENANCE.json'));manifest=json.loads(z.read('MANIFEST.json'))
  if set(provenance['copied_members'])!=set(manifest)-{'identity.json','PROVENANCE.json'}:raise ValueError('C25 origin coverage mismatch')
  for n,r in provenance['copied_members'].items():
   if r['sha256']!=manifest[n] or not isinstance(r['origin_member'],str):raise ValueError('C25 origin digest mismatch')
  if provenance['parent_c24_sha256']!=ci['parent_c24_sha256']:raise ValueError('Parent C24 SHA mismatch')
  with tempfile.TemporaryDirectory(prefix='c25_nested_validate_') as tmp:
   nested=Path(tmp)/'c24_capsule.zip'
   with z.open('c24_capsule.zip') as src,nested.open('wb') as dst:
    while True:
     block=src.read(8*1024*1024)
     if not block:break
     dst.write(block)
   if sha(nested)!=i['parent_capsule_sha256']:raise ValueError('Inherited C24 capsule changed')
   validate_training_capsule(nested)
  return {'identity':i,'summary':s,'freeze':f,'capsule_identity':ci}

def make_capsule(parent_path,destination):
 parent_path=Path(parent_path);destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True);validate_parent(parent_path);parent_sha=sha(parent_path);pins=pins_at_root();tmp=destination.with_suffix('.tmp');manifest={};copied={}
 with zipfile.ZipFile(parent_path) as source,zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as target:
  def copy(old,new):
   h=hashlib.sha256()
   with source.open(old) as src,target.open(new,'w',force_zip64=True) as dst:
    while True:
     block=src.read(8*1024*1024)
     if not block:break
     h.update(block);dst.write(block)
   manifest[new]=h.hexdigest();copied[new]={'origin_member':'C24/'+old,'sha256':h.hexdigest()}
  copy(START_MEMBER,'start_checkpoint.pt');copy('parent_capsule.zip','c24_capsule.zip')
  for n in RECORDS:copy(n,'C24_RECORDS/'+n)
  for n in pins:copy('source/'+n,'C24_RECORDS/source/'+n)
  for t in previous.test_specs():copy('evaluation/inspected/'+t['id']+'/COMPLETE.json','C24_RECORDS/evaluation/inspected/'+t['id']+'/COMPLETE.json')
  ci={'protocol':CAPSULE_PROTOCOL,'start_head_sha256':START_HEAD_SHA,'parent_c24_sha256':parent_sha,'full_historical_result_archives_embedded':False}
  provenance={'parent_c24_sha256':parent_sha,'copied_members':copied,'policy':'Required immutable bytes only; full C24 result remains external and unmodified.'}
  for n,v in [('identity.json',ci),('PROVENANCE.json',provenance)]:
   bits=json.dumps(v,indent=2).encode();target.writestr(n,bits);manifest[n]=hashlib.sha256(bits).hexdigest()
  target.writestr('MANIFEST.json',json.dumps(manifest,indent=2))
 validate_capsule(tmp);os.replace(tmp,destination);print('C25 compact C24 parent verified:',destination,destination.stat().st_size,flush=True);return destination
