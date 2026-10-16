"""Import exact completed C21 into a compact immutable C22 parent."""
import hashlib,json,os,tempfile,zipfile
from pathlib import Path
import c21_protocol as previous
from c21_parent import validate_capsule as validate_training_capsule
from c17_protocol import sha,verified_zip,verify_open_zip

CAPSULE_PROTOCOL='GPT2_C22_C21_PARENT_CAPSULE_V1'
START_MEMBER='heads/calibrated/checkpoint_000016.pt'
START_CP_SHA='407e463e255ae7f31937b57b7df9b71800ff298282402f722348379177044ce8'
START_HEAD_SHA='09e4815ef29322665ce363ac7fe4f38a0906bcc05f6412a3037f2f53cb4c539a'

def pins_at_root():return json.loads((Path(__file__).parent/'C21_SOURCE_PINS.json').read_text())
def check_records(i,s,f,p,read,pins):
 if i.get('protocol')!=previous.PROTOCOL or i.get('settings')!=previous.SETTINGS or i.get('source_hashes')!=pins or i.get('start_head_sha256')!=previous.START_HEAD_SHA:raise ValueError('Different C21 source/settings/start identity')
 for n,h in pins.items():
  if hashlib.sha256(read('source/'+n)).hexdigest()!=h:raise ValueError('Archived C21 source changed: '+n)
 if not s.get('complete') or s.get('decision')!='C21_RAW_TRAINING_FUNCTION_FAILURE' or s.get('writer_head_sha256')!=START_HEAD_SHA:raise ValueError('Exact completed C21 final head required')
 if s['training']['calibrated']['n_requested']!=60 or s['training']['calibrated']['passed']!=50 or s['original36_training']['calibrated']['passed']!=30 or s['prospective_local_state_development']['calibrated']['n_requested']!=16:raise ValueError('Wrong recorded C21 cohort/results')
 if p.get('checkpoint')!='checkpoint_000016.pt' or p.get('rounds')!=16 or p.get('checkpoint_sha256')!=START_CP_SHA:raise ValueError('Wrong actual C21 checkpoint pointer')
 if f.get('selected_checkpoint')!=START_MEMBER or f.get('head_sha256')!=START_HEAD_SHA:raise ValueError('Wrong frozen C21 identity')

def validate_parent(path,pins=None,verified_identity=None):
 with zipfile.ZipFile(path) as z:
  i=verified_identity or verify_open_zip(z,previous.PROTOCOL);s=json.loads(z.read('SUMMARY.json'));f=json.loads(z.read('WRITER_FROZEN.json'));p=json.loads(z.read('heads/calibrated/PROGRESS.json'))
  check_records(i,s,f,p,z.read,pins or pins_at_root())
  if hashlib.sha256(z.read(START_MEMBER)).hexdigest()!=START_CP_SHA:raise ValueError('Actual C21 checkpoint digest mismatch')
  if hashlib.sha256(z.read('parent_capsule.zip')).hexdigest()!=i['parent_capsule_sha256']:raise ValueError('Inherited training capsule digest mismatch')
  return {'identity':i,'summary':s,'freeze':f}

def required_members(pins):
 n={'identity.json','MANIFEST.json','PROVENANCE.json','start_checkpoint.pt','training_capsule.zip'}
 n.update('C21_RECORDS/'+m for m in ('identity.json','SUMMARY.json','WRITER_FROZEN.json','heads/calibrated/PROGRESS.json','FINAL_TRAINING.json','SPLIT.json'))
 n.update('C21_RECORDS/source/'+m for m in pins);return n

def validate_capsule(path,pins=None):
 pins=pins or pins_at_root();ci=verified_zip(path,CAPSULE_PROTOCOL)
 with zipfile.ZipFile(path) as z:
  if set(z.namelist())!=required_members(pins):raise ValueError('C22 capsule required member coverage mismatch')
  read=lambda n:z.read('C21_RECORDS/'+n);i=json.loads(read('identity.json'));s=json.loads(read('SUMMARY.json'));f=json.loads(read('WRITER_FROZEN.json'));p=json.loads(read('heads/calibrated/PROGRESS.json'))
  check_records(i,s,f,p,read,pins)
  if hashlib.sha256(z.read('start_checkpoint.pt')).hexdigest()!=START_CP_SHA:raise ValueError('Capsule actual C21 checkpoint changed')
  provenance=json.loads(z.read('PROVENANCE.json'));manifest=json.loads(z.read('MANIFEST.json'));expected=set(manifest)-{'identity.json','PROVENANCE.json'}
  if set(provenance['copied_members'])!=expected:raise ValueError('Capsule origin coverage mismatch')
  for n,r in provenance['copied_members'].items():
   if r['sha256']!=manifest[n] or not isinstance(r['origin_member'],str):raise ValueError('Capsule origin digest mismatch')
  if ci['start_head_sha256']!=START_HEAD_SHA or provenance['parent_c21_sha256']!=ci['parent_c21_sha256']:raise ValueError('Capsule parent identity mismatch')
  with tempfile.TemporaryDirectory(prefix='c22_training_validate_') as tmp:
   inherited=Path(tmp)/'training.zip'
   with z.open('training_capsule.zip') as src,open(inherited,'wb') as dst:
    while True:
     block=src.read(8*1024*1024)
     if not block:break
     dst.write(block)
   if sha(inherited)!=i['parent_capsule_sha256']:raise ValueError('Nested training capsule changed')
   validate_training_capsule(inherited)
  return {'identity':i,'summary':s,'freeze':f,'capsule_identity':ci}

def make_capsule(parent_path,destination):
 parent_path=Path(parent_path);destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True);info=validate_parent(parent_path);parent_sha=sha(parent_path);pins=pins_at_root();tmp=destination.with_suffix('.tmp');manifest={};copied={}
 print('C22 completed C21 source/checkpoint/cohorts verified; importing immutable training capsule.',flush=True)
 with zipfile.ZipFile(parent_path) as source,zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as target:
  def copy_member(old,new):
   h=hashlib.sha256()
   with source.open(old) as src,target.open(new,'w',force_zip64=True) as dst:
    while True:
     block=src.read(8*1024*1024)
     if not block:break
     h.update(block);dst.write(block)
   manifest[new]=h.hexdigest();copied[new]={'origin_member':'C21/'+old,'sha256':h.hexdigest()}
  copy_member(START_MEMBER,'start_checkpoint.pt');copy_member('parent_capsule.zip','training_capsule.zip')
  for n in ('identity.json','SUMMARY.json','WRITER_FROZEN.json','heads/calibrated/PROGRESS.json','FINAL_TRAINING.json','SPLIT.json'):copy_member(n,'C21_RECORDS/'+n)
  for n in pins:copy_member('source/'+n,'C21_RECORDS/source/'+n)
  identity={'protocol':CAPSULE_PROTOCOL,'start_head_sha256':START_HEAD_SHA,'parent_c21_sha256':parent_sha,'historical_full_result_archives_embedded':False}
  provenance={'parent_c21_sha256':parent_sha,'copied_members':copied,'copy_policy':'Unchanged required member bytes. Full historical result ZIPs external, original input unmodified.'}
  for n,value in [('identity.json',identity),('PROVENANCE.json',provenance)]:
   bits=json.dumps(value,indent=2).encode();target.writestr(n,bits);manifest[n]=hashlib.sha256(bits).hexdigest()
  target.writestr('MANIFEST.json',json.dumps(manifest,indent=2))
 validate_capsule(tmp);os.replace(tmp,destination);print('C22 compact C21 parent OK:',destination,destination.stat().st_size,flush=True);return destination
