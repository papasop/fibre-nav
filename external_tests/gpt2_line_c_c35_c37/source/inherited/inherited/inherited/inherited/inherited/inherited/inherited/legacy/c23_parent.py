"""Import exact completed C22 into a compact immutable C23 parent."""
import hashlib,json,os,tempfile,zipfile
from pathlib import Path
import c22_protocol as previous
from c22_parent import validate_capsule as validate_training_capsule
from c17_protocol import sha,verified_zip,verify_open_zip

CAPSULE_PROTOCOL='GPT2_C23_C22_PARENT_CAPSULE_V1'
START_MEMBER='heads/calibrated/checkpoint_000015.pt'
START_CP_SHA='7c202d680eeee294c21c456d0561135fbd2e98d18025c236f492b90fe8f7b7f9'
START_HEAD_SHA='3909cd5126c5995dba5ae170980260294f55c0295096ff2390f03ba2083c318c'

def pins_at_root():return json.loads((Path(__file__).parent/'C22_SOURCE_PINS.json').read_text())
def check_records(i,s,f,p,read,pins):
 if i.get('protocol')!=previous.PROTOCOL or i.get('settings')!=previous.SETTINGS or i.get('source_hashes')!=pins or i.get('start_head_sha256')!=previous.START_HEAD_SHA:raise ValueError('Different C22 source/settings/start identity')
 for n,h in pins.items():
  if hashlib.sha256(read('source/'+n)).hexdigest()!=h:raise ValueError('Archived C22 source changed: '+n)
 if not s.get('complete') or s.get('decision')!='C22_RAW_TRAINING_FUNCTION_FAILURE' or s.get('writer_head_sha256')!=START_HEAD_SHA:raise ValueError('Exact completed C22 final head required')
 if s['training']['calibrated']['n_requested']!=60 or s['training']['calibrated']['passed']!=54 or s['original36_training']['calibrated']['passed']!=32 or s['prospective_local_state_development']['calibrated']['n_requested']!=16:raise ValueError('Wrong recorded C22 cohort/results')
 if p.get('checkpoint')!='checkpoint_000015.pt' or p.get('rounds')!=15 or p.get('accepted_updates')!=13 or not p.get('done') or p.get('checkpoint_sha256')!=START_CP_SHA:raise ValueError('Wrong actual C22 checkpoint pointer')
 if f.get('selected_checkpoint')!=START_MEMBER or f.get('head_sha256')!=START_HEAD_SHA:raise ValueError('Wrong frozen C22 identity')

def validate_parent(path,pins=None,verified_identity=None):
 with zipfile.ZipFile(path) as z:
  i=verified_identity or verify_open_zip(z,previous.PROTOCOL);s=json.loads(z.read('SUMMARY.json'));f=json.loads(z.read('WRITER_FROZEN.json'));p=json.loads(z.read('heads/calibrated/PROGRESS.json'))
  check_records(i,s,f,p,z.read,pins or pins_at_root())
  if hashlib.sha256(z.read(START_MEMBER)).hexdigest()!=START_CP_SHA:raise ValueError('Actual C22 checkpoint digest mismatch')
  if hashlib.sha256(z.read('parent_capsule.zip')).hexdigest()!=i['parent_capsule_sha256']:raise ValueError('Inherited training capsule digest mismatch')
  g=json.loads(z.read('calibration/round_000014/START.json'))
  if g.get('head_sha256')!=START_HEAD_SHA or g.get('case')!='xor_10' or g.get('round')!=14:raise ValueError('Cached round14 geometry at wrong actual frozen head')
  for t in previous.training_specs():
   if t['case']=='xor_10':z.getinfo('calibration/round_000014/states/'+t['id']+'/JACOBIAN.pt')
  for ri in range(3):z.getinfo('calibration/round_000014/COEFFICIENTS_%02d.pt'%ri)
  return {'identity':i,'summary':s,'freeze':f}

def required_members(pins):
 n={'identity.json','MANIFEST.json','PROVENANCE.json','start_checkpoint.pt','training_capsule.zip','GEOMETRY_START.json'}
 n.update('geometry/'+t['id']+'/JACOBIAN.pt' for t in previous.training_specs() if t['case']=='xor_10')
 n.update('coefficients/COEFFICIENTS_%02d.pt'%ri for ri in range(3))
 n.update('direction_info/DIRECTION_INFO_%02d_%02d.json'%(rdi,ri) for rdi in range(3) for ri in range(3))
 n.update('C22_RECORDS/'+m for m in ('identity.json','SUMMARY.json','WRITER_FROZEN.json','heads/calibrated/PROGRESS.json','FINAL_TRAINING.json','SPLIT.json'))
 n.update('C22_RECORDS/source/'+m for m in pins);return n

def validate_capsule(path,pins=None):
 pins=pins or pins_at_root();ci=verified_zip(path,CAPSULE_PROTOCOL)
 with zipfile.ZipFile(path) as z:
  if set(z.namelist())!=required_members(pins):raise ValueError('C23 capsule required member coverage mismatch')
  read=lambda n:z.read('C22_RECORDS/'+n);i=json.loads(read('identity.json'));s=json.loads(read('SUMMARY.json'));f=json.loads(read('WRITER_FROZEN.json'));p=json.loads(read('heads/calibrated/PROGRESS.json'))
  check_records(i,s,f,p,read,pins)
  if hashlib.sha256(z.read('start_checkpoint.pt')).hexdigest()!=START_CP_SHA:raise ValueError('Capsule actual C22 checkpoint changed')
  g=json.loads(z.read('GEOMETRY_START.json'))
  if g.get('head_sha256')!=START_HEAD_SHA or g.get('case')!='xor_10' or g.get('round')!=14:raise ValueError('Capsule geometry at wrong frozen head')
  provenance=json.loads(z.read('PROVENANCE.json'));manifest=json.loads(z.read('MANIFEST.json'));expected=set(manifest)-{'identity.json','PROVENANCE.json'}
  if set(provenance['copied_members'])!=expected:raise ValueError('Capsule origin coverage mismatch')
  for n,r in provenance['copied_members'].items():
   if r['sha256']!=manifest[n] or not isinstance(r['origin_member'],str):raise ValueError('Capsule origin digest mismatch')
  if ci['start_head_sha256']!=START_HEAD_SHA or provenance['parent_c22_sha256']!=ci['parent_c22_sha256']:raise ValueError('Capsule parent identity mismatch')
  with tempfile.TemporaryDirectory(prefix='c23_training_validate_') as tmp:
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
 print('C23 completed C22 source/checkpoint/cohorts verified; importing immutable training capsule.',flush=True)
 with zipfile.ZipFile(parent_path) as source,zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as target:
  def copy_member(old,new):
   h=hashlib.sha256()
   with source.open(old) as src,target.open(new,'w',force_zip64=True) as dst:
    while True:
     block=src.read(8*1024*1024)
     if not block:break
     h.update(block);dst.write(block)
   manifest[new]=h.hexdigest();copied[new]={'origin_member':'C22/'+old,'sha256':h.hexdigest()}
  copy_member(START_MEMBER,'start_checkpoint.pt');copy_member('parent_capsule.zip','training_capsule.zip')
  base='calibration/round_000014/'
  copy_member(base+'START.json','GEOMETRY_START.json')
  for t in previous.training_specs():
   if t['case']=='xor_10':copy_member(base+'states/'+t['id']+'/JACOBIAN.pt','geometry/'+t['id']+'/JACOBIAN.pt')
  for ri in range(3):copy_member(base+'COEFFICIENTS_%02d.pt'%ri,'coefficients/COEFFICIENTS_%02d.pt'%ri)
  for rdi in range(3):
   for ri in range(3):copy_member(base+'DIRECTION_INFO_%02d_%02d.json'%(rdi,ri),'direction_info/DIRECTION_INFO_%02d_%02d.json'%(rdi,ri))
  for n in ('identity.json','SUMMARY.json','WRITER_FROZEN.json','heads/calibrated/PROGRESS.json','FINAL_TRAINING.json','SPLIT.json'):copy_member(n,'C22_RECORDS/'+n)
  for n in pins:copy_member('source/'+n,'C22_RECORDS/source/'+n)
  identity={'protocol':CAPSULE_PROTOCOL,'start_head_sha256':START_HEAD_SHA,'parent_c22_sha256':parent_sha,'historical_full_result_archives_embedded':False}
  provenance={'parent_c22_sha256':parent_sha,'copied_members':copied,'copy_policy':'Unchanged required member bytes. Full historical result ZIPs external, original input unmodified.'}
  for n,value in [('identity.json',identity),('PROVENANCE.json',provenance)]:
   bits=json.dumps(value,indent=2).encode();target.writestr(n,bits);manifest[n]=hashlib.sha256(bits).hexdigest()
  target.writestr('MANIFEST.json',json.dumps(manifest,indent=2))
 validate_capsule(tmp);os.replace(tmp,destination);print('C23 compact C22 parent OK:',destination,destination.stat().st_size,flush=True);return destination
