"""Read-only verified C20 import into a compact, byte-preserving parent capsule."""
import hashlib,json,os,tempfile,zipfile
from contextlib import ExitStack
from pathlib import Path
import c20_protocol as previous
from c17_protocol import C15_SHA,C9_SHA,sha,verified_zip,verify_open_zip
from confirmation_protocol import CHECKPOINT,CHECKPOINT_SHA,C8_SHA

CAPSULE_PROTOCOL='GPT2_C21_C20_PARENT_CAPSULE_V1'
START_MEMBER='heads/calibrated/checkpoint_000016.pt'
START_CP_SHA='a64691a9428a28a7da6a7bc0be1c68880f280584493850a302330eca8f1d15f8'
START_HEAD_SHA='03f995055d3d21655fcc7d78f473744f56d16adeee2827705c1d363d8492de1e'

def pins_at_root():return json.loads((Path(__file__).parent/'C20_SOURCE_PINS.json').read_text())

def check_records(ident,s,f,p,read,pins):
 if ident.get('protocol')!=previous.PROTOCOL or ident.get('settings')!=previous.SETTINGS or ident.get('source_hashes')!=pins or ident.get('start_head_sha256')!=previous.START_HEAD_SHA:raise ValueError('Different C20 source/settings/start identity')
 for n,h in pins.items():
  if hashlib.sha256(read('source/'+n)).hexdigest()!=h:raise ValueError('Archived C20 source changed: '+n)
 if not s.get('complete') or s.get('decision')!='C20_RAW_TRAINING_FUNCTION_FAILURE' or s.get('writer_head_sha256')!=START_HEAD_SHA:raise ValueError('Requires exact completed recorded C20 head')
 if s['training']['calibrated']['n_requested']!=60 or s['training']['calibrated']['passed']!=36 or s['prospective_local_state_development']['calibrated']['n_requested']!=16:raise ValueError('Wrong actual C20 cohort/results')
 if p.get('checkpoint')!='checkpoint_000016.pt' or p.get('rounds')!=16 or p.get('checkpoint_sha256')!=START_CP_SHA:raise ValueError('Wrong actual C20 checkpoint pointer')
 if f.get('selected_checkpoint')!=START_MEMBER or f.get('head_sha256')!=START_HEAD_SHA:raise ValueError('Wrong C20 frozen writer identity')

def validate_parent(path,pins=None,verified_identity=None):
 with zipfile.ZipFile(path) as z:
  ident=verified_identity or verify_open_zip(z,previous.PROTOCOL)
  s=json.loads(z.read('SUMMARY.json'));f=json.loads(z.read('WRITER_FROZEN.json'));p=json.loads(z.read('heads/calibrated/PROGRESS.json'))
  check_records(ident,s,f,p,z.read,pins or pins_at_root())
  if hashlib.sha256(z.read(START_MEMBER)).hexdigest()!=START_CP_SHA:raise ValueError('Actual C20 checkpoint digest mismatch')
  return {'identity':ident,'summary':s,'freeze':f}

def required_capsule_members(pins):
 names={'identity.json','MANIFEST.json','PROVENANCE.json','start_checkpoint.pt','c9_checkpoint.pt','STATE_SKETCH.pt'}
 names.update('C20_RECORDS/'+n for n in ('identity.json','SUMMARY.json','WRITER_FROZEN.json','heads/calibrated/PROGRESS.json','FINAL_TRAINING.json','SPLIT.json'))
 names.update('C20_RECORDS/source/'+n for n in pins)
 for t in previous.training_specs():names.update('training/'+t['id']+'/'+n for n in ('prepared.pt','teacher.pt','INPUT.pt','REFERENCE.json'))
 names.update('base_geometry/'+sid+'/jacobian.pt' for sid in previous.STATES)
 return names

def validate_capsule(path,pins=None):
 pins=pins or pins_at_root();ci=verified_zip(path,CAPSULE_PROTOCOL)
 with zipfile.ZipFile(path) as z:
  if set(z.namelist())!=required_capsule_members(pins):raise ValueError('Capsule required member coverage mismatch')
  read=lambda n:z.read('C20_RECORDS/'+n)
  ident=json.loads(read('identity.json'));s=json.loads(read('SUMMARY.json'));f=json.loads(read('WRITER_FROZEN.json'));p=json.loads(read('heads/calibrated/PROGRESS.json'))
  check_records(ident,s,f,p,read,pins)
  if hashlib.sha256(z.read('start_checkpoint.pt')).hexdigest()!=START_CP_SHA or hashlib.sha256(z.read('c9_checkpoint.pt')).hexdigest()!=CHECKPOINT_SHA:raise ValueError('Capsule actual checkpoint changed')
  provenance=json.loads(z.read('PROVENANCE.json'));manifest=json.loads(z.read('MANIFEST.json'))
  expected=set(manifest)-{'identity.json','PROVENANCE.json'}
  if set(provenance['copied_members'])!=expected:raise ValueError('Capsule origin coverage mismatch')
  for n,r in provenance['copied_members'].items():
   if r['sha256']!=manifest[n] or not isinstance(r['origin_member'],str):raise ValueError('Capsule copied-member origin digest mismatch')
  if provenance['parent_c20_sha256']!=ci['parent_c20_sha256'] or ci['start_head_sha256']!=START_HEAD_SHA:raise ValueError('Capsule parent identity mismatch')
  return {'identity':ident,'summary':s,'freeze':f,'capsule_identity':ci}

def make_capsule(parent_path,destination):
 """Validate full source archive first, copy required bytes; never write the input."""
 parent_path=Path(parent_path);destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
 info=validate_parent(parent_path);parent_digest=sha(parent_path);pins=pins_at_root();tmp=destination.with_suffix('.tmp')
 print('C21 full C20 CRC/member hashes/source/checkpoint verified; extracting immutable training capsule.',flush=True)
 copied={};manifest={}
 with ExitStack() as stack:
  top=stack.enter_context(zipfile.ZipFile(parent_path))
  temp_root=Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='c21_parent_extract_',dir=destination.parent)))
  def nested(z,member,expected=None):
   # Historical containers can exceed GB: stream to scratch, never retain the full chain in RAM.
   target_path=temp_root/member;digester=hashlib.sha256()
   with z.open(member) as src,open(target_path,'wb') as dst:
    while True:
     block=src.read(8*1024*1024)
     if not block:break
     digester.update(block);dst.write(block)
   if expected is not None and digester.hexdigest()!=expected:raise ValueError('Parent lineage digest mismatch: '+member)
   return stack.enter_context(zipfile.ZipFile(target_path))
  p19=nested(top,'parent_c19.zip',info['identity']['parent_c19_sha256']);p18=nested(p19,'parent_c18.zip');p17=nested(p18,'parent_c17.zip');p15=nested(p17,'parent_c15.zip',C15_SHA)
  c9=nested(p15,'parent_c9.zip',C9_SHA);c8=nested(c9,'parent_c8.zip',C8_SHA)
  print('C21 original C15/C9/C8 lineage verified; no old archive is modified.',flush=True)
  target=stack.enter_context(zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED))
  def copy_member(source,source_member,name,origin):
   bits=source.read(source_member);digest=hashlib.sha256(bits).hexdigest();target.writestr(name,bits)
   manifest[name]=digest;copied[name]={'origin_member':origin,'sha256':digest}
  for n in ('identity.json','SUMMARY.json','WRITER_FROZEN.json','heads/calibrated/PROGRESS.json','FINAL_TRAINING.json','SPLIT.json'):
   copy_member(top,n,'C20_RECORDS/'+n,'C20/'+n)
  for n in pins:copy_member(top,'source/'+n,'C20_RECORDS/source/'+n,'C20/source/'+n)
  copy_member(top,START_MEMBER,'start_checkpoint.pt','C20/'+START_MEMBER)
  copy_member(c9,CHECKPOINT,'c9_checkpoint.pt','C20/ancestry/C9/'+CHECKPOINT)
  copy_member(p15,'STATE_SKETCH.pt','STATE_SKETCH.pt','C20/ancestry/C15/STATE_SKETCH.pt')
  original_ids={t['id'] for t in previous.original_training_specs()}
  for index,t in enumerate(previous.training_specs()):
   sid=t['id'];dest='training/'+sid+'/'
   if sid in original_ids:
    base='states/training/'+sid+'/'
    for n in ('prepared.pt','teacher.pt','REFERENCE.json'):copy_member(p15,base+n,dest+n,'C20/ancestry/C15/'+base+n)
    n='evaluation/training/'+sid+'/ONLINE_INPUT.pt';copy_member(p15,n,dest+'INPUT.pt','C20/ancestry/C15/'+n)
   else:
    base='states/coverage_training/'+sid+'/'
    for n in ('prepared.pt','teacher.pt','INPUT.pt','REFERENCE.json'):copy_member(top,base+n,dest+n,'C20/'+base+n)
   if (index+1)%15==0:print('C21 capsule copied training states',index+1,'/60',flush=True)
  for sid in previous.STATES:
   n='states/'+sid+'/jacobian.pt';copy_member(c8,n,'base_geometry/'+sid+'/jacobian.pt','C20/ancestry/C8/'+n)
  ci={'protocol':CAPSULE_PROTOCOL,'parent_c20_sha256':parent_digest,'start_head_sha256':START_HEAD_SHA,'original_full_parent_archives_embedded':False}
  provenance={'parent_c20_sha256':parent_digest,'copied_members':copied,'copy_policy':'Required original member bytes copied unchanged after full C20 verification. Full historical containers remain external; their whole ZIP hashes cannot be recomputed from this capsule alone.'}
  for n,obj in [('identity.json',ci),('PROVENANCE.json',provenance)]:
   bits=json.dumps(obj,indent=2).encode();target.writestr(n,bits);manifest[n]=hashlib.sha256(bits).hexdigest()
  target.writestr('MANIFEST.json',json.dumps(manifest,indent=2))
 validate_capsule(tmp);os.replace(tmp,destination)
 print('C21 compact parent capsule OK:',destination,'bytes',destination.stat().st_size,flush=True)
 return destination
