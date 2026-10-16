"""Verified C23 candidate plus unchanged C22 inspected evaluation states."""
import hashlib,json,os,tempfile,zipfile
from pathlib import Path
import c23_protocol as previous
import c23_parent as inherited
import c22_protocol as old
from c17_protocol import sha,verified_zip,verify_open_zip
CAPSULE_PROTOCOL='GPT2_C24_C23_CANDIDATE_CAPSULE_V1'
START_HEAD_SHA='86b71471cd25e76ea518f1f5b411e8a8cdf365c35acd40afc72f7fead3fe9522'
START_SPEC={'id':'candidate_022','radius_index':0,'ridge_index':2,'alpha_index':2,'backtrack_index':2,'increment_radius':.05,'ridge':1e-10,'alpha':.25,'case':'xor_10'}
LAYER_MEMBER='best_candidate/CANDIDATE_LAYER.pt'
AUDIT_MEMBER='candidate_audits/candidate_022/AUDIT.json'
RECORDS=('identity.json','SUMMARY.json','BEST_CANDIDATE.json',AUDIT_MEMBER)

def pins_at_root():return json.loads((Path(__file__).parent/'C23_SOURCE_PINS.json').read_text())
def check_records(i,s,b,a,read,pins):
 if i.get('protocol')!=previous.PROTOCOL or i.get('settings')!=previous.SETTINGS or i.get('source_hashes')!=pins or i.get('origin_head_sha256')!=previous.START_HEAD_SHA:raise ValueError('Different C23 source/settings/origin')
 for n,h in pins.items():
  if hashlib.sha256(read('source/'+n)).hexdigest()!=h:raise ValueError('C23 source changed: '+n)
 if not s.get('complete') or s.get('decision')!='C23_FULL60_ACCEPTABLE_CANDIDATE_FOUND' or s.get('n_completed_candidate_specs')!=90 or s.get('n_unique_candidate_heads')!=70 or not s.get('origin_head_unchanged') or not s.get('no_head_committed'):raise ValueError('Exact complete C23 scan required')
 if s.get('best_candidate')!=b or b.get('spec')!=START_SPEC or b.get('candidate_head_sha256')!=START_HEAD_SHA or not b.get('full60_verified') or b.get('head_committed') is not False:raise ValueError('Wrong verified C23 candidate')
 if a.get('spec')!=START_SPEC or a.get('head_sha256')!=START_HEAD_SHA or not a.get('full60_verified') or a.get('replay_error') is not None or not a['guard']['accepted']:raise ValueError('C23 candidate full60 audit required')
 rows=a['full60_rows']
 if set(rows)!={t['id'] for t in old.training_specs()}:raise ValueError('C23 all60 denominator required')
 q=old.quality(rows)
 if len(q['passed_ids'])!=54 or q!=b['quality'] or q!=a['guard']['after']:raise ValueError('C23 candidate actual54 quality mismatch')
 if not old.guard(rows,rows)['before']==q:raise ValueError('Invalid audit metrics')

def validate_parent(path,pins=None,verified_identity=None):
 with zipfile.ZipFile(path) as z:
  i=verified_identity or verify_open_zip(z,previous.PROTOCOL);s=json.loads(z.read('SUMMARY.json'));b=json.loads(z.read('BEST_CANDIDATE.json'));a=json.loads(z.read(AUDIT_MEMBER))
  check_records(i,s,b,a,z.read,pins or pins_at_root())
  if hashlib.sha256(z.read('parent_capsule.zip')).hexdigest()!=i['parent_capsule_sha256']:raise ValueError('C23 inherited capsule changed')
  z.getinfo(LAYER_MEMBER)
  if sum(n.startswith('candidates/') and n.endswith('/COMPLETE.json') for n in z.namelist())!=90 or sum(n.startswith('zero_controls/') and n.endswith('.json') for n in z.namelist())!=90:raise ValueError('C23 scan/control coverage mismatch')
  return {'identity':i,'summary':s,'best':b,'audit':a}

def validate_companion(path,c23_info):
 info=inherited.validate_parent(path)
 if sha(path)!=c23_info['identity']['parent_c22_sha256']:raise ValueError('Select the original C22 archive used by C23')
 with zipfile.ZipFile(path) as z:
  for t in old.test_specs():
   for n in ('prepared.pt','teacher.pt','INPUT.pt','REFERENCE.json'):z.getinfo('states/new/'+t['id']+'/'+n)
   z.getinfo('evaluation/new/'+t['id']+'/COMPLETE.json')
 return info

def required_members(pins):
 names={'identity.json','MANIFEST.json','PROVENANCE.json','c23_capsule.zip','CANDIDATE_LAYER.pt'}
 names.update('C23_RECORDS/'+n for n in RECORDS);names.update('C23_RECORDS/source/'+n for n in pins)
 for t in old.test_specs():
  names.update('evaluation_states/'+t['id']+'/'+n for n in ('prepared.pt','teacher.pt','INPUT.pt','REFERENCE.json'))
  names.add('C22_NEW_RESULTS/'+t['id']+'/COMPLETE.json')
 return names

def validate_capsule(path,pins=None):
 pins=pins or pins_at_root();ci=verified_zip(path,CAPSULE_PROTOCOL)
 with zipfile.ZipFile(path) as z:
  if set(z.namelist())!=required_members(pins):raise ValueError('C24 capsule coverage mismatch')
  read=lambda n:z.read('C23_RECORDS/'+n)
  i,s,b,a=[json.loads(read(n)) for n in RECORDS];check_records(i,s,b,a,read,pins)
  if ci['start_head_sha256']!=START_HEAD_SHA or ci['c23_capsule_sha256']!=i['parent_capsule_sha256']:raise ValueError('Candidate capsule identity mismatch')
  provenance=json.loads(z.read('PROVENANCE.json'));manifest=json.loads(z.read('MANIFEST.json'))
  if set(provenance['copied_members'])!=set(manifest)-{'identity.json','PROVENANCE.json'}:raise ValueError('C24 origin receipts coverage mismatch')
  for n,r in provenance['copied_members'].items():
   if r['sha256']!=manifest[n] or not isinstance(r['origin_member'],str):raise ValueError('C24 origin receipt mismatch')
  if provenance['parent_c23_sha256']!=ci['parent_c23_sha256'] or provenance['companion_c22_sha256']!=i['parent_c22_sha256']:raise ValueError('C24 parent SHA mismatch')
  for t in old.test_specs():
   ref=json.loads(z.read('evaluation_states/'+t['id']+'/REFERENCE.json'));row=json.loads(z.read('C22_NEW_RESULTS/'+t['id']+'/COMPLETE.json'))
   if ref['spec']!=t or row['id']!=t['id'] or ref['valid']!=row['reference_valid']:raise ValueError('Inspected evaluation state identity changed')
  with tempfile.TemporaryDirectory(prefix='c24_parent_verify_') as tmp:
   nested=Path(tmp)/'c23_capsule.zip'
   with z.open('c23_capsule.zip') as src,nested.open('wb') as dst:
    while True:
     block=src.read(8*1024*1024)
     if not block:break
     dst.write(block)
   if sha(nested)!=i['parent_capsule_sha256']:raise ValueError('Nested C23 capsule digest mismatch')
   inherited.validate_capsule(nested)
  return {'identity':i,'summary':s,'best':b,'audit':a,'capsule_identity':ci}

def make_capsule(parent_path,companion_path,destination):
 parent_path=Path(parent_path);companion_path=Path(companion_path);destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
 info=validate_parent(parent_path);validate_companion(companion_path,info);parent_sha=sha(parent_path);pins=pins_at_root();tmp=destination.with_suffix('.tmp');manifest={};copied={}
 with zipfile.ZipFile(parent_path) as parent,zipfile.ZipFile(companion_path) as companion,zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as target:
  def copy(src,old,new,label):
   h=hashlib.sha256()
   with src.open(old) as inp,target.open(new,'w',force_zip64=True) as out:
    while True:
     block=inp.read(8*1024*1024)
     if not block:break
     h.update(block);out.write(block)
   manifest[new]=h.hexdigest();copied[new]={'origin_member':label+'/'+old,'sha256':h.hexdigest()}
  copy(parent,'parent_capsule.zip','c23_capsule.zip','C23');copy(parent,LAYER_MEMBER,'CANDIDATE_LAYER.pt','C23')
  for n in RECORDS:copy(parent,n,'C23_RECORDS/'+n,'C23')
  for n in pins:copy(parent,'source/'+n,'C23_RECORDS/source/'+n,'C23')
  for t in old.test_specs():
   for n in ('prepared.pt','teacher.pt','INPUT.pt','REFERENCE.json'):copy(companion,'states/new/'+t['id']+'/'+n,'evaluation_states/'+t['id']+'/'+n,'C22')
   copy(companion,'evaluation/new/'+t['id']+'/COMPLETE.json','C22_NEW_RESULTS/'+t['id']+'/COMPLETE.json','C22')
  ci={'protocol':CAPSULE_PROTOCOL,'start_head_sha256':START_HEAD_SHA,'parent_c23_sha256':parent_sha,'c23_capsule_sha256':info['identity']['parent_capsule_sha256'],'historical_full_result_archives_embedded':False}
  provenance={'parent_c23_sha256':parent_sha,'companion_c22_sha256':info['identity']['parent_c22_sha256'],'copied_members':copied,'policy':'Required immutable member bytes only; no full result ZIP chain; original archives untouched.'}
  for n,v in [('identity.json',ci),('PROVENANCE.json',provenance)]:
   raw=json.dumps(v,indent=2).encode();target.writestr(n,raw);manifest[n]=hashlib.sha256(raw).hexdigest()
  target.writestr('MANIFEST.json',json.dumps(manifest,indent=2))
 validate_capsule(tmp);os.replace(tmp,destination);print('C24 compact candidate and original16 states imported:',destination,destination.stat().st_size,flush=True);return destination
