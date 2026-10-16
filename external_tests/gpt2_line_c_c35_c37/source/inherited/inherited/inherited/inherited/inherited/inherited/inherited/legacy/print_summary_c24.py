"""Bounded, read-only C24 result inspection."""
import argparse,hashlib,json,zipfile
from pathlib import Path
from collections import Counter

def main():
 ap=argparse.ArgumentParser();ap.add_argument('zip_path');args=ap.parse_args();path=Path(args.zip_path)
 def emit(title,obj):print(title+': '+json.dumps(obj,ensure_ascii=False,sort_keys=True))
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 emit('ARCHIVE',{'path':str(path),'bytes':path.stat().st_size,'sha256':h.hexdigest()})
 with zipfile.ZipFile(path) as z:
  names=z.namelist()
  if len(names)!=len(set(names)) or any(Path(n).is_absolute() or '..' in Path(n).parts or '\\' in n for n in names):raise ValueError('Duplicate/unsafe archive paths')
  manifest=json.loads(z.read('MANIFEST.json'))
  if {n for n in names if not n.endswith('/')}!=set(manifest)|{'MANIFEST.json'}:raise ValueError('Manifest coverage mismatch')
  for n,expected in manifest.items():
   d=hashlib.sha256()
   with z.open(n) as f:
    for b in iter(lambda:f.read(8*1024*1024),b''):d.update(b)
   if d.hexdigest()!=expected:raise ValueError('Member digest mismatch: '+n)
  print('CRC_AND_MEMBER_HASHES_OK',len(manifest))
  def read(n):return json.loads(z.read(n))
  if read('identity.json')['protocol']!='GPT2_C24_RELINEARISED_CALIBRATION_A100_V1':raise ValueError('Choose C24 RESULT ZIP')
  if 'SUMMARY.json' in names:emit('SUMMARY',read('SUMMARY.json'))
  else:print('INCOMPLETE_NO_FINAL_SUMMARY')
  for n in sorted(names):
   if n.startswith('calibration/') and n.endswith('/RESULT.json'):
    r=read(n);emit('ROUND',{k:r[k] for k in ('round','case','accepted','chosen','head_sha256','quality','scan')})
  if 'FINAL_TRAINING.json' in names:
   rows=read('FINAL_TRAINING.json');failed=[];passed=Counter()
   for sid,row in rows.items():
    m=row['metrics'];ok=row['reference_valid'] and all(m['gates'].values())
    if ok:passed[sid.split('__')[0]]+=1
    else:failed.append({'id':sid,'R':m['drift'],'KL':m['kl'],'failed_gates':[g for g,ok in m['gates'].items() if not ok],'response_residual':m['response_residual']})
   emit('TRAINING_RECOUNT',{'n':len(rows),'passes_by_case':dict(passed),'failed':failed})
  paired=[]
  for n in sorted(names):
   if n.startswith('evaluation/inspected/') and n.endswith('/COMPLETE.json'):
    r=read(n);a,b=r['arms']['c23'],r['arms']['calibrated'];paired.append({'id':r['id'],'valid':r['reference_valid'],'c23_pass':bool(r['reference_valid'] and a['episode_pass']),'c24_pass':bool(r['reference_valid'] and b['episode_pass']),'c23_R':a['drift'],'c24_R':b['drift']})
  emit('INSPECTED16_PAIRED',paired)
  if 'WRITER_FROZEN.json' in names:
   f=read('WRITER_FROZEN.json');p=read('heads/calibrated/PROGRESS.json');member=f['selected_checkpoint'];actual=hashlib.sha256(z.read(member)).hexdigest()
   if member!='heads/calibrated/'+p['checkpoint'] or actual!=p['checkpoint_sha256']:raise ValueError('Actual frozen checkpoint pointer mismatch')
   emit('FROZEN_CHECKPOINT',{'member':member,'checkpoint_sha256':actual,'head_sha256':f['head_sha256'],'progress':p})
 print('READ_ONLY_COMPLETE; original archive unchanged')

if __name__=='__main__':main()
