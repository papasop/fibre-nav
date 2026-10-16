"""Read-only bounded summary for C25 result archives."""
import argparse,hashlib,json,zipfile
from pathlib import Path
from collections import Counter

def main():
 ap=argparse.ArgumentParser();ap.add_argument('zip_path');args=ap.parse_args();path=Path(args.zip_path)
 def emit(title,value):print(title+': '+json.dumps(value,ensure_ascii=False,sort_keys=True))
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
  if read('identity.json')['protocol']!='GPT2_C25_FOUR_ROUTE_COVERAGE_A100_V1':raise ValueError('Choose C25 RESULT ZIP')
  if 'SUMMARY.json' in names:emit('SUMMARY',read('SUMMARY.json'))
  else:print('INCOMPLETE_NO_FINAL_SUMMARY')
  for n in sorted(names):
   if n.startswith('calibration/') and n.endswith('/RESULT.json'):
    r=read(n);q=r['quality'];emit('ROUND',{'round':r['round'],'case':r['case'],'accepted':r['accepted'],'chosen':r['chosen'],'head_sha256':r['head_sha256'],'passed':len(q['passed_ids']),'maximum_R':q['maximum_drift'],'original60_maximum_R':q['original60_maximum_drift'],'violation_score':q['violation_score'],'scan':r['scan'],'stalled_cases':r['stalled_cases']})
  if 'FINAL_TRAINING.json' in names:
   rows=read('FINAL_TRAINING.json');failed=[];passed=Counter()
   for sid,row in rows.items():
    m=row['metrics'];ok=row['reference_valid'] and all(m['gates'].values())
    if ok:passed[sid.split('__')[0]]+=1
    else:failed.append({'id':sid,'R':m['drift'],'KL':m['kl'],'failed_gates':[g for g,v in m['gates'].items() if not v],'response_residual':m['response_residual']})
   emit('EXPANDED76_RECOUNT',{'n':len(rows),'passes_by_case':dict(passed),'failed':failed})
  new=[]
  for n in sorted(names):
   if n.startswith('evaluation/new/') and n.endswith('/COMPLETE.json'):
    r=read(n);a,b=r['arms']['c24'],r['arms']['calibrated'];new.append({'id':r['id'],'valid':r['reference_valid'],'c24_pass':bool(r['reference_valid'] and a['episode_pass']),'c25_pass':bool(r['reference_valid'] and b['episode_pass']),'c24_R':a['drift'],'c25_R':b['drift'],'failed_gates':[g for g,v in b['gates'].items() if not v],'c25_response_residual':b['response_residual']})
  emit('PROSPECTIVE16_PAIRED',new)
  if 'WRITER_FROZEN.json' in names:
   f=read('WRITER_FROZEN.json');p=read('heads/calibrated/PROGRESS.json');n=f['selected_checkpoint'];actual=hashlib.sha256(z.read(n)).hexdigest()
   if n!='heads/calibrated/'+p['checkpoint'] or actual!=p['checkpoint_sha256']:raise ValueError('Actual frozen checkpoint pointer mismatch')
   emit('FROZEN_CHECKPOINT',{'member':n,'checkpoint_sha256':actual,'head_sha256':f['head_sha256'],'progress':p})
 print('READ_ONLY_COMPLETE; original archive unchanged')

if __name__=='__main__':main()
