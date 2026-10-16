"""Read-only C23 result inspection, with bounded diagnostic output."""
import argparse,hashlib,json,zipfile
from collections import Counter
from pathlib import Path

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
  ident=read('identity.json')
  if ident['protocol']!='GPT2_C23_FROZEN_STEP_DIAGNOSIS_A100_V1':raise ValueError('Choose C23 RESULT ZIP')
  specs=[read(n) for n in sorted(names) if n.startswith('candidates/') and n.endswith('/COMPLETE.json')]
  audits=[read(n) for n in sorted(names) if n.startswith('candidate_audits/') and n.endswith('/AUDIT.json')]
  emit('PROGRESS',{'completed_candidate_specs':len(specs),'unique_heads':len({r['candidate_head_sha256'] for r in specs}),
   'route_eligible_specs':sum(r['route_guard_eligible'] for r in specs),'full60_audits':len(audits),
   'zero_probe_records':sum(n.startswith('zero_controls/') and n.endswith('.json') for n in names)})
  if 'SUMMARY.json' in names:emit('SUMMARY',read('SUMMARY.json'))
  else:print('INCOMPLETE_NO_FINAL_SUMMARY')
  reasons=Counter(reason for r in specs for reason in r['route_guard']['reasons'])
  emit('GUARD_REJECTIONS',dict(reasons))
  scan=[]
  for r in specs:
   spec=r['spec'];q=r['route_guard']['after']
   scan.append({'id':spec['id'],'radius':spec['increment_radius'],'ridge':spec['ridge'],'alpha':spec['alpha'],
    'alias_of':r['byte_identical_alias_of'],'route_eligible':r['route_guard_eligible'],'merged_training_passes':len(q['passed_ids']),
    'maximum_R':q['maximum_drift'],'violation_score':q['violation_score'],'prediction':r['prediction'],
    'float64_response_passes_diagnostic':r['float64_aggregation_response_passes_diagnostic'],
    'aggregation_gate_disagreements':r['aggregation_gate_disagreements'],'maximum_aggregation_difference':r['maximum_aggregation_difference']})
  emit('FIXED_ORIGIN_SCAN',scan)
  emit('FULL60_AUDITS',[{'id':a['spec']['id'],'head_sha256':a['head_sha256'],'full60_verified':a['full60_verified'],
   'replay_error':a['replay_error'],'guard':a['guard']} for a in audits])
  if 'BEST_CANDIDATE.json' in names:emit('BEST_DIAGNOSTIC_NOT_COMMITTED',read('BEST_CANDIDATE.json'))
 print('READ_ONLY_COMPLETE; original archive unchanged')

if __name__=='__main__':main()
