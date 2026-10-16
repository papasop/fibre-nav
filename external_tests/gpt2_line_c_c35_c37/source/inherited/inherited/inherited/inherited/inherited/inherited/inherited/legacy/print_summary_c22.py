"""Bounded read-only C22 diagnostic output; never print full Gram/residual matrices."""
import argparse,hashlib,json,zipfile
from pathlib import Path
from collections import Counter

def main():
 ap=argparse.ArgumentParser();ap.add_argument('zip_path');a=ap.parse_args();path=Path(a.zip_path)
 def emit(title,obj):print(title+': '+json.dumps(obj,ensure_ascii=False,sort_keys=True))
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 emit('ARCHIVE',{'path':str(path),'bytes':path.stat().st_size,'sha256':h.hexdigest()})
 with zipfile.ZipFile(path) as z:
  names=z.namelist()
  if len(names)!=len(set(names)) or any(Path(n).is_absolute() or '..' in Path(n).parts or '\\' in n for n in names):raise ValueError('Duplicate/unsafe archive paths')
  manifest=json.loads(z.read('MANIFEST.json'))
  if {n for n in names if not n.endswith('/')}!=set(manifest)|{'MANIFEST.json'}:raise ValueError('Manifest coverage mismatch')
  for n,expected in manifest.items():
   d=hashlib.sha256()
   with z.open(n) as f:
    for block in iter(lambda:f.read(8*1024*1024),b''):d.update(block)
   if d.hexdigest()!=expected:raise ValueError('Member digest mismatch: '+n)
  print('CRC_AND_MEMBER_HASHES_OK',len(manifest))
  def read(n):return json.loads(z.read(n))
  identity=read('identity.json')
  if identity['protocol']!='GPT2_C22_BUDGET_CALIBRATION_A100_V1':raise ValueError('Choose C22 RESULT ZIP')
  if 'SUMMARY.json' not in names:print('INCOMPLETE_NO_FINAL_SUMMARY');return
  summary=read('SUMMARY.json');emit('SUMMARY',{k:summary[k] for k in ('protocol','complete','decision','training','original36_training','supplemental24_training','prospective_local_state_development','writer_head_sha256','offline_calibration_rounds','accepted_layer_updates','offline_stop')})
  final=read('FINAL_TRAINING.json');initial=read('C21_INHERITED60_BASELINE.json');passes=Counter();failed=[]
  for sid,row in final.items():
   m=row['metrics'];ok=row['reference_valid'] and m['episode_pass']
   if ok:passes[sid.split('__')[0]]+=1
   else:failed.append({'id':sid,'R':m['drift'],'failed_gates':[g for g,ok in m['gates'].items() if not ok]})
  emit('TRAINING_RECOUNT',{'n':len(final),'passes_by_case':dict(passes),'failed':failed,'lost_prior_passes':[sid for sid in initial if initial[sid]['metrics']['episode_pass'] and not final[sid]['metrics']['episode_pass']]})
  for n in sorted(names):
   if n.startswith('calibration/') and n.endswith('/RESULT.json'):
    r=read(n);c=r.get('chosen');chosen=None;solver=None;direction=None
    if c:
     chosen=next(t for t in r['trials'] if (t['radius_index'],t['ridge_index'],t['backtrack_index'])==(c['radius_index'],c['ridge_index'],c['backtrack_index']))
     di=read(n.rsplit('/',1)[0]+'/DIRECTION_INFO_%02d_%02d.json'%(c['radius_index'],c['ridge_index']));direction={k:di['info'][k] for k in ('trust_clip_factor','unclipped_max_training_increment_norm','unclipped_linear_predicted_R','numerical_Gram_rank')};solver=di['interval_solver_info']
    emit('ROUND',{'round':r['round'],'case':r['case'],'accepted':r['accepted'],'chosen':c,'quality':{k:v for k,v in r['quality'].items() if k!='passed_ids'},'direction':direction,'solver':solver,'prediction_max_R':chosen['prediction']['maximum_drift'] if chosen else None,'prediction_error':chosen.get('maximum_prediction_error') if chosen else None})
  paired=[]
  for n in sorted(names):
   if n.startswith('evaluation/new/') and n.endswith('/COMPLETE.json'):
    r=read(n);x=r['arms']['c21'];y=r['arms']['calibrated'];paired.append({'id':r['id'],'valid':r['reference_valid'],'c21_pass':bool(r['reference_valid'] and x['episode_pass']),'c22_pass':bool(r['reference_valid'] and y['episode_pass']),'c21_R':x['drift'],'c22_R':y['drift']})
  emit('NEW_PAIRED',{'n':len(paired),'gained':[r['id'] for r in paired if r['c22_pass'] and not r['c21_pass']],'lost':[r['id'] for r in paired if r['c21_pass'] and not r['c22_pass']],'states':paired})
  freeze=read('WRITER_FROZEN.json');progress=read('heads/calibrated/PROGRESS.json');member=freeze['selected_checkpoint'];actual=hashlib.sha256(z.read(member)).hexdigest()
  if member!='heads/calibrated/'+progress['checkpoint'] or actual!=progress['checkpoint_sha256'] or freeze['head_sha256']!=summary['writer_head_sha256']:raise ValueError('Frozen pointer identity mismatch')
  emit('FINAL_CHECKPOINT',{'member':member,'sha256':actual,'head_sha256':freeze['head_sha256'],'progress':progress})
 print('READ_ONLY_COMPLETE; original archive unchanged')

if __name__=='__main__':main()
