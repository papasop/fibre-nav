"""Read-only paired checkpoint diagnosis on the full recorded C15 cohort."""
import hashlib,json,math,zipfile
from pathlib import Path
import confirmation_protocol as archive
import c15_protocol as c15
from c15_protocol import (CASES,STATES,REVISION,C9_SHA,HEAD_SHA,CHECKPOINT,CHECKPOINT_SHA,sha,verified_zip,verify_open_zip,
 endpoint_gate,episode_gate,target_from_reads)
PROTOCOL='GPT2_C16_CHECKPOINT_DIAGNOSIS_A100_V1'
PARENT_PROTOCOL=c15.PROTOCOL
RESULT='gpt2_c16_checkpoint_diagnosis_results'
FINAL_HEAD_SHA='36309dbe0774584470962b44273f38ac42e17f048b778dc793b27a9372c1c44a'
ARMS=('c9_baseline','warm1000','functional720')
SETTINGS={'response_budget':.002,'kl_budget':.005,'margin_gate':.05,
 'parameter_replay_tolerance':2e-6,'recorded_metric_tolerance':1e-4,'fresh_feature_tolerance':1e-4,
 'raw_input_mismatch_is_diagnostic_not_replacement':True,'primary_input':'exact saved C15 ONLINE_INPUT.pt',
 'n_training':36,'n_inspected_new_states':16,'new_training_steps':0,'new_states':0,
 'Jacobian_builds':0,'SVD_calls':0,'correction_steps':0,'archive_every_states':8,
 'fresh_evaluation':False,'native_update':False,'controller_dispatch_tested':False,
 'selection':'diagnosis only; all three heads retained; no production head selected',
 'scope':'Same saved C15 states and original gates; independent checkpoint replays and paired actual forward diagnostics only'}

def state_specs():
 train=[{'id':s,'case':s.split('__')[0],'cohort':'training'} for s in STATES]+[
  {'id':s['id'],'case':s['case'],'cohort':'training'} for s in c15.train_specs()]
 new=[{'id':s['id'],'case':s['case'],'cohort':'new'} for s in c15.test_specs()]
 return train+new

def source_hashes(root):return {p.name:sha(p) for p in Path(root).iterdir() if p.is_file() and not p.name.startswith('build_')}

def inherited_check(root):
 for name,h in json.loads((Path(root)/'INHERITED_SOURCE_HASHES.json').read_text()).items():
  if sha(Path(root)/name)!=h:raise ValueError('Inherited C15 module changed: '+name)

def checkpoint_pointer(z,stage,total):
 base='heads/'+stage+'/';p=json.loads(z.read(base+'PROGRESS.json'));n=p['checkpoint']
 if Path(n).name!=n or '\\' in n or n!=('checkpoint_%06d.pt'%total):raise ValueError('Wrong or unsafe final checkpoint pointer')
 if p.get('steps')!=total:raise ValueError('Incomplete C15 checkpoint stage')
 member=base+n;digest=hashlib.sha256(z.read(member)).hexdigest()
 if digest!=p['checkpoint_sha256']:raise ValueError('Actual head checkpoint digest mismatch')
 return {'member':member,'checkpoint_sha256':digest,'steps':total,'stage':stage}

def validate_parent(path,pins=None):
 with zipfile.ZipFile(path) as z:
  ident=verify_open_zip(z,PARENT_PROTOCOL);s=json.loads(z.read('SUMMARY.json'))
  if ident.get('settings')!=c15.SETTINGS or ident.get('parent_c9_sha256')!=C9_SHA:raise ValueError('Different C15 protocol or C9 lineage')
  pins=pins or json.loads((Path(__file__).parent/'C15_SOURCE_PINS.json').read_text())
  if ident.get('source_hashes')!=pins:raise ValueError('C15 source differs from the delivered V1')
  for n,h in pins.items():
   if hashlib.sha256(z.read('source/'+n)).hexdigest()!=h:raise ValueError('Archived C15 source member changed: '+n)
  if not s.get('complete') or s.get('training_raw',{}).get('n_requested')!=36 or s.get('new_state_raw',{}).get('n_requested')!=16:raise ValueError('Complete C15 36/16 results required')
  freeze=json.loads(z.read('WRITER_FROZEN.json'))
  if s.get('writer_head_sha256')!=FINAL_HEAD_SHA or freeze.get('head_sha256')!=FINAL_HEAD_SHA:raise ValueError('Requires the final C15 head recorded in the supplied completed log')
  if s.get('baseline_c9_head_sha256')!=HEAD_SHA or freeze.get('selected_checkpoint')!='heads/functional/checkpoint_000720.pt':raise ValueError('Wrong frozen checkpoint identity')
  if hashlib.sha256(z.read('parent_c9.zip')).hexdigest()!=C9_SHA:raise ValueError('Original C9 parent changed')
  for t in state_specs():
   base='states/'+t['cohort']+'/'+t['id']+'/'
   evalbase='evaluation/'+t['cohort']+'/'+t['id']+'/'
   for n in ['prepared.pt','teacher.pt','FEATURES.pt','REFERENCE.json']:z.getinfo(base+n)
   for n in ['ONLINE_INPUT.pt','c15_raw.pt','c15_raw.json','frozen_c9_raw.pt','frozen_c9_raw.json','COMPLETE.json']:z.getinfo(evalbase+n)
  points={a:checkpoint_pointer(z,a,n) for a,n in [('warm',1000),('functional',720)]}
  return {'identity':ident,'summary':s,'freeze':freeze,'checkpoints':points}

def cohort_counts(rows,arm):
 m=[r['arms'][arm] for r in rows]
 return {'n_requested':len(rows),'valid_references':sum(bool(r['reference_valid']) for r in rows),
  'passed':sum(bool(r['reference_valid'] and x['metrics'].get('episode_pass')) for r,x in zip(rows,m)),
  'gates':{g:sum(bool(x['metrics'].get('gates',{}).get(g)) for x in m) for g in ('bits_match','output_flip','unselected_preserved','response','KL','exact_read_margin','development_read_margin')},
  'maximum_response_drift':max([x['metrics']['drift'] for x in m],default=0.),
  'median_teacher_delta_error_l2':median([x['teacher_error']['l2'] for x in m])}

def median(values):
 v=sorted(values);n=len(v)
 return None if not n else v[n//2] if n%2 else (v[n//2-1]+v[n//2])/2

def paired(rows):
 if not rows:raise ValueError('Nonempty paired cohort required')
 pairs=[(r['arms']['warm1000']['metrics'],r['arms']['functional720']['metrics']) for r in rows]
 return {'n':len(rows),'complete_gained':sum(not a.get('episode_pass') and b.get('episode_pass',False) for a,b in pairs),
  'complete_lost':sum(a.get('episode_pass',False) and not b.get('episode_pass') for a,b in pairs),
  'response_gained':sum(not a['gates']['response'] and b['gates']['response'] for a,b in pairs),
  'response_lost':sum(a['gates']['response'] and not b['gates']['response'] for a,b in pairs),
  'drift_increased':sum(b['drift']>a['drift'] for a,b in pairs),'drift_decreased':sum(b['drift']<a['drift'] for a,b in pairs),
  'drift_equal':sum(b['drift']==a['drift'] for a,b in pairs),
  'median_drift_change':median([b['drift']-a['drift'] for a,b in pairs])}

def interpretation(training,replay_consistent):
 if len(training)!=36:raise ValueError('All 36 training states required for stage diagnosis')
 if not replay_consistent:return 'REFERENCE_OR_REPLAY_MISMATCH_STAGE_ATTRIBUTION_UNRESOLVED'
 a=cohort_counts(training,'warm1000');b=cohort_counts(training,'functional720')
 if a['passed']==36 and b['passed']<36:return 'FUNCTIONAL_STAGE_LOSES_COMPLETE_TRAINING_ACCEPTANCE'
 if a['passed']<36 and (b['passed']<a['passed'] or b['maximum_response_drift']>a['maximum_response_drift']):return 'WARM_ACCEPTANCE_INSUFFICIENT_FUNCTIONAL_STAGE_WORSENS_RECORDED_COHORT'
 if a['passed']<36 and b['passed']<36:return 'BOTH_CHECKPOINTS_FAIL_COMPLETE_TRAINING_ACCEPTANCE'
 if a['passed']<36 and b['passed']==36:return 'FUNCTIONAL_STAGE_REACHES_TRAINING_ACCEPTANCE'
 return 'BOTH_CHECKPOINTS_PASS_TRAINING_ACCEPTANCE'

def resume_score(path):
 ident=verified_zip(path,PROTOCOL)
 if ident.get('settings')!=SETTINGS or ident.get('source_hashes')!=source_hashes(Path(__file__).parent):raise ValueError('Different C16 source/settings')
 with zipfile.ZipFile(path) as z:
  if hashlib.sha256(z.read('parent_c15.zip')).hexdigest()!=ident['parent_c15_sha256']:raise ValueError('C15 parent corrupt')
  names=z.namelist();return (int('SUMMARY.json' in names),sum(n.startswith('evaluation/') and n.endswith('/COMPLETE.json') for n in names),sum(n.startswith('evaluation/') and n.endswith('/RESULT.json') for n in names))

def snapshot(out,drive=''):
 archive.PROTOCOL=PROTOCOL;archive.RESULT=RESULT
 return archive.snapshot(out,drive)
