"""Guarded offline head refinement; prospective frozen raw writes on known classes."""
import hashlib,json,math,zipfile
from pathlib import Path
import confirmation_protocol as archive
import c16_protocol as c16
from c16_protocol import CASES,STATES,REVISION,C9_SHA,HEAD_SHA,CHECKPOINT,sha,verified_zip,verify_open_zip,endpoint_gate,episode_gate,target_from_reads
PROTOCOL='GPT2_C17_GUARDED_RAW_WRITER_A100_V1'
RESULT='gpt2_c17_guarded_raw_writer_results'
C15_SHA='38ff708fa145d7e0baffae0f0ae2093937e79d3a7347cebaf47685a2061b58d1'
C16_SHA='f3deec2e4c31da82c177e5580013a0defe1a639ebbd7d1c9b3597e783409cdf7'
WARM_MEMBER='heads/warm/checkpoint_001000.pt'
WARM_CP_SHA='fabbf47d553ec2357f18caf5bde5d711a919b53830e562fb579a358afce92de6'
WARM_HEAD_SHA='24261f3fdc4e243153fd9697a14944f272e0b20cf5332dfa6ac4474b28416e0a'
GATES=('bits_match','output_flip','unselected_preserved','response','KL','exact_read_margin','development_read_margin')
NON_RESPONSE=tuple(g for g in GATES if g!='response')
SETTINGS={'response_budget':.002,'kl_budget':.005,'margin_gate':.05,
 'n_training':36,'n_requested_new_states':16,'start_checkpoint':WARM_MEMBER,'start_head_sha256':WARM_HEAD_SHA,
 'max_offline_attempts':64,'max_consecutive_rejections':8,'head_lr':2e-6,'minimum_lr':1e-10,
 'head_backtracking':8,'gradient_clip':1.,'weight_decay':0.,'internal_response_target':.001,
 'teacher_mse_weight':.1,'guard_relative_score_improvement':1e-6,
 'guard':'all 36 actual-forward gates; preserve every current complete pass and every non-response gate; worst drift cannot rise; improve pass count or mean squared physical drift',
 'loss':'offline response mean squared residual on all nine states of one operand route, plus training teacher increment MSE; fresh AdamW; no per-state online fit',
 'routing':'two actual operand-read signs; requested output bit supplied',
 'optimizer_rollback':'all rejected attempts restore head AND optimiser; accepted fractional steps keep proposed Adam moments',
 'test_seed_base':1715031,'test_norms':[.375,.75],'replicates_per_norm':2,
 'correction_ridge':1e-6,'correction_max_steps':8,'correction_backtracking':6,
 'offline_geometry_allowed':True,'online_Jacobian_builds':0,'online_SVD_calls':0,'online_correction_steps':0,
 'online_backward_calls':0,'online_proposals_per_arm':1,'online_parameter_commits_per_arm':1,
 'teacher_fallback_allowed':False,'state_replacement_allowed':False,'test_based_checkpoint_selection':False,
 'archive_every_attempts':1,'archive_every_new_states':2,'fresh_state_seeds':True,'fresh_evaluation':False,
 'independent_model_seed':False,'internal_parameter_update':False,'controller_dispatch_tested':False,
 'test_generation_after_final_writer_freeze':True,'old_C15_inspected_new_states_used_for_training':False,
 'scope':'Prospective local-state development in the same GPT-2/chart/prompts and four trained XOR classes. Offline preparation uses geometry. One frozen externally applied learned increment per arm; no native update or unseen-task claim.'}

def training_specs():return [t for t in c16.state_specs() if t['cohort']=='training']
def test_specs():
 return [{'id':case+'__c17_new%d'%i,'case':case,'base_id':case+'__v%d'%(i%3),'seed':SETTINGS['test_seed_base']+100*c+i,
  'norm':SETTINGS['test_norms'][i//2],'cohort':'new'} for c,case in enumerate(CASES) for i in range(4)]

def source_hashes(root):return {p.name:sha(p) for p in Path(root).iterdir() if p.is_file() and not p.name.startswith('build_')}
def inherited_check(root):
 for n,h in json.loads((Path(root)/'C17_INHERITED_PINS.json').read_text()).items():
  if sha(Path(root)/n)!=h:raise ValueError('Inherited source changed: '+n)

def validate_c15(path):
 if sha(path)!=C15_SHA:raise ValueError('Requires the exact completed recorded C15 parent')
 r=c16.validate_parent(path)
 if r['checkpoints']['warm']['member']!=WARM_MEMBER or r['checkpoints']['warm']['checkpoint_sha256']!=WARM_CP_SHA:raise ValueError('True warm1000 checkpoint identity mismatch')
 return r

def input_kind(path):
 ident=verified_zip(path);p=ident['protocol']
 if p==PROTOCOL:resume_score(path);return 'resume'
 if p==c16.PARENT_PROTOCOL:validate_c15(path);return 'c15'
 if p==c16.PROTOCOL:
  pins=json.loads((Path(__file__).parent/'C16_SOURCE_PINS.json').read_text())
  if ident.get('settings')!=c16.SETTINGS or ident.get('source_hashes')!=pins:raise ValueError('Different C16 source/settings')
  with zipfile.ZipFile(path) as z:
   for n,h in pins.items():
    if hashlib.sha256(z.read('source/'+n)).hexdigest()!=h:raise ValueError('C16 archived source mismatch: '+n)
   s=json.loads(z.read('SUMMARY.json'))
   if not s.get('complete') or not s.get('stage_attribution_premises_met') or s.get('checkpoint_head_sha256',{}).get('warm1000')!=WARM_HEAD_SHA:raise ValueError('C16 diagnosis incomplete or wrong checkpoint')
   if hashlib.sha256(z.read('parent_c15.zip')).hexdigest()!=C15_SHA:raise ValueError('C16 embedded C15 mismatch')
  return 'c16'
 raise ValueError('Upload C15/C16 RESULT ZIP or matching C17 resume, not source/launcher ZIP')

def ordered_metrics(rows):
 ids=[t['id'] for t in training_specs()]
 if set(rows)!=set(ids):raise ValueError('All 36 fixed training states required')
 for sid in ids:
  r=rows[sid]
  if not r.get('reference_valid'):raise ValueError('Invalid training positive reference: '+sid)
  if not math.isfinite(r['metrics']['drift']) or r['metrics']['drift']<0:raise ValueError('Invalid physical response drift')
  if set(r['metrics']['gates'])!=set(GATES):raise ValueError('Missing gate data')
 return [(sid,rows[sid]['metrics']) for sid in ids]

def quality(rows):
 r=ordered_metrics(rows)
 return {'passed_ids':[sid for sid,m in r if all(m['gates'].values())],
  'maximum_drift':max(m['drift'] for _,m in r),
  'score':sum((m['drift']/SETTINGS['response_budget'])**2 for _,m in r)/len(r)}

def guard(previous,candidate):
 a=quality(previous);b=quality(candidate);reasons=[]
 if not set(a['passed_ids']).issubset(b['passed_ids']):reasons.append('LOST_PREVIOUS_COMPLETE_PASS')
 for sid,m in ordered_metrics(previous):
  for g in NON_RESPONSE:
   if m['gates'][g] and not candidate[sid]['metrics']['gates'][g]:reasons.append('LOST_NON_RESPONSE_GATE:'+sid+':'+g)
 if b['maximum_drift']>a['maximum_drift']:reasons.append('WORST_RESPONSE_DRIFT_INCREASED')
 if len(b['passed_ids'])<=len(a['passed_ids']) and not b['score']<a['score']*(1-SETTINGS['guard_relative_score_improvement']):reasons.append('NO_DECLARED_TRAINING_IMPROVEMENT')
 return {'accepted':not reasons,'reasons':reasons,'before':a,'after':b}

def counts(rows,arm):
 return {'n_requested':len(rows),'valid_state_references':sum(r['reference_valid'] for r in rows),
 'passed':sum(bool(r['reference_valid'] and r['arms'][arm]['episode_pass']) for r in rows),
 'gate_counts':{g:sum(bool(r['arms'][arm]['gates'].get(g)) for r in rows) for g in GATES},
 'maximum_response_drift':max([r['arms'][arm]['drift'] for r in rows],default=None)}

def decide(train,new):
 if len(train)!=36 or len(new)!=16:raise ValueError('Fixed 36/16 denominators required')
 if not all(r['reference_valid'] for r in train+new):return 'C17_STATE_REFERENCE_FAILURE'
 if counts(train,'guarded')['passed']!=36:return 'C17_RAW_TRAINING_FUNCTION_FAILURE'
 return 'C17_RAW_LOCAL_STATE_'+('PASS' if counts(new,'guarded')['passed']==16 else 'FAIL')

def resume_score(path):
 i=verified_zip(path,PROTOCOL)
 if i.get('settings')!=SETTINGS or i.get('source_hashes')!=source_hashes(Path(__file__).parent) or i.get('parent_c15_sha256')!=C15_SHA:raise ValueError('C17 source/settings/parent changed')
 with zipfile.ZipFile(path) as z:
  if hashlib.sha256(z.read('parent_c15.zip')).hexdigest()!=C15_SHA:raise ValueError('Parent digest mismatch')
  names=set(z.namelist());attempts=0
  if 'heads/guarded/PROGRESS.json' in names:
   p=json.loads(z.read('heads/guarded/PROGRESS.json'));n=p['checkpoint']
   if Path(n).name!=n or '\\' in n or not n.endswith('.pt'):raise ValueError('Unsafe checkpoint pointer')
   if hashlib.sha256(z.read('heads/guarded/'+n)).hexdigest()!=p['checkpoint_sha256']:raise ValueError('Checkpoint digest mismatch')
   attempts=p['attempts']
   if type(attempts)!=int or not 0<=attempts<=SETTINGS['max_offline_attempts']:raise ValueError('Invalid actual attempt progress')
  return (int('SUMMARY.json' in names),int('WRITER_FROZEN.json' in names),sum(n.startswith('evaluation/new/') and n.endswith('/COMPLETE.json') for n in names),attempts)

def snapshot(out,drive=''):
 archive.PROTOCOL=PROTOCOL;archive.RESULT=RESULT;return archive.snapshot(out,drive)
