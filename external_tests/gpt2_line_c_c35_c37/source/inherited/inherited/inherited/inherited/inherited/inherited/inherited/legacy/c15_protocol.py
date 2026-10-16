"""Fixed offline expansion/training, then one raw proposal with no online repair."""
import hashlib,json,zipfile
from pathlib import Path
import confirmation_protocol as c9c
from confirmation_protocol import (C9_SHA,HEAD_SHA,CHECKPOINT,CHECKPOINT_SHA,CASES,STATES,REVISION,C6_SHA,C8_SHA,
 sha,verified_zip,verify_open_zip,validate_parent,inherited_check,source_hashes,endpoint_gate,episode_gate,target_from_reads)
PROTOCOL='GPT2_C15_RAW_WRITER_A100_V1'
PARENT_PROTOCOL='GPT2_C9_STATE_GENERALISATION_A100_V1'
RESULT='gpt2_c15_raw_writer_results'
SETTINGS={'response_budget':.002,'kl_budget':.005,'margin_gate':.05,
 'correction_ridge':1e-6,'correction_max_steps':8,'correction_backtracking':6,
 'training_seed_base':415031,'test_seed_base':515031,'sketch_seed':615031,'head_seed':715031,
 'sketch_dim':64,'extras_dim':70,'input_dim':3147,'expert_width':64,
 'new_training_states':24,'total_requested_training_states':36,'n_requested_new_states':16,
 'warm_steps':1000,'warm_lr':.001,'functional_steps':720,'functional_lr':.0001,
 'checkpoint_every':20,'warm_archive_every':250,'functional_archive_every':120,
 'gradient_clip':5.,'weight_decay':0.,'internal_response_target':.001,'internal_read_target':.10,
 'loss_weights':{'teacher_mse':.1,'response':1.,'read':1.,'KL':1.},
 'offline_geometry_allowed':True,'online_Jacobian_builds':0,'online_SVD_calls':0,'online_correction_steps':0,
 'online_proposals_per_arm':1,'online_parameter_commits_per_arm':1,'teacher_fallback_allowed':False,
 'test_based_checkpoint_selection':False,'requested_output_bit_supplied':True,
 'fresh_state_seeds':True,'fresh_evaluation':False,'independent_model_seed':False,
 'internal_parameter_update':False,'controller_dispatch_tested':False,
 'routing':'first two actual binary readout-margin signs; same four trained XOR classes',
 'state_features':'original C9 features plus protected-response residual and fixed random sketch of current B; one head call',
 'test_generation_after_final_writer_freeze':True,'state_replacement_allowed':False,
 'scope':'Uncorrected external learned-writer evaluation on new local states in a known model/chart and known input classes; no native update or unseen-XOR claim'}

def train_specs():
 return [{'id':case+'__c15_train%d'%i,'case':case,'base_id':case+'__v%d'%(i%3),'seed':SETTINGS['training_seed_base']+100*c+i,'norm':.375 if i<3 else .75}
  for c,case in enumerate(CASES) for i in range(6)]

def test_specs():
 return [{'id':case+'__c15_new%d'%i,'case':case,'base_id':case+'__v%d'%(i%3),'seed':SETTINGS['test_seed_base']+100*c+i,'norm':.375 if i<2 else .75}
  for c,case in enumerate(CASES) for i in range(4)]

def counts(rows):
 return {'n_requested':len(rows),'valid_state_references':sum(bool(r['reference_valid']) for r in rows),
  'raw_passed':sum(bool(r['reference_valid'] and r['raw'].get('episode_pass')) for r in rows),
  'baseline_c9_raw_passed':sum(bool(r['reference_valid'] and r['baseline'].get('episode_pass')) for r in rows),
  'raw_gate_counts':{g:sum(bool(r['raw'].get('gates',{}).get(g)) for r in rows) for g in ('bits_match','output_flip','unselected_preserved','response','KL','exact_read_margin','development_read_margin')}}

def decision(training,test):
 if len(training)!=36 or len(test)!=16:raise ValueError('Complete 36/16 requested denominators required')
 if not all(r['reference_valid'] for r in training+test):return 'C15_STATE_REFERENCE_FAILURE'
 if not all(r['raw'].get('episode_pass') for r in training):return 'C15_RAW_TRAINING_FUNCTION_FAILURE'
 return 'C15_RAW_LOCAL_STATE_'+('PASS' if all(r['raw'].get('episode_pass') for r in test) else 'FAIL')

def snapshot(out,drive=''):
 # Reuse only the generic verified archive writer, with an explicit own-version protocol.
 c9c.PROTOCOL=PROTOCOL;c9c.RESULT=RESULT
 return c9c.snapshot(out,drive)

def resume_score(path):
 ident=verified_zip(path,PROTOCOL)
 if ident.get('settings')!=SETTINGS or ident.get('parent_c9_sha256')!=C9_SHA or ident.get('source_hashes')!=source_hashes(Path(__file__).parent):raise ValueError('Different C15 source/parent/settings')
 with zipfile.ZipFile(path) as z:
  if hashlib.sha256(z.read('parent_c9.zip')).hexdigest()!=C9_SHA:raise ValueError('Parent C9 corrupt')
  names=set(z.namelist());steps=[]
  for stage,total in [('warm',1000),('functional',720)]:
   base='heads/'+stage+'/';n=base+'PROGRESS.json'
   if n not in names:steps.append(0);continue
   p=json.loads(z.read(n));name=p['checkpoint']
   if Path(name).name!=name or '\\' in name or not name.endswith('.pt'):raise ValueError('Unsafe head checkpoint path')
   if hashlib.sha256(z.read(base+name)).hexdigest()!=p['checkpoint_sha256']:raise ValueError('Head checkpoint digest mismatch')
   if not isinstance(p['steps'],int) or not 0<=p['steps']<=total:raise ValueError('Invalid actual head steps')
   steps.append(p['steps'])
  return (int('SUMMARY.json' in names),sum(n.startswith('evaluation/') and n.endswith('/COMPLETE.json') for n in names),*steps,sum(n.startswith('states/training/') and n.endswith('/REFERENCE.json') for n in names))
