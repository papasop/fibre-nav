"""Offline joint response calibration of the frozen writer's output layers."""
import hashlib,io,json,zipfile
from pathlib import Path
import confirmation_protocol as archive
import c19_protocol as previous
from c17_protocol import (CASES,STATES,GATES,REVISION,C15_SHA,C9_SHA,CHECKPOINT,sha,verified_zip,verify_open_zip,
 endpoint_gate,episode_gate,target_from_reads,counts)
from coverage_guard import original_training_specs,supplemental_training_specs,training_specs,quality,guard,TRAIN_SEED_BASE
PROTOCOL='GPT2_C20_COVERAGE_CALIBRATION_A100_V1'
PARENT_PROTOCOL=previous.PROTOCOL
RESULT='gpt2_c20_coverage_calibration_results'
START_MEMBER='heads/calibrated/checkpoint_000016.pt'
START_CP_SHA='f78d533a39870996457cfc231598991479ea1a8bee38711bac195f24fbc73648'
START_HEAD_SHA='351fe0d1327e7441202c319fdb839f9f9ef358dc54e7c86d9cab4d8d212677f7'
SETTINGS={'response_budget':.002,'kl_budget':.005,'margin_gate':.05,'n_training':60,'n_original_training':36,'n_supplemental_training':24,'training_seed_base':TRAIN_SEED_BASE,'n_requested_new_states':16,
 'max_calibration_rounds':16,'branch_rejection_limit':2,'head_backtracking':8,'ridge_grid':[1e-6,1e-8,1e-10],
 'normalisation_floor':1e-12,'gram_rank_relative_cutoff':1e-8,'layer_frobenius_radius':.5,'training_increment_radius':.05,
 'initial_branch_trust':1.,'minimum_branch_trust':1e-4,'guard_relative_score_improvement':1e-6,
 'functional_response_replay_tolerance':2e-6,'layer_width_with_bias':65,
 'calibration':'90 measured residual equations per route; minimum-norm ridge Gauss-Newton output-layer direction; row-normalised Kronecker Gram formed in float64',
 'linear_protection':'skip physical trial if actual float32 proposal displacement predicts loss of any currently passing state in the selected route',
 'physical_guard':'fifteen changed-route physical forwards plus invariant other45; full60 physical replay before commit; preserve passes/non-response gates, all60 worst drift and original36 worst drift; improve pass count or squared drift',
 'candidate_comparison':'at the same round-start head, test every ridge; first physically admissible backtracking scale per ridge; compare by most passes, then smallest maximum drift, then smallest score, then ridge/backtrack index',
 'unchanged_routes':'all inputs/frozen input layers and unselected output tensors must be byte-identical; predictions use fixed saved input features; recompute all60 before commit',
 'trainable_domain':'only the final weight and bias of each existing writer expert; all input layers, normalisation and runtime specifications frozen',
 'branch_trust':'independent per operand route; a rejection in one route does not shrink the others',
 'optimiser_restored':False,'new_Adam_steps':0,'test_seed_base':2015031,'test_norms':[.375,.75],
 'correction_ridge':1e-6,'correction_max_steps':8,'correction_backtracking':6,
 'offline_geometry_allowed':True,'online_Jacobian_builds':0,'online_SVD_calls':0,'online_correction_steps':0,'online_backward_calls':0,
 'online_proposals_per_arm':1,'teacher_fallback_allowed':False,'state_replacement_allowed':False,'test_based_checkpoint_selection':False,
 'archive_every_rounds':1,'archive_every_geometry_states':15,'archive_every_new_states':2,'archive_every_training_states':4,'fresh_state_seeds':True,'fresh_evaluation':False,
 'independent_model_seed':False,'internal_parameter_update':False,'controller_dispatch_tested':False,
 'test_generation_after_final_writer_freeze':True,'old_inspected_C15_C17_C18_C19_new_states_used_for_training':False,
 'scope':'Prospective raw learned-writer development on known XOR classes in the same GPT-2 chart. Offline joint calibration and state preparation use external autograd/geometry; frozen online writing is one proposal and external commit, without repair or teacher access.'}

def test_specs():return [{'id':c+'__c20_new%d'%i,'case':c,'base_id':c+'__v%d'%(i%3),'seed':SETTINGS['test_seed_base']+100*k+i,'norm':SETTINGS['test_norms'][i//2],'cohort':'new'} for k,c in enumerate(CASES) for i in range(4)]
def source_hashes(root):return {p.name:sha(p) for p in Path(root).iterdir() if p.is_file() and not p.name.startswith('build_')}
def inherited_check(root):
 for n,h in json.loads((Path(root)/'C20_INHERITED_PINS.json').read_text()).items():
  if sha(Path(root)/n)!=h:raise ValueError('Inherited module changed: '+n)

def validate_parent(path,pins=None):
 with zipfile.ZipFile(path) as z:
  ident=verify_open_zip(z,PARENT_PROTOCOL);s=json.loads(z.read('SUMMARY.json'));f=json.loads(z.read('WRITER_FROZEN.json'))
  pins=pins or json.loads((Path(__file__).parent/'C19_SOURCE_PINS.json').read_text())
  if ident.get('settings')!=previous.SETTINGS or ident.get('source_hashes')!=pins or ident.get('start_head_sha256')!=previous.START_HEAD_SHA:raise ValueError('Different C19 protocol/source/start head')
  for n,h in pins.items():
   if hashlib.sha256(z.read('source/'+n)).hexdigest()!=h:raise ValueError('Archived C19 source mismatch: '+n)
  if hashlib.sha256(z.read('parent_c18.zip')).hexdigest()!=ident.get('parent_c18_sha256'):raise ValueError('Nested original C18 changed')
  with zipfile.ZipFile(io.BytesIO(z.read('parent_c18.zip'))) as p18,zipfile.ZipFile(io.BytesIO(p18.read('parent_c17.zip'))) as p17:
   if hashlib.sha256(p17.read('parent_c15.zip')).hexdigest()!=C15_SHA:raise ValueError('Exact original C15 training archive changed')
  if not s.get('complete') or s.get('decision')!='C19_RAW_TRAINING_FUNCTION_FAILURE' or s.get('writer_head_sha256')!=START_HEAD_SHA:raise ValueError('Requires completed recorded C19 final head')
  if s['training']['calibrated']['n_requested']!=36 or s['training']['calibrated']['passed']!=22 or s['prospective_local_state_development']['calibrated']['n_requested']!=16:raise ValueError('Wrong C19 cohort or actual result')
  p=json.loads(z.read('heads/calibrated/PROGRESS.json'))
  if p.get('checkpoint')!='checkpoint_000016.pt' or p.get('rounds')!=16 or p.get('checkpoint_sha256')!=START_CP_SHA:raise ValueError('Wrong actual C19 checkpoint pointer')
  if hashlib.sha256(z.read(START_MEMBER)).hexdigest()!=START_CP_SHA:raise ValueError('Actual C19 checkpoint digest mismatch')
  if f.get('head_sha256')!=START_HEAD_SHA or f.get('selected_checkpoint')!=START_MEMBER:raise ValueError('C19 frozen head mismatch')
  return {'identity':ident,'summary':s,'freeze':f}

def input_kind(path):
 try:i=verified_zip(path)
 except KeyError as e:raise ValueError('Choose a RESULT ZIP, not source/launcher ZIP') from e
 if i['protocol']==PARENT_PROTOCOL:validate_parent(path);return 'parent'
 if i['protocol']==PROTOCOL:resume_score(path);return 'resume'
 raise ValueError('C20 requires completed C19 results or its own compatible resume')

def decide(train,new):
 if len(train)!=60 or len(new)!=16:raise ValueError('All requested60/16 states must remain in denominator')
 if not all(r['reference_valid'] for r in train+new):return 'C20_STATE_REFERENCE_FAILURE'
 if counts(train,'calibrated')['passed']!=60:return 'C20_RAW_TRAINING_FUNCTION_FAILURE'
 return 'C20_RAW_LOCAL_STATE_'+('PASS' if counts(new,'calibrated')['passed']==16 else 'FAIL')

def resume_score(path):
 i=verified_zip(path,PROTOCOL)
 if i.get('settings')!=SETTINGS or i.get('source_hashes')!=source_hashes(Path(__file__).parent) or i.get('start_head_sha256')!=START_HEAD_SHA:raise ValueError('Different C20 source/settings/start head')
 with zipfile.ZipFile(path) as z:
  if hashlib.sha256(z.read('parent_c19.zip')).hexdigest()!=i['parent_c19_sha256']:raise ValueError('C19 parent digest mismatch')
  names=set(z.namelist());rounds=0
  if 'heads/calibrated/PROGRESS.json' in names:
   p=json.loads(z.read('heads/calibrated/PROGRESS.json'));n=p['checkpoint']
   if Path(n).name!=n or '\\' in n or not n.endswith('.pt'):raise ValueError('Unsafe checkpoint pointer')
   if hashlib.sha256(z.read('heads/calibrated/'+n)).hexdigest()!=p['checkpoint_sha256']:raise ValueError('Real C20 checkpoint digest mismatch')
   rounds=p['rounds']
   if type(rounds)!=int or not 0<=rounds<=SETTINGS['max_calibration_rounds']:raise ValueError('Invalid actual round count')
  return (int('SUMMARY.json' in names),int('WRITER_FROZEN.json' in names),sum(n.startswith('evaluation/new/') and n.endswith('/COMPLETE.json') for n in names),rounds,
   sum(n.startswith('calibration/') and n.endswith('/JACOBIAN.pt') for n in names),
   sum(n.startswith('states/coverage_training/') and n.endswith('/REFERENCE.json') for n in names))

def snapshot(out,drive=''):
 archive.PROTOCOL=PROTOCOL;archive.RESULT=RESULT;return archive.snapshot(out,drive)
