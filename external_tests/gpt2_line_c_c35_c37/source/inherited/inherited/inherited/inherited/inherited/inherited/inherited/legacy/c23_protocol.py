"""Predeclared fixed-origin step scan; no training, no accepted-head commitment."""
import hashlib,json,zipfile
from pathlib import Path
import confirmation_protocol as archive
import c22_protocol as previous
from c22_protocol import CASES,STATES,GATES,REVISION,sha,verified_zip,endpoint_gate,episode_gate,target_from_reads,training_specs,original_training_specs,quality,guard,counts
from budget_guard import merge_route,check_replay,candidate_key
from c23_parent import START_MEMBER,START_CP_SHA,START_HEAD_SHA,validate_parent,validate_capsule
PROTOCOL='GPT2_C23_FROZEN_STEP_DIAGNOSIS_A100_V1'
PARENT_PROTOCOL=previous.PROTOCOL
RESULT='gpt2_c23_frozen_step_diagnosis_results'
SETTINGS={
 'response_budget':.002,'kl_budget':.005,'margin_gate':.05,'scan_case':'xor_10','n_training':60,'n_scanned_states':15,
 'radius_grid':[.05,.1,.2],'ridge_grid':[1e-6,1e-8,1e-10],
 'alpha_grid':[1.,.5,.25,.125,.03125,.0078125,.001953125,.00048828125,.000244140625,.0001220703125],
 'layer_frobenius_radius':.5,'cached_geometry_round':14,'new_training_steps':0,'new_Jacobian_builds':0,'new_backward_calls':0,'new_linear_solver_calls':0,
 'zero_repeats_per_phase':3,'zero_phases':['before_scan','after_scan'],'archive_every_candidates':5,
 'guard':'unchanged C22 physical guard: preserve passes/non-response gates and worst all60/original36 drift; improve count or squared budget excess',
 'scan_rule':'all90 specs from the same actual C22 final head; no inherited trust shrinkage, early pair stopping or linear rejection',
 'duplicate_rule':'reuse only byte-identical candidate-head hash; keep all90 requested specs in report',
 'full60_audit_rule':'rank distinct route-eligible heads under original guard; physically audit each head at most once in rank order until one passes or none remain; others are route-eligible, not full60-verified',
 'origin_restore':'restore the exact frozen origin after every trial and audit, including failure/interruption; no accepted-head checkpoint produced',
 'precision_diagnostic':'float64 aggregation of the SAME float32 forward logits and reference logits; primary acceptance remains original float32 aggregation',
 'full_model_float64_forward':False,'float64_aggregation_changes_primary_acceptance':False,
 'new_evaluation_states':0,'fresh_evaluation':False,'controller_dispatch_tested':False,'internal_parameter_update':False,
 'scope':'Development diagnosis on the inspected C22 training states. Temporary output-layer trials and full endpoint tests, cached geometry, no new learning or new-state confirmation.'}

def candidate_specs():
 return [{'id':'candidate_%03d'%(rdi*30+ri*10+ai),'radius_index':rdi,'ridge_index':ri,'alpha_index':ai,'backtrack_index':ai,
          'increment_radius':radius,'ridge':ridge,'alpha':alpha,'case':SETTINGS['scan_case']}
         for rdi,radius in enumerate(SETTINGS['radius_grid']) for ri,ridge in enumerate(SETTINGS['ridge_grid']) for ai,alpha in enumerate(SETTINGS['alpha_grid'])]
def source_hashes(root):return {p.name:sha(p) for p in Path(root).iterdir() if p.is_file() and not p.name.startswith('build_')}
def inherited_check(root):
 for n,h in json.loads((Path(root)/'C23_INHERITED_PINS.json').read_text()).items():
  if sha(Path(root)/n)!=h:raise ValueError('Inherited C22 file changed: '+n)
def input_kind(path):
 try:i=verified_zip(path)
 except KeyError as e:raise ValueError('Choose C22/C23 RESULT ZIP, not source ZIP') from e
 if i['protocol']==PARENT_PROTOCOL:validate_parent(path,verified_identity=i);return 'parent'
 if i['protocol']==PROTOCOL:resume_score(path);return 'resume'
 raise ValueError('C23 requires exact completed C22 or matching own results')
def resume_score(path):
 i=verified_zip(path,PROTOCOL)
 if i.get('settings')!=SETTINGS or i.get('source_hashes')!=source_hashes(Path(__file__).parent) or i.get('origin_head_sha256')!=START_HEAD_SHA:raise ValueError('Different C23 source/settings/origin')
 with zipfile.ZipFile(path) as z:
  if hashlib.sha256(z.read('parent_capsule.zip')).hexdigest()!=i['parent_capsule_sha256']:raise ValueError('C23 parent capsule changed')
  n=set(z.namelist())
  if 'SUMMARY.json' in n:
   s=json.loads(z.read('SUMMARY.json'))
   if not s.get('complete') or s.get('protocol')!=PROTOCOL or s.get('origin_head_sha256')!=START_HEAD_SHA or not s.get('origin_head_unchanged') or s.get('n_completed_candidate_specs')!=90:raise ValueError('Invalid C23 complete summary')
   if sum(x.startswith('candidates/') and x.endswith('/COMPLETE.json') for x in n)!=90 or sum(x.startswith('zero_controls/') and x.endswith('.json') for x in n)!=90:raise ValueError('Incomplete C23 candidate/control denominator')
  return (int('SUMMARY.json' in n),sum(x.startswith('candidates/') and x.endswith('/COMPLETE.json') for x in n),sum(x.startswith('zero_controls/') and x.endswith('.json') for x in n),sum(x.startswith('candidate_audits/') and x.endswith('/AUDIT.json') for x in n),int('BASELINE_REPLAY.json' in n))
def snapshot(out,drive=''):
 archive.PROTOCOL=PROTOCOL;archive.RESULT=RESULT;return archive.snapshot(out,drive)
def decide(records,best,origin_unchanged):
 if len(records)!=90 or {r['spec']['id'] for r in records}!={s['id'] for s in candidate_specs()}:raise ValueError('All90 requested specs required')
 if not origin_unchanged:raise ValueError('Origin changed; diagnosis invalid')
 return 'C23_'+('FULL60_ACCEPTABLE_CANDIDATE_FOUND' if best is not None else 'NO_FULL60_ACCEPTABLE_CANDIDATE_FOUND')
