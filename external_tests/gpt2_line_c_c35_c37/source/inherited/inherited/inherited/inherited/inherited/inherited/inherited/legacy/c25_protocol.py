"""C25 four-route coverage expansion; prior evaluated states explicitly promoted."""
import hashlib,json,zipfile
from pathlib import Path
import confirmation_protocol as archive
import c24_protocol as previous
from c24_protocol import CASES,STATES,GATES,REVISION,sha,verified_zip,endpoint_gate,episode_gate,target_from_reads,original_training_specs,counts
from c25_guard import training_specs,original60_specs,promoted_specs,quality,guard,merge_route,check_replay,candidate_key,choose_case
from c25_parent import START_MEMBER,START_CP_SHA,START_HEAD_SHA,validate_parent,validate_capsule
PROTOCOL='GPT2_C25_FOUR_ROUTE_COVERAGE_A100_V1'
RESULT='gpt2_c25_four_route_coverage_results'
PARENT_PROTOCOL=previous.PROTOCOL
SETTINGS={**previous.SETTINGS,'n_training':76,'n_original60':60,'n_promoted_inspected':16,'n_training_per_route':19,'max_calibration_rounds':32,
 'scan_cases':list(CASES),'response_equations_per_route':114,'branch_stall_rule':'a full fixed grid has no actual full76 acceptable candidate; skip that route for this run; not infeasibility certification',
 'physical_guard':'full76 before every commit; preserve all original60 and every prior complete pass/non-response gate; worst76 drift nonincreasing; improve pass count or squared budget excess; original60 drift may vary INSIDE unchanged 0.002 budget',
 'guard_change_from_C24':'expanded76 denominator; all60 acceptance protected, without requiring its measured maximum to be nonincreasing inside budget',
 'evaluation_state_policy':'C22 inspected16 promoted to training; disjoint16 predeclared fixed-seed local states generated only after final writer freeze; no resampling or evaluation-based selection',
 'test_seed_base':2515031,'test_norms':[.375,.75],'n_requested_new_states':16,'fresh_state_seeds':True,'test_generation_after_final_writer_freeze':True,
 'test_access_after_final_writer_freeze':True,'new_state_generation':True,'n_new_preparation_states':16,'fresh_evaluation':False,'promoted_old_evaluation_states':16,'evaluation_influenced_protocol_design':True,
 'scope':'Same-model/chart development: original60 plus16 explicitly promoted inspected states; offline four-route calibration. Frozen one-proposal raw writing on16 prospectively fixed new local preparation seeds, no online geometry/repair, no unseen-XOR-class or independent/native claim.'}
SETTINGS.pop('scan_case',None)
def candidate_specs(case):
 if case not in CASES:raise ValueError('Unknown case')
 return [{**s,'case':case} for s in previous.candidate_specs()]
def test_specs():return [{'id':c+'__c25_new%d'%i,'case':c,'base_id':c+'__v%d'%(i%3),'seed':SETTINGS['test_seed_base']+100*k+i,'norm':SETTINGS['test_norms'][i//2],'cohort':'new'} for k,c in enumerate(CASES) for i in range(4)]
def source_hashes(root):return {p.name:sha(p) for p in Path(root).iterdir() if p.is_file() and not p.name.startswith('build_')}
def inherited_check(root):
 for n,h in json.loads((Path(root)/'C25_INHERITED_PINS.json').read_text()).items():
  if sha(Path(root)/n)!=h:raise ValueError('Inherited C24 file changed: '+n)
def input_kind(path):
 try:i=verified_zip(path)
 except KeyError as e:raise ValueError('Choose C24/C25 RESULT ZIP, not source ZIP') from e
 if i['protocol']==PARENT_PROTOCOL:validate_parent(path,verified_identity=i);return 'parent'
 if i['protocol']==PROTOCOL:resume_score(path);return 'resume'
 raise ValueError('C25 requires exact completed C24 or matching own results')
def decide(train,new):
 if len(train)!=76 or len(new)!=16:raise ValueError('All76/16 requested states required')
 if not all(r['reference_valid'] for r in train+new):return 'C25_STATE_REFERENCE_FAILURE'
 if counts(train,'calibrated')['passed']!=76:return 'C25_RAW_EXPANDED_TRAINING_FUNCTION_FAILURE'
 return 'C25_RAW_PROSPECTIVE_LOCAL_STATE_'+('PASS' if counts(new,'calibrated')['passed']==16 else 'FAIL')
def resume_score(path):
 i=verified_zip(path,PROTOCOL)
 if i.get('settings')!=SETTINGS or i.get('source_hashes')!=source_hashes(Path(__file__).parent) or i.get('start_head_sha256')!=START_HEAD_SHA:raise ValueError('Different C25 source/settings/start')
 with zipfile.ZipFile(path) as z:
  if hashlib.sha256(z.read('parent_capsule.zip')).hexdigest()!=i['parent_capsule_sha256']:raise ValueError('C25 capsule changed')
  names=set(z.namelist());rounds=0
  if 'heads/calibrated/PROGRESS.json' in names:
   p=json.loads(z.read('heads/calibrated/PROGRESS.json'));n=p['checkpoint']
   if Path(n).name!=n or '\\' in n or not n.endswith('.pt'):raise ValueError('Unsafe checkpoint pointer')
   if hashlib.sha256(z.read('heads/calibrated/'+n)).hexdigest()!=p['checkpoint_sha256']:raise ValueError('Actual C25 checkpoint digest mismatch')
   rounds=p['rounds']
   if type(rounds)!=int or not 0<=rounds<=SETTINGS['max_calibration_rounds']:raise ValueError('Invalid real round count')
  if 'SUMMARY.json' in names:
   s=json.loads(z.read('SUMMARY.json'))
   if not s.get('complete') or s.get('protocol')!=PROTOCOL or s.get('actual_start_head_sha256')!=START_HEAD_SHA or sum(n.startswith('evaluation/new/') and n.endswith('/COMPLETE.json') for n in names)!=16:raise ValueError('Incomplete C25 final summary')
  return (int('SUMMARY.json' in names),int('WRITER_FROZEN.json' in names),sum(n.startswith('evaluation/new/') and n.endswith('/COMPLETE.json') for n in names),rounds,sum(n.startswith('calibration/') and n.endswith('/COMPLETE.json') for n in names),sum(n.startswith('calibration/') and n.endswith('/JACOBIAN.pt') for n in names))
def snapshot(out,drive=''):
 archive.PROTOCOL=PROTOCOL;archive.RESULT=RESULT;return archive.snapshot(out,drive)
