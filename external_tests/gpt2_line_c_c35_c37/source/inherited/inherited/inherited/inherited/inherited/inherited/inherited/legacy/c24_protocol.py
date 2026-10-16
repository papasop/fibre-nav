"""C24 offline recalibration from an actually verified C23 candidate."""
import hashlib,json,zipfile
from pathlib import Path
import confirmation_protocol as archive
import c23_protocol as previous
import c22_protocol as legacy
from c22_protocol import CASES,STATES,GATES,REVISION,sha,verified_zip,endpoint_gate,episode_gate,target_from_reads,original_training_specs,training_specs,quality,guard,counts
from budget_guard import merge_route,check_replay,candidate_key
from c24_parent import START_HEAD_SHA,START_SPEC,validate_parent,validate_capsule
PROTOCOL='GPT2_C24_RELINEARISED_CALIBRATION_A100_V1'
RESULT='gpt2_c24_relinearised_calibration_results'
PARENT_PROTOCOL=previous.PROTOCOL
SETTINGS={**legacy.SETTINGS,
 'max_calibration_rounds':16,'scan_case':'xor_10','radius_grid':[.05,.1,.2],'alpha_grid':[1.,.5,.25,.125,.03125,.0078125,.001953125,.00048828125,.000244140625,.0001220703125],
 'archive_every_candidates':5,'archive_every_geometry_states':5,'fixed_start_head_sha256':START_HEAD_SHA,
 'linear_protection':'diagnostic only; no candidate excluded by the linear prediction',
 'candidate_comparison':'all90 fixed radius/ridge/alpha specs from each exact round start; byte-identical heads reuse measurements; all specs retained; no inherited trust or early pair stopping',
 'branch_trust':'not used: every new geometry round resets the same declared alpha grid',
 'stop_rule':'all60 pass, max16 completed rounds, or a complete grid has no full60 acceptable distinct candidate; no claim of infeasibility',
 'evaluation_state_policy':'reuse EXACT sixteen C22 already inspected prepared/input/reference member bytes; no generation, resampling, promotion to training or evaluation-based selection',
 'n_new_preparation_states':0,'fresh_state_seeds':False,'test_generation_after_final_writer_freeze':False,
 'test_access_after_final_writer_freeze':True,'evaluation_influenced_protocol_design':True,
 'full60_physical_audit_before_each_commit':True,
 'scope':'Development-only offline relinearised calibration from C23 candidate_022. Fixed original60 training and previously inspected C22 sixteen evaluation states; frozen online writer makes one proposal with external commit, no online geometry/repair, no independent/native claim.'}

def candidate_specs():
 return [{'id':'candidate_%03d'%(rdi*30+ri*10+ai),'radius_index':rdi,'ridge_index':ri,'alpha_index':ai,'backtrack_index':ai,'increment_radius':r,'ridge':ridge,'alpha':alpha,'case':SETTINGS['scan_case']}
  for rdi,r in enumerate(SETTINGS['radius_grid']) for ri,ridge in enumerate(SETTINGS['ridge_grid']) for ai,alpha in enumerate(SETTINGS['alpha_grid'])]
def test_specs():return legacy.test_specs()
def source_hashes(root):return {p.name:sha(p) for p in Path(root).iterdir() if p.is_file() and not p.name.startswith('build_')}
def inherited_check(root):
 for n,h in json.loads((Path(root)/'C24_INHERITED_PINS.json').read_text()).items():
  if sha(Path(root)/n)!=h:raise ValueError('Inherited C23 file changed: '+n)
def input_kind(path):
 try:i=verified_zip(path)
 except KeyError as e:raise ValueError('Choose C23/C24 RESULT ZIP, not source ZIP') from e
 if i['protocol']==PARENT_PROTOCOL:validate_parent(path,verified_identity=i);return 'parent'
 if i['protocol']==PROTOCOL:resume_score(path);return 'resume'
 raise ValueError('C24 requires completed C23 or matching own results')
def decide(train,new):
 if len(train)!=60 or len(new)!=16:raise ValueError('All60/16 requested states required')
 if not all(r['reference_valid'] for r in train+new):return 'C24_STATE_REFERENCE_FAILURE'
 if counts(train,'calibrated')['passed']!=60:return 'C24_RAW_TRAINING_FUNCTION_FAILURE'
 return 'C24_RAW_INSPECTED_LOCAL_STATE_'+('PASS' if counts(new,'calibrated')['passed']==16 else 'FAIL')
def resume_score(path):
 i=verified_zip(path,PROTOCOL)
 if i.get('settings')!=SETTINGS or i.get('source_hashes')!=source_hashes(Path(__file__).parent) or i.get('start_head_sha256')!=START_HEAD_SHA:raise ValueError('Different C24 source/settings/start')
 with zipfile.ZipFile(path) as z:
  if hashlib.sha256(z.read('parent_capsule.zip')).hexdigest()!=i['parent_capsule_sha256']:raise ValueError('C24 capsule changed')
  names=set(z.namelist());rounds=0
  if 'heads/calibrated/PROGRESS.json' in names:
   p=json.loads(z.read('heads/calibrated/PROGRESS.json'));n=p['checkpoint']
   if Path(n).name!=n or '\\' in n or not n.endswith('.pt'):raise ValueError('Unsafe checkpoint pointer')
   if hashlib.sha256(z.read('heads/calibrated/'+n)).hexdigest()!=p['checkpoint_sha256']:raise ValueError('Real C24 checkpoint digest mismatch')
   rounds=p['rounds']
   if type(rounds)!=int or not 0<=rounds<=SETTINGS['max_calibration_rounds']:raise ValueError('Invalid actual round count')
  if 'SUMMARY.json' in names:
   s=json.loads(z.read('SUMMARY.json'))
   if not s.get('complete') or s.get('protocol')!=PROTOCOL or s.get('actual_start_head_sha256')!=START_HEAD_SHA or sum(n.startswith('evaluation/inspected/') and n.endswith('/COMPLETE.json') for n in names)!=16:raise ValueError('Incomplete final C24 summary')
  return (int('SUMMARY.json' in names),int('WRITER_FROZEN.json' in names),sum(n.startswith('evaluation/inspected/') and n.endswith('/COMPLETE.json') for n in names),rounds,sum(n.startswith('calibration/') and n.endswith('/COMPLETE.json') for n in names),sum(n.startswith('calibration/') and n.endswith('/JACOBIAN.pt') for n in names))
def snapshot(out,drive=''):
 archive.PROTOCOL=PROTOCOL;archive.RESULT=RESULT;return archive.snapshot(out,drive)
