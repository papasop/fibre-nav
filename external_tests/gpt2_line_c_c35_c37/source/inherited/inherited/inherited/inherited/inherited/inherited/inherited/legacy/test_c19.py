import ast,hashlib,io,json,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
import c19_protocol as p
from namespace_audit import global_audit
from test_c16 import write_zip
ROOT=Path(__file__).parent

def parent_fixture():
 source=b'# source fixture';pins={'a.py':hashlib.sha256(source).hexdigest()};c15=b'parent15fixture';cp=b'checkpoint-fixture'
 buf=io.BytesIO()
 with zipfile.ZipFile(buf,'w') as nested:nested.writestr('parent_c15.zip',c15)
 parent=buf.getvalue()
 m={'identity.json':json.dumps({'protocol':p.PARENT_PROTOCOL,'settings':p.previous.SETTINGS,'source_hashes':pins,'start_head_sha256':p.previous.START_HEAD_SHA,'parent_c17_sha256':hashlib.sha256(parent).hexdigest()}).encode(),
  'source/a.py':source,'parent_c17.zip':parent,p.START_MEMBER:cp,
  'SUMMARY.json':json.dumps({'complete':True,'decision':'C18_RAW_TRAINING_FUNCTION_FAILURE','writer_head_sha256':p.START_HEAD_SHA,'training':{'calibrated':{'n_requested':36,'passed':18}},'prospective_local_state_development':{'calibrated':{'n_requested':16}}}).encode(),
  'WRITER_FROZEN.json':json.dumps({'head_sha256':p.START_HEAD_SHA,'selected_checkpoint':p.START_MEMBER}).encode(),
  'heads/calibrated/PROGRESS.json':json.dumps({'checkpoint':'checkpoint_000012.pt','rounds':12,'checkpoint_sha256':hashlib.sha256(cp).hexdigest()}).encode()}
 return m,pins,hashlib.sha256(c15).hexdigest(),hashlib.sha256(cp).hexdigest()

def metric(d):
 gates={n:True for n in p.GATES};gates['response']=d<=.002;return {'drift':d,'gates':gates,'episode_pass':all(gates.values())}
def rows(n,d=.001,valid=True):return [{'reference_valid':valid,'arms':{a:metric(d) for a in ('c18','calibrated')}} for _ in range(n)]
class Tests(unittest.TestCase):
 def test_true_start_pointer(self):self.assertEqual(p.START_MEMBER,'heads/calibrated/checkpoint_000012.pt');self.assertEqual(len(p.START_CP_SHA),64)
 def test_36_training_unchanged(self):self.assertEqual(p.training_specs(),p.previous.training_specs());self.assertEqual(len(p.training_specs()),36)
 def test_16_new_states_balanced(self):
  self.assertEqual(len(p.test_specs()),16)
  for c in p.CASES:self.assertEqual(sum(t['case']==c for t in p.test_specs()),4)
 def test_new_seeds_disjoint_C17(self):self.assertFalse(set(t['seed'] for t in p.test_specs())&set(t['seed'] for t in p.previous.test_specs()))
 def test_no_inspected_states_promoted(self):self.assertFalse(p.SETTINGS['old_inspected_C15_C17_C18_new_states_used_for_training'])
 def test_thresholds_unchanged(self):self.assertEqual([p.SETTINGS[k] for k in ('response_budget','kl_budget','margin_gate')],[.002,.005,.05])
 def test_no_online_geometry(self):
  for k in ('online_Jacobian_builds','online_SVD_calls','online_correction_steps','online_backward_calls'):self.assertEqual(p.SETTINGS[k],0)
 def test_not_native_or_independent_confirmation(self):self.assertFalse(p.SETTINGS['internal_parameter_update']);self.assertFalse(p.SETTINGS['fresh_evaluation']);self.assertFalse(p.SETTINGS['independent_model_seed'])
 def test_full_denominator_required(self):
  with self.assertRaises(ValueError):p.decide(rows(36),rows(15))
 def test_full_pass(self):self.assertEqual(p.decide(rows(36),rows(16)),'C19_RAW_LOCAL_STATE_PASS')
 def test_training_failure(self):self.assertEqual(p.decide(rows(36,.01),rows(16)),'C19_RAW_TRAINING_FUNCTION_FAILURE')
 def test_new_failure(self):self.assertEqual(p.decide(rows(36),rows(16,.01)),'C19_RAW_LOCAL_STATE_FAIL')
 def test_invalid_new_blocks_pass(self):self.assertEqual(p.decide(rows(36),rows(16,valid=False)),'C19_STATE_REFERENCE_FAILURE')
 def test_parent_structural_fixture(self):
  m,pins,ph,ch=parent_fixture()
  with tempfile.TemporaryDirectory() as d,patch.object(p,'C15_SHA',ph),patch.object(p,'START_CP_SHA',ch):
   f=Path(d)/'p.zip';write_zip(f,m);self.assertTrue(p.validate_parent(f,pins)['summary']['complete'])
 def test_wrong_checkpoint_digest_rejected(self):
  m,pins,ph,ch=parent_fixture();m[p.START_MEMBER]=b'changed'
  with tempfile.TemporaryDirectory() as d,patch.object(p,'C15_SHA',ph),patch.object(p,'START_CP_SHA',ch):
   f=Path(d)/'p.zip';write_zip(f,m)
   with self.assertRaises(ValueError):p.validate_parent(f,pins)
 def test_wrong_head_rejected(self):
  m,pins,ph,ch=parent_fixture();r=json.loads(m['WRITER_FROZEN.json']);r['head_sha256']='wrong';m['WRITER_FROZEN.json']=json.dumps(r).encode()
  with tempfile.TemporaryDirectory() as d,patch.object(p,'C15_SHA',ph),patch.object(p,'START_CP_SHA',ch):
   f=Path(d)/'p.zip';write_zip(f,m)
   with self.assertRaises(ValueError):p.validate_parent(f,pins)
 def test_incomplete_parent_rejected(self):
  m,pins,ph,ch=parent_fixture();r=json.loads(m['SUMMARY.json']);r['complete']=False;m['SUMMARY.json']=json.dumps(r).encode()
  with tempfile.TemporaryDirectory() as d,patch.object(p,'C15_SHA',ph),patch.object(p,'START_CP_SHA',ch):
   f=Path(d)/'p.zip';write_zip(f,m)
   with self.assertRaises(ValueError):p.validate_parent(f,pins)
 def test_changed_source_rejected(self):
  m,pins,ph,ch=parent_fixture();m['source/a.py']=b'changed'
  with tempfile.TemporaryDirectory() as d,patch.object(p,'C15_SHA',ph),patch.object(p,'START_CP_SHA',ch):
   f=Path(d)/'p.zip';write_zip(f,m)
   with self.assertRaises(ValueError):p.validate_parent(f,pins)
 def test_source_archive_friendly_rejection(self):
  with tempfile.TemporaryDirectory() as d:
   f=Path(d)/'source.zip'
   with zipfile.ZipFile(f,'w') as z:z.writestr('SOURCE_MANIFEST.json','{}')
   with self.assertRaisesRegex(ValueError,'[Rr]esult ZIP'):p.input_kind(f)
 def test_actual_resume_round_not_filename(self):
  parent=b'parent';cp=b'cp';m={'identity.json':json.dumps({'protocol':p.PROTOCOL,'settings':p.SETTINGS,'source_hashes':p.source_hashes(ROOT),'start_head_sha256':p.START_HEAD_SHA,'parent_c18_sha256':hashlib.sha256(parent).hexdigest()}).encode(),'parent_c18.zip':parent,
   'heads/calibrated/PROGRESS.json':json.dumps({'checkpoint':'checkpoint_000003.pt','rounds':3,'checkpoint_sha256':hashlib.sha256(cp).hexdigest()}).encode(),'heads/calibrated/checkpoint_000003.pt':cp}
  with tempfile.TemporaryDirectory() as d:
   f=Path(d)/'9999.zip';write_zip(f,m);self.assertEqual(p.resume_score(f)[-2],3)
   m['heads/calibrated/checkpoint_000003.pt']=b'changed';write_zip(f,m)
   with self.assertRaises(ValueError):p.resume_score(f)
 def test_namespace_audit(self):
  for n in ('worker_c19.py','launch_c19.py','c19_protocol.py','joint_calibration.py','output_layer.py'):self.assertEqual(global_audit(ROOT/n),set())
 def test_missing_import_negative_control(self):
  s=(ROOT/'worker_c19.py').read_text().replace('REVISION,CASES,STATES,GATES,','REVISION,CASES,GATES,')
  with tempfile.TemporaryDirectory() as d:
   f=Path(d)/'bad.py';f.write_text(s);self.assertIn('STATES',global_audit(f))
 def test_online_function_only_one_proposal_no_geometry(self):
  tree=ast.parse((ROOT/'worker_c19.py').read_text());f=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='online_raw_arm')
  banned={'autograd','grad','svd','eigh','solve','run_correction','offline_correct','build_route_geometry','set_layer_matrix','teacher'}
  self.assertFalse({n.id for n in ast.walk(f) if isinstance(n,ast.Name)}&banned)
  calls=[n for n in ast.walk(f) if isinstance(n,ast.Call)];self.assertFalse({n.func.attr for n in calls if isinstance(n.func,ast.Attribute)}&banned)
  self.assertEqual(sum(isinstance(n.func,ast.Name) and n.func.id=='one_raw_proposal' for n in calls),1)
 def test_new_generation_after_freeze(self):
  s=(ROOT/'worker_c19.py').read_text();self.assertLess(s.index("dump(out/'WRITER_FROZEN.json'"),s.index('def prepare_new('))
 def test_real_response_residuals_recorded(self):self.assertIn("v['response_residual']=residual.cpu().double().tolist()",(ROOT/'worker_c19.py').read_text())
 def test_c17_guard_still_protects_all_training_passes(self):
  a={t['id']:{'reference_valid':True,'metrics':metric(.003)} for t in p.training_specs()};b={t['id']:{'reference_valid':True,'metrics':metric(.001)} for t in p.training_specs()};sid=p.training_specs()[0]['id'];a[sid]['metrics']=metric(.001);b[sid]['metrics']=metric(.003)
  self.assertFalse(p.guard(a,b)['accepted'])
 def test_latest_three_own_backups_only(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/'session'/'run';out.mkdir(parents=True);(out/'identity.json').write_text(json.dumps({'protocol':p.PROTOCOL}));drive=Path(d)/'drive';drive.mkdir();old=drive/'older.zip';old.write_bytes(b'untouched')
   for _ in range(4):p.snapshot(out,drive)
   self.assertEqual(len(list(drive.glob(p.RESULT+'_*.zip'))),3);self.assertEqual(old.read_bytes(),b'untouched')



from candidate_selection import select_candidate,merge_route,check_replay

def cohort(d=.003):
 return {t['id']:{'reference_valid':True,'metrics':{**metric(d),'kl':.001}} for t in p.training_specs()}
def candidate(rows,ri=0,bt=0):return {'rows':rows,'ridge_index':ri,'backtrack_index':bt}
class ComparisonTests(unittest.TestCase):
 def test_later_ridge_beats_first_admissible(self):
  old=cohort(.004);a=cohort(.003);b=cohort(.001)
  self.assertIs(select_candidate(old,[candidate(a),candidate(b,1)])['rows'],b)
 def test_pass_count_precedes_average_score(self):
  old=cohort(.004);a=cohort(.003);b=cohort(.0035)
  for sid in list(b)[:2]:b[sid]['metrics']=dict(metric(.001),kl=.001)
  self.assertIs(select_candidate(old,[candidate(a),candidate(b,1)])['rows'],b)
 def test_maximum_drift_breaks_equal_pass_count(self):
  old=cohort(.008);a=cohort(.004);b=cohort(.0035)
  self.assertIs(select_candidate(old,[candidate(a),candidate(b,1)])['rows'],b)
 def test_rejected_impressive_candidate_cannot_win(self):
  old=cohort(.004);sid=list(old)[0];old[sid]['metrics']=dict(metric(.001),kl=.001)
  bad=cohort(.001);bad[sid]['metrics']=dict(metric(.003),kl=.001);good=cohort(.003);good[sid]=old[sid]
  self.assertIs(select_candidate(old,[candidate(bad),candidate(good,1)])['rows'],good)
 def test_no_improvement_returns_none(self):self.assertIsNone(select_candidate(cohort(),[candidate(cohort())]))
 def test_tie_deterministic_not_input_order(self):
  old=cohort(.004);a=candidate(cohort(.003),2);b=candidate(cohort(.003),0)
  self.assertIs(select_candidate(old,[a,b]),b);self.assertIs(select_candidate(old,[b,a]),b)
 def test_route_merge_requires_all_nine(self):
  old=cohort();ids=list(old)[:9]
  with self.assertRaises(ValueError):merge_route(old,{k:old[k] for k in ids[:8]},ids)
 def test_route_merge_retains_other_27_and_does_not_mutate_start(self):
  old=cohort();ids=list(old)[:9];part={k:{'reference_valid':True,'metrics':dict(metric(.001),kl=.001)} for k in ids};merged=merge_route(old,part,ids)
  self.assertEqual(len(p.quality(merged)['passed_ids']),9);self.assertEqual(len(p.quality(old)['passed_ids']),0)
  for sid in set(old)-set(ids):self.assertIs(merged[sid],old[sid])
 def test_full_audit_detects_untouched_route_gate_change(self):
  import copy
  old=cohort();new=copy.deepcopy(old);new[list(new)[-1]]['metrics']['gates']['KL']=False
  with self.assertRaises(ValueError):check_replay(old,new)
 def test_full_audit_detects_metric_disagreement(self):
  import copy
  old=cohort();new=copy.deepcopy(old);new[list(new)[-1]]['metrics']['kl']+=.001
  with self.assertRaises(ValueError):check_replay(old,new)
 def test_all_ridges_loop_is_not_terminated_after_first_acceptance(self):
  tree=ast.parse((ROOT/'worker_c19.py').read_text())
  loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and ast.unparse(n.iter)=="enumerate(SETTINGS['ridge_grid'])")
  self.assertFalse(any(isinstance(n,ast.Break) for n in loop.body));self.assertIn('select_candidate',ast.unparse(tree))
 def test_new_seeds_disjoint_all_inspected_splits(self):
  import c15_protocol,c17_protocol,c18_protocol
  new={t['seed'] for t in p.test_specs()}
  self.assertFalse(new & {t['seed'] for t in c17_protocol.test_specs()})
  self.assertFalse(new & {t['seed'] for t in c18_protocol.test_specs()})
  self.assertFalse(new & {t['seed'] for t in c15_protocol.test_specs()})
 def test_exact_c18_pinned_source_and_modules(self):
  p.inherited_check(ROOT);pins=json.loads((ROOT/'C18_SOURCE_PINS.json').read_text());self.assertEqual(len(pins),53)
 def test_route_cache_requires_candidate_identity(self):
  s=(ROOT/'worker_c19.py').read_text();self.assertIn("'head_sha256','start_head_sha256','case','prediction'",s)
 def test_full36_audit_precedes_checkpoint_commit(self):
  s=(ROOT/'worker_c19.py').read_text();self.assertLess(s.index("check_replay(winner['rows'],audited)"),s.index('rounds+=1;'))
 def test_new_namespace(self):self.assertEqual(global_audit(ROOT/'candidate_selection.py'),set())
 def test_resume_new_rounds_not_parent_twelve(self):self.assertIn('rounds=accepted=0;', (ROOT/'worker_c19.py').read_text())
 def test_parent_data_not_generated_states_used_in_fit(self):
  s=(ROOT/'worker_c19.py').read_text();self.assertIn("prep=vector(load_obj(parent,b+'prepared.pt'))",s);self.assertNotIn("parent18.read('states/new",s)

if __name__=='__main__':unittest.main()
