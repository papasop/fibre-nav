import ast,copy,hashlib,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import c23_protocol as p
from test_c16 import write_zip
from test_c23_parent import CapsuleTests
from namespace_audit import global_audit
from precision_diagnostics import controls_summary
ROOT=Path(__file__).parent

def metric(d=.001):
 g={k:True for k in p.GATES};g['response']=d<=.002
 return {'drift':d,'kl':.001,'gates':g,'episode_pass':all(g.values())}
def cohort(d=.003):return {t['id']:{'reference_valid':True,'metrics':metric(d)} for t in p.training_specs()}
def records():return [{'spec':s} for s in p.candidate_specs()]
def zero_records():
 return [{'phase':phase,'repeat':rep,'state_id':sid,'probe':{'response_residual':[.001,.002],'response_residual_float64_aggregation':[.00101,.00199],'full_logits_sha256':'same','maximum_aggregation_difference':.00001}}
 for phase in ('before_scan','after_scan') for rep in range(3) for sid in ('a','b')]

class ProtocolTests(unittest.TestCase):
 def test_exact_parent_identity(self):self.assertEqual(p.START_MEMBER,'heads/calibrated/checkpoint_000015.pt');self.assertEqual(p.START_CP_SHA,'7c202d680eeee294c21c456d0561135fbd2e98d18025c236f492b90fe8f7b7f9')
 def test_source104_unchanged(self):p.inherited_check(ROOT);self.assertEqual(len(json.loads((ROOT/'C22_SOURCE_PINS.json').read_text())),104)
 def test_all90_unique_specs(self):self.assertEqual(len(p.candidate_specs()),90);self.assertEqual(len({s['id'] for s in p.candidate_specs()}),90)
 def test_three_radii_three_ridges_ten_alphas(self):
  specs=p.candidate_specs();self.assertEqual(len({s['increment_radius'] for s in specs}),3);self.assertEqual(len({s['ridge'] for s in specs}),3);self.assertEqual(len({s['alpha'] for s in specs}),10)
 def test_large_and_historical_small_steps_present(self):self.assertIn(1.,p.SETTINGS['alpha_grid']);self.assertIn(.00048828125,p.SETTINGS['alpha_grid']);self.assertIn(.000244140625,p.SETTINGS['alpha_grid'])
 def test_grid_does_not_use_parent_trust(self):self.assertTrue(all(s['alpha']>0 for s in p.candidate_specs()));self.assertNotIn('trust',str(p.candidate_specs()))
 def test_zero_controls_are_separate(self):self.assertNotIn(0.,p.SETTINGS['alpha_grid']);self.assertEqual(p.SETTINGS['zero_repeats_per_phase'],3)
 def test_thresholds_unchanged(self):self.assertEqual([p.SETTINGS[k] for k in ('response_budget','kl_budget','margin_gate')],[.002,.005,.05])
 def test_no_new_learning_or_geometry(self):
  for k in ('new_training_steps','new_Jacobian_builds','new_backward_calls','new_linear_solver_calls','new_evaluation_states'):self.assertEqual(p.SETTINGS[k],0)
 def test_high_precision_only_aggregation(self):self.assertFalse(p.SETTINGS['full_model_float64_forward']);self.assertFalse(p.SETTINGS['float64_aggregation_changes_primary_acceptance'])
 def test_scope_development_not_native(self):self.assertFalse(p.SETTINGS['fresh_evaluation']);self.assertFalse(p.SETTINGS['internal_parameter_update']);self.assertFalse(p.SETTINGS['controller_dispatch_tested'])
 def test_same60_training_specs(self):self.assertEqual(p.training_specs(),p.previous.training_specs())
 def test_cached_round14_is_fixed(self):self.assertEqual(p.SETTINGS['cached_geometry_round'],14)
 def test_preserve54_current_passes(self):
  a=cohort();b=cohort(.001)
  for sid in list(a)[:54]:a[sid]['metrics']=metric(.001)
  b[list(a)[53]]['metrics']=metric(.003);self.assertFalse(p.guard(a,b)['accepted'])
 def test_maximum_drift_cannot_increase(self):self.assertFalse(p.guard(cohort(.003),cohort(.004))['accepted'])
 def test_disallow_lost_nonresponse_gate(self):
  a=cohort();b=cohort(.001);b[next(iter(b))]['metrics']['gates']['KL']=False;self.assertFalse(p.guard(a,b)['accepted'])
 def test_diagnosis_decision_is_not_LC1_pass(self):self.assertEqual(p.decide(records(),{},True),'C23_FULL60_ACCEPTABLE_CANDIDATE_FOUND')
 def test_no_candidate_is_not_a_mathematical_impossibility_claim(self):self.assertEqual(p.decide(records(),None,True),'C23_NO_FULL60_ACCEPTABLE_CANDIDATE_FOUND')
 def test_missing_spec_rejected(self):
  with self.assertRaises(ValueError):p.decide(records()[:-1],None,True)
 def test_duplicate_spec_rejected(self):
  r=records();r[-1]=r[0]
  with self.assertRaises(ValueError):p.decide(r,None,True)
 def test_changed_origin_rejected(self):
  with self.assertRaises(ValueError):p.decide(records(),None,False)
 def test_controls_complete_and_stable(self):
  s=controls_summary(zero_records(),['a','b']);self.assertEqual(s['maximum_vector_spread_float32'],0.);self.assertFalse(s['pure_roundoff_cause_established'])
 def test_controls_detect_changed_logits_and_response(self):
  r=zero_records();r[-1]['probe']['response_residual']=[.00102,.002];r[-1]['probe']['full_logits_sha256']='changed';s=controls_summary(r,['a','b']);self.assertGreater(s['maximum_vector_spread_float32'],0.);self.assertEqual(s['states'][1]['distinct_full_logit_hashes'],2)
 def test_missing_control_rejected(self):
  with self.assertRaises(ValueError):controls_summary(zero_records()[:-1],['a','b'])
 def test_duplicate_control_rejected(self):
  r=zero_records();r[-1]=r[0]
  with self.assertRaises(ValueError):controls_summary(r,['a','b'])
 def test_nonfinite_control_rejected(self):
  r=zero_records();r[0]['probe']['response_residual'][0]=float('nan')
  with self.assertRaises(ValueError):controls_summary(r,['a','b'])
 def test_namespace_audit(self):
  for n in ('c23_protocol.py','c23_parent.py','prepare_c23.py','worker_c23.py','precision_diagnostics.py','launch_c23.py'):self.assertEqual(global_audit(ROOT/n),set())
 def test_worker_no_fresh_solver_or_gradient_calls(self):
  s=(ROOT/'worker_c23.py').read_text();tree=ast.parse(s);calls={ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)}
  self.assertFalse(calls & {'torch.autograd.grad','torch.linalg.solve','np.linalg.solve','np.linalg.eigh','np.linalg.svd','interval_coefficients','prepare_system','run_correction'});self.assertNotIn('requires_grad_(True)',s)
 def test_origin_restore_is_finally_protected(self):
  tree=ast.parse((ROOT/'worker_c23.py').read_text());self.assertGreaterEqual(sum(any(isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='restore_origin' for n in t.finalbody) for t in ast.walk(tree) if isinstance(t,ast.Try)),2)
 def test_scan_no_early_break_or_linear_rejection(self):
  s=(ROOT/'worker_c23.py').read_text();a=s.index('  for index,spec in enumerate(candidate_specs())');b=s.index('  best=None;',a);self.assertNotIn('break',s[a:b]);self.assertNotIn('LINEAR_PROTECTION_REJECT',s[a:b])
 def test_no_accepted_head_checkpoint_written(self):
  s=(ROOT/'worker_c23.py').read_text();self.assertNotIn('WRITER_FROZEN.json',s);self.assertNotIn("out/'heads'",s);self.assertIn("'diagnostic_only_not_committed':True",s)
 def test_cached_candidate_state_and_guard_checked(self):
  s=(ROOT/'worker_c23.py').read_text();self.assertIn('Cached completed state differs from actual candidate',s);self.assertIn('Cached candidate guard changed',s)
 def test_each_actual_head_audited_at_most_once(self):self.assertIn('audited_heads.add', (ROOT/'worker_c23.py').read_text())
 def test_resume_score_counts_actual_completed_grid(self):
  parent=b'parent';m={'identity.json':json.dumps({'protocol':p.PROTOCOL,'settings':p.SETTINGS,'source_hashes':p.source_hashes(ROOT),'origin_head_sha256':p.START_HEAD_SHA,'parent_capsule_sha256':hashlib.sha256(parent).hexdigest()}).encode(),'parent_capsule.zip':parent,'candidates/candidate_000/COMPLETE.json':b'{}'}
  with tempfile.TemporaryDirectory() as d:
   f=Path(d)/'99999.zip';write_zip(f,m);self.assertEqual(p.resume_score(f)[1],1)
 def test_incomplete_false_summary_rejected(self):
  parent=b'parent';m={'identity.json':json.dumps({'protocol':p.PROTOCOL,'settings':p.SETTINGS,'source_hashes':p.source_hashes(ROOT),'origin_head_sha256':p.START_HEAD_SHA,'parent_capsule_sha256':hashlib.sha256(parent).hexdigest()}).encode(),'parent_capsule.zip':parent,'SUMMARY.json':b'{"complete":false}'}
  with tempfile.TemporaryDirectory() as d:
   f=Path(d)/'p.zip';write_zip(f,m)
   with self.assertRaises(ValueError):p.resume_score(f)
 def test_wrong_resume_capsule_digest_rejected(self):
  m={'identity.json':json.dumps({'protocol':p.PROTOCOL,'settings':p.SETTINGS,'source_hashes':p.source_hashes(ROOT),'origin_head_sha256':p.START_HEAD_SHA,'parent_capsule_sha256':'bad'}).encode(),'parent_capsule.zip':b'changed'}
  with tempfile.TemporaryDirectory() as d:
   f=Path(d)/'p.zip';write_zip(f,m)
   with self.assertRaises(ValueError):p.resume_score(f)
 def test_old_archives_untouched_own_latest_three(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/'session'/'run';out.mkdir(parents=True);(out/'identity.json').write_text(json.dumps({'protocol':p.PROTOCOL}));drive=Path(d)/'drive';drive.mkdir();old=drive/'old.zip';old.write_bytes(b'keep')
   for _ in range(4):p.snapshot(out,drive)
   self.assertEqual(len(list(drive.glob(p.RESULT+'_*.zip'))),3);self.assertEqual(old.read_bytes(),b'keep')
 def test_already_mounted_drive_not_reauthorised(self):self.assertIn("if not Path('/content/drive/MyDrive').is_dir()",(ROOT/'launch_c23.py').read_text())

if __name__=='__main__':unittest.main()
