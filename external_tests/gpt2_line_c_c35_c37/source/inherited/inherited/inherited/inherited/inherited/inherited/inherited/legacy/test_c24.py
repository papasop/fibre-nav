import ast,copy,hashlib,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import c24_protocol as p
from test_c16 import write_zip
from test_c24_parent import CapsuleTests,cohort,metric
from namespace_audit import global_audit
ROOT=Path(__file__).parent

class ProtocolTests(unittest.TestCase):
 def test_exact_actual_candidate_identity(self):self.assertEqual(p.START_HEAD_SHA,'86b71471cd25e76ea518f1f5b411e8a8cdf365c35acd40afc72f7fead3fe9522');self.assertEqual(p.START_SPEC['id'],'candidate_022')
 def test_inherited_source_byte_identity(self):p.inherited_check(ROOT);self.assertEqual(len(json.loads((ROOT/'C23_SOURCE_PINS.json').read_text())),119)
 def test_grid_all90(self):self.assertEqual(len(p.candidate_specs()),90);self.assertEqual(len({s['id'] for s in p.candidate_specs()}),90)
 def test_grid_fixed_and_not_parent_trust(self):self.assertIn(1.,p.SETTINGS['alpha_grid']);self.assertIn(.0001220703125,p.SETTINGS['alpha_grid']);self.assertNotIn('trust',str(p.candidate_specs()))
 def test_gates_unchanged(self):self.assertEqual([p.SETTINGS[k] for k in ('response_budget','kl_budget','margin_gate')],[.002,.005,.05])
 def test_same60_training_and_original16_eval_specs(self):self.assertEqual(p.training_specs(),p.legacy.training_specs());self.assertEqual(p.test_specs(),p.legacy.test_specs())
 def test_evaluation_is_inspected_not_fresh(self):self.assertFalse(p.SETTINGS['fresh_evaluation']);self.assertFalse(p.SETTINGS['fresh_state_seeds']);self.assertTrue(p.SETTINGS['evaluation_influenced_protocol_design'])
 def test_no_new_preparations_or_native_claim(self):self.assertEqual(p.SETTINGS['n_new_preparation_states'],0);self.assertFalse(p.SETTINGS['internal_parameter_update']);self.assertFalse(p.SETTINGS['controller_dispatch_tested'])
 def test_preserve_prior54(self):
  a=cohort();b=cohort(.001);b[next(sid for sid,r in a.items() if r['metrics']['episode_pass'])]['metrics']=metric(.003);self.assertFalse(p.guard(a,b)['accepted'])
 def test_preserve_nonresponse_gate(self):
  a=cohort();b=cohort(.001);b[next(iter(b))]['metrics']['gates']['KL']=False;self.assertFalse(p.guard(a,b)['accepted'])
 def test_drift_guard_not_loss_only(self):self.assertFalse(p.guard(cohort(),cohort(.004))['accepted'])
 def test_partial_training_cannot_be_pass(self):
  train=[{'reference_valid':True,'arms':{'calibrated':r['metrics']}} for r in cohort().values()];new=[{'reference_valid':True,'arms':{'calibrated':metric(.001)}}]*16;self.assertEqual(p.decide(train,new),'C24_RAW_TRAINING_FUNCTION_FAILURE')
 def test_complete_decision_still_inspected_scope(self):
  train=[{'reference_valid':True,'arms':{'calibrated':metric(.001)}}]*60;new=[{'reference_valid':True,'arms':{'calibrated':metric(.001)}}]*16;self.assertEqual(p.decide(train,new),'C24_RAW_INSPECTED_LOCAL_STATE_PASS')
 def test_invalid_state_retained_failure(self):
  train=[{'reference_valid':True,'arms':{'calibrated':metric(.001)}}]*60;new=[{'reference_valid':False,'arms':{'calibrated':metric(.001)}}]*16;self.assertEqual(p.decide(train,new),'C24_STATE_REFERENCE_FAILURE')
 def test_missing_denominator_rejected(self):
  with self.assertRaises(ValueError):p.decide([],[])
 def test_namespace_audit_new_modules(self):
  for n in ('c24_parent.py','c24_protocol.py','c24_identity.py','c24_scan.py','prepare_c24.py','worker_c24.py','launch_c24.py'):self.assertFalse(global_audit(ROOT/n),n)
 def test_online_no_geometry_or_backward(self):
  tree=ast.parse((ROOT/'worker_c24.py').read_text());fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='online_raw_arm');calls={ast.unparse(n.func) for n in ast.walk(fn) if isinstance(n,ast.Call)}
  self.assertEqual(sum(x=='one_raw_proposal' for x in calls),1);self.assertFalse(any(x for x in calls if any(w in x for w in ('autograd','jacobian','svd','correction','solve','backward'))))
 def test_offline_new_derivatives_are_actual_autograd(self):self.assertIn('torch.autograd.grad(residual[k],candidate',(ROOT/'worker_c24.py').read_text())
 def test_eval_access_after_freeze(self):
  s=(ROOT/'worker_c24.py').read_text();self.assertGreater(s.index("b='evaluation_states/'"),s.index("freeze=json.loads"));self.assertNotIn('prepare_state(',s)
 def test_full_grid_no_linear_or_pair_early_stop(self):
  s=(ROOT/'c24_scan.py').read_text();self.assertNotIn('linear_protection',s);tree=ast.parse(s);loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and ast.unparse(n.iter)=='enumerate(candidate_specs())');self.assertFalse(any(isinstance(n,ast.Break) for n in ast.walk(loop)))
 def test_each_trial_restores_finally(self):self.assertGreaterEqual((ROOT/'c24_scan.py').read_text().count('finally:restore()'),2)
 def test_distinct_head_full60_audit_not_alias_retries(self):self.assertIn("if record['head_sha256'] in unique_audits:continue",(ROOT/'c24_scan.py').read_text())
 def test_commit_requires_full60(self):self.assertIn("if not winner['full60_verified'] or not guard(current,winner['rows'])['accepted']",(ROOT/'worker_c24.py').read_text())
 def test_resume_priority_progress_not_filename(self):
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'x.zip';cp=b'actual';m={'parent_capsule.zip':b'parent','identity.json':json.dumps({'protocol':p.PROTOCOL,'settings':p.SETTINGS,'source_hashes':p.source_hashes(ROOT),'start_head_sha256':p.START_HEAD_SHA,'parent_capsule_sha256':hashlib.sha256(b'parent').hexdigest()}).encode(),'heads/calibrated/PROGRESS.json':json.dumps({'checkpoint':'checkpoint_000003.pt','checkpoint_sha256':hashlib.sha256(cp).hexdigest(),'rounds':3}).encode(),'heads/calibrated/checkpoint_000003.pt':cp};write_zip(path,m);self.assertEqual(p.resume_score(path)[3],3)
 def test_corrupt_resume_checkpoint_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'x.zip';m={'parent_capsule.zip':b'parent','identity.json':json.dumps({'protocol':p.PROTOCOL,'settings':p.SETTINGS,'source_hashes':p.source_hashes(ROOT),'start_head_sha256':p.START_HEAD_SHA,'parent_capsule_sha256':hashlib.sha256(b'parent').hexdigest()}).encode(),'heads/calibrated/PROGRESS.json':json.dumps({'checkpoint':'checkpoint_000003.pt','checkpoint_sha256':'wrong','rounds':3}).encode(),'heads/calibrated/checkpoint_000003.pt':b'actual'};write_zip(path,m)
   with self.assertRaisesRegex(ValueError,'digest'):p.resume_score(path)
 def test_only_own_three_backups_old_untouched(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);out=root/'session'/'run';out.mkdir(parents=True);(out/'identity.json').write_text(json.dumps({'protocol':p.PROTOCOL}));drive=root/'drive';drive.mkdir();old=drive/'gpt2_c23_frozen_step_diagnosis_results_old.zip';old.write_bytes(b'keep')
   with __import__('contextlib').redirect_stdout(io.StringIO()):
    for i in range(4):p.snapshot(out,str(drive))
   self.assertEqual(len(list(drive.glob(p.RESULT+'_*.zip'))),3);self.assertEqual(old.read_bytes(),b'keep')
 def test_launcher_parent_and_companion_routes(self):
  s=(ROOT/'launch_c24.py').read_text();self.assertIn('validate_companion',s);self.assertIn('GPT2_C23_FROZEN_STEP_DIAGNOSIS_A100_V1_RECOVERY',s);self.assertIn("if not Path('/content/drive/MyDrive').is_dir()",s)

if __name__=='__main__':unittest.main()
