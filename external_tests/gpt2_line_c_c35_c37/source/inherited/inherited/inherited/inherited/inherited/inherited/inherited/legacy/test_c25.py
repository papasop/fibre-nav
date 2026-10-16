import ast,copy,hashlib,io,json,tempfile,unittest
from pathlib import Path
from contextlib import redirect_stdout
import c25_protocol as p
from test_c25_parent import CapsuleTests,metric
from test_c16 import write_zip
from namespace_audit import global_audit
ROOT=Path(__file__).parent

def cohort(failing=.006):
 bad={'xor_00__c22_new3','xor_01__c22_new3','xor_10__c22_new2','xor_10__c22_new3','xor_11__c22_new1','xor_11__c22_new2','xor_11__c22_new3'}
 return {s['id']:{'reference_valid':True,'metrics':metric(failing if s['id'] in bad else .001)} for s in p.training_specs()}

class ProtocolTests(unittest.TestCase):
 def test_actual_checkpoint_identity(self):self.assertEqual(p.START_MEMBER,'heads/calibrated/checkpoint_000008_final.pt');self.assertEqual(p.START_CP_SHA,'b804fb325e7bfda784aa35249338520942928f2705a6410c78f3d13aa6a0b5b1')
 def test_inherited_source134_unchanged(self):p.inherited_check(ROOT);self.assertEqual(len(json.loads((ROOT/'C24_SOURCE_PINS.json').read_text())),134)
 def test_training76_unique_and_each_route19(self):
  specs=p.training_specs();self.assertEqual(len(specs),76);self.assertEqual(len({s['id'] for s in specs}),76)
  for c in p.CASES:self.assertEqual(sum(s['case']==c for s in specs),19)
 def test_explicit_promotion_preserves_ids(self):self.assertEqual({s['id'] for s in p.training_specs()}-{s['id'] for s in p.original60_specs()},{s['id'] for s in p.promoted_specs()});self.assertEqual(p.SETTINGS['promoted_old_evaluation_states'],16)
 def test_new_states_disjoint_ids_and_seeds(self):
  new=p.test_specs();self.assertEqual(len(new),16);self.assertEqual(len({s['seed'] for s in new}),16);self.assertFalse({s['id'] for s in new}&{s['id'] for s in p.training_specs()});self.assertFalse({s['seed'] for s in new}&{s['seed'] for s in p.promoted_specs()})
 def test_all_four_case_grids_are90(self):
  for c in p.CASES:self.assertEqual(len(p.candidate_specs(c)),90);self.assertTrue(all(s['case']==c for s in p.candidate_specs(c)))
 def test_unknown_route_rejected(self):
  with self.assertRaises(ValueError):p.candidate_specs('wrong')
 def test_gates_unchanged(self):self.assertEqual([p.SETTINGS[k] for k in ('response_budget','kl_budget','margin_gate')],[.002,.005,.05])
 def test_bootstrap69(self):self.assertEqual(len(p.quality(cohort())['passed_ids']),69)
 def test_original60_cannot_be_lost(self):
  a=cohort();b=cohort(.001);sid=p.original60_specs()[0]['id'];b[sid]['metrics']=metric(.003);self.assertFalse(p.guard(a,b)['accepted'])
 def test_promoted_prior_pass_cannot_be_lost(self):
  a=cohort();b=cohort(.001);sid=p.promoted_specs()[0]['id'];b[sid]['metrics']=metric(.003);self.assertFalse(p.guard(a,b)['accepted'])
 def test_original60_can_change_inside_budget(self):
  a=cohort();b=cohort(.004);sid=p.original60_specs()[0]['id'];b[sid]['metrics']=metric(.0019);self.assertTrue(p.guard(a,b)['accepted'])
 def test_original60_max_guard_change_explicit(self):self.assertIn('without requiring',p.SETTINGS['guard_change_from_C24'])
 def test_nonresponse_gate_loss_rejected(self):
  a=cohort();b=cohort(.001);b[next(iter(b))]['metrics']['gates']['KL']=False;self.assertFalse(p.guard(a,b)['accepted'])
 def test_worst76_cannot_increase(self):self.assertFalse(p.guard(cohort(),cohort(.007))['accepted'])
 def test_coverage_merit_not_just_passcount(self):self.assertTrue(p.guard(cohort(),cohort(.004))['accepted'])
 def test_round_robin_and_stalled_skip(self):
  r=cohort();sequence=[];last=-1
  for _ in range(4):c=p.choose_case(r,[],last);sequence.append(c);last=p.CASES.index(c)
  self.assertEqual(sequence,list(p.CASES));self.assertEqual(p.choose_case(r,[p.CASES[0]],-1),p.CASES[1]);self.assertIsNone(p.choose_case(r,p.CASES,-1))
 def test_completed_case_skipped(self):
  r=cohort();r.update({s['id']:{'reference_valid':True,'metrics':metric()} for s in p.training_specs() if s['case']==p.CASES[0]});self.assertEqual(p.choose_case(r,[],-1),p.CASES[1])
 def test_merge_requires_all19(self):
  r=cohort();ids=[s['id'] for s in p.training_specs() if s['case']==p.CASES[0]];self.assertEqual(len(p.merge_route(r,{sid:r[sid] for sid in ids},ids)),76)
  with self.assertRaises(ValueError):p.merge_route(r,{sid:r[sid] for sid in ids[:-1]},ids[:-1])
 def test_missing_training_denominator_rejected(self):
  r=cohort();r.pop(next(iter(r)))
  with self.assertRaises(ValueError):p.quality(r)
 def test_no_independent_native_claim(self):self.assertFalse(p.SETTINGS['fresh_evaluation']);self.assertFalse(p.SETTINGS['internal_parameter_update']);self.assertFalse(p.SETTINGS['controller_dispatch_tested']);self.assertTrue(p.SETTINGS['fresh_state_seeds'])
 def test_new_eval_not_accessed_in_training_scan(self):
  s=(ROOT/'worker_c25.py').read_text();self.assertGreater(s.index("folder,reference=prepare_state(t,freeze['head_sha256'])"),s.index('freeze=json.loads'));self.assertNotIn('test_specs',(ROOT/'c25_scan.py').read_text())
 def test_no_replacement_or_eval_selection(self):self.assertFalse(p.SETTINGS['state_replacement_allowed']);self.assertFalse(p.SETTINGS['test_based_checkpoint_selection'])
 def test_actual_grid_no_linear_rejection_or_early_pair_break(self):
  s=(ROOT/'c25_scan.py').read_text();self.assertNotIn('linear_protection',s);tree=ast.parse(s);loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and ast.unparse(n.iter)=='enumerate(candidate_specs(case))');self.assertFalse(any(isinstance(n,ast.Break) for n in ast.walk(loop)))
 def test_online_no_geometry(self):
  tree=ast.parse((ROOT/'worker_c25.py').read_text());fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='online_raw_arm');calls={ast.unparse(n.func) for n in ast.walk(fn) if isinstance(n,ast.Call)};self.assertIn('one_raw_proposal',calls);self.assertFalse(any(x for x in calls if any(w in x for w in ('autograd','jacobian','svd','correction','solve','backward'))))
 def test_namespace_audit_new_files(self):
  for n in ('c25_parent.py','c25_guard.py','c25_protocol.py','c25_identity.py','c25_scan.py','worker_c25.py','launch_c25.py','prepare_c25.py'):self.assertFalse(global_audit(ROOT/n),n)
 def test_decision_requires76_and16(self):
  with self.assertRaises(ValueError):p.decide([],[])
 def test_training_fail_cannot_be_newstate_pass(self):
  train=[{'reference_valid':True,'arms':{'calibrated':r['metrics']}} for r in cohort().values()];new=[{'reference_valid':True,'arms':{'calibrated':metric()}}]*16;self.assertEqual(p.decide(train,new),'C25_RAW_EXPANDED_TRAINING_FUNCTION_FAILURE')
 def test_all_pass_still_development_scope(self):
  train=[{'reference_valid':True,'arms':{'calibrated':metric()}}]*76;new=[{'reference_valid':True,'arms':{'calibrated':metric()}}]*16;self.assertEqual(p.decide(train,new),'C25_RAW_PROSPECTIVE_LOCAL_STATE_PASS')
 def test_invalid_reference_retained(self):
  train=[{'reference_valid':True,'arms':{'calibrated':metric()}}]*76;new=[{'reference_valid':False,'arms':{'calibrated':metric()}}]*16;self.assertEqual(p.decide(train,new),'C25_STATE_REFERENCE_FAILURE')
 def test_backups_only_own_last3(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/'session'/'run';out.mkdir(parents=True);(out/'identity.json').write_text(json.dumps({'protocol':p.PROTOCOL}));drive=Path(d)/'drive';drive.mkdir();old=drive/'gpt2_c24_relinearised_calibration_results_old.zip';old.write_bytes(b'keep')
   with redirect_stdout(io.StringIO()):
    for i in range(4):p.snapshot(out,str(drive))
   self.assertEqual(len(list(drive.glob(p.RESULT+'_*.zip'))),3);self.assertEqual(old.read_bytes(),b'keep')
 def test_resume_uses_real_round_count(self):
  with tempfile.TemporaryDirectory() as d:
   cp=b'actual';m={'parent_capsule.zip':b'parent','identity.json':json.dumps({'protocol':p.PROTOCOL,'settings':p.SETTINGS,'source_hashes':p.source_hashes(ROOT),'start_head_sha256':p.START_HEAD_SHA,'parent_capsule_sha256':hashlib.sha256(b'parent').hexdigest()}).encode(),'heads/calibrated/PROGRESS.json':json.dumps({'checkpoint':'checkpoint_000003.pt','checkpoint_sha256':hashlib.sha256(cp).hexdigest(),'rounds':3}).encode(),'heads/calibrated/checkpoint_000003.pt':cp};path=Path(d)/'x.zip';write_zip(path,m);self.assertEqual(p.resume_score(path)[3],3)

if __name__=='__main__':unittest.main()
