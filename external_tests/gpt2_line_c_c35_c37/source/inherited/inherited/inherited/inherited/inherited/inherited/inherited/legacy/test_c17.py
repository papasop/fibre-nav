import ast,copy,hashlib,json,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
import c17_protocol as p
from namespace_audit import global_audit
from test_c16 import cpu_fixture,write_zip
ROOT=Path(__file__).parent

def metric(d=.001,**changes):
 g={n:True for n in p.GATES};g['response']=d<=.002;g.update(changes)
 return {'reference_valid':True,'metrics':{'drift':d,'gates':g,'episode_pass':all(g.values())}}
def rows(d=.001):return {t['id']:metric(d) for t in p.training_specs()}
def final_rows(n,ok=True,valid=True):return [{'reference_valid':valid,'arms':{a:metric(.001 if ok else .01)['metrics'] for a in ('warm','guarded')}} for _ in range(n)]

class Tests(unittest.TestCase):
 def test_all_36_training(self):self.assertEqual(len(p.training_specs()),36)
 def test_16_new(self):self.assertEqual(len(p.test_specs()),16)
 def test_known_four_classes_balanced(self):
  for c in p.CASES:self.assertEqual(sum(t['case']==c for t in p.test_specs()),4)
 def test_new_seeds_disjoint_from_c15(self):self.assertFalse(set(t['seed'] for t in p.test_specs())&set(t['seed'] for t in p.c16.c15.test_specs()+p.c16.c15.train_specs()))
 def test_no_old_diagnostic_ids_used(self):self.assertFalse(set(t['id'] for t in p.training_specs())&set(t['id'] for t in p.c16.c15.test_specs()))
 def test_thresholds_unchanged(self):self.assertEqual([p.SETTINGS[x] for x in ('response_budget','kl_budget','margin_gate')],[.002,.005,.05])
 def test_warm_actual_identity(self):self.assertEqual(p.WARM_MEMBER,'heads/warm/checkpoint_001000.pt');self.assertEqual(len(p.WARM_HEAD_SHA),64)
 def test_online_work_disabled(self):
  for n in ('online_Jacobian_builds','online_SVD_calls','online_correction_steps','online_backward_calls'):self.assertEqual(p.SETTINGS[n],0)
 def test_native_and_fresh_claim_not_made(self):self.assertFalse(p.SETTINGS['fresh_evaluation']);self.assertFalse(p.SETTINGS['internal_parameter_update'])
 def test_response_improvement_accepted(self):self.assertTrue(p.guard(rows(.004),rows(.003))['accepted'])
 def test_exact_same_candidate_rejected(self):self.assertFalse(p.guard(rows(),rows())['accepted'])
 def test_lost_pass_rejected_even_if_mean_improves(self):
  a=rows(.008);b=rows(.001);sid=p.training_specs()[0]['id'];a[sid]=metric(.001);b[sid]=metric(.0021)
  self.assertIn('LOST_PREVIOUS_COMPLETE_PASS',p.guard(a,b)['reasons'])
 def test_more_passes_not_override_old_pass_loss(self):
  a=rows(.003);b=rows(.001);sid=p.training_specs()[0]['id'];a[sid]=metric(.001);b[sid]=metric(.003)
  self.assertFalse(p.guard(a,b)['accepted'])
 def test_non_response_gates_protected(self):
  for g in p.NON_RESPONSE:
   a=rows(.004);b=rows(.003);sid=p.training_specs()[0]['id'];b[sid]['metrics']['gates'][g]=False
   self.assertFalse(p.guard(a,b)['accepted'])
 def test_worst_drift_cannot_rise(self):
  a=rows(.004);b=rows(.001);b[p.training_specs()[0]['id']]=metric(.00401);self.assertFalse(p.guard(a,b)['accepted'])
 def test_missing_training_row_rejected(self):
  r=rows();r.pop(next(iter(r)))
  with self.assertRaises(ValueError):p.quality(r)
 def test_extra_training_row_rejected(self):
  r=rows();r['extra']=metric()
  with self.assertRaises(ValueError):p.quality(r)
 def test_invalid_reference_rejected(self):
  r=rows();r[next(iter(r))]['reference_valid']=False
  with self.assertRaises(ValueError):p.quality(r)
 def test_nan_drift_rejected(self):
  r=rows();r[next(iter(r))]['metrics']['drift']=float('nan')
  with self.assertRaises(ValueError):p.quality(r)
 def test_missing_gate_rejected(self):
  r=rows();del r[next(iter(r))]['metrics']['gates']['KL']
  with self.assertRaises(ValueError):p.quality(r)
 def test_full_pass_decision(self):self.assertEqual(p.decide(final_rows(36),final_rows(16)),'C17_RAW_LOCAL_STATE_PASS')
 def test_training_failure_not_local_pass(self):self.assertEqual(p.decide(final_rows(36,False),final_rows(16)),'C17_RAW_TRAINING_FUNCTION_FAILURE')
 def test_new_failure(self):self.assertEqual(p.decide(final_rows(36),final_rows(16,False)),'C17_RAW_LOCAL_STATE_FAIL')
 def test_invalid_new_not_dropped(self):self.assertEqual(p.decide(final_rows(36),final_rows(16,True,False)),'C17_STATE_REFERENCE_FAILURE')
 def test_all_denominators_required(self):
  with self.assertRaises(ValueError):p.decide(final_rows(36),final_rows(15))
 def test_namespace(self):
  for n in ('worker_c17.py','c17_protocol.py','launch_c17.py','guarded_update.py'):self.assertEqual(global_audit(ROOT/n),set(),n)
 def test_missing_import_negative_control(self):
  s=(ROOT/'worker_c17.py').read_text().replace('REVISION,CASES,STATES,GATES,','REVISION,CASES,GATES,')
  with tempfile.TemporaryDirectory() as d:
   f=Path(d)/'bad.py';f.write_text(s);self.assertIn('STATES',global_audit(f))
 def test_online_function_has_no_repair_or_training(self):
  tree=ast.parse((ROOT/'worker_c17.py').read_text());func=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='online_raw_arm')
  banned={'backward','step','svd','jacobian','offline_jacobian','offline_correct','run_correction','null_project','AdamW','teacher'}
  self.assertFalse({n.id for n in ast.walk(func) if isinstance(n,ast.Name)}&banned)
  calls=[n for n in ast.walk(func) if isinstance(n,ast.Call)];self.assertFalse({n.func.attr for n in calls if isinstance(n.func,ast.Attribute)}&banned)
  self.assertEqual(sum(isinstance(n.func,ast.Name) and n.func.id=='one_raw_proposal' for n in calls),1)
 def test_test_generation_follows_freeze(self):
  s=(ROOT/'worker_c17.py').read_text();self.assertLess(s.index("dump(out/'WRITER_FROZEN.json'"),s.index('def prepare_new('))
 def test_no_summary_claim_from_test_selection(self):self.assertFalse(p.SETTINGS['test_based_checkpoint_selection'])
 def test_structural_c15_parent_guards_remain(self):
  m,pins,ch=cpu_fixture()
  with tempfile.TemporaryDirectory() as d,patch.object(p.c16,'C9_SHA',ch):
   f=Path(d)/'x.zip';write_zip(f,m);self.assertEqual(p.c16.validate_parent(f,pins)['checkpoints']['warm']['steps'],1000)
 def test_wrong_full_parent_digest_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   f=Path(d)/'bad.zip';write_zip(f,{'identity.json':json.dumps({'protocol':p.c16.PARENT_PROTOCOL}).encode()})
   with self.assertRaises(ValueError):p.validate_c15(f)
 def test_source_zip_not_result(self):
  with tempfile.TemporaryDirectory() as d:
   f=Path(d)/'s.zip'
   with zipfile.ZipFile(f,'w') as z:z.writestr('SOURCE_MANIFEST.json','{}')
   with self.assertRaises((ValueError,KeyError)):p.input_kind(f)
 def test_resume_checkpoint_digest_checked(self):
  members={'identity.json':json.dumps({'protocol':p.PROTOCOL,'settings':p.SETTINGS,'source_hashes':p.source_hashes(ROOT),'parent_c15_sha256':hashlib.sha256(b'parent').hexdigest()}).encode(),
   'parent_c15.zip':b'parent','heads/guarded/PROGRESS.json':json.dumps({'checkpoint':'checkpoint_000001.pt','checkpoint_sha256':'wrong','attempts':1}).encode(),
   'heads/guarded/checkpoint_000001.pt':b'cp'}
  with tempfile.TemporaryDirectory() as d,patch.object(p,'C15_SHA',hashlib.sha256(b'parent').hexdigest()):
   f=Path(d)/'r.zip';write_zip(f,members)
   with self.assertRaises(ValueError):p.resume_score(f)
 def test_real_resume_score_uses_checkpoint_not_filename(self):
  parent=b'parent';cp=b'checkpoint';digest=hashlib.sha256(parent).hexdigest()
  members={'identity.json':json.dumps({'protocol':p.PROTOCOL,'settings':p.SETTINGS,'source_hashes':p.source_hashes(ROOT),'parent_c15_sha256':digest}).encode(),'parent_c15.zip':parent,
   'heads/guarded/PROGRESS.json':json.dumps({'checkpoint':'checkpoint_000021.pt','checkpoint_sha256':hashlib.sha256(cp).hexdigest(),'attempts':21}).encode(),'heads/guarded/checkpoint_000021.pt':cp}
  with tempfile.TemporaryDirectory() as d,patch.object(p,'C15_SHA',digest):
   f=Path(d)/'fake_timestamp.zip';write_zip(f,members);self.assertEqual(p.resume_score(f)[-1],21)
 def test_archive_latest_three_only_own(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/'session'/'run';out.mkdir(parents=True);(out/'identity.json').write_text(json.dumps({'protocol':p.PROTOCOL}));(out/'a').write_text('payload');drive=Path(d)/'drive';drive.mkdir();other=drive/'older_other_protocol.zip';other.write_bytes(b'untouched')
   for _ in range(4):p.snapshot(out,drive)
   self.assertEqual(len(list(drive.glob(p.RESULT+'_*.zip'))),3);self.assertEqual(other.read_bytes(),b'untouched')

if __name__=='__main__':unittest.main()
