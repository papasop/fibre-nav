import ast,hashlib,json,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
import c18_protocol as p
from namespace_audit import global_audit
from test_c16 import write_zip
ROOT=Path(__file__).parent

def parent_fixture():
 source=b'# source fixture';pins={'a.py':hashlib.sha256(source).hexdigest()};parent=b'parent15fixture';cp=b'checkpoint-fixture'
 m={'identity.json':json.dumps({'protocol':p.PARENT_PROTOCOL,'settings':p.previous.SETTINGS,'source_hashes':pins,'parent_c15_sha256':hashlib.sha256(parent).hexdigest()}).encode(),
  'source/a.py':source,'parent_c15.zip':parent,p.START_MEMBER:cp,
  'SUMMARY.json':json.dumps({'complete':True,'decision':'C17_RAW_TRAINING_FUNCTION_FAILURE','writer_head_sha256':p.START_HEAD_SHA,'training':{'guarded':{'n_requested':36}},'prospective_local_state_development':{'guarded':{'n_requested':16}}}).encode(),
  'WRITER_FROZEN.json':json.dumps({'head_sha256':p.START_HEAD_SHA,'selected_checkpoint':p.START_MEMBER}).encode(),
  'heads/guarded/PROGRESS.json':json.dumps({'checkpoint':'checkpoint_000028.pt','attempts':28,'checkpoint_sha256':hashlib.sha256(cp).hexdigest()}).encode()}
 return m,pins,hashlib.sha256(parent).hexdigest(),hashlib.sha256(cp).hexdigest()

def metric(d):
 gates={n:True for n in p.GATES};gates['response']=d<=.002;return {'drift':d,'gates':gates,'episode_pass':all(gates.values())}
def rows(n,d=.001,valid=True):return [{'reference_valid':valid,'arms':{a:metric(d) for a in ('c17','calibrated')}} for _ in range(n)]
class Tests(unittest.TestCase):
 def test_true_start_pointer(self):self.assertEqual(p.START_MEMBER,'heads/guarded/checkpoint_000028.pt');self.assertEqual(len(p.START_CP_SHA),64)
 def test_36_training_unchanged(self):self.assertEqual(p.training_specs(),p.previous.training_specs());self.assertEqual(len(p.training_specs()),36)
 def test_16_new_states_balanced(self):
  self.assertEqual(len(p.test_specs()),16)
  for c in p.CASES:self.assertEqual(sum(t['case']==c for t in p.test_specs()),4)
 def test_new_seeds_disjoint_C17(self):self.assertFalse(set(t['seed'] for t in p.test_specs())&set(t['seed'] for t in p.previous.test_specs()))
 def test_no_inspected_states_promoted(self):self.assertFalse(p.SETTINGS['old_inspected_C15_C17_new_states_used_for_training'])
 def test_thresholds_unchanged(self):self.assertEqual([p.SETTINGS[k] for k in ('response_budget','kl_budget','margin_gate')],[.002,.005,.05])
 def test_no_online_geometry(self):
  for k in ('online_Jacobian_builds','online_SVD_calls','online_correction_steps','online_backward_calls'):self.assertEqual(p.SETTINGS[k],0)
 def test_not_native_or_independent_confirmation(self):self.assertFalse(p.SETTINGS['internal_parameter_update']);self.assertFalse(p.SETTINGS['fresh_evaluation']);self.assertFalse(p.SETTINGS['independent_model_seed'])
 def test_full_denominator_required(self):
  with self.assertRaises(ValueError):p.decide(rows(36),rows(15))
 def test_full_pass(self):self.assertEqual(p.decide(rows(36),rows(16)),'C18_RAW_LOCAL_STATE_PASS')
 def test_training_failure(self):self.assertEqual(p.decide(rows(36,.01),rows(16)),'C18_RAW_TRAINING_FUNCTION_FAILURE')
 def test_new_failure(self):self.assertEqual(p.decide(rows(36),rows(16,.01)),'C18_RAW_LOCAL_STATE_FAIL')
 def test_invalid_new_blocks_pass(self):self.assertEqual(p.decide(rows(36),rows(16,valid=False)),'C18_STATE_REFERENCE_FAILURE')
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
  parent=b'parent';cp=b'cp';m={'identity.json':json.dumps({'protocol':p.PROTOCOL,'settings':p.SETTINGS,'source_hashes':p.source_hashes(ROOT),'start_head_sha256':p.START_HEAD_SHA,'parent_c17_sha256':hashlib.sha256(parent).hexdigest()}).encode(),'parent_c17.zip':parent,
   'heads/calibrated/PROGRESS.json':json.dumps({'checkpoint':'checkpoint_000003.pt','rounds':3,'checkpoint_sha256':hashlib.sha256(cp).hexdigest()}).encode(),'heads/calibrated/checkpoint_000003.pt':cp}
  with tempfile.TemporaryDirectory() as d:
   f=Path(d)/'9999.zip';write_zip(f,m);self.assertEqual(p.resume_score(f)[-2],3)
   m['heads/calibrated/checkpoint_000003.pt']=b'changed';write_zip(f,m)
   with self.assertRaises(ValueError):p.resume_score(f)
 def test_namespace_audit(self):
  for n in ('worker_c18.py','launch_c18.py','c18_protocol.py','joint_calibration.py','output_layer.py'):self.assertEqual(global_audit(ROOT/n),set())
 def test_missing_import_negative_control(self):
  s=(ROOT/'worker_c18.py').read_text().replace('REVISION,CASES,STATES,GATES,','REVISION,CASES,GATES,')
  with tempfile.TemporaryDirectory() as d:
   f=Path(d)/'bad.py';f.write_text(s);self.assertIn('STATES',global_audit(f))
 def test_online_function_only_one_proposal_no_geometry(self):
  tree=ast.parse((ROOT/'worker_c18.py').read_text());f=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='online_raw_arm')
  banned={'autograd','grad','svd','eigh','solve','run_correction','offline_correct','build_route_geometry','set_layer_matrix','teacher'}
  self.assertFalse({n.id for n in ast.walk(f) if isinstance(n,ast.Name)}&banned)
  calls=[n for n in ast.walk(f) if isinstance(n,ast.Call)];self.assertFalse({n.func.attr for n in calls if isinstance(n.func,ast.Attribute)}&banned)
  self.assertEqual(sum(isinstance(n.func,ast.Name) and n.func.id=='one_raw_proposal' for n in calls),1)
 def test_new_generation_after_freeze(self):
  s=(ROOT/'worker_c18.py').read_text();self.assertLess(s.index("dump(out/'WRITER_FROZEN.json'"),s.index('def prepare_new('))
 def test_real_response_residuals_recorded(self):self.assertIn("v['response_residual']=residual.cpu().double().tolist()",(ROOT/'worker_c18.py').read_text())
 def test_c17_guard_still_protects_all_training_passes(self):
  a={t['id']:{'reference_valid':True,'metrics':metric(.003)} for t in p.training_specs()};b={t['id']:{'reference_valid':True,'metrics':metric(.001)} for t in p.training_specs()};sid=p.training_specs()[0]['id'];a[sid]['metrics']=metric(.001);b[sid]['metrics']=metric(.003)
  self.assertFalse(p.guard(a,b)['accepted'])
 def test_latest_three_own_backups_only(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/'session'/'run';out.mkdir(parents=True);(out/'identity.json').write_text(json.dumps({'protocol':p.PROTOCOL}));drive=Path(d)/'drive';drive.mkdir();old=drive/'older.zip';old.write_bytes(b'untouched')
   for _ in range(4):p.snapshot(out,drive)
   self.assertEqual(len(list(drive.glob(p.RESULT+'_*.zip'))),3);self.assertEqual(old.read_bytes(),b'untouched')

if __name__=='__main__':unittest.main()
