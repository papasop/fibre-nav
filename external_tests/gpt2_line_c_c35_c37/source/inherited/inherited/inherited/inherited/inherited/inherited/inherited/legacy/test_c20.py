import ast,copy,hashlib,io,json,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
import c20_protocol as p
from namespace_audit import global_audit
from test_c16 import write_zip
from coverage_guard import select_candidate,merge_route,check_replay
ROOT=Path(__file__).parent

def metric(d=.001):
 gates={g:True for g in p.GATES};gates['response']=d<=.002
 return {'drift':d,'kl':.001,'gates':gates,'episode_pass':all(gates.values())}
def cohort(d=.003):return {t['id']:{'reference_valid':True,'metrics':metric(d)} for t in p.training_specs()}
def episodes(n,d=.001,valid=True):return [{'reference_valid':valid,'arms':{a:metric(d) for a in ('c19','calibrated')}} for _ in range(n)]
def zipped(members):
 b=io.BytesIO()
 with zipfile.ZipFile(b,'w') as z:
  for n,bits in members.items():z.writestr(n,bits)
 return b.getvalue()
def fixture():
 source=b'source fixture';pins={'a.py':hashlib.sha256(source).hexdigest()};c15=b'original15';cp=b'actualcp';parent=zipped({'parent_c17.zip':zipped({'parent_c15.zip':c15})})
 return {'identity.json':json.dumps({'protocol':p.PARENT_PROTOCOL,'settings':p.previous.SETTINGS,'source_hashes':pins,'start_head_sha256':p.previous.START_HEAD_SHA,'parent_c18_sha256':hashlib.sha256(parent).hexdigest()}).encode(),
  'source/a.py':source,'parent_c18.zip':parent,p.START_MEMBER:cp,
  'SUMMARY.json':json.dumps({'complete':True,'decision':'C19_RAW_TRAINING_FUNCTION_FAILURE','writer_head_sha256':p.START_HEAD_SHA,'training':{'calibrated':{'n_requested':36,'passed':22}},'prospective_local_state_development':{'calibrated':{'n_requested':16}}}).encode(),
  'WRITER_FROZEN.json':json.dumps({'head_sha256':p.START_HEAD_SHA,'selected_checkpoint':p.START_MEMBER}).encode(),
  'heads/calibrated/PROGRESS.json':json.dumps({'checkpoint':'checkpoint_000016.pt','rounds':16,'checkpoint_sha256':hashlib.sha256(cp).hexdigest()}).encode()},pins,hashlib.sha256(c15).hexdigest(),hashlib.sha256(cp).hexdigest()

class Tests(unittest.TestCase):
 def test_actual_C19_pointer(self):self.assertEqual(p.START_MEMBER,'heads/calibrated/checkpoint_000016.pt');self.assertEqual(len(p.START_CP_SHA),64)
 def test_original36_bytewise_spec_unchanged(self):self.assertEqual(p.original_training_specs(),p.previous.training_specs())
 def test_total60_unique(self):self.assertEqual(len(p.training_specs()),60);self.assertEqual(len({t['id'] for t in p.training_specs()}),60)
 def test_each_route_has15(self):
  for c in p.CASES:self.assertEqual(sum(t['case']==c for t in p.training_specs()),15)
 def test_supplemental24_balanced(self):
  for c in p.CASES:
   group=[t for t in p.supplemental_training_specs() if t['case']==c]
   self.assertEqual(len(group),6);self.assertEqual(sum(t['norm']==.375 for t in group),3);self.assertEqual(sum(t['norm']==.75 for t in group),3)
 def test_each_original_base_covered_both_training_radii(self):
  for sid in p.STATES:self.assertEqual({t['norm'] for t in p.supplemental_training_specs() if t['base_id']==sid},{.375,.75})
 def test_test16_balanced(self):self.assertEqual(len(p.test_specs()),16)
 def test_train_test_seeds_disjoint(self):self.assertFalse({t['seed'] for t in p.test_specs()} & {t['seed'] for t in p.supplemental_training_specs()})
 def test_all_inspected_seeds_disjoint(self):
  import c15_protocol,c17_protocol,c18_protocol,c19_protocol
  seeds={t['seed'] for t in p.test_specs()+p.supplemental_training_specs()}
  for old in (c15_protocol,c17_protocol,c18_protocol,c19_protocol):self.assertFalse(seeds & {t['seed'] for t in old.test_specs()})
 def test_no_old_inspected_states_promoted(self):self.assertFalse(p.SETTINGS['old_inspected_C15_C17_C18_C19_new_states_used_for_training'])
 def test_fixed_thresholds(self):self.assertEqual([p.SETTINGS[k] for k in ('response_budget','kl_budget','margin_gate')],[.002,.005,.05])
 def test_no_online_geometry_or_fallback(self):
  for k in ('online_Jacobian_builds','online_SVD_calls','online_correction_steps','online_backward_calls'):self.assertEqual(p.SETTINGS[k],0)
  self.assertFalse(p.SETTINGS['teacher_fallback_allowed']);self.assertEqual(p.SETTINGS['online_proposals_per_arm'],1)
 def test_development_scope(self):self.assertFalse(p.SETTINGS['fresh_evaluation']);self.assertFalse(p.SETTINGS['internal_parameter_update']);self.assertFalse(p.SETTINGS['controller_dispatch_tested'])
 def test_full_denominator(self):
  with self.assertRaises(ValueError):p.decide(episodes(59),episodes(16))
 def test_all_pass(self):self.assertEqual(p.decide(episodes(60),episodes(16)),'C20_RAW_LOCAL_STATE_PASS')
 def test_training_fail(self):self.assertEqual(p.decide(episodes(60,.003),episodes(16)),'C20_RAW_TRAINING_FUNCTION_FAILURE')
 def test_new_fail(self):self.assertEqual(p.decide(episodes(60),episodes(16,.003)),'C20_RAW_LOCAL_STATE_FAIL')
 def test_invalid_reference_retained_blocks_acceptance(self):self.assertEqual(p.decide(episodes(60,valid=False),episodes(16)),'C20_STATE_REFERENCE_FAILURE')
 def test_expanded_quality_rejects_incomplete_set(self):
  with self.assertRaises(ValueError):p.quality({k:v for k,v in list(cohort().items())[:-1]})
 def test_invalid_reference_not_counted_as_pass(self):
  r=cohort(.001);r[next(iter(r))]['reference_valid']=False;self.assertEqual(len(p.quality(r)['passed_ids']),59);self.assertEqual(p.quality(r)['valid_references'],59)
 def test_invalid_training_reference_prevents_fit_acceptance(self):
  a=cohort(.003);b=cohort(.001);a[next(iter(a))]['reference_valid']=False;self.assertIn('INVALID_TRAINING_REFERENCE',p.guard(a,b)['reasons'])
 def test_preserve_original_passes(self):
  a=cohort(.003);sid=p.original_training_specs()[0]['id'];a[sid]['metrics']=metric(.001);b=cohort(.001);b[sid]['metrics']=metric(.003);self.assertFalse(p.guard(a,b)['accepted'])
 def test_new_outlier_cannot_hide_original_max_regression(self):
  a=cohort(.006);b=cohort(.004);oldids={t['id'] for t in p.original_training_specs()}
  for sid in oldids:a[sid]['metrics']=metric(.003);b[sid]['metrics']=metric(.0035)
  g=p.guard(a,b);self.assertIn('ORIGINAL36_WORST_DRIFT_INCREASED',g['reasons']);self.assertNotIn('WORST_RESPONSE_DRIFT_INCREASED',g['reasons'])
 def test_new_passes_protected_too(self):
  a=cohort(.003);sid=p.supplemental_training_specs()[0]['id'];a[sid]['metrics']=metric(.001);b=cohort(.001);b[sid]['metrics']=metric(.003);self.assertFalse(p.guard(a,b)['accepted'])
 def test_later_ridge_wins(self):
  a=cohort(.004);b={'rows':cohort(.003),'ridge_index':0,'backtrack_index':0};c={'rows':cohort(.001),'ridge_index':1,'backtrack_index':0};self.assertIs(select_candidate(a,[b,c]),c)
 def test_route_merge15_and_remaining45_unchanged(self):
  a=cohort();ids=[t['id'] for t in p.training_specs() if t['case']==p.CASES[0]];part={k:{'reference_valid':True,'metrics':metric(.001)} for k in ids};b=merge_route(a,part,ids)
  self.assertEqual(len(p.quality(b)['passed_ids']),15)
  for k in set(a)-set(ids):self.assertIs(a[k],b[k])
 def test_old9_route_merge_rejected(self):
  a=cohort();ids=list(a)[:9]
  with self.assertRaises(ValueError):merge_route(a,{k:a[k] for k in ids},ids)
 def test_full60_replay_detects_changed_untouched_gate(self):
  a=cohort();b=copy.deepcopy(a);b[list(b)[-1]]['metrics']['gates']['KL']=False
  with self.assertRaises(ValueError):check_replay(a,b)
 def test_parent_structural_fixture(self):
  m,pins,ph,ch=fixture()
  with tempfile.TemporaryDirectory() as d,patch.object(p,'C15_SHA',ph),patch.object(p,'START_CP_SHA',ch):
   path=Path(d)/'p.zip';write_zip(path,m);self.assertTrue(p.validate_parent(path,pins)['summary']['complete'])
 def test_wrong_real_checkpoint_rejected(self):
  m,pins,ph,ch=fixture();m[p.START_MEMBER]=b'changed'
  with tempfile.TemporaryDirectory() as d,patch.object(p,'C15_SHA',ph),patch.object(p,'START_CP_SHA',ch):
   path=Path(d)/'p.zip';write_zip(path,m)
   with self.assertRaises(ValueError):p.validate_parent(path,pins)
 def test_wrong_real_head_rejected(self):
  m,pins,ph,ch=fixture();f=json.loads(m['WRITER_FROZEN.json']);f['head_sha256']='wrong';m['WRITER_FROZEN.json']=json.dumps(f).encode()
  with tempfile.TemporaryDirectory() as d,patch.object(p,'C15_SHA',ph),patch.object(p,'START_CP_SHA',ch):
   path=Path(d)/'p.zip';write_zip(path,m)
   with self.assertRaises(ValueError):p.validate_parent(path,pins)
 def test_incomplete_parent_rejected(self):
  m,pins,ph,ch=fixture();f=json.loads(m['SUMMARY.json']);f['complete']=False;m['SUMMARY.json']=json.dumps(f).encode()
  with tempfile.TemporaryDirectory() as d,patch.object(p,'C15_SHA',ph),patch.object(p,'START_CP_SHA',ch):
   path=Path(d)/'p.zip';write_zip(path,m)
   with self.assertRaises(ValueError):p.validate_parent(path,pins)
 def test_source_archive_friendly_rejection(self):
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'source.zip';path.write_bytes(zipped({'SOURCE_MANIFEST.json':b'{}'}))
   with self.assertRaisesRegex(ValueError,'[Rr]esult ZIP'):p.input_kind(path)
 def test_resume_round_not_filename(self):
  cp=b'cp';parent=b'parent';m={'identity.json':json.dumps({'protocol':p.PROTOCOL,'settings':p.SETTINGS,'source_hashes':p.source_hashes(ROOT),'start_head_sha256':p.START_HEAD_SHA,'parent_c19_sha256':hashlib.sha256(parent).hexdigest()}).encode(),'parent_c19.zip':parent,
  'heads/calibrated/PROGRESS.json':json.dumps({'checkpoint':'checkpoint_000003.pt','rounds':3,'checkpoint_sha256':hashlib.sha256(cp).hexdigest()}).encode(),'heads/calibrated/checkpoint_000003.pt':cp}
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'99999.zip';write_zip(path,m);self.assertEqual(p.resume_score(path)[3],3)
   m['heads/calibrated/checkpoint_000003.pt']=b'bad';write_zip(path,m)
   with self.assertRaises(ValueError):p.resume_score(path)
 def test_pre_checkpoint_preparation_progress_counted(self):
  parent=b'parent';m={'identity.json':json.dumps({'protocol':p.PROTOCOL,'settings':p.SETTINGS,'source_hashes':p.source_hashes(ROOT),'start_head_sha256':p.START_HEAD_SHA,'parent_c19_sha256':hashlib.sha256(parent).hexdigest()}).encode(),'parent_c19.zip':parent,'states/coverage_training/x/REFERENCE.json':b'{}'}
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'p.zip';write_zip(path,m);self.assertEqual(p.resume_score(path)[-1],1)
 def test_namespace_audit_and_negative_control(self):
  for n in ('worker_c20.py','launch_c20.py','c20_protocol.py','coverage_guard.py'):self.assertEqual(global_audit(ROOT/n),set())
  s=(ROOT/'worker_c20.py').read_text().replace('REVISION,CASES,STATES,GATES,','REVISION,CASES,GATES,')
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'bad.py';path.write_text(s);self.assertIn('STATES',global_audit(path))
 def test_inherited63_modules_unchanged(self):p.inherited_check(ROOT);self.assertEqual(len(json.loads((ROOT/'C19_SOURCE_PINS.json').read_text())),63)
 def test_online_arm_no_geometry(self):
  tree=ast.parse((ROOT/'worker_c20.py').read_text());f=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='online_raw_arm');banned={'autograd','grad','svd','eigh','solve','run_correction','offline_correct','prepare_state','set_layer_matrix','teacher'}
  self.assertFalse({n.id for n in ast.walk(f) if isinstance(n,ast.Name)}&banned);calls=[n for n in ast.walk(f) if isinstance(n,ast.Call)];self.assertEqual(sum(isinstance(n.func,ast.Name) and n.func.id=='one_raw_proposal' for n in calls),1)
 def test_preparation_then_fit_then_frozen_test_order(self):
  s=(ROOT/'worker_c20.py').read_text();self.assertLess(s.index('for index,t in enumerate(supplemental_training_specs())'),s.index("while rounds<SETTINGS['max_calibration_rounds']"));self.assertLess(s.index("dump(out/'WRITER_FROZEN.json'"),s.index("prepare_state(t,freeze['head_sha256'])"))
 def test_invalid_training_stops_without_resampling(self):
  s=(ROOT/'worker_c20.py').read_text();self.assertIn("stop=None if initial_quality['valid_references']==60 else 'TRAINING_STATE_REFERENCE_FAILURE'",s);self.assertIn("'generation_retries':0,'replacement_used':False",s)
 def test_ridge_comparison_and_full60_audit(self):
  s=(ROOT/'worker_c20.py').read_text();self.assertIn("'full60_physical_audit':True",s);tree=ast.parse(s);loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and ast.unparse(n.iter)=="enumerate(SETTINGS['ridge_grid'])");self.assertFalse(any(isinstance(n,ast.Break) for n in loop.body))
 def test_drive_already_mounted_not_reauthorised(self):self.assertIn("if not Path('/content/drive/MyDrive').is_dir()",(ROOT/'launch_c20.py').read_text())
 def test_latest_three_backups_older_archives_untouched(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/'session'/'run';out.mkdir(parents=True);(out/'identity.json').write_text(json.dumps({'protocol':p.PROTOCOL}));drive=Path(d)/'drive';drive.mkdir();old=drive/'old.zip';old.write_bytes(b'untouched')
   for _ in range(4):p.snapshot(out,drive)
   self.assertEqual(len(list(drive.glob(p.RESULT+'_*.zip'))),3);self.assertEqual(old.read_bytes(),b'untouched')

if __name__=='__main__':unittest.main()
