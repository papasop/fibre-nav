import ast,copy,hashlib,io,json,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
import c16_protocol as p
from namespace_audit import global_audit
ROOT=Path(__file__).parent

def rows(n,warm_pass=False,final_pass=False,warm_drift=.1,final_drift=.2):
 def arm(ok,d):return {'metrics':{'episode_pass':ok,'drift':d,'gates':{'response':d<=.002}},'teacher_error':{'l2':1.}}
 return [{'reference_valid':True,'arms':{'warm1000':arm(warm_pass,warm_drift),'functional720':arm(final_pass,final_drift),'c9_baseline':arm(False,.1)}} for _ in range(n)]

def cpu_fixture():
 # Structural ZIP fixture only; dummy PT payloads are not Torch/physical evidence.
 source=b'# fixture source';pins={'a.py':hashlib.sha256(source).hexdigest()};c9=b'fixture-parent-c9';c9hash=hashlib.sha256(c9).hexdigest()
 members={'parent_c9.zip':c9,'source/a.py':source,
  'identity.json':json.dumps({'protocol':p.PARENT_PROTOCOL,'settings':p.c15.SETTINGS,'parent_c9_sha256':c9hash,'source_hashes':pins}).encode(),
  'SUMMARY.json':json.dumps({'complete':True,'training_raw':{'n_requested':36},'new_state_raw':{'n_requested':16},'writer_head_sha256':p.FINAL_HEAD_SHA,'baseline_c9_head_sha256':p.HEAD_SHA}).encode(),
  'WRITER_FROZEN.json':json.dumps({'head_sha256':p.FINAL_HEAD_SHA,'selected_checkpoint':'heads/functional/checkpoint_000720.pt'}).encode()}
 for stage,total in [('warm',1000),('functional',720)]:
  name='checkpoint_%06d.pt'%total;b=b'fixture-'+stage.encode();base='heads/'+stage+'/'
  members[base+name]=b;members[base+'PROGRESS.json']=json.dumps({'checkpoint':name,'steps':total,'checkpoint_sha256':hashlib.sha256(b).hexdigest()}).encode()
 for t in p.state_specs():
  base='states/'+t['cohort']+'/'+t['id']+'/';ev='evaluation/'+t['cohort']+'/'+t['id']+'/'
  for n in ['prepared.pt','teacher.pt','FEATURES.pt','REFERENCE.json']:members[base+n]=b'fixture'
  for n in ['ONLINE_INPUT.pt','c15_raw.pt','c15_raw.json','frozen_c9_raw.pt','frozen_c9_raw.json','COMPLETE.json']:members[ev+n]=b'fixture'
 return members,pins,c9hash

def write_zip(path,members):
 members=dict(members);members['MANIFEST.json']=json.dumps({n:hashlib.sha256(b).hexdigest() for n,b in members.items()}).encode()
 with zipfile.ZipFile(path,'w') as z:
  for n,b in members.items():z.writestr(n,b)

class Tests(unittest.TestCase):
 def test_full_52_cohort(self):self.assertEqual(len(p.state_specs()),52)
 def test_36_16_split(self):self.assertEqual([sum(s['cohort']==c for s in p.state_specs()) for c in ('training','new')],[36,16])
 def test_ids_unique(self):self.assertEqual(len({(s['cohort'],s['id']) for s in p.state_specs()}),52)
 def test_new_class_coverage(self):
  for c in p.CASES:self.assertEqual(sum(s['case']==c and s['cohort']=='new' for s in p.state_specs()),4)
 def test_fixed_three_heads(self):self.assertEqual(p.ARMS,('c9_baseline','warm1000','functional720'))
 def test_original_thresholds(self):self.assertEqual((p.SETTINGS['response_budget'],p.SETTINGS['kl_budget'],p.SETTINGS['margin_gate']),(.002,.005,.05))
 def test_no_new_work(self):
  for k in ('new_training_steps','new_states','Jacobian_builds','SVD_calls','correction_steps'):self.assertEqual(p.SETTINGS[k],0)
 def test_inspected_not_fresh(self):self.assertFalse(p.SETTINGS['fresh_evaluation']);self.assertFalse(p.SETTINGS['native_update'])
 def test_median_odd(self):self.assertEqual(p.median([3,1,2]),2)
 def test_median_even(self):self.assertEqual(p.median([1,3]),2)
 def test_empty_median_none(self):self.assertIsNone(p.median([]))
 def test_invalid_reference_not_pass(self):
  r=rows(36,True,True);r[0]['reference_valid']=False;c=p.cohort_counts(r,'warm1000');self.assertEqual(c['n_requested'],36);self.assertEqual(c['passed'],35)
 def test_paired_loss(self):
  a=p.paired(rows(36,True,False,.001,.1));self.assertEqual(a['complete_lost'],36);self.assertEqual(a['response_lost'],36)
 def test_paired_gain(self):
  a=p.paired(rows(36,False,True,.1,.001));self.assertEqual(a['complete_gained'],36);self.assertEqual(a['response_gained'],36)
 def test_paired_equal(self):self.assertEqual(p.paired(rows(36,False,False,.1,.1))['drift_equal'],36)
 def test_warm_failure_plus_worsening(self):self.assertEqual(p.interpretation(rows(36),True),'WARM_ACCEPTANCE_INSUFFICIENT_FUNCTIONAL_STAGE_WORSENS_RECORDED_COHORT')
 def test_warm_pass_final_failure(self):self.assertEqual(p.interpretation(rows(36,True,False,.001,.1),True),'FUNCTIONAL_STAGE_LOSES_COMPLETE_TRAINING_ACCEPTANCE')
 def test_both_fail_without_deterioration(self):self.assertEqual(p.interpretation(rows(36,False,False,.2,.1),True),'BOTH_CHECKPOINTS_FAIL_COMPLETE_TRAINING_ACCEPTANCE')
 def test_functional_reaches_training_pass(self):self.assertEqual(p.interpretation(rows(36,False,True,.1,.001),True),'FUNCTIONAL_STAGE_REACHES_TRAINING_ACCEPTANCE')
 def test_replay_mismatch_blocks_stage_attribution(self):self.assertIn('UNRESOLVED',p.interpretation(rows(36),False))
 def test_missing_rows_block_interpretation(self):
  with self.assertRaises(ValueError):p.interpretation(rows(35),True)
 def test_no_empty_pair(self):
  with self.assertRaises(ValueError):p.paired([])
 def test_complete_structural_parent(self):
  m,pins,ch=cpu_fixture()
  with tempfile.TemporaryDirectory() as d,patch.object(p,'C9_SHA',ch):
   f=Path(d)/'parent.zip';write_zip(f,m);r=p.validate_parent(f,pins);self.assertEqual(r['checkpoints']['warm']['steps'],1000)
 def test_incomplete_parent_rejected(self):
  m,pins,ch=cpu_fixture();s=json.loads(m['SUMMARY.json']);s['complete']=False;m['SUMMARY.json']=json.dumps(s).encode()
  with tempfile.TemporaryDirectory() as d,patch.object(p,'C9_SHA',ch):
   f=Path(d)/'parent.zip';write_zip(f,m)
   with self.assertRaises(ValueError):p.validate_parent(f,pins)
 def test_checkpoint_digest_mismatch_rejected(self):
  m,pins,ch=cpu_fixture();m['heads/warm/checkpoint_001000.pt']=b'changed'
  with tempfile.TemporaryDirectory() as d,patch.object(p,'C9_SHA',ch):
   f=Path(d)/'parent.zip';write_zip(f,m)
   with self.assertRaises(ValueError):p.validate_parent(f,pins)
 def test_wrong_final_head_rejected(self):
  m,pins,ch=cpu_fixture();s=json.loads(m['WRITER_FROZEN.json']);s['head_sha256']='wrong';m['WRITER_FROZEN.json']=json.dumps(s).encode()
  with tempfile.TemporaryDirectory() as d,patch.object(p,'C9_SHA',ch):
   f=Path(d)/'parent.zip';write_zip(f,m)
   with self.assertRaises(ValueError):p.validate_parent(f,pins)
 def test_source_identity_mismatch_rejected(self):
  m,pins,ch=cpu_fixture();m['source/a.py']=b'changed'
  with tempfile.TemporaryDirectory() as d,patch.object(p,'C9_SHA',ch):
   f=Path(d)/'parent.zip';write_zip(f,m)
   with self.assertRaises(ValueError):p.validate_parent(f,pins)
 def test_missing_state_rejected(self):
  m,pins,ch=cpu_fixture();del m['states/new/xor_11__c15_new3/prepared.pt']
  with tempfile.TemporaryDirectory() as d,patch.object(p,'C9_SHA',ch):
   f=Path(d)/'parent.zip';write_zip(f,m)
   with self.assertRaises(KeyError):p.validate_parent(f,pins)
 def test_wrong_checkpoint_steps_rejected(self):
  m,pins,ch=cpu_fixture();s=json.loads(m['heads/warm/PROGRESS.json']);s['steps']=999;m['heads/warm/PROGRESS.json']=json.dumps(s).encode()
  with tempfile.TemporaryDirectory() as d,patch.object(p,'C9_SHA',ch):
   f=Path(d)/'parent.zip';write_zip(f,m)
   with self.assertRaises(ValueError):p.validate_parent(f,pins)
 def test_unsafe_pointer_rejected(self):
  m,pins,ch=cpu_fixture();s=json.loads(m['heads/warm/PROGRESS.json']);s['checkpoint']='../bad.pt';m['heads/warm/PROGRESS.json']=json.dumps(s).encode()
  with tempfile.TemporaryDirectory() as d,patch.object(p,'C9_SHA',ch):
   f=Path(d)/'parent.zip';write_zip(f,m)
   with self.assertRaises(ValueError):p.validate_parent(f,pins)
 def test_namespace_complete(self):self.assertEqual(global_audit(ROOT/'worker_c16.py'),set())
 def test_missing_import_negative_control(self):
  text=(ROOT/'worker_c16.py').read_text().replace('episode_gate,target_from_reads,','episode_gate,')
  with tempfile.TemporaryDirectory() as d:
   f=Path(d)/'bad.py';f.write_text(text);self.assertIn('target_from_reads',global_audit(f))
 def test_no_training_or_geometry_in_worker(self):
  tree=ast.parse((ROOT/'worker_c16.py').read_text());forbidden={'backward','step','svd','jacobian','correct','ridge_correction','null_project','fit_stage','AdamW'}
  calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
  self.assertFalse([n for n in calls if (isinstance(n.func,ast.Name) and n.func.id in forbidden) or (isinstance(n.func,ast.Attribute) and n.func.attr in forbidden)])
 def test_models_loaded_once(self):
  tree=ast.parse((ROOT/'worker_c16.py').read_text());self.assertEqual(sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='make_model' for n in ast.walk(tree)),1)
 def test_all_local_modules_present(self):
  for n in ('raw_writer','proposer','writer_core','run_l5_audit','c16_protocol','runtime_a100','confirmation_protocol','c15_protocol','namespace_audit'):self.assertTrue((ROOT/(n+'.py')).exists())
 def test_inherited_modules_unchanged(self):p.inherited_check(ROOT)
 def test_no_production_head_selected(self):self.assertIn('no production head selected',p.SETTINGS['selection'])
 def test_resume_prefers_completed_records(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/'run';out.mkdir();parent=b'fixture parent';(out/'parent_c15.zip').write_bytes(parent)
   ident={'protocol':p.PROTOCOL,'settings':p.SETTINGS,'source_hashes':p.source_hashes(ROOT),'parent_c15_sha256':hashlib.sha256(parent).hexdigest()}
   (out/'identity.json').write_text(json.dumps(ident));z=p.snapshot(out);self.assertEqual(p.resume_score(z),(0,0,0))
   f=out/'evaluation'/'training'/'xor_00__v0'/'warm1000'/'RESULT.json';f.parent.mkdir(parents=True);f.write_text('{}')
   z=p.snapshot(out);self.assertEqual(p.resume_score(z),(0,0,1))
   (f.parent.parent/'COMPLETE.json').write_text('{}');z=p.snapshot(out);self.assertEqual(p.resume_score(z),(0,1,1))
 def test_resume_changed_settings_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/'run';out.mkdir();(out/'parent_c15.zip').write_bytes(b'fixture')
   ident={'protocol':p.PROTOCOL,'settings':dict(p.SETTINGS,response_budget=.01),'source_hashes':p.source_hashes(ROOT),'parent_c15_sha256':hashlib.sha256(b'fixture').hexdigest()}
   (out/'identity.json').write_text(json.dumps(ident));z=p.snapshot(out)
   with self.assertRaises(ValueError):p.resume_score(z)
 def test_snapshot_own_protocol(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/'run';out.mkdir();(out/'identity.json').write_text(json.dumps({'protocol':p.PROTOCOL}));f=p.snapshot(out)
   self.assertEqual(f.name,p.RESULT+'.zip');self.assertEqual(p.verified_zip(f)['protocol'],p.PROTOCOL)
 def test_drive_preserves_old_versions(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/'run';out.mkdir();drive=Path(d)/'drive';drive.mkdir();old=drive/'c15_old.zip';old.write_bytes(b'keep')
   (out/'identity.json').write_text(json.dumps({'protocol':p.PROTOCOL}))
   for i in range(4):(out/'a.txt').write_text(str(i));p.snapshot(out,drive)
   self.assertEqual(len(list(drive.glob(p.RESULT+'_*.zip'))),3);self.assertEqual(old.read_bytes(),b'keep')

if __name__=='__main__':unittest.main()
