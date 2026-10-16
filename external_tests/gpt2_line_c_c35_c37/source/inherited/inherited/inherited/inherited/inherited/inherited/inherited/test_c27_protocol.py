import hashlib,io,json,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
from c27_protocol import *


def row(d=.001):
 return {'reference_valid':True,'metrics':{'drift':d,'kl':.001,'gates':{g:(d<=.002 if g=='response' else True) for g in GATES}}}


class Tests(unittest.TestCase):
 def test_grid_18_complete_unique(self):
  s=specs();self.assertEqual(len(s),18);self.assertEqual(len({x['id'] for x in s}),18)
  self.assertEqual({x['ridge'] for x in s},{1e-6,1e-8});self.assertEqual({x['alpha'] for x in s},{1.,.5,.25})
 def test_budget_unchanged(self):self.assertEqual([SETTINGS[k] for k in ('response_budget','kl_budget','margin_gate')],[.002,.005,.05])
 def test_no_fresh_native_claim(self):
  self.assertFalse(SETTINGS['fresh_evaluation']);self.assertFalse(SETTINGS['internal_parameter_update']);self.assertFalse(SETTINGS['new_state_generation'])
 def test_count_invalid_not_dropped(self):
  r={'a':row(),'b':row()};r['b']['reference_valid']=False;x=metrics_counts(r);self.assertEqual(x['n'],2);self.assertEqual(x['passed'],1)
 def test_training_failure_precedes_dev(self):
  tr={a:{'n':76,'valid':76,'passed':74} for a in ARMS};de={a:{'n':16,'valid':16,'passed':16} for a in ARMS}
  self.assertEqual(decide(tr,de),'C27_RESIDUAL_TRAINING_INCOMPLETE')
 def test_full_train_not_full_dev(self):
  tr={a:{'n':76,'valid':76,'passed':76} for a in ARMS};de={a:{'n':16,'valid':16,'passed':5} for a in ARMS}
  self.assertEqual(decide(tr,de),'C27_INSPECTED_STATE_GENERALISATION_FAILURE')
 def test_only_inspected_pass_not_LC1(self):
  tr={a:{'n':76,'valid':76,'passed':76} for a in ARMS};de={a:{'n':16,'valid':16,'passed':16} for a in ARMS}
  self.assertEqual(decide(tr,de),'C27_INSPECTED_STATE_RESIDUAL_PASS')
 def test_invalid_reference_failure(self):
  tr={'state_bypass':{'n':76,'valid':75,'passed':75}};de={'state_bypass':{'n':16,'valid':16,'passed':16}}
  self.assertEqual(decide(tr,de),'C27_STATE_REFERENCE_FAILURE')
 def test_guard_protects_previous_pass(self):
  ids=['xor_10__a','xor_10__b'];before={ids[0]:row(),ids[1]:row(.005)};after={ids[0]:row(.0021),ids[1]:row(.003)}
  self.assertFalse(guard(before,after,ids,[ids[0]])['accepted'])
 def test_snapshot_manifest_nested_and_no_parent_touch(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);out=p/'session/run';out.mkdir(parents=True);dump(out/'identity.json',{'protocol':PROTOCOL})
   dump(out/'nested/MANIFEST.json',{'retained':True});dump(out/'state.json',{'x':1})
   unrelated=p/'C25.zip';unrelated.write_bytes(b'parent');result=snapshot(out)
   with zipfile.ZipFile(result) as z:
    m=verify_archive(z);self.assertIn('nested/MANIFEST.json',m)
   self.assertEqual(unrelated.read_bytes(),b'parent')
 def test_backup_prunes_only_verified_own_results(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);out=p/'session/run';out.mkdir(parents=True);dump(out/'identity.json',{'protocol':PROTOCOL});dd=p/'drive';dd.mkdir()
   unrelated=dd/'qwen_old.zip';unrelated.write_bytes(b'untouched')
   for i in range(4):dump(out/'state.json',{'i':i});snapshot(out,str(dd))
   self.assertEqual(len(list(dd.glob(RESULT+'_*.zip'))),3);self.assertEqual(unrelated.read_bytes(),b'untouched')
 def test_resume_requires_identical_source(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);root=p/'source';root.mkdir();(root/'a.py').write_text('x=1')
   out=p/'session/run';out.mkdir(parents=True)
   dump(out/'identity.json',{'protocol':PROTOCOL,'settings':SETTINGS,'source_hashes':source_hashes(root),'parent_C25_sha256':PARENT_SHA})
   result=snapshot(out);self.assertEqual(resume_score(result,root),(0,0,0,0,0))
   (root/'a.py').write_text('x=2')
   with self.assertRaises(ValueError):resume_score(result,root)
 def test_unsafe_archive_rejected(self):
  b=io.BytesIO()
  with zipfile.ZipFile(b,'w') as z:z.writestr('../a',b'x');z.writestr('MANIFEST.json',json.dumps({'../a':hashlib.sha256(b'x').hexdigest()}))
  b.seek(0)
  with zipfile.ZipFile(b) as z:
   with self.assertRaises(ValueError):verify_archive(z)
 def test_wrong_source_digest_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'legacy').mkdir();(p/'legacy/a').write_text('bad');dump(p/'C25_SOURCE_PINS.json',{'a':'0'*64})
   with self.assertRaises(ValueError):check_legacy(p)


if __name__=='__main__':unittest.main()
