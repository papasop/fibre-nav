import hashlib,io,json,tempfile,unittest,zipfile
from pathlib import Path
from c28_protocol import *

def row(d):return {'reference_valid':True,'metrics':{'drift':d,'kl':.001,'gates':{g:(d<=.002 if g=='response' else True) for g in GATES}}}
class Tests(unittest.TestCase):
 def test_eighteen_unique_specs(self):self.assertEqual(len(specs()),18);self.assertEqual(len({s['id'] for s in specs()}),18)
 def test_same_thresholds(self):self.assertEqual([SETTINGS[k] for k in ('response_budget','kl_budget','margin_gate')],[.002,.005,.05])
 def test_no_fresh_native_claim(self):self.assertFalse(SETTINGS['fresh_evaluation']);self.assertFalse(SETTINGS['internal_parameter_update'])
 def test_no_unseen_state_generation(self):self.assertFalse(SETTINGS['new_state_generation'])
 def test_all_four_routes(self):self.assertEqual(SETTINGS['routes'],list(CASES))
 def test_same_parameter_count(self):self.assertEqual(SETTINGS['features_plus_bias'],9)
 def test_route_loss_rejected(self):self.assertFalse(route_guard({'x':row(.0019)},{'x':row(.0021)},['x'])['accepted'])
 def test_nonimprovement_rejected(self):self.assertFalse(route_guard({'x':row(.001)},{'x':row(.001)},['x'])['accepted'])
 def test_improvement_accepted(self):self.assertTrue(route_guard({'x':row(.0019)},{'x':row(.0012)},['x'])['accepted'])
 def test_worst_not_traded_for_average(self):self.assertFalse(route_guard({'a':row(.0015),'b':row(.0015)},{'a':row(.0001),'b':row(.0016)},['a','b'])['accepted'])
 def test_invalid_reference_rejected(self):
  b={'x':row(.001)};b['x']['reference_valid']=False;self.assertFalse(route_guard({'x':row(.0019)},b,['x'])['accepted'])
 def test_denominator_not_changed(self):
  with self.assertRaises(ValueError):route_guard({'x':row(.001)},{},['x'])
 def test_nested_manifest_covered(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/'session/run';out.mkdir(parents=True);dump(out/'identity.json',{'protocol':PROTOCOL});dump(out/'source/inherited/MANIFEST.json',{'x':1});p=snapshot(out)
   with zipfile.ZipFile(p) as z:self.assertIn('source/inherited/MANIFEST.json',verify_archive(z))
 def test_backup_only_own_recent3(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/'session/run';out.mkdir(parents=True);dd=Path(d)/'drive';dd.mkdir();other=dd/'C27.zip';other.write_bytes(b'parent');dump(out/'identity.json',{'protocol':PROTOCOL})
   for i in range(4):dump(out/'step.json',{'i':i});snapshot(out,str(dd))
   self.assertEqual(len(list(dd.glob(RESULT+'_*.zip'))),3);self.assertEqual(other.read_bytes(),b'parent')
 def test_source_changed_resume_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)/'source';root.mkdir();(root/'x.py').write_text('x=1');out=Path(d)/'session/run';out.mkdir(parents=True);dump(out/'identity.json',{'protocol':PROTOCOL,'settings':SETTINGS,'source_hashes':source_hashes(root)});p=snapshot(out);self.assertEqual(resume_score(p,root),(0,0,0,0,0,0));(root/'x.py').write_text('x=2')
   with self.assertRaises(ValueError):resume_score(p,root)
 def test_wrong_parent_not_accepted(self):
  b=io.BytesIO()
  with zipfile.ZipFile(b,'w') as z:z.writestr('identity.json',json.dumps({'protocol':'SOURCE'}));z.writestr('SUMMARY.json','{}');z.writestr('arms/state_bypass/FROZEN.json','{}')
  b.seek(0)
  with zipfile.ZipFile(b) as z:
   with self.assertRaises((ValueError,KeyError)):parent_probe(z,Path(__file__).parent)
if __name__=='__main__':unittest.main()
