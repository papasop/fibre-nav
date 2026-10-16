import io,json,tempfile,unittest,zipfile
from pathlib import Path
from c29_protocol import *
class Tests(unittest.TestCase):
 def test_not_training(self):self.assertEqual(SETTINGS['new_head_training_steps'],0);self.assertEqual(SETTINGS['new_candidates'],0);self.assertEqual(SETTINGS['new_heads_committed'],0)
 def test_only_inspected_not_fresh(self):self.assertFalse(SETTINGS['fresh_evaluation']);self.assertFalse(SETTINGS['new_state_generation']);self.assertFalse(SETTINGS['internal_parameter_update'])
 def test_fixed_denominators(self):self.assertEqual([SETTINGS['training_states'],SETTINGS['inspected_development_states']],[76,16])
 def test_unchanged_gates(self):self.assertEqual([SETTINGS[k] for k in ('response_budget','kl_budget','margin_gate')],[.002,.005,.05])
 def test_half_probe_not_selection(self):self.assertEqual(SETTINGS['diagnostic_half_step'],.5);self.assertTrue(SETTINGS['half_step_is_not_selected_or_accepted'])
 def test_two_exact_frozen_arms(self):self.assertEqual(set(PINS),set(ARMS));self.assertTrue(all(len(v)==64 for p in PINS.values() for v in p.values()))
 def test_snapshot_nested_manifests(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/'s/run';out.mkdir(parents=True);dump(out/'identity.json',{'protocol':PROTOCOL});dump(out/'nested/MANIFEST.json',{'original':'kept'});p=snapshot(out)
   with zipfile.ZipFile(p) as z:self.assertIn('nested/MANIFEST.json',verify_archive(z))
 def test_only_own_last_three(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/'s/run';out.mkdir(parents=True);dd=Path(d)/'drive';dd.mkdir();other=dd/'C28.zip';other.write_bytes(b'untouched');dump(out/'identity.json',{'protocol':PROTOCOL})
   for i in range(4):dump(out/'state.json',{'i':i});snapshot(out,str(dd))
   self.assertEqual(len(list(dd.glob(RESULT+'_*.zip'))),3);self.assertEqual(other.read_bytes(),b'untouched')
 def test_strict_source_resume(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)/'src';root.mkdir();(root/'x').write_text('a');out=Path(d)/'s/run';out.mkdir(parents=True);dump(out/'identity.json',{'protocol':PROTOCOL,'settings':SETTINGS,'source_hashes':source_hashes(root)});p=snapshot(out);self.assertEqual(resume_score(p,root),(0,0,0));(root/'x').write_text('b')
   with self.assertRaises(ValueError):resume_score(p,root)
 def test_source_bytes_checked(self):self.assertGreater(len(check_inherited(Path(__file__).parent)),180)
 def test_wrong_parent_rejected(self):
  b=io.BytesIO()
  with zipfile.ZipFile(b,'w') as z:z.writestr('identity.json',json.dumps({'protocol':'WRONG'}));z.writestr('SUMMARY.json','{}')
  b.seek(0)
  with zipfile.ZipFile(b) as z:
   with self.assertRaises((KeyError,ValueError)):parent_probe(z,Path(__file__).parent)
 def test_incomplete_not_final(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)/'src';root.mkdir();out=Path(d)/'s/run';out.mkdir(parents=True);dump(out/'identity.json',{'protocol':PROTOCOL,'settings':SETTINGS,'source_hashes':source_hashes(root)});dump(out/'diagnosis/training/x/COMPLETE.json',{});p=snapshot(out);self.assertEqual(resume_score(p,root),(0,1,0))
if __name__=='__main__':unittest.main()
