"""Negative controls for the offline record reviewer; no torch/model download."""
import copy,json,unittest
from pathlib import Path
from verify_evidence import verify_records
ROOT=Path(__file__).parent
class Tests(unittest.TestCase):
 def setUp(self):
  load=lambda folder,name:json.loads((ROOT/folder/(name+'.json')).read_text())
  self.args=[load('results','logic_rows'),load('results','replay_rows'),load('results','stage1_rows'),load('results','summary'),load('results','audits'),load('source','protection_cases'),load('results','protocol')]
 def test_recorded_success(self):self.assertEqual(verify_records(*self.args)['known_raw_success'],4)
 def test_missing_case(self):
  self.args[0].pop()
  with self.assertRaises(ValueError):verify_records(*self.args)
 def test_duplicate_case(self):
  self.args[0][-1]=copy.deepcopy(self.args[0][0])
  with self.assertRaises(ValueError):verify_records(*self.args)
 def test_external_prompt_correction(self):
  self.args[0][0]['prompt']='corrected value supplied by evaluator'
  with self.assertRaises(ValueError):verify_records(*self.args)
 def test_wrong_answer_score(self):
  r=next(r for r in self.args[0] if r['id']=='10-raw-xor');r['raw']='register_00: 1\nregister_01: 0\nRESULT: 0'
  with self.assertRaises(ValueError):verify_records(*self.args)
 def test_changed_stage1(self):
  self.args[2]['10']['raw']='register_00: 0\nregister_01: 0'
  with self.assertRaises(ValueError):verify_records(*self.args)
 def test_budget_relaxation(self):
  for a in self.args[4].values():a['on']['10']['response']=.021
  with self.assertRaises(ValueError):verify_records(*self.args)
 def test_overclaim(self):
  self.args[3]['formal_l6_pass']=True
  with self.assertRaises(ValueError):verify_records(*self.args)
if __name__=='__main__':unittest.main()
