import unittest,json,ast
from pathlib import Path
from two_stage_protocol import cases,parse_read,intermediate,compute_prompt,bind,score,summarize
ROOT=Path(__file__).parent

def source(s,raw=None):
 values='11' if s=='zero' else s
 return dict(id=s+'-joint-0',snapshot=s,arm='on',phase='before',raw=raw if raw is not None else f'register_00: {values[0]}\nregister_01: {values[1]}',truncated=False)

class Tests(unittest.TestCase):
 def test_plan(self):
  self.assertEqual(len(cases()),30);self.assertEqual(len({c['id'] for c in cases()}),30)
 def test_main_raw_not_reconstructed(self):
  raw='  register_00: 1\r\nregister_01: 0\n'
  self.assertEqual(intermediate(raw,'raw'),raw)
  self.assertTrue(compute_prompt(raw,'xor','raw').startswith(raw+'\n'))
  bad='wrong unparseable output'
  self.assertTrue(compute_prompt(bad,'xor','raw').startswith(bad+'\n'))
  self.assertIsNone(compute_prompt(bad,'xor','flip0'))
 def test_flip_only_named_character(self):
  raw=' register_00: 1\r\nregister_01: 0\n'
  self.assertEqual(intermediate(raw,'flip0'),' register_00: 0\r\nregister_01: 0\n')
  self.assertEqual(intermediate(raw,'flip1'),' register_00: 1\r\nregister_01: 1\n')
  self.assertEqual(intermediate(raw,'flip_both'),' register_00: 0\r\nregister_01: 1\n')
 def test_no_intermediate_independent_of_source(self):
  self.assertEqual(compute_prompt('anything','xor','no_intermediate'),compute_prompt('else','xor','no_intermediate'))
 def test_wrong_read_not_repaired_by_answer(self):
  c=next(c for c in cases() if c['id']=='10-raw-xor');b=bind(c,source('10','register_00: 0\nregister_01: 1'))
  s=score(b,'register_00: 1\nregister_01: 0\nRESULT: 1')
  self.assertFalse(s['end_to_end_success']);self.assertFalse(s['source_read_correct'])
 def test_zero_has_no_memory_credit(self):
  c=bind(cases()[0],source('zero'));s=score(c,'register_00: 1\nregister_01: 1\nRESULT: 0')
  self.assertIsNone(s['end_to_end_success']);self.assertTrue(s['follows_intermediate'])
 def test_scores_provenance_and_controls(self):
  rr=[];re=[source(s) for s in ('zero','10','11')]
  for c in cases():
   b=bind(c,next(x for x in re if x['snapshot']==c['snapshot']));v=b['supplied_values'] or '01';a=b['expected_from_intermediate'] or '0'
   raw=f'register_00: {v[0]}\nregister_01: {v[1]}\nRESULT: {a}'
   rr.append(dict(**b,raw=raw,output_ids=[],truncated=False,skipped=False,**score(b,raw)))
  s=summarize(rr,re);self.assertTrue(s['known_raw_four_pass']);self.assertTrue(s['known_single_flip_eight_pass']);self.assertTrue(s['known_double_flip_four_pass']);self.assertFalse(s['formal_l6_pass'])
  with self.assertRaises(ValueError):summarize(rr+[rr[0]],re)
  rr[0]['prompt']='injected corrected value'
  with self.assertRaises(ValueError):summarize(rr,re)
 def test_truncated_source_and_no_format_repair(self):
  c=next(c for c in cases() if c['id']=='10-raw-xor');s=source('10');s['truncated']=True;b=bind(c,s)
  self.assertFalse(score(b,'register_00: 1\nregister_01: 0\nRESULT: 1')['end_to_end_success'])
  b=bind(c,source('10'));self.assertFalse(score(b,'register_00:1\nregister_01:0\nRESULT:1')['format_pass'])
 def test_frozen_code_and_source_hashes(self):
  from snapshot_io import verify_files
  verify_files(ROOT)
  tree=ast.parse((ROOT/'run_cpu_audit.py').read_text());names=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
  for n in ('backward','step','add_'):self.assertNotIn(n,names)
if __name__=='__main__':unittest.main()
