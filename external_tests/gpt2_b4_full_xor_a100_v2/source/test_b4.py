import unittest
from protocol import *
import ast,types
from pathlib import Path
node=next(n for n in ast.parse((Path(__file__).parent/"run_l5_audit.py").read_text()).body if isinstance(n,ast.FunctionDef) and n.name=="prompts")
namespace={}
exec(compile(ast.Module(body=[node],type_ignores=[]),"prompts","exec"),namespace)
audit=types.SimpleNamespace(prompts=namespace["prompts"])
from archive_io import restore_archive
from pathlib import Path
import tempfile,zipfile,hashlib,json
class ProtocolTests(unittest.TestCase):
 def test_four_cases_and_seed_separation(self):
  self.assertEqual(CASES,[(0,0),(0,1),(1,0),(1,1)])
  self.assertEqual(len(set(SEEDS)),3);self.assertEqual(SEEDS[0],DEVELOPMENT_SEED)
 def test_new_read_family_disjoint(self):
  cells,exact,train,dev=audit.prompts()
  self.assertFalse(set(fresh_prompts(cells))&set(exact+train+dev))
  self.assertEqual(len(fresh_prompts(cells)),16)
 def test_controller_actual_reads_and_override(self):
  for a,b in CASES:
   self.assertEqual(controller(lambda i:(a,b)[i])['value'],a^b)
   def forbidden(i):raise AssertionError('Blocked reader was called')
   self.assertEqual(controller(forbidden,True,(1^a,b))['value'],1^(a^b))
  with self.assertRaises(ValueError):controller(lambda i:0,True)
 def test_flip_and_control_gates(self):
  for a,b in CASES:
   x=a^b;stages={k:{'observed_bits':v,'pass':True} for k,v in {
    'prepare':[a,b,1,1^x],'normal_writeback':[a,b,1,x],
    'blocked_writeback':[a,b,1,1^x],'input_substitution':[1^a,b,1,x],
    'substituted_writeback':[1^a,b,1,1^x]}.items()}
   traces={'normal':controller(lambda i:(a,b)[i]),'blocked':controller(None,True,(1^a,b)), 'substituted':controller(lambda i:(1^a,b)[i])}
   reloads={k:True for k in stages}
   self.assertTrue(all(case_checks(a,b,stages,traces,[0,0,0,0],True,reloads).values()))
   stages['substituted_writeback']['observed_bits']=stages['input_substitution']['observed_bits'][:]
   self.assertFalse(case_checks(a,b,stages,traces,[0,0,0,0],True,reloads)['substituted_required_output_flip'])
 def test_gate_margins_and_budgets(self):
  cfg={'response_budget':.002,'endpoint_kl_gate':.005}
  ev={'exact_accuracy':1,'heldout_accuracy':1,'minimum_exact_margin':.05,'minimum_heldout_margin':.05}
  self.assertTrue(endpoint_pass(ev,.002,.005,cfg))
  self.assertFalse(endpoint_pass(ev,.0021,.005,cfg))
 def test_resume_actual_progress_beats_stale_completed_steps(self):
  from archive_io import archive_progress
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);p=root/'seeds'/'82931';p.mkdir(parents=True)
   (p/'records.json').write_text(json.dumps({'00_prepare':{}}))
   (root/'progress.json').write_text(json.dumps({'seed':82931,'phase':'00_prepare','steps':600}))
   previous=archive_progress(root)
   (root/'progress.json').write_text(json.dumps({'seed':82931,'phase':'00_normal_writeback','steps':5}))
   self.assertGreater(archive_progress(root),previous)
 def test_archive_restore_and_corruption(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'run.zip';data={'identity.json':json.dumps({'protocol':PROTOCOL}).encode(),'progress.json':b'{}'}
   with zipfile.ZipFile(p,'w') as z:
    for n,b in data.items():z.writestr(n,b)
    z.writestr('MANIFEST.json',json.dumps({n:hashlib.sha256(b).hexdigest() for n,b in data.items()}))
   restore_archive(p,Path(td)/'restored')
   with zipfile.ZipFile(p,'w') as z:
    z.writestr('identity.json',data['identity.json']);z.writestr('progress.json',b'bad')
    z.writestr('MANIFEST.json',json.dumps({n:hashlib.sha256(b).hexdigest() for n,b in data.items()}))
   with self.assertRaises(ValueError):restore_archive(p,Path(td)/'bad')
if __name__=='__main__':unittest.main(verbosity=2)
