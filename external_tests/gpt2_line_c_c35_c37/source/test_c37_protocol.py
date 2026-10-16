"""Root-only import, prospective centre specification and archive tests."""
import sys,unittest,tempfile,subprocess,ast,contextlib,io,zipfile
from pathlib import Path
from c37_paths import configure
ROOT=Path(__file__).resolve().parent;configure(ROOT)
from c37_protocol import *
class Tests(unittest.TestCase):
 def test_32_unique_seeds_and_ids(self):
  self.assertEqual(len(specs()),32);self.assertEqual(len({s['seed'] for s in specs()}),32);self.assertEqual(len({s['id'] for s in specs()}),32)
 def test_16_fixed_centres_each_two_children(self):
  self.assertEqual(len(centres()),16);self.assertTrue(all(sum(s['centre_id']==c['id'] for s in specs())==2 for c in centres()))
 def test_centres_are_preparations_and_previously_inspected(self):
  self.assertTrue(SETTINGS['centres_previously_inspected']);self.assertTrue(all('__c36_new' in c['id'] and int(c['id'][-2:])<4 for c in centres()))
 def test_uniform_routes(self):self.assertEqual([sum(s['case']==c for s in specs()) for c in CASES],[8]*4)
 def test_fresh_seeds(self):
  from c36_protocol import specs as old
  self.assertFalse({s['seed'] for s in specs()}&{s['seed'] for s in old()})
 def test_budgets(self):self.assertEqual([SETTINGS[k] for k in ('response_budget','KL_budget','margin_gate')],[.002,.005,.05])
 def test_root_only_import(self):
  r=subprocess.run([sys.executable,'-I','-c','import sys;sys.path.insert(0,sys.argv[1]);import launch_c37;import c30_core',str(ROOT)],capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr)
 def test_pinned_C36_sources(self):check_sources(ROOT)
 def test_no_training_in_real_worker(self):
  tree=ast.parse((ROOT/'worker_c37.py').read_text());calls={n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
  self.assertFalse(calls&{'fit','fit_all','select','train','backward'})
 def test_heavy_selftest_unlimited(self):
  tree=ast.parse((ROOT/'launch_c37.py').read_text());calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and any(isinstance(a,ast.Constant) and a.value=='C37 archive/tensor/seal/resume tests' for a in n.args)]
  self.assertEqual(len(calls),1);self.assertTrue(any(k.arg=='timeout' and isinstance(k.value,ast.Constant) and k.value.value is None for k in calls[0].keywords))
 def test_archive_tamper_and_path_rejected(self):
  for n in ('x','../x'):
   with tempfile.TemporaryDirectory() as d:
    p=Path(d)/'bad.zip'
    with zipfile.ZipFile(p,'w') as z:z.writestr(n,'x');z.writestr('MANIFEST.json',json.dumps({n:'0'*64}))
    with zipfile.ZipFile(p) as z:
     with self.assertRaises(ValueError):verify(z,True)
 def test_real_snapshot_and_last3_retention(self):
  with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
   out=Path(d)/'session/run';dump(out/'identity.json',{'protocol':PROTOCOL});drive=Path(d)/'drive'
   for i in range(4):dump(out/'count.json',{'i':i});p=snapshot(out,drive)
   with zipfile.ZipFile(p) as z:self.assertIn('count.json',verify(z,True))
   self.assertEqual(len(list(drive.glob('*.zip'))),3)
 def test_not_native_or_independent_claim(self):
  self.assertFalse(SETTINGS['internal_parameter_update']);self.assertFalse(SETTINGS['independent_model_or_chart_replication']);self.assertFalse(SETTINGS['fresh_evaluation'])
if __name__=='__main__':unittest.main()
