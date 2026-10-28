"""Isolated current entrypoint and no-model/no-solver import checks."""
from pathlib import Path
import sys,subprocess,ast,unittest
ROOT=Path(__file__).resolve().parent
class Checks(unittest.TestCase):
 def test_current_worker_import_without_model_or_solver(self):
  s="import sys;sys.path.insert(0,sys.argv[1]);import worker187;assert worker187.PROTOCOL=='GPT2_LC2_C187_READONLY_CONTINUATION_TRAIN_AUDIT_CPU_V1';assert not any(n in sys.modules for n in ('torch','transformers','peft','scipy'))";r=subprocess.run([sys.executable,'-I','-c',s,str(ROOT)],capture_output=True,text=True,timeout=90);self.assertEqual(r.returncode,0,r.stderr)
 def test_current_launcher_routes_current_scripts(self):
  s=(ROOT/'launch187.py').read_text();ast.parse(s)
  for name in ('test_replay187.py','test_focus187.py','worker187.py','test_math187.py','test_diagnostics187.py','test_outcomes187.py','test187.py','test_activity187.py','test_precision187.py','test_bootstrap187.py','test_launcher187.py'):self.assertIn(name,s)
  self.assertNotIn('torch_smoke',s);self.assertNotIn('pip install',s);self.assertNotIn('from runtime170',s)
 def test_current_worker_help_contains_all32_input_paths(self):
  s='import sys,runpy;sys.path.insert(0,sys.argv.pop(1));p=sys.argv.pop(1);sys.argv=[p,"--help"];runpy.run_path(p,run_name="__main__")';r=subprocess.run([sys.executable,'-I','-c',s,str(ROOT),str(ROOT/'worker187.py')],capture_output=True,text=True,timeout=90);self.assertEqual(r.returncode,0,r.stderr)
  for k in range(155,187):self.assertIn('--c'+str(k),r.stdout)
 def test_production_original_hashes_and_fixed_start_head(self):
  import json
  pins=json.loads((ROOT/'INPUT_PINS.json').read_text());self.assertEqual(set(pins),{'C'+str(i) for i in range(155,187)});self.assertEqual(pins['C186']['sha256'],'e1114eae903a7445d1b37dc83fe5602c30f7edfed01b717419b7b74cb01ecb80');self.assertEqual(pins['C185']['sha256'],'036dbe198c9fd0c92a63736c6e211f8c83efd231fdefca576675e24d393eedcd');self.assertIn("ANCHOR_SHA256='aca4e6546dbeaf2eaa8e1f2f3a9c098aa9cebacf209fe3993fee0bf8e1c20d74'",(ROOT/'vendor186/core186.py').read_text())
if __name__=='__main__':unittest.main(verbosity=2)
