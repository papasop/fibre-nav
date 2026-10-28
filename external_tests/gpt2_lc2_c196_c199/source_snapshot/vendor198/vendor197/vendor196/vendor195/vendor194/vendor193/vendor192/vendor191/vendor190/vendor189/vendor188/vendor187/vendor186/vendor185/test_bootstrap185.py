"""Isolated current entrypoint and no-model/no-solver import checks."""
from pathlib import Path
import sys,subprocess,ast,unittest
ROOT=Path(__file__).resolve().parent
class Checks(unittest.TestCase):
 def test_current_worker_import_without_model_or_solver(self):
  s="import sys;sys.path.insert(0,sys.argv[1]);import worker185;assert worker185.PROTOCOL=='GPT2_LC2_C185_READONLY_CONTINUATION_TRAIN_AUDIT_CPU_V1';assert not any(n in sys.modules for n in ('torch','transformers','peft','scipy'))";r=subprocess.run([sys.executable,'-I','-c',s,str(ROOT)],capture_output=True,text=True,timeout=90);self.assertEqual(r.returncode,0,r.stderr)
 def test_current_launcher_routes_current_scripts(self):
  s=(ROOT/'launch185.py').read_text();ast.parse(s)
  for name in ('test_focus185.py','worker185.py','test_math185.py','test_diagnostics185.py','test_outcomes185.py','test185.py','test_activity185.py','test_precision185.py','test_bootstrap185.py','test_launcher185.py'):self.assertIn(name,s)
  self.assertNotIn('torch_smoke',s);self.assertNotIn('pip install',s);self.assertNotIn('from runtime170',s)
 def test_current_worker_help_contains_all30_input_paths(self):
  s='import sys,runpy;sys.path.insert(0,sys.argv.pop(1));p=sys.argv.pop(1);sys.argv=[p,"--help"];runpy.run_path(p,run_name="__main__")';r=subprocess.run([sys.executable,'-I','-c',s,str(ROOT),str(ROOT/'worker185.py')],capture_output=True,text=True,timeout=90);self.assertEqual(r.returncode,0,r.stderr)
  for k in range(155,185):self.assertIn('--c'+str(k),r.stdout)
 def test_production_original_hashes_and_fixed_start_head(self):
  import json
  pins=json.loads((ROOT/'INPUT_PINS.json').read_text());self.assertEqual(set(pins),{'C'+str(i) for i in range(155,185)});self.assertEqual(pins['C184']['sha256'],'ecbb043714f7ac6ea7a7d014182a0a677348eed2dbcb5b4aeb7a914e2539ef94');self.assertEqual(pins['C183']['sha256'],'d85356de305b8ac7c4b5a3fac445abc7585ed5ac8c2b88071e04a03f31c313b9');self.assertIn("ANCHOR_SHA256='ed0b980e8c09f4940aec8f84e8cf7edc3136923a1aec675b9cfa0af4cd56b50c'",(ROOT/'vendor184/core184.py').read_text())
if __name__=='__main__':unittest.main(verbosity=2)
