"""Isolated current entrypoint and no-model/no-solver import checks."""
from pathlib import Path
import sys,subprocess,ast,unittest
ROOT=Path(__file__).resolve().parent
class Checks(unittest.TestCase):
 def test_current_worker_import_without_model_or_solver(self):
  s="import sys;sys.path.insert(0,sys.argv[1]);import worker197;assert worker197.PROTOCOL=='GPT2_LC2_C197_READONLY_CONTINUATION_TRAIN_AUDIT_CPU_V1';assert not any(n in sys.modules for n in ('torch','transformers','peft','scipy'))";r=subprocess.run([sys.executable,'-I','-c',s,str(ROOT)],capture_output=True,text=True,timeout=90);self.assertEqual(r.returncode,0,r.stderr)
 def test_current_launcher_routes_current_scripts(self):
  s=(ROOT/'launch197.py').read_text();ast.parse(s)
  for name in ('test_replay197.py','test_focus197.py','worker197.py','test_math197.py','test_diagnostics197.py','test_outcomes197.py','test197.py','test_activity197.py','test_precision197.py','test_bootstrap197.py','test_launcher197.py'):self.assertIn(name,s)
  self.assertNotIn('torch_smoke',s);self.assertNotIn('pip install',s);self.assertNotIn('from runtime170',s)
 def test_current_worker_help_contains_all42_input_paths(self):
  s='import sys,runpy;sys.path.insert(0,sys.argv.pop(1));p=sys.argv.pop(1);sys.argv=[p,"--help"];runpy.run_path(p,run_name="__main__")';r=subprocess.run([sys.executable,'-I','-c',s,str(ROOT),str(ROOT/'worker197.py')],capture_output=True,text=True,timeout=90);self.assertEqual(r.returncode,0,r.stderr)
  for k in range(155,197):self.assertIn('--c'+str(k),r.stdout)
 def test_production_original_hashes_and_fixed_start_head(self):
  import json
  pins=json.loads((ROOT/'INPUT_PINS.json').read_text());self.assertEqual(set(pins),{'C'+str(i) for i in range(155,197)});self.assertEqual(pins['C196']['sha256'],'2b96c9ac30e05d7bc65ac2b30c97de1798a943da219fcbc0e6bed58ce92400df');self.assertEqual(pins['C195']['sha256'],'8390f73598c0c9446c0ee7ddcd63298470e539e552d20ae44cec90cc302d044b');self.assertIn("ANCHOR_SHA256='2ff29f06c399a501de74387f9f3caebadecb763c89d9c6ef9db8a84bef478aaf'",(ROOT/'vendor196/core196.py').read_text())
if __name__=='__main__':unittest.main(verbosity=2)
