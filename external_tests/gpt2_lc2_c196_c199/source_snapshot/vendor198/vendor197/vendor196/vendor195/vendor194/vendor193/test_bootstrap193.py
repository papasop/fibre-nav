"""Isolated current entrypoint and no-model/no-solver import checks."""
from pathlib import Path
import sys,subprocess,ast,unittest
ROOT=Path(__file__).resolve().parent
class Checks(unittest.TestCase):
 def test_current_worker_import_without_model_or_solver(self):
  s="import sys;sys.path.insert(0,sys.argv[1]);import worker193;assert worker193.PROTOCOL=='GPT2_LC2_C193_READONLY_CONTINUATION_TRAIN_AUDIT_CPU_V1';assert not any(n in sys.modules for n in ('torch','transformers','peft','scipy'))";r=subprocess.run([sys.executable,'-I','-c',s,str(ROOT)],capture_output=True,text=True,timeout=90);self.assertEqual(r.returncode,0,r.stderr)
 def test_current_launcher_routes_current_scripts(self):
  s=(ROOT/'launch193.py').read_text();ast.parse(s)
  for name in ('test_replay193.py','test_focus193.py','worker193.py','test_math193.py','test_diagnostics193.py','test_outcomes193.py','test193.py','test_activity193.py','test_precision193.py','test_bootstrap193.py','test_launcher193.py'):self.assertIn(name,s)
  self.assertNotIn('torch_smoke',s);self.assertNotIn('pip install',s);self.assertNotIn('from runtime170',s)
 def test_current_worker_help_contains_all38_input_paths(self):
  s='import sys,runpy;sys.path.insert(0,sys.argv.pop(1));p=sys.argv.pop(1);sys.argv=[p,"--help"];runpy.run_path(p,run_name="__main__")';r=subprocess.run([sys.executable,'-I','-c',s,str(ROOT),str(ROOT/'worker193.py')],capture_output=True,text=True,timeout=90);self.assertEqual(r.returncode,0,r.stderr)
  for k in range(155,193):self.assertIn('--c'+str(k),r.stdout)
 def test_production_original_hashes_and_fixed_start_head(self):
  import json
  pins=json.loads((ROOT/'INPUT_PINS.json').read_text());self.assertEqual(set(pins),{'C'+str(i) for i in range(155,193)});self.assertEqual(pins['C192']['sha256'],'da563a3f767f74580ef8076dfccc6cb57a5e5eef3f7f8ab5a8b30f7bd23eba3d');self.assertEqual(pins['C191']['sha256'],'043b24b0656a20f23b0d489cbee38de40a2d30ceb19a2b6fcd43eedd811a3b75');self.assertIn("ANCHOR_SHA256='be0dbfe05156f4c791d609a8a7ce31b1a0f184f7748c710f64e46580552fb8ed'",(ROOT/'vendor192/core192.py').read_text())
if __name__=='__main__':unittest.main(verbosity=2)
