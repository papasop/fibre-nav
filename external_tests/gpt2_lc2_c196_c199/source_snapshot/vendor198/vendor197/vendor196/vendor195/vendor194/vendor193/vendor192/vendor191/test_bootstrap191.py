"""Isolated current entrypoint and no-model/no-solver import checks."""
from pathlib import Path
import sys,subprocess,ast,unittest
ROOT=Path(__file__).resolve().parent
class Checks(unittest.TestCase):
 def test_current_worker_import_without_model_or_solver(self):
  s="import sys;sys.path.insert(0,sys.argv[1]);import worker191;assert worker191.PROTOCOL=='GPT2_LC2_C191_READONLY_CONTINUATION_TRAIN_AUDIT_CPU_V1';assert not any(n in sys.modules for n in ('torch','transformers','peft','scipy'))";r=subprocess.run([sys.executable,'-I','-c',s,str(ROOT)],capture_output=True,text=True,timeout=90);self.assertEqual(r.returncode,0,r.stderr)
 def test_current_launcher_routes_current_scripts(self):
  s=(ROOT/'launch191.py').read_text();ast.parse(s)
  for name in ('test_replay191.py','test_focus191.py','worker191.py','test_math191.py','test_diagnostics191.py','test_outcomes191.py','test191.py','test_activity191.py','test_precision191.py','test_bootstrap191.py','test_launcher191.py'):self.assertIn(name,s)
  self.assertNotIn('torch_smoke',s);self.assertNotIn('pip install',s);self.assertNotIn('from runtime170',s)
 def test_current_worker_help_contains_all36_input_paths(self):
  s='import sys,runpy;sys.path.insert(0,sys.argv.pop(1));p=sys.argv.pop(1);sys.argv=[p,"--help"];runpy.run_path(p,run_name="__main__")';r=subprocess.run([sys.executable,'-I','-c',s,str(ROOT),str(ROOT/'worker191.py')],capture_output=True,text=True,timeout=90);self.assertEqual(r.returncode,0,r.stderr)
  for k in range(155,191):self.assertIn('--c'+str(k),r.stdout)
 def test_production_original_hashes_and_fixed_start_head(self):
  import json
  pins=json.loads((ROOT/'INPUT_PINS.json').read_text());self.assertEqual(set(pins),{'C'+str(i) for i in range(155,191)});self.assertEqual(pins['C190']['sha256'],'7774d439bb75dfc3a9372a74f2b05d51f5051f454e00ddae32524c12217bb4d7');self.assertEqual(pins['C189']['sha256'],'16570ba6517fd7276d1b7d0944567cbeb015678b08cd2f9c7fd8474f83637480');self.assertIn("ANCHOR_SHA256='2bfa9158e2ea1e2175c948722f997036a2aa03446a075555989f96f8a71d7813'",(ROOT/'vendor190/core190.py').read_text())
if __name__=='__main__':unittest.main(verbosity=2)
