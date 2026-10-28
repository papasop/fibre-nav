"""Isolated current-entrypoint checks, no model or solver imports."""
from pathlib import Path
import sys,subprocess,ast,unittest
ROOT=Path(__file__).resolve().parent
class Checks(unittest.TestCase):
 def test_current_worker_imports_without_model_or_solver(self):
  script="import sys;sys.path.insert(0,sys.argv[1]);import worker192;assert worker192.PROTOCOL=='GPT2_LC2_C192_ALL_COORDINATE_CONTINUATION_TRAIN_CPU_V1';assert not any(n in sys.modules for n in ('torch','transformers','peft','scipy'))";r=subprocess.run([sys.executable,'-I','-c',script,str(ROOT)],capture_output=True,text=True,timeout=90);self.assertEqual(r.returncode,0,r.stderr)
 def test_current_launcher_scripts_and_fixed_dependency_versions(self):
  s=(ROOT/'launch192.py').read_text();ast.parse(s)
  for name in ('worker192.py','test192.py','test_endpoint192.py','test_binding192.py','torch_smoke192.py','test_bootstrap192.py','test_launcher192.py'):self.assertIn(name,s)
  self.assertIn('scipy==1.17.0',s);self.assertIn('transformers==4.56.2',s);self.assertNotIn('torch_smoke168.py',s)
 def test_current_worker_help_has_all37_originals(self):
  script='import sys,runpy;sys.path.insert(0,sys.argv.pop(1));p=sys.argv.pop(1);sys.argv=[p,"--help"];runpy.run_path(p,run_name="__main__")';r=subprocess.run([sys.executable,'-I','-c',script,str(ROOT),str(ROOT/'worker192.py')],capture_output=True,text=True,timeout=90);self.assertEqual(r.returncode,0,r.stderr)
  for i in range(155,192):self.assertIn('--c'+str(i),r.stdout)
 def test_production_anchor_and_original_C191_pin_are_exact(self):
  import json
  s=(ROOT/'core192.py').read_text();self.assertIn("ANCHOR_SHA256='be0dbfe05156f4c791d609a8a7ce31b1a0f184f7748c710f64e46580552fb8ed'",s)
  pins=json.loads((ROOT/'INPUT_PINS.json').read_text());self.assertEqual(set(pins),{'C'+str(i) for i in range(155,192)});self.assertEqual(pins['C191']['sha256'],'043b24b0656a20f23b0d489cbee38de40a2d30ceb19a2b6fcd43eedd811a3b75');self.assertEqual(pins['C190']['sha256'],'7774d439bb75dfc3a9372a74f2b05d51f5051f454e00ddae32524c12217bb4d7')
if __name__=='__main__':unittest.main(verbosity=2)
