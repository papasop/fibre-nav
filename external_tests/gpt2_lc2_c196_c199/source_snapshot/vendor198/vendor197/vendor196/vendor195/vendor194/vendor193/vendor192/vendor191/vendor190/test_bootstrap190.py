"""Isolated current-entrypoint checks, no model or solver imports."""
from pathlib import Path
import sys,subprocess,ast,unittest
ROOT=Path(__file__).resolve().parent
class Checks(unittest.TestCase):
 def test_current_worker_imports_without_model_or_solver(self):
  script="import sys;sys.path.insert(0,sys.argv[1]);import worker190;assert worker190.PROTOCOL=='GPT2_LC2_C190_ALL_COORDINATE_CONTINUATION_TRAIN_CPU_V1';assert not any(n in sys.modules for n in ('torch','transformers','peft','scipy'))";r=subprocess.run([sys.executable,'-I','-c',script,str(ROOT)],capture_output=True,text=True,timeout=90);self.assertEqual(r.returncode,0,r.stderr)
 def test_current_launcher_scripts_and_fixed_dependency_versions(self):
  s=(ROOT/'launch190.py').read_text();ast.parse(s)
  for name in ('worker190.py','test190.py','test_endpoint190.py','test_binding190.py','torch_smoke190.py','test_bootstrap190.py','test_launcher190.py'):self.assertIn(name,s)
  self.assertIn('scipy==1.17.0',s);self.assertIn('transformers==4.56.2',s);self.assertNotIn('torch_smoke168.py',s)
 def test_current_worker_help_has_all35_originals(self):
  script='import sys,runpy;sys.path.insert(0,sys.argv.pop(1));p=sys.argv.pop(1);sys.argv=[p,"--help"];runpy.run_path(p,run_name="__main__")';r=subprocess.run([sys.executable,'-I','-c',script,str(ROOT),str(ROOT/'worker190.py')],capture_output=True,text=True,timeout=90);self.assertEqual(r.returncode,0,r.stderr)
  for i in range(155,190):self.assertIn('--c'+str(i),r.stdout)
 def test_production_anchor_and_original_C189_pin_are_exact(self):
  import json
  s=(ROOT/'core190.py').read_text();self.assertIn("ANCHOR_SHA256='2bfa9158e2ea1e2175c948722f997036a2aa03446a075555989f96f8a71d7813'",s)
  pins=json.loads((ROOT/'INPUT_PINS.json').read_text());self.assertEqual(set(pins),{'C'+str(i) for i in range(155,190)});self.assertEqual(pins['C189']['sha256'],'16570ba6517fd7276d1b7d0944567cbeb015678b08cd2f9c7fd8474f83637480');self.assertEqual(pins['C188']['sha256'],'78ab7f82f6961482fa1895f03f6143e0b6e204a23ef21db56b38adce9f979f65')
if __name__=='__main__':unittest.main(verbosity=2)
