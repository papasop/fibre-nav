"""Isolated current-entrypoint checks, no model or solver imports."""
from pathlib import Path
import sys,subprocess,ast,unittest
ROOT=Path(__file__).resolve().parent
class Checks(unittest.TestCase):
 def test_current_worker_imports_without_model_or_solver(self):
  script="import sys;sys.path.insert(0,sys.argv[1]);import worker184;assert worker184.PROTOCOL=='GPT2_LC2_C184_ALL_COORDINATE_CONTINUATION_TRAIN_CPU_V1';assert not any(n in sys.modules for n in ('torch','transformers','peft','scipy'))";r=subprocess.run([sys.executable,'-I','-c',script,str(ROOT)],capture_output=True,text=True,timeout=90);self.assertEqual(r.returncode,0,r.stderr)
 def test_current_launcher_scripts_and_fixed_dependency_versions(self):
  s=(ROOT/'launch184.py').read_text();ast.parse(s)
  for name in ('worker184.py','test184.py','test_endpoint184.py','test_binding184.py','torch_smoke184.py','test_bootstrap184.py','test_launcher184.py'):self.assertIn(name,s)
  self.assertIn('scipy==1.17.0',s);self.assertIn('transformers==4.56.2',s);self.assertNotIn('torch_smoke168.py',s)
 def test_current_worker_help_has_all29_originals(self):
  script='import sys,runpy;sys.path.insert(0,sys.argv.pop(1));p=sys.argv.pop(1);sys.argv=[p,"--help"];runpy.run_path(p,run_name="__main__")';r=subprocess.run([sys.executable,'-I','-c',script,str(ROOT),str(ROOT/'worker184.py')],capture_output=True,text=True,timeout=90);self.assertEqual(r.returncode,0,r.stderr)
  for i in range(155,184):self.assertIn('--c'+str(i),r.stdout)
 def test_production_anchor_and_original_C183_pin_are_exact(self):
  import json
  s=(ROOT/'core184.py').read_text();self.assertIn("ANCHOR_SHA256='ed0b980e8c09f4940aec8f84e8cf7edc3136923a1aec675b9cfa0af4cd56b50c'",s)
  pins=json.loads((ROOT/'INPUT_PINS.json').read_text());self.assertEqual(set(pins),{'C'+str(i) for i in range(155,184)});self.assertEqual(pins['C183']['sha256'],'d85356de305b8ac7c4b5a3fac445abc7585ed5ac8c2b88071e04a03f31c313b9');self.assertEqual(pins['C182']['sha256'],'8c00c1e048433fca7c813d57be8ed8efae133ed7dfb80b7d2f27cdea81fa63fc')
if __name__=='__main__':unittest.main(verbosity=2)
