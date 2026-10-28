"""Isolated current-entrypoint checks, no model or solver imports."""
from pathlib import Path
import sys,subprocess,ast,unittest
ROOT=Path(__file__).resolve().parent
class Checks(unittest.TestCase):
 def test_current_worker_imports_without_model_or_solver(self):
  script="import sys;sys.path.insert(0,sys.argv[1]);import worker188;assert worker188.PROTOCOL=='GPT2_LC2_C188_ALL_COORDINATE_CONTINUATION_TRAIN_CPU_V1';assert not any(n in sys.modules for n in ('torch','transformers','peft','scipy'))";r=subprocess.run([sys.executable,'-I','-c',script,str(ROOT)],capture_output=True,text=True,timeout=90);self.assertEqual(r.returncode,0,r.stderr)
 def test_current_launcher_scripts_and_fixed_dependency_versions(self):
  s=(ROOT/'launch188.py').read_text();ast.parse(s)
  for name in ('worker188.py','test188.py','test_endpoint188.py','test_binding188.py','torch_smoke188.py','test_bootstrap188.py','test_launcher188.py'):self.assertIn(name,s)
  self.assertIn('scipy==1.17.0',s);self.assertIn('transformers==4.56.2',s);self.assertNotIn('torch_smoke168.py',s)
 def test_current_worker_help_has_all33_originals(self):
  script='import sys,runpy;sys.path.insert(0,sys.argv.pop(1));p=sys.argv.pop(1);sys.argv=[p,"--help"];runpy.run_path(p,run_name="__main__")';r=subprocess.run([sys.executable,'-I','-c',script,str(ROOT),str(ROOT/'worker188.py')],capture_output=True,text=True,timeout=90);self.assertEqual(r.returncode,0,r.stderr)
  for i in range(155,188):self.assertIn('--c'+str(i),r.stdout)
 def test_production_anchor_and_original_C187_pin_are_exact(self):
  import json
  s=(ROOT/'core188.py').read_text();self.assertIn("ANCHOR_SHA256='005c3da405a9dff4f1168009b7857e5e8eb2247810647020ea49417d1e432c06'",s)
  pins=json.loads((ROOT/'INPUT_PINS.json').read_text());self.assertEqual(set(pins),{'C'+str(i) for i in range(155,188)});self.assertEqual(pins['C187']['sha256'],'70ad0f3e235f92b5f542ebdcbf79b945f6e963f39f8764386c2042c661f82d37');self.assertEqual(pins['C186']['sha256'],'e1114eae903a7445d1b37dc83fe5602c30f7edfed01b717419b7b74cb01ecb80')
if __name__=='__main__':unittest.main(verbosity=2)
