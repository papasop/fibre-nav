"""Isolated current-entrypoint checks, no model or solver imports."""
from pathlib import Path
import sys,subprocess,ast,unittest
ROOT=Path(__file__).resolve().parent
class Checks(unittest.TestCase):
 def test_current_worker_imports_without_model_or_solver(self):
  script="import sys;sys.path.insert(0,sys.argv[1]);import worker186;assert worker186.PROTOCOL=='GPT2_LC2_C186_ALL_COORDINATE_CONTINUATION_TRAIN_CPU_V1';assert not any(n in sys.modules for n in ('torch','transformers','peft','scipy'))";r=subprocess.run([sys.executable,'-I','-c',script,str(ROOT)],capture_output=True,text=True,timeout=90);self.assertEqual(r.returncode,0,r.stderr)
 def test_current_launcher_scripts_and_fixed_dependency_versions(self):
  s=(ROOT/'launch186.py').read_text();ast.parse(s)
  for name in ('worker186.py','test186.py','test_endpoint186.py','test_binding186.py','torch_smoke186.py','test_bootstrap186.py','test_launcher186.py'):self.assertIn(name,s)
  self.assertIn('scipy==1.17.0',s);self.assertIn('transformers==4.56.2',s);self.assertNotIn('torch_smoke168.py',s)
 def test_current_worker_help_has_all31_originals(self):
  script='import sys,runpy;sys.path.insert(0,sys.argv.pop(1));p=sys.argv.pop(1);sys.argv=[p,"--help"];runpy.run_path(p,run_name="__main__")';r=subprocess.run([sys.executable,'-I','-c',script,str(ROOT),str(ROOT/'worker186.py')],capture_output=True,text=True,timeout=90);self.assertEqual(r.returncode,0,r.stderr)
  for i in range(155,186):self.assertIn('--c'+str(i),r.stdout)
 def test_production_anchor_and_original_C185_pin_are_exact(self):
  import json
  s=(ROOT/'core186.py').read_text();self.assertIn("ANCHOR_SHA256='aca4e6546dbeaf2eaa8e1f2f3a9c098aa9cebacf209fe3993fee0bf8e1c20d74'",s)
  pins=json.loads((ROOT/'INPUT_PINS.json').read_text());self.assertEqual(set(pins),{'C'+str(i) for i in range(155,186)});self.assertEqual(pins['C185']['sha256'],'036dbe198c9fd0c92a63736c6e211f8c83efd231fdefca576675e24d393eedcd');self.assertEqual(pins['C184']['sha256'],'ecbb043714f7ac6ea7a7d014182a0a677348eed2dbcb5b4aeb7a914e2539ef94')
if __name__=='__main__':unittest.main(verbosity=2)
