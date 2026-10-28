"""Isolated current-entrypoint checks, no model or solver imports."""
from pathlib import Path
import sys,subprocess,ast,unittest
ROOT=Path(__file__).resolve().parent
class Checks(unittest.TestCase):
 def test_current_worker_imports_without_model_or_solver(self):
  script="import sys;sys.path.insert(0,sys.argv[1]);import worker176;assert worker176.PROTOCOL=='GPT2_LC2_C176_ALL_COORDINATE_RELINEARISED_TRAIN_CPU_V1';assert not any(n in sys.modules for n in ('torch','transformers','peft','scipy'))";r=subprocess.run([sys.executable,'-I','-c',script,str(ROOT)],capture_output=True,text=True,timeout=90);self.assertEqual(r.returncode,0,r.stderr)
 def test_current_launcher_scripts_and_fixed_dependency_versions(self):
  s=(ROOT/'launch176.py').read_text();ast.parse(s)
  for name in ('worker176.py','test176.py','test_binding176.py','torch_smoke176.py','test_bootstrap176.py','test_launcher176.py'):self.assertIn(name,s)
  self.assertIn('scipy==1.17.0',s);self.assertIn('transformers==4.56.2',s);self.assertNotIn('torch_smoke168.py',s)
 def test_current_worker_help_has_all21_originals(self):
  script='import sys,runpy;sys.path.insert(0,sys.argv.pop(1));p=sys.argv.pop(1);sys.argv=[p,"--help"];runpy.run_path(p,run_name="__main__")';r=subprocess.run([sys.executable,'-I','-c',script,str(ROOT),str(ROOT/'worker176.py')],capture_output=True,text=True,timeout=90);self.assertEqual(r.returncode,0,r.stderr)
  for i in range(155,176):self.assertIn('--c'+str(i),r.stdout)
if __name__=='__main__':unittest.main(verbosity=2)
