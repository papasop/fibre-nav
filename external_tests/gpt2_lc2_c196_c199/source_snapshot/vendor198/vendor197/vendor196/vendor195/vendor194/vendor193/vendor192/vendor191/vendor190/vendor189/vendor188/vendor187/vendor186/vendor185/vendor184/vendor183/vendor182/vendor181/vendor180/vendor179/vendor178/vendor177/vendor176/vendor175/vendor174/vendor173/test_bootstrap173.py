"""Isolated current entrypoint and no-model/no-solver import checks."""
from pathlib import Path
import sys,subprocess,ast,unittest
ROOT=Path(__file__).resolve().parent
class Checks(unittest.TestCase):
 def test_current_worker_import_without_model_or_solver(self):
  s="import sys;sys.path.insert(0,sys.argv[1]);import worker173;assert worker173.PROTOCOL=='GPT2_LC2_C173_READONLY_FINITE_STEP_COORDINATE_AUDIT_CPU_V1';assert not any(n in sys.modules for n in ('torch','transformers','peft','scipy'))";r=subprocess.run([sys.executable,'-I','-c',s,str(ROOT)],capture_output=True,text=True,timeout=90);self.assertEqual(r.returncode,0,r.stderr)
 def test_current_launcher_routes_current_scripts(self):
  s=(ROOT/'launch173.py').read_text();ast.parse(s)
  for name in ('worker173.py','test173.py','test_bootstrap173.py','test_launcher173.py'):self.assertIn(name,s)
  self.assertNotIn('torch_smoke',s);self.assertNotIn('pip install',s);self.assertNotIn('from runtime170',s)
 def test_current_worker_help_contains_all18_input_paths(self):
  s='import sys,runpy;sys.path.insert(0,sys.argv.pop(1));p=sys.argv.pop(1);sys.argv=[p,"--help"];runpy.run_path(p,run_name="__main__")';r=subprocess.run([sys.executable,'-I','-c',s,str(ROOT),str(ROOT/'worker173.py')],capture_output=True,text=True,timeout=90);self.assertEqual(r.returncode,0,r.stderr)
  for k in range(155,173):self.assertIn('--c'+str(k),r.stdout)
if __name__=='__main__':unittest.main(verbosity=2)
