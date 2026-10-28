"""Isolated entrypoint/import checks; do not load model packages."""
from pathlib import Path
import subprocess,sys,unittest,ast
ROOT=Path(__file__).resolve().parent
class Checks(unittest.TestCase):
 def isolated(self,script):
  return subprocess.run([sys.executable,'-I','-c','import sys;sys.path.insert(0,sys.argv[1]);'+script,str(ROOT)],capture_output=True,text=True,timeout=90)
 def test_current_worker_imports_without_model_or_solver(self):
  r=self.isolated("import worker169;assert not any(n in sys.modules for n in ('torch','transformers','peft','huggingface_hub','scipy'));assert worker169.PROTOCOL=='GPT2_LC2_C169_READONLY_FINITE_PROBE_AUDIT_CPU_V1'");self.assertEqual(r.returncode,0,r.stderr)
 def test_actual_launcher_points_to_current_scripts(self):
  s=(ROOT/'launch169.py').read_text();ast.parse(s)
  for name in ('worker169.py','test169.py','test_bootstrap169.py','test_launcher169.py'):self.assertIn(name,s)
  self.assertNotIn('torch_smoke',s);self.assertNotIn('pip install',s);self.assertNotIn('from runtime168',s)
 def test_actual_isolated_worker_help_has_fourteen_inputs(self):
  r=subprocess.run([sys.executable,'-I','-c','import sys,runpy;sys.path.insert(0,sys.argv.pop(1));p=sys.argv.pop(1);sys.argv=[p,"--help"];runpy.run_path(p,run_name="__main__")',str(ROOT),str(ROOT/'worker169.py')],capture_output=True,text=True,timeout=90);self.assertEqual(r.returncode,0,r.stderr)
  for k in range(155,169):self.assertIn('--c'+str(k),r.stdout)
if __name__=='__main__':unittest.main(verbosity=2)
