"""Actual current isolated worker entrypoint; no model import/download."""
import ast,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent
class Checks(unittest.TestCase):
 def test_current_worker_imports_without_model_or_solver(self):
  code="import sys;sys.path.insert(0,sys.argv[1]);import worker168,inputs168;assert not any(n in sys.modules for n in ('torch','transformers','scipy'));assert worker168.PROTOCOL=='GPT2_LC2_C168_CERTIFIED_DIRECTION_PROBES_CPU_V1'"
  p=subprocess.run([sys.executable,'-I','-c',code,str(ROOT)],capture_output=True,text=True,timeout=60);self.assertEqual(p.returncode,0,p.stdout+p.stderr)
 def test_actual_current_launcher_routes_all_current_scripts(self):
  s=(ROOT/'launch168.py').read_text()
  for n in ['test168.py','test_bootstrap168.py','test_launcher168.py','torch_smoke168.py','worker168.py']:self.assertIn(n,s)
  self.assertIn('new_model_forwards_requested=3699',s);self.assertIn('scalar_backward_calls_requested=0',s)
 def test_actual_isolated_child_boot_finds_current_worker(self):
  tree=ast.parse((ROOT/'launch168.py').read_text());start=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='start');boot=next(n.value.value for n in ast.walk(start) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='boot' for t in n.targets));boot=boot.replace('runpy.run_path(p,run_name="__main__")','runpy.run_path(p,run_name="import_only")');code="import sys;r=sys.argv[1];sys.argv=[sys.argv[0],r+'/empty',r,r+'/worker168.py'];"+boot
  with tempfile.TemporaryDirectory() as d:p=subprocess.run([sys.executable,'-I','-c',code,str(ROOT)],cwd=d,capture_output=True,text=True,timeout=60)
  self.assertEqual(p.returncode,0,p.stdout+p.stderr)
if __name__=='__main__':unittest.main(verbosity=2)
