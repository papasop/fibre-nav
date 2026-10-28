"""Fresh isolated import and actual worker routing checks."""
import ast,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent
class Checks(unittest.TestCase):
 def test_worker_and_verifier_import_no_model_no_solver(self):
  code="import sys;sys.path.insert(0,sys.argv[1]);import worker167,certificate167;assert not any(n in sys.modules for n in ('scipy','torch','transformers'));assert worker167.PROTOCOL=='GPT2_LC2_C167_RECORDED_GRADIENT_FEASIBILITY_CPU_V1'"
  p=subprocess.run([sys.executable,'-I','-c',code,str(ROOT)],capture_output=True,text=True,timeout=60);self.assertEqual(p.returncode,0,p.stdout+p.stderr)
 def test_actual_launcher_routes_current_scripts(self):
  s=(ROOT/'launch167.py').read_text()
  for n in ('test167.py','test_bootstrap167.py','test_launcher167.py','worker167.py'):self.assertIn(n,s)
  self.assertIn('new_model_forwards_requested=0',s);self.assertIn('offline_LP_calls_if_active=2',s)
 def test_isolated_boot_finds_worker(self):
  tree=ast.parse((ROOT/'launch167.py').read_text());start=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='start');boot=next(n.value.value for n in ast.walk(start) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='boot' for t in n.targets));boot=boot.replace('runpy.run_path(p,run_name="__main__")','runpy.run_path(p,run_name="import_only")');code="import sys;r=sys.argv[1];sys.argv=[sys.argv[0],r+'/empty',r,r+'/worker167.py'];"+boot
  with tempfile.TemporaryDirectory() as d:p=subprocess.run([sys.executable,'-I','-c',code,str(ROOT)],cwd=d,capture_output=True,text=True,timeout=60)
  self.assertEqual(p.returncode,0,p.stdout+p.stderr)
if __name__=='__main__':unittest.main(verbosity=2)
