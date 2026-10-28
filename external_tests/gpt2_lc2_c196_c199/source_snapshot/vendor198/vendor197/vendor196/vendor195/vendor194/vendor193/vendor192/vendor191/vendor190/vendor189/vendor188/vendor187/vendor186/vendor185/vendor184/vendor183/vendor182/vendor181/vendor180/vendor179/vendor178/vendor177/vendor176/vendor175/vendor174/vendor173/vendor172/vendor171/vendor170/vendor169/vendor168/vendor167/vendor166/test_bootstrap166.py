"""Actual fresh isolated imports/entrypoint routing; no model download."""
import ast,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent
class Checks(unittest.TestCase):
    def test_worker_and_pure_audit_import_without_Torch(self):
        code="import sys;sys.path.insert(0,sys.argv[1]);import worker166,audit166;assert 'torch' not in sys.modules;assert 'transformers' not in sys.modules;assert worker166.PROTOCOL=='GPT2_LC2_C166_READONLY_DIRECTION_AUDIT_CPU_V1'"
        p=subprocess.run([sys.executable,'-I','-c',code,str(ROOT)],capture_output=True,text=True,timeout=60);self.assertEqual(p.returncode,0,p.stdout+p.stderr)
    def test_actual_launcher_routes_current_scripts(self):
        s=(ROOT/'launch166.py').read_text()
        for n in ['test166.py','test_bootstrap166.py','worker166.py']:self.assertIn(n,s)
        self.assertIn('new_model_forwards_requested=0',s);self.assertIn('scalar_backward_calls_requested=0',s)
    def test_actual_child_boot_finds_worker_in_fresh_directory(self):
        tree=ast.parse((ROOT/'launch166.py').read_text());boot=next(n.value.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='boot' for x in n.targets));boot=boot.replace('runpy.run_path(p,run_name="__main__")','runpy.run_path(p,run_name="import_only")');code="import sys;r=sys.argv[1];sys.argv=[sys.argv[0],r+'/empty_deps',r,r+'/worker166.py'];"+boot
        with tempfile.TemporaryDirectory() as d:p=subprocess.run([sys.executable,'-I','-c',code,str(ROOT)],cwd=d,capture_output=True,text=True,timeout=60)
        self.assertEqual(p.returncode,0,p.stdout+p.stderr)
if __name__=='__main__':unittest.main(verbosity=2)
