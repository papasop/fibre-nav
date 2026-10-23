"""Isolated actual entrypoint/import discovery; Torch stub is not tensor validation."""
import ast,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent
STUB="import types;sys.modules['torch']=types.ModuleType('torch')"
class Checks(unittest.TestCase):
    def child(self,code):
        with tempfile.TemporaryDirectory() as t:return subprocess.run([sys.executable,'-I','-c',code,str(ROOT)],cwd=t,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=60)
    def test_real_launch_boot_discovers_all_four_forward_Torch_checks(self):
        tree=ast.parse((ROOT/'launch154.py').read_text());boots=[n.value.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='boot' for x in n.targets) and isinstance(n.value,ast.Constant)];self.assertEqual(len(boots),1);boot=boots[0].replace('runpy.run_path(p,run_name="__main__")','ns=runpy.run_path(p,run_name="import_only")');p=self.child("import sys,unittest;r=sys.argv[1];sys.argv=[sys.argv[0],r+'/empty_deps',r,r+'/torch_smoke154.py'];"+STUB+'\n'+boot+"\nassert len(unittest.defaultTestLoader.getTestCaseNames(ns['Checks']))==4");self.assertEqual(p.returncode,0,p.stdout)
    def test_actual_worker_default_factory_can_resolve_without_model_libraries(self):
        p=self.child("import sys;sys.path.insert(0,sys.argv[1]);import worker154;f=worker154.resolve_default_runtime_factory();assert f.__module__=='runtime154';assert 'torch' not in sys.modules;assert 'transformers' not in sys.modules");self.assertEqual(p.returncode,0,p.stdout)
    def test_no_legacy_functional_backend_import_in_worker(self):
        tree=ast.parse((ROOT/'worker154.py').read_text());bad=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom) and n.module and n.module.startswith('functional')];self.assertEqual(bad,[])
if __name__=='__main__':unittest.main(verbosity=2)
