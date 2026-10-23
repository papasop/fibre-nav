"""Isolated import checks use a Torch stub; they never validate tensor work."""
from pathlib import Path
import ast,os,subprocess,sys,tempfile,unittest
ROOT=Path(__file__).resolve().parent
STUB="""
import types
t=types.ModuleType('torch');t.nn=types.SimpleNamespace(Module=object)
f=types.ModuleType('torch.func');f.functional_call=lambda *a,**k:(_ for _ in ()).throw(AssertionError('No tensor work during import test'));t.func=f
sys.modules['torch']=t;sys.modules['torch.func']=f
"""
CHECK="""
import unittest
assert sum(len(unittest.defaultTestLoader.getTestCaseNames(ns[k])) for k in ('OriginalGraphChecks','PriorDirectionalChecks','PriorWorstResponseChecks','CouplingChecks'))==23
assert ns['Backend'].__module__=='functional150'
assert ns['scalar_gradients'].__module__=='functional150'
print('IMPORT_ONLY_OK 23 actual Torch tests discoverable; no tensor validation')
"""
class Checks(unittest.TestCase):
    def child(self,code,env=None):
        with tempfile.TemporaryDirectory() as cwd:return subprocess.run([sys.executable,'-I','-c',code,str(ROOT)],cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=60)
    def test_actual_torch_entrypoint_imports_pinned_dependencies_first(self):
        p=self.child('import sys,runpy;r=sys.argv[1];sys.path.insert(0,r);'+STUB+"\nns=runpy.run_path(r+'/torch_smoke150.py',run_name='import_only')\n"+CHECK);self.assertEqual(p.returncode,0,p.stdout)
    def test_actual_launcher_boot_imports_all_23_tests(self):
        tree=ast.parse((ROOT/'launch150.py').read_text());boots=[n.value.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='boot' for x in n.targets) and isinstance(n.value,ast.Constant)];self.assertEqual(len(boots),1);boot=boots[0].replace('runpy.run_path(p,run_name="__main__")','ns=runpy.run_path(p,run_name="import_only")');p=self.child("import sys;r=sys.argv[1];sys.argv=[sys.argv[0],r+'/empty_deps',r,r+'/torch_smoke150.py'];"+STUB+'\n'+boot+'\n'+CHECK);self.assertEqual(p.returncode,0,p.stdout)
    def test_foreign_pythonpath_does_not_override_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp,'core146.py').write_text("raise AssertionError('foreign')");p=self.child('import sys,runpy;r=sys.argv[1];sys.path.insert(0,r);'+STUB+"\nns=runpy.run_path(r+'/torch_smoke150.py',run_name='import_only')\n"+CHECK,dict(os.environ,PYTHONPATH=tmp));self.assertEqual(p.returncode,0,p.stdout)

if __name__=='__main__':unittest.main(verbosity=2)
