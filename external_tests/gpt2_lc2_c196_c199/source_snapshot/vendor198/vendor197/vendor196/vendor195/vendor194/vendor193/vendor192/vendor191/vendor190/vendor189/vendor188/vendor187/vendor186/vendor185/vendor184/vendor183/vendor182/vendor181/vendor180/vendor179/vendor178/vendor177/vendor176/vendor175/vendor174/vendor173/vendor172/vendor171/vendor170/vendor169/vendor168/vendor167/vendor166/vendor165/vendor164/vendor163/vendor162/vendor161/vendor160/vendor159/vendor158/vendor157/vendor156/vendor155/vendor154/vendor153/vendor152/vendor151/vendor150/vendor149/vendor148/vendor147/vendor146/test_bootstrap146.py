"""Import-only regression checks. Torch stubs never validate tensor or GPT2 work."""
from pathlib import Path
import ast,json,os,subprocess,sys,tempfile,unittest

ROOT=Path(__file__).resolve().parent
STUB="""
import types
t=types.ModuleType('torch');t.nn=types.SimpleNamespace(Module=object)
f=types.ModuleType('torch.func')
def forbidden(*a,**k):raise AssertionError('Import test must not execute tensor work')
f.functional_call=forbidden;t.func=f
sys.modules['torch']=t;sys.modules['torch.func']=f
"""
CHECK="""
from pathlib import Path
import unittest
assert Path(ns['previous'].__file__).resolve()==Path(r,'vendor145/torch_smoke145.py').resolve()
assert len(unittest.defaultTestLoader.getTestCaseNames(ns['OriginalGraphChecks']))==8
assert len(unittest.defaultTestLoader.getTestCaseNames(ns['PriorDirectionalChecks']))==4
assert len(unittest.defaultTestLoader.getTestCaseNames(ns['WorstResponseChecks']))==6
assert ns['scalar_gradients'].__module__=='functional146'
assert ns['Endpoint'].__module__=='functional143'
print('PINNED_IMPORT_ONLY_OK 18 real Torch tests discoverable; no tensor execution')
"""

class BootstrapChecks(unittest.TestCase):
    def child(self,code,args=(),env=None):
        with tempfile.TemporaryDirectory() as outside:
            return subprocess.run([sys.executable,'-I','-c',code,str(ROOT),*args],cwd=outside,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
    def success(self,p):
        self.assertEqual(p.returncode,0,p.stdout);self.assertIn('PINNED_IMPORT_ONLY_OK',p.stdout)
    def test_original_order_reproduces_missing_module(self):
        code="import sys; r=sys.argv[1]; sys.path.insert(0,r);"+STUB+"\nimport torch_smoke145\n"
        p=self.child(code);self.assertNotEqual(p.returncode,0);self.assertIn("No module named 'torch_smoke145'",p.stdout)
    def test_fixed_entrypoint_bootstraps_all_18_tests(self):
        code="import sys,runpy; r=sys.argv[1]; sys.path.insert(0,r);"+STUB+"\nns=runpy.run_path(r+'/torch_smoke146.py',run_name='import_only')\n"+CHECK
        self.success(self.child(code))
    def test_actual_launcher_boot_is_isolated_and_keeps_18_tests(self):
        tree=ast.parse((ROOT/'launch146.py').read_text());boots=[n.value.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='boot' for x in n.targets) and isinstance(n.value,ast.Constant)]
        self.assertEqual(len(boots),1)
        boot=boots[0].replace('runpy.run_path(p,run_name="__main__")','ns=runpy.run_path(p,run_name="import_only")')
        code="import sys; root=sys.argv[1]; sys.argv=[sys.argv[0],root+'/empty_deps',root,root+'/torch_smoke146.py'];"+STUB+'\n'+boot+'\n'+CHECK
        self.success(self.child(code))
    def test_foreign_pythonpath_is_ignored_by_isolated_child(self):
        with tempfile.TemporaryDirectory() as foreign:
            Path(foreign,'torch_smoke145.py').write_text("raise AssertionError('FOREIGN MODULE MUST NOT LOAD')\n")
            env=dict(os.environ,PYTHONPATH=foreign)
            code="import sys,runpy; r=sys.argv[1]; sys.path.insert(0,r);"+STUB+"\nns=runpy.run_path(r+'/torch_smoke146.py',run_name='import_only')\n"+CHECK
            self.success(self.child(code,env=env))

if __name__=='__main__':unittest.main(verbosity=2)
