"""Actual default import routing without model libraries; not a tensor validation."""
import ast,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent
class Checks(unittest.TestCase):
    def test_actual_worker_default_runtime_resolves_without_Torch_or_HF(self):
        with tempfile.TemporaryDirectory() as t:p=subprocess.run([sys.executable,'-I','-c',"import sys;sys.path.insert(0,sys.argv[1]);import worker155;assert worker155.resolve_default_runtime_factory().__module__=='runtime155';assert 'torch' not in sys.modules;assert 'transformers' not in sys.modules",str(ROOT)],cwd=t,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=60)
        self.assertEqual(p.returncode,0,p.stdout)
    def test_actual_launcher_dispatches_current_worker_smoke_and_self_tests(self):
        s=(ROOT/'launch155.py').read_text()
        for name in ['test155.py','test_bootstrap155.py','torch_smoke155.py','worker155.py']:self.assertIn(name,s)
        tree=ast.parse(s);boots=[n.value.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='boot' for x in n.targets) and isinstance(n.value,ast.Constant)];self.assertEqual(len(boots),1);self.assertIn('runpy.run_path',boots[0])
    def test_six_actual_Torch_cases_are_present_and_no_training_dispatch(self):
        tree=ast.parse((ROOT/'torch_smoke155.py').read_text());cases=[n.name for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name.startswith('test_')];self.assertEqual(len(cases),6);s=(ROOT/'worker155.py').read_text();self.assertNotIn('autograd.grad(',s);self.assertNotIn('.backward(',s);self.assertNotIn('torch.optim',s)
if __name__=='__main__':unittest.main(verbosity=2)
