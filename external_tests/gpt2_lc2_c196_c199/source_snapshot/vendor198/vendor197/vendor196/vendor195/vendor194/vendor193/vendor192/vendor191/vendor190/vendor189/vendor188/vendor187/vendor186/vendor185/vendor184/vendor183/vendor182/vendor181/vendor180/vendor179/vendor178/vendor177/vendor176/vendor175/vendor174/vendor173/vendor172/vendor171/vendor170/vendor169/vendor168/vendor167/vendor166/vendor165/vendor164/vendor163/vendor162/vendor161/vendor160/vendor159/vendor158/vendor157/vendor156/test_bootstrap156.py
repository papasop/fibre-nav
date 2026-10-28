"""Isolated actual CPU worker imports and launch routing, no model libraries."""
import ast,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent
class Checks(unittest.TestCase):
    def test_actual_worker_and_auditor_import_without_Torch_HF_or_model(self):
        code="import sys;sys.path.insert(0,sys.argv[1]);import worker156,audit156;assert 'torch' not in sys.modules;assert 'transformers' not in sys.modules;assert worker156.PROTOCOL=='GPT2_LC2_C156_READONLY_PRECISION_TRAJECTORY_CPU_V1'"
        with tempfile.TemporaryDirectory() as t:p=subprocess.run([sys.executable,'-I','-c',code,str(ROOT)],cwd=t,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=60)
        self.assertEqual(p.returncode,0,p.stdout)
    def test_actual_launcher_has_only_NumPy_dependencies_and_current_scripts(self):
        s=(ROOT/'launch156.py').read_text()
        for name in ['test156.py','test_bootstrap156.py','worker156.py']:self.assertIn(name,s)
        self.assertNotIn('torch_smoke',s);self.assertNotIn("'pip'",s);self.assertIn('CPU_NUMPY_READY',s)
    def test_actual_boot_invocation_can_import_current_worker_in_isolation(self):
        t=ast.parse((ROOT/'launch156.py').read_text());boots=[n.value.value for n in ast.walk(t) if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='boot' for x in n.targets) and isinstance(n.value,ast.Constant)];self.assertEqual(len(boots),1);boot=boots[0].replace('runpy.run_path(p,run_name="__main__")','runpy.run_path(p,run_name="import_only")');code="import sys;r=sys.argv[1];sys.argv=[sys.argv[0],r+'/empty_deps',r,r+'/worker156.py'];"+boot+";assert 'torch' not in sys.modules"
        with tempfile.TemporaryDirectory() as tmp:p=subprocess.run([sys.executable,'-I','-c',code,str(ROOT)],cwd=tmp,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=60)
        self.assertEqual(p.returncode,0,p.stdout)
if __name__=='__main__':unittest.main(verbosity=2)
