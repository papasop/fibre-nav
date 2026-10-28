"""Isolated current-entrypoint checks, no model or solver imports."""
from pathlib import Path
import sys,subprocess,ast,unittest
ROOT=Path(__file__).resolve().parent
class Checks(unittest.TestCase):
 def test_current_worker_imports_without_model_or_solver(self):
  script="import sys;sys.path.insert(0,sys.argv[1]);import worker196;assert worker196.PROTOCOL=='GPT2_LC2_C196_ALL_COORDINATE_CONTINUATION_TRAIN_CPU_V1';assert not any(n in sys.modules for n in ('torch','transformers','peft','scipy'))";r=subprocess.run([sys.executable,'-I','-c',script,str(ROOT)],capture_output=True,text=True,timeout=90);self.assertEqual(r.returncode,0,r.stderr)
 def test_current_launcher_scripts_and_fixed_dependency_versions(self):
  s=(ROOT/'launch196.py').read_text();ast.parse(s)
  for name in ('worker196.py','test196.py','test_endpoint196.py','test_binding196.py','torch_smoke196.py','test_bootstrap196.py','test_launcher196.py'):self.assertIn(name,s)
  self.assertIn('scipy==1.17.0',s);self.assertIn('transformers==4.56.2',s);self.assertNotIn('torch_smoke168.py',s)
 def test_current_worker_help_has_all41_originals(self):
  script='import sys,runpy;sys.path.insert(0,sys.argv.pop(1));p=sys.argv.pop(1);sys.argv=[p,"--help"];runpy.run_path(p,run_name="__main__")';r=subprocess.run([sys.executable,'-I','-c',script,str(ROOT),str(ROOT/'worker196.py')],capture_output=True,text=True,timeout=90);self.assertEqual(r.returncode,0,r.stderr)
  for i in range(155,196):self.assertIn('--c'+str(i),r.stdout)
 def test_production_anchor_and_original_C195_pin_are_exact(self):
  import json
  s=(ROOT/'core196.py').read_text();self.assertIn("ANCHOR_SHA256='2ff29f06c399a501de74387f9f3caebadecb763c89d9c6ef9db8a84bef478aaf'",s)
  pins=json.loads((ROOT/'INPUT_PINS.json').read_text());self.assertEqual(set(pins),{'C'+str(i) for i in range(155,196)});self.assertEqual(pins['C195']['sha256'],'8390f73598c0c9446c0ee7ddcd63298470e539e552d20ae44cec90cc302d044b');self.assertEqual(pins['C194']['sha256'],'ca41d78a423094f070c4abcfad1ede42c73ad49ac241f3554b95c184820d2ce4')
if __name__=='__main__':unittest.main(verbosity=2)
