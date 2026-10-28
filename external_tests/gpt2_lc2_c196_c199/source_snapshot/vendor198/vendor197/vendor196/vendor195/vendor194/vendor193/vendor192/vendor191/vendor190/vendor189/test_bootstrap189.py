"""Isolated current entrypoint and no-model/no-solver import checks."""
from pathlib import Path
import sys,subprocess,ast,unittest
ROOT=Path(__file__).resolve().parent
class Checks(unittest.TestCase):
 def test_current_worker_import_without_model_or_solver(self):
  s="import sys;sys.path.insert(0,sys.argv[1]);import worker189;assert worker189.PROTOCOL=='GPT2_LC2_C189_READONLY_CONTINUATION_TRAIN_AUDIT_CPU_V1';assert not any(n in sys.modules for n in ('torch','transformers','peft','scipy'))";r=subprocess.run([sys.executable,'-I','-c',s,str(ROOT)],capture_output=True,text=True,timeout=90);self.assertEqual(r.returncode,0,r.stderr)
 def test_current_launcher_routes_current_scripts(self):
  s=(ROOT/'launch189.py').read_text();ast.parse(s)
  for name in ('test_replay189.py','test_focus189.py','worker189.py','test_math189.py','test_diagnostics189.py','test_outcomes189.py','test189.py','test_activity189.py','test_precision189.py','test_bootstrap189.py','test_launcher189.py'):self.assertIn(name,s)
  self.assertNotIn('torch_smoke',s);self.assertNotIn('pip install',s);self.assertNotIn('from runtime170',s)
 def test_current_worker_help_contains_all34_input_paths(self):
  s='import sys,runpy;sys.path.insert(0,sys.argv.pop(1));p=sys.argv.pop(1);sys.argv=[p,"--help"];runpy.run_path(p,run_name="__main__")';r=subprocess.run([sys.executable,'-I','-c',s,str(ROOT),str(ROOT/'worker189.py')],capture_output=True,text=True,timeout=90);self.assertEqual(r.returncode,0,r.stderr)
  for k in range(155,189):self.assertIn('--c'+str(k),r.stdout)
 def test_production_original_hashes_and_fixed_start_head(self):
  import json
  pins=json.loads((ROOT/'INPUT_PINS.json').read_text());self.assertEqual(set(pins),{'C'+str(i) for i in range(155,189)});self.assertEqual(pins['C188']['sha256'],'78ab7f82f6961482fa1895f03f6143e0b6e204a23ef21db56b38adce9f979f65');self.assertEqual(pins['C187']['sha256'],'70ad0f3e235f92b5f542ebdcbf79b945f6e963f39f8764386c2042c661f82d37');self.assertIn("ANCHOR_SHA256='005c3da405a9dff4f1168009b7857e5e8eb2247810647020ea49417d1e432c06'",(ROOT/'vendor188/core188.py').read_text())
if __name__=='__main__':unittest.main(verbosity=2)
