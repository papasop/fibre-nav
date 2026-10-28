"""Isolated current entrypoint and no-model/no-solver import checks."""
from pathlib import Path
import sys,subprocess,ast,unittest
ROOT=Path(__file__).resolve().parent
class Checks(unittest.TestCase):
 def test_current_worker_import_without_model_or_solver(self):
  s="import sys;sys.path.insert(0,sys.argv[1]);import worker195;assert worker195.PROTOCOL=='GPT2_LC2_C195_READONLY_CONTINUATION_TRAIN_AUDIT_CPU_V1';assert not any(n in sys.modules for n in ('torch','transformers','peft','scipy'))";r=subprocess.run([sys.executable,'-I','-c',s,str(ROOT)],capture_output=True,text=True,timeout=90);self.assertEqual(r.returncode,0,r.stderr)
 def test_current_launcher_routes_current_scripts(self):
  s=(ROOT/'launch195.py').read_text();ast.parse(s)
  for name in ('test_replay195.py','test_focus195.py','worker195.py','test_math195.py','test_diagnostics195.py','test_outcomes195.py','test195.py','test_activity195.py','test_precision195.py','test_bootstrap195.py','test_launcher195.py'):self.assertIn(name,s)
  self.assertNotIn('torch_smoke',s);self.assertNotIn('pip install',s);self.assertNotIn('from runtime170',s)
 def test_current_worker_help_contains_all40_input_paths(self):
  s='import sys,runpy;sys.path.insert(0,sys.argv.pop(1));p=sys.argv.pop(1);sys.argv=[p,"--help"];runpy.run_path(p,run_name="__main__")';r=subprocess.run([sys.executable,'-I','-c',s,str(ROOT),str(ROOT/'worker195.py')],capture_output=True,text=True,timeout=90);self.assertEqual(r.returncode,0,r.stderr)
  for k in range(155,195):self.assertIn('--c'+str(k),r.stdout)
 def test_production_original_hashes_and_fixed_start_head(self):
  import json
  pins=json.loads((ROOT/'INPUT_PINS.json').read_text());self.assertEqual(set(pins),{'C'+str(i) for i in range(155,195)});self.assertEqual(pins['C194']['sha256'],'ca41d78a423094f070c4abcfad1ede42c73ad49ac241f3554b95c184820d2ce4');self.assertEqual(pins['C193']['sha256'],'26ed68937c444d0a360571094543701c805dd366241bc1ef0fd22fba613246e4');self.assertIn("ANCHOR_SHA256='57603d1f7b701f41daef788ee7f65fe9cf9d2e94bb7d35e8a8fab04a67855621'",(ROOT/'vendor194/core194.py').read_text())
if __name__=='__main__':unittest.main(verbosity=2)
