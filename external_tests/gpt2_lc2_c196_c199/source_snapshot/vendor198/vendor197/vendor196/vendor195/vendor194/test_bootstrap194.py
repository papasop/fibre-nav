"""Isolated current-entrypoint checks, no model or solver imports."""
from pathlib import Path
import sys,subprocess,ast,unittest
ROOT=Path(__file__).resolve().parent
class Checks(unittest.TestCase):
 def test_current_worker_imports_without_model_or_solver(self):
  script="import sys;sys.path.insert(0,sys.argv[1]);import worker194;assert worker194.PROTOCOL=='GPT2_LC2_C194_ALL_COORDINATE_CONTINUATION_TRAIN_CPU_V1';assert not any(n in sys.modules for n in ('torch','transformers','peft','scipy'))";r=subprocess.run([sys.executable,'-I','-c',script,str(ROOT)],capture_output=True,text=True,timeout=90);self.assertEqual(r.returncode,0,r.stderr)
 def test_current_launcher_scripts_and_fixed_dependency_versions(self):
  s=(ROOT/'launch194.py').read_text();ast.parse(s)
  for name in ('worker194.py','test194.py','test_endpoint194.py','test_binding194.py','torch_smoke194.py','test_bootstrap194.py','test_launcher194.py'):self.assertIn(name,s)
  self.assertIn('scipy==1.17.0',s);self.assertIn('transformers==4.56.2',s);self.assertNotIn('torch_smoke168.py',s)
 def test_current_worker_help_has_all39_originals(self):
  script='import sys,runpy;sys.path.insert(0,sys.argv.pop(1));p=sys.argv.pop(1);sys.argv=[p,"--help"];runpy.run_path(p,run_name="__main__")';r=subprocess.run([sys.executable,'-I','-c',script,str(ROOT),str(ROOT/'worker194.py')],capture_output=True,text=True,timeout=90);self.assertEqual(r.returncode,0,r.stderr)
  for i in range(155,194):self.assertIn('--c'+str(i),r.stdout)
 def test_production_anchor_and_original_C193_pin_are_exact(self):
  import json
  s=(ROOT/'core194.py').read_text();self.assertIn("ANCHOR_SHA256='57603d1f7b701f41daef788ee7f65fe9cf9d2e94bb7d35e8a8fab04a67855621'",s)
  pins=json.loads((ROOT/'INPUT_PINS.json').read_text());self.assertEqual(set(pins),{'C'+str(i) for i in range(155,194)});self.assertEqual(pins['C193']['sha256'],'26ed68937c444d0a360571094543701c805dd366241bc1ef0fd22fba613246e4');self.assertEqual(pins['C192']['sha256'],'da563a3f767f74580ef8076dfccc6cb57a5e5eef3f7f8ab5a8b30f7bd23eba3d')
if __name__=='__main__':unittest.main(verbosity=2)
