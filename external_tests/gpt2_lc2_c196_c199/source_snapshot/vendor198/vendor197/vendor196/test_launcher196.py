"""Actual launcher success/failure packaging, stub children only."""
import os,contextlib,io,tempfile,unittest,zipfile
from unittest.mock import patch
from core196 import *
import launch196
class Checks(unittest.TestCase):
 def stub(self,fail=False):
  t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);root=Path(t.name);src=root/'source';src.mkdir();pins={}
  for k in INPUT_KEYS:q=root/(k+'.zip');q.write_bytes(b'Synthetic launcher only');pins[k]=dict(path='/content/drive/MyDrive/'+q.name,local_path=str(q),sha256=sha(q))
  dump(src/'INPUT_PINS.json',pins)
  for name in ('test_math196.py','test_endpoint196.py','test196.py','test_bootstrap196.py','test_launcher196.py','test_binding196.py','torch_smoke196.py'):(src/name).write_text("print('Synthetic launcher stub only')\n"+("raise RuntimeError('intentional child failure')\n" if fail and name=='test_binding196.py' else ''))
  s="import argparse,json,os;from pathlib import Path\np=argparse.ArgumentParser()\n"
  for k in INPUT_KEYS:s+="p.add_argument('--"+k.lower()+"')\n"
  s+="p.add_argument('--out');a=p.parse_args();assert os.environ['CUDA_VISIBLE_DEVICES']=='';Path(a.out,'SUMMARY.json').write_text(json.dumps(dict(protocol='SYNTHETIC_LAUNCHER_ONLY',complete=True,synthetic_fixture=True,new_model_forwards=0,new_backward_calls=0,LC2_closed=False)))\n";(src/'worker196.py').write_text(s);dump(src/'SOURCE_MANIFEST.json',{p.name:sha(p) for p in src.iterdir()});return root,src
 def test_actual_launcher_success_manifest_and_parent_GPU_mask(self):
  root,src=self.stub()
  with contextlib.redirect_stdout(io.StringIO()),patch.object(launch196,'ROOT',src),patch.object(launch196,'dependencies',lambda *a:None),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'7'}):
   result=launch196.start(False,None,False,root);self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'7')
  with zipfile.ZipFile(result) as z:
   s=clean_json(z.read('SUMMARY.json'));self.assertTrue(s['complete']);self.assertFalse(s['LC2_closed']);m=clean_json(z.read('RESULT_MANIFEST.json'))
   for n,h in m.items():self.assertEqual(hashlib.sha256(z.read(n)).hexdigest(),h)
 def test_child_failure_still_packages_one_incomplete_result(self):
  root,src=self.stub(True)
  with contextlib.redirect_stdout(io.StringIO()),patch.object(launch196,'ROOT',src),patch.object(launch196,'dependencies',lambda *a:None):
   with self.assertRaises(RuntimeError):launch196.start(False,None,False,root)
  results=list(root.rglob('*_results.zip'));self.assertEqual(len(results),1)
  with zipfile.ZipFile(results[0]) as z:self.assertFalse(clean_json(z.read('SUMMARY.json'))['complete']);self.assertIn('LAUNCHER_ERROR.json',z.namelist())
 def test_wrong_source_rejected_before_execution(self):
  root,src=self.stub();(src/'test196.py').write_text('changed')
  with patch.object(launch196,'ROOT',src):
   with self.assertRaises(ValueError):launch196.start(False,None,False,root)
 def test_wrong_C195_whole_hash_packages_failure_before_dependencies(self):
  root,src=self.stub();(root/'C195.zip').write_bytes(b'changed')
  with contextlib.redirect_stdout(io.StringIO()),patch.object(launch196,'ROOT',src),patch.object(launch196,'dependencies',lambda *a:(_ for _ in ()).throw(AssertionError('No deps before pinned inputs'))):
   with self.assertRaises(RuntimeError):launch196.start(False,None,False,root)
  results=list(root.rglob('*_results.zip'));self.assertEqual(len(results),1)
if __name__=='__main__':unittest.main(verbosity=2)
