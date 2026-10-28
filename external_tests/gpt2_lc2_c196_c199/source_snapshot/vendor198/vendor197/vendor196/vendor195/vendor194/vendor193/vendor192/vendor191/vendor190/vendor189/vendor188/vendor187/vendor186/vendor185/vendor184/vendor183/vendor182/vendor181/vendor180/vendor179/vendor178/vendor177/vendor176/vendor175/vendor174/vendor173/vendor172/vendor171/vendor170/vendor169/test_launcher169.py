"""Actual launcher packaging with stub children only, never scientific evidence."""
import contextlib,io,os,tempfile,unittest,zipfile
from unittest.mock import patch
from core169 import *
import launch169
class Checks(unittest.TestCase):
 def stub(self,fail=False):
  tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);root=Path(tmp.name);src=root/'source';src.mkdir();pins={}
  for k in INPUT_KEYS:
   p=root/(k+'.zip');p.write_bytes(b'Synthetic launcher input only');pins[k]=dict(path='/content/drive/MyDrive/'+p.name,local_path=str(p),sha256=sha(p))
  dump(src/'INPUT_PINS.json',pins)
  for name in ('test169.py','test_bootstrap169.py','test_launcher169.py'):(src/name).write_text("print('synthetic launcher stub only')\n"+("raise RuntimeError('deliberate stub failure')\n" if fail and name=='test169.py' else ''))
  text="import argparse,json,os;from pathlib import Path\np=argparse.ArgumentParser()\n"
  for k in INPUT_KEYS:text+="p.add_argument('--"+k.lower()+"')\n"
  text+="p.add_argument('--out');a=p.parse_args();assert os.environ['CUDA_VISIBLE_DEVICES']=='';Path(a.out,'SUMMARY.json').write_text(json.dumps(dict(protocol='SYNTHETIC_LAUNCHER_ONLY',complete=True,synthetic_fixture=True,new_model_forwards=0,new_backward_calls=0,LC2_closed=False)))\n";(src/'worker169.py').write_text(text);dump(src/'SOURCE_MANIFEST.json',{p.name:sha(p) for p in src.iterdir()});return root,src
 def test_launcher_success_manifest_and_parent_GPU_mask_unchanged(self):
  root,src=self.stub()
  with contextlib.redirect_stdout(io.StringIO()),patch.object(launch169,'ROOT',src),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'7'}):
   result=launch169.start(False,None,False,root);self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'7')
  with zipfile.ZipFile(result) as z:
   s=clean_json(z.read('SUMMARY.json'));self.assertTrue(s['complete']);self.assertFalse(s['LC2_closed']);m=clean_json(z.read('RESULT_MANIFEST.json'))
   for n,h in m.items():self.assertEqual(hashlib.sha256(z.read(n)).hexdigest(),h)
 def test_child_failure_still_packages_one_incomplete_result(self):
  root,src=self.stub(True)
  with contextlib.redirect_stdout(io.StringIO()),patch.object(launch169,'ROOT',src):
   with self.assertRaises(RuntimeError):launch169.start(False,None,False,root)
  results=list(root.rglob('*_results.zip'));self.assertEqual(len(results),1)
  with zipfile.ZipFile(results[0]) as z:self.assertFalse(clean_json(z.read('SUMMARY.json'))['complete']);self.assertIn('LAUNCHER_ERROR.json',z.namelist())
 def test_wrong_source_rejected_before_child(self):
  root,src=self.stub();(src/'test169.py').write_text('changed')
  with patch.object(launch169,'ROOT',src):
   with self.assertRaises(ValueError):launch169.start(False,None,False,root)
 def test_wrong_C168_whole_hash_packages_failure(self):
  root,src=self.stub();p=root/'C168.zip';p.write_bytes(b'changed input')
  with contextlib.redirect_stdout(io.StringIO()),patch.object(launch169,'ROOT',src):
   with self.assertRaises(RuntimeError):launch169.start(False,None,False,root)
  self.assertEqual(len(list(root.rglob('*_results.zip'))),1)
if __name__=='__main__':unittest.main(verbosity=2)
