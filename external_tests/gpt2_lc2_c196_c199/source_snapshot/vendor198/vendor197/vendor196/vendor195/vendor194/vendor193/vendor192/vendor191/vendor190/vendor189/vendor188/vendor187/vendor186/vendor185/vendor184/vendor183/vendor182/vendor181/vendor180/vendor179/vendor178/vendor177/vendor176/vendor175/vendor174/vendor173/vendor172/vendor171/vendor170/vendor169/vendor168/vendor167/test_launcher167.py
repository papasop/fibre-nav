"""Launcher success/failure ZIP packaging with stub child scripts only; no scientific claims."""
import contextlib,hashlib,io,json,os,tempfile,unittest,zipfile
from unittest.mock import patch
from core167 import *
import launch167
class Checks(unittest.TestCase):
    def setup_stub(self,fail=False):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);root=Path(tmp.name);src=root/'source';src.mkdir();pins={}
        for k in INPUT_KEYS:
            p=root/(k+'.zip');p.write_bytes(b'Synthetic launcher input only');pins[k]=dict(path='/content/drive/MyDrive/'+p.name,local_path=str(p),sha256=sha(p))
        dump(src/'INPUT_PINS.json',pins)
        for name in ['test167.py','test_worker167.py','test_bootstrap167.py','test_launcher167.py']:(src/name).write_text("print('synthetic launcher stub only')\n"+("raise RuntimeError('deliberate stub failure')\n" if fail and name=='test_launcher167.py' else ''))
        (src/'worker167.py').write_text("import argparse,json,os;from pathlib import Path\np=argparse.ArgumentParser();p.add_argument('--c155');p.add_argument('--c156');p.add_argument('--c157');p.add_argument('--c158');p.add_argument('--c159');p.add_argument('--c160');p.add_argument('--c161');p.add_argument('--c162');p.add_argument('--c163');p.add_argument('--c164');p.add_argument('--c165');p.add_argument('--c166');p.add_argument('--out');a=p.parse_args();assert os.environ['CUDA_VISIBLE_DEVICES']=='';Path(a.out,'SUMMARY.json').write_text(json.dumps(dict(protocol='SYNTHETIC_LAUNCHER_ONLY',complete=True,synthetic_fixture=True,new_model_forwards=0,new_backward_calls=0,LC2_closed=False)))\n")
        dump(src/'SOURCE_MANIFEST.json',{p.name:sha(p) for p in src.iterdir()});return root,src
    def test_actual_launcher_success_ZIP_manifest_and_parent_GPU_mask(self):
        root,src=self.setup_stub()
        with contextlib.redirect_stdout(io.StringIO()),patch.object(launch167,'ROOT',src),patch.object(launch167,'dependencies',lambda *a:None),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'7'}):
            result=launch167.start(False,None,False,root);self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'7')
        with zipfile.ZipFile(result) as z:
            s=json.loads(z.read('SUMMARY.json'));self.assertTrue(s['complete']);self.assertTrue(s['synthetic_fixture']);self.assertFalse(s['LC2_closed']);m=json.loads(z.read('RESULT_MANIFEST.json'))
            for n,h in m.items():self.assertEqual(hashlib.sha256(z.read(n)).hexdigest(),h)
    def test_child_failure_still_packages_incomplete_result(self):
        root,src=self.setup_stub(True)
        with contextlib.redirect_stdout(io.StringIO()),patch.object(launch167,'ROOT',src),patch.object(launch167,'dependencies',lambda *a:None):
            with self.assertRaises(RuntimeError):launch167.start(False,None,False,root)
        results=list(root.rglob('*_results.zip'));self.assertEqual(len(results),1)
        with zipfile.ZipFile(results[0]) as z:self.assertFalse(json.loads(z.read('SUMMARY.json'))['complete']);self.assertIn('LAUNCHER_ERROR.json',z.namelist())
    def test_wrong_source_rejected_before_child_execution(self):
        root,src=self.setup_stub();(src/'test167.py').write_text('changed')
        with patch.object(launch167,'ROOT',src):
            with self.assertRaises(ValueError):launch167.start(False,None,False,root)
if __name__=='__main__':unittest.main(verbosity=2)
