"""Current isolated entrypoint and launcher failure packaging; no science runs."""
import unittest,tempfile,subprocess,os,zipfile
from unittest import mock
from core199 import *
import launch199
class Checks(unittest.TestCase):
 def test_current_entrypoint_isolated_import(self):
  code='import sys;sys.path.insert(0,sys.argv[1]);import launch199;assert launch199.PROTOCOL=="'+PROTOCOL+'";assert "torch" not in sys.modules;assert "scipy" not in sys.modules;print("C199_ENTRY_OK")'
  r=subprocess.run([sys.executable,'-I','-c',code,str(ROOT)],capture_output=True,text=True,timeout=30);self.assertEqual(r.returncode,0,r.stderr);self.assertIn('C199_ENTRY_OK',r.stdout)
 def test_child_gpu_mask_preserves_kernel_environment(self):
  with mock.patch.dict(os.environ,CUDA_VISIBLE_DEVICES='kernel-test'):
   env=launch199.child_environment();self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'');self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'kernel-test')
 def test_wrong_explicit_input_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'wrong.zip';p.write_bytes(b'wrong')
   with self.assertRaises(ValueError):launch199.choose('C197',str(p))
 def test_no_matching_input_reports_original_kind(self):
  with tempfile.TemporaryDirectory() as t:
   with self.assertRaisesRegex(FileNotFoundError,'C197'):launch199.choose('C197','',Path(t))
 def test_launcher_failure_still_packages_source_and_summary(self):
  with tempfile.TemporaryDirectory() as t:
   with mock.patch.object(launch199,'choose',side_effect=FileNotFoundError('synthetic missing original')):
    with self.assertRaises(RuntimeError):launch199.start(False,{},False,t)
   archives=list(Path(t).rglob('*_results.zip'));self.assertEqual(len(archives),1)
   with zipfile.ZipFile(archives[0]) as z:
    s=clean_json(z.read('SUMMARY.json'));self.assertFalse(s['complete']);self.assertFalse(s['LC2_closed']);self.assertIn('source/worker199.py',z.namelist());self.assertIn('LAUNCHER_ERROR.json',z.namelist());self.assertIn('RESULT_MANIFEST.json',z.namelist())
 def test_visible_subprocess_failure_is_bounded_and_recorded(self):
  import io
  with self.assertRaises(ValueError):launch199.visible([sys.executable,'-I','-c','raise SystemExit(2)'],'synthetic exit',10,launch199.child_environment(),io.StringIO())
if __name__=='__main__':unittest.main(verbosity=2)
