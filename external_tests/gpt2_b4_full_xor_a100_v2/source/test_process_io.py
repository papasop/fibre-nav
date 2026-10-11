import contextlib,io,sys,tempfile,unittest,time
from pathlib import Path
from process_io import visible
class Tests(unittest.TestCase):
 def test_streams_child_stdout_and_stderr(self):
  buf=io.StringIO()
  with tempfile.TemporaryDirectory() as t,contextlib.redirect_stdout(buf):
   p=Path(t)/'log.txt'
   code=visible([sys.executable,'-u','-c',"import sys;print('VISIBLE_STDOUT');print('VISIBLE_STDERR',file=sys.stderr)"],'test',timeout=2,log_path=p)
   self.assertEqual(code,0);self.assertIn('VISIBLE_STDOUT',p.read_text());self.assertIn('VISIBLE_STDERR',p.read_text())
  self.assertIn('VISIBLE_STDOUT',buf.getvalue());self.assertIn('VISIBLE_STDERR',buf.getvalue())
 def test_stall_is_timed_out_and_stopped(self):
  started=time.monotonic();buf=io.StringIO()
  with contextlib.redirect_stdout(buf):
   with self.assertRaises(TimeoutError):visible([sys.executable,'-c','import time;time.sleep(30)'],'stall',timeout=.3,heartbeat=.1)
  self.assertLess(time.monotonic()-started,3);self.assertIn('等待输出',buf.getvalue())
 def test_failure_is_visible(self):
  buf=io.StringIO()
  with contextlib.redirect_stdout(buf):
   code=visible([sys.executable,'-c',"print('IMPORT_FAILED');raise SystemExit(3)"],'bad',timeout=2,check=False)
  self.assertEqual(code,3);self.assertIn('IMPORT_FAILED',buf.getvalue())
if __name__=='__main__':unittest.main()
