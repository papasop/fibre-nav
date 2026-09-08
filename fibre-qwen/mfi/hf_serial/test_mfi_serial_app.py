import io
import json
from pathlib import Path
import shutil
import subprocess
import threading
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import zipfile
import mfi_serial_app as app

class Tests(unittest.TestCase):
    def setUp(self):
        with app.LOCK: app.STATE.update(status='idle',download=False)
    def test_completed_reference_preserved_and_downloadable(self):
        def fake_run(args, **kw):
            out=Path(args[args.index('--out')+1]);out.mkdir();(out/'report.json').write_text(json.dumps({'passed':True,'scores':{'mfi':{'correct':33}}}))
            kw['stdout'].write('fixture: earlier reference results, not a new model run\n')
            self.assertEqual(kw['timeout'],1800)
            self.assertEqual(kw['env']['CUDA_VISIBLE_DEVICES'],'')
            return subprocess.CompletedProcess(args,0)
        with patch.object(app.subprocess,'run',side_effect=fake_run):app.run_job()
        self.assertEqual(app.STATE['status'],'completed')
        self.assertEqual(app.STATE['returncode'],0)
        self.assertTrue(app.STATE['report']['passed'])
        self.assertEqual(app.STATE['report']['scores']['mfi']['correct'],33)
        with zipfile.ZipFile(app.BUNDLE) as z:self.assertIn('run/report.json',z.namelist())
    def test_timeout_is_incomplete_not_a_score(self):
        with patch.object(app.subprocess,'run',side_effect=subprocess.TimeoutExpired('fixture',600)):app.run_job()
        self.assertEqual(app.STATE['status'],'timeout')
        self.assertEqual(app.STATE['returncode'],124)
        self.assertIsNone(app.STATE['report'])
    def test_http_token_duplicate_and_reference_page(self):
        server=app.ThreadingHTTPServer(('127.0.0.1',0),app.Handler)
        threading.Thread(target=server.serve_forever,daemon=True).start()
        base='http://127.0.0.1:'+str(server.server_port)
        try:
            with urlopen(base) as r:self.assertIn('MFI'.encode(),r.read())
            with self.assertRaises(HTTPError) as e:urlopen(Request(base+'/run',method='POST'))
            self.assertEqual(e.exception.code,403)
            app.STATE['status']='running'
            with self.assertRaises(HTTPError) as e:urlopen(Request(base+'/run',method='POST',headers={'X-MFI-Token':app.TOKEN}))
            self.assertEqual(e.exception.code,409)
            with urlopen(base+'/status') as r:self.assertEqual(json.load(r)['status'],'running')
        finally:server.shutdown();server.server_close()

if __name__=='__main__':unittest.main()
