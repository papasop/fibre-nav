"""End-to-end twelve-original synthetic history; never model evidence."""
import os,contextlib,io,tempfile,unittest,shutil
from core167 import *
from worker167 import run,bind166
class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)
  import test_archive166 as previous
  cls.old=previous.Checks;cls.old.setUpClass();cls.paths=dict(cls.old.paths)
  from worker166 import run as oldrun
  from fixture166 import pack
  with contextlib.redirect_stdout(io.StringIO()):s=oldrun(cls.paths,cls.root/'C166',cls.old.c)
  need(s['complete'],'Full synthetic C166: '+str(s['errors']));shutil.copytree(ROOT/'vendor166',cls.root/'C166/source',ignore=shutil.ignore_patterns('__pycache__'));path=cls.root/'C166.zip';pack(cls.root/'C166',path);cls.paths['C166']=path;cls.c=contract();cls.c.update(production=False,parent=cls.old.c,pins={k:dict(path=str(p),sha256=sha(p)) for k,p in cls.paths.items()});cls.fresh=s
 @classmethod
 def tearDownClass(cls):cls.old.tearDownClass();cls.tmp.cleanup()
 def test_all12_originals_bound_complete_LPs_exact_certificate_no_model(self):
  with contextlib.redirect_stdout(io.StringIO()):s=run(self.paths,self.root/'C167',self.c)
  self.assertTrue(s['complete'],s['errors']);self.assertTrue(s['all_twelve_originals_independently_bound']);self.assertFalse(s['LC2_closed']);self.assertEqual(s['new_model_forwards'],0);self.assertEqual(s['linear_program_calls_attempted'],2 if s['active_response_loss_rows'] else 0);self.assertTrue(s['exact_certificate_available']);self.assertFalse('torch' in sys.modules)
  dest=os.environ.get('C167_VALIDATION_REPORT')
  if dest:dump(Path(dest),s)
 def test_rebound_C166_direction_forgery_blocks(self):
  import zipfile
  p=self.root/'forged.zip'
  with zipfile.ZipFile(self.paths['C166']) as z:m={n:z.read(n) for n in z.namelist()}
  s=clean_json(m['SUMMARY.json']);s['direction_diagnostics']['active_count']+=1;m['SUMMARY.json']=(json.dumps(s)+'\n').encode();m['RESULT_MANIFEST.json']=(json.dumps({k:hashlib.sha256(v).hexdigest() for k,v in m.items() if k!='RESULT_MANIFEST.json'})+'\n').encode()
  with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
   for k,v in m.items():z.writestr(k,v)
  with self.assertRaises(ValueError):bind166(p,self.fresh,self.root/'C166',self.c)
if __name__=='__main__':unittest.main(verbosity=2)
