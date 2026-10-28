from pathlib import Path
import hashlib,runpy,tempfile,unittest,zipfile,subprocess,sys,json
ROOT=Path('/workspace/scratch/b1f94791eb89');FRONT=ROOT/'C198_download/C198_CPU_LAUNCHER.py';BUNDLE=ROOT/'C198_download/C198_CPU_LAUNCHER_BUNDLE.zip'
class Checks(unittest.TestCase):
 def ns(self):return runpy.run_path(str(FRONT))['extract_bundle'].__globals__
 def test_matching_pair_extracts_verified_current_worker(self):
  g=self.ns();r,m=g['extract_bundle'](BUNDLE);self.assertIn('worker198.py',m);self.assertIn('vendor197/core197.py',m);self.assertEqual(json.loads((r/'PROTOCOL_FROZEN.json').read_text())['counts']['model_forwards'],1513)
 def test_mismatched_archive_rejected_before_extract(self):
  g=self.ns()
  with self.assertRaises(ValueError):g['extract_bundle'](ROOT/'C197_download/C197_CPU_LAUNCHER_BUNDLE.zip')
 def test_unsafe_path_rejected(self):
  g=self.ns()
  with tempfile.TemporaryDirectory() as t:
   z=Path(t)/'unsafe.zip'
   with zipfile.ZipFile(z,'w') as f:f.writestr('../escape',b'{}')
   g['BUNDLE_SHA256']=hashlib.sha256(z.read_bytes()).hexdigest()
   with self.assertRaisesRegex(ValueError,'不安全'):g['extract_bundle'](z)
 def test_inventory_limit_rejects_1701_files(self):
  g=self.ns()
  with tempfile.TemporaryDirectory() as t:
   z=Path(t)/'large.zip'
   with zipfile.ZipFile(z,'w') as f:
    for i in range(1701):f.writestr(str(i),b'')
   g['BUNDLE_SHA256']=hashlib.sha256(z.read_bytes()).hexdigest()
   with self.assertRaisesRegex(ValueError,'大小'):g['extract_bundle'](z)
 def test_frontend_imports_launch198_only_and_restores_module_state(self):
  code='''import runpy,sys,types
n=runpy.run_path(sys.argv[1]);g=n['start'].__globals__;g['BUNDLE_ZIP']=sys.argv[2];g['USE_GOOGLE_DRIVE']=False;g['AUTO_DOWNLOAD']=False
old=g['importlib'].import_module;called=[]
def custom(name,*a,**k):
 if name=='launch198':
  called.append(name)
  def run(d,inputs,download):
   assert not d and not download and len(inputs)==43
   return 'C198_STUB_ONLY'
  return types.SimpleNamespace(start=run)
 return old(name,*a,**k)
g['importlib'].import_module=custom
try:
 assert g['start']()=='C198_STUB_ONLY'
 assert called==['launch198']
finally:g['importlib'].import_module=old
print('CURRENT_FRONTEND_OK')
'''
  p=subprocess.run([sys.executable,'-I','-c',code,str(FRONT),str(BUNDLE)],capture_output=True,text=True,timeout=30);self.assertEqual(p.returncode,0,p.stderr);self.assertIn('CURRENT_FRONTEND_OK',p.stdout)
if __name__=='__main__':unittest.main(verbosity=2)
