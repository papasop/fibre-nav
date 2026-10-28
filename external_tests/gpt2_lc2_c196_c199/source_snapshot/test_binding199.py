"""Reconstruct a complete synthetic C198 original; tampering cannot supply confirmation."""
import copy,contextlib,io,tempfile,unittest,shutil,zipfile
from core199 import *
from inputs199 import audit198,load
from fixture198 import fixture,Fake
from worker198 import run as run198
class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);paths,cp,ctx=fixture(cls.root/'upstream')
  with contextlib.redirect_stdout(io.StringIO()):s=run198(paths,cls.root/'C198',cp,lambda *a:copy.deepcopy(ctx),lambda out,ctx,calls:Fake(out,ctx,calls))
  need(s['complete'] and s['frozen_training_replay_passed'],'Synthetic C198 prerequisite/'+str(s['errors']));shutil.copytree(ROOT/'vendor198',cls.root/'C198/source',ignore=shutil.ignore_patterns('__pycache__'));cls.ctx=ctx;cls.paths=paths
  cls.path=cls.root/'C198.zip';cls.pack(cls.root/'C198',cls.path);paths['C198']=cls.path;c=contract();c.update(production=False,parent=cp,pins={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()});cls.c=c
 @staticmethod
 def pack(root,path):
  files={p.relative_to(root).as_posix():p.read_bytes() for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts};files['RESULT_MANIFEST.json']=(json.dumps({n:hashlib.sha256(b).hexdigest() for n,b in files.items() if n!='RESULT_MANIFEST.json'})+'\n').encode()
  with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
   for n,b in files.items():z.writestr(n,b)
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup()
 def changed(self,name,fn):
  with zipfile.ZipFile(self.path) as z:m={n:z.read(n) for n in z.namelist()}
  m[name]=fn(m[name]);m['RESULT_MANIFEST.json']=(json.dumps({n:hashlib.sha256(b).hexdigest() for n,b in m.items() if n!='RESULT_MANIFEST.json'})+'\n').encode();p=self.root/(self._testMethodName+'.zip')
  with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
   for n,b in m.items():z.writestr(n,b)
  return p
 def j(self,fn):
  def mutate(b):x=clean_json(b);fn(x);return(json.dumps(x)+'\n').encode()
  return mutate
 def arr(self,fn):
  def mutate(b):
   with np.load(io.BytesIO(b),allow_pickle=False) as z:x={n:z[n] for n in z.files}
   fn(x);out=io.BytesIO();np.savez_compressed(out,**x);return out.getvalue()
  return mutate
 def audit(self,p=None):return audit198(p or self.path,self.ctx,self.root/(self._testMethodName+'_out'),self.c)
 def test_full_C198672_records168_features504_native_reconstructed(self):
  r=self.audit();self.assertEqual(r['receipt']['ordered_records_reconstructed'],672);self.assertEqual(r['receipt']['feature_records_reconstructed'],168);self.assertEqual(r['receipt']['native_records_reconstructed'],504);self.assertEqual(len(r['points']['0']),84);self.assertFalse(r['receipt']['model_outputs_independently_recomputed'])
 def test_source_change_blocks_rebound_manifest(self):
  p=self.changed('source/runtime198.py',lambda b:b+b'\n# tamper\n')
  with self.assertRaises(ValueError):self.audit(p)
 def test_forged_full_replay_flag_blocks(self):
  p=self.changed('SUMMARY.json',self.j(lambda s:s.__setitem__('frozen_training_replay_passed',False)))
  with self.assertRaises(ValueError):self.audit(p)
 def test_per_state_capture_change_blocks_even_rebound_manifest(self):
  n='prewrite/round0/'+IDS[0]+'.npz';p=self.changed(n,self.arr(lambda a:a['capture3147'].__setitem__(0,1.)))
  with self.assertRaises(ValueError):self.audit(p)
 def test_route_change_blocks(self):
  n='prewrite/round0/'+IDS[0]+'.npz';p=self.changed(n,self.arr(lambda a:a['weights'].__setitem__(0,.9)))
  with self.assertRaises(ValueError):self.audit(p)
 def test_record_order_change_blocks(self):
  def fn(b):r=b.splitlines();r[0],r[1]=r[1],r[0];return b'\n'.join(r)+b'\n'
  p=self.changed('ALL672_RECORDS.jsonl',fn)
  with self.assertRaises(ValueError):self.audit(p)
 def test_cached_reference_change_blocks(self):
  p=self.changed('FIXED_ARCHIVED_REFERENCES.npz',self.arr(lambda a:a['response'].__setitem__(0,.001)))
  with self.assertRaises(ValueError):self.audit(p)
 def test_wrong_C198_whole_hash_blocks_before_parent_load(self):
  c=copy.deepcopy(self.c);c['pins']['C198']['sha256']='0'*64
  with self.assertRaises(ValueError):load(self.paths,self.root/'wronghash',c)
if __name__=='__main__':unittest.main(verbosity=2)
