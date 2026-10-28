"""Whole synthetic C197 archive binding controls; no real science originals."""
import copy,contextlib,io,tempfile,unittest,shutil,zipfile
from core198 import *
from inputs198 import bind197,load
from fixture197 import fixture
from worker197 import run as run197
from fixture171 import pack
class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)
  with contextlib.redirect_stdout(io.StringIO()):
   paths,cp,ctx,src=fixture(cls.root/'upstream');s=run197(paths,cls.root/'C197',cp,lambda *a:copy.deepcopy(ctx))
  need(s['complete'],'Synthetic C197 audit generation/'+str(s['errors']));shutil.copytree(ROOT/'vendor197',cls.root/'C197/source',ignore=shutil.ignore_patterns('__pycache__'));paths['C197']=pack(cls.root/'C197',cls.root/'C197.zip');c=contract();c.update(production=False,parent=cp,pins={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()});cls.paths=paths;cls.c=c;cls.recon=cls.root/'C197';cls.s=s
  with np.load(src/'INITIAL_C194_FIXED_FINAL_HEAD.npz',allow_pickle=False) as z:p0=z['P'].copy()
  with np.load(src/'FIXED_FINAL_HEAD_STEP8.npz',allow_pickle=False) as z:p1=z['P'].copy()
  cls.a=dict(summary=clean_json((src/'SUMMARY.json').read_bytes()),heads=[p0,p1],native_summaries=s['native_summaries'],paired=s['paired_by_round'],precision=s['endpoint_precision_reconstruction']);cls.d=s['diagnostics'];cls.focus=s['rescue_and_remaining_failure_focus'];cls.replay=s['cross_run_anchor_replay']
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup()
 def bind(self,path=None):return bind197(path or self.paths['C197'],self.a,self.d,self.focus,self.replay,self.recon,self.root/(self._testMethodName+'_out'),self.c)
 def altered(self,name,mut):
  with zipfile.ZipFile(self.paths['C197']) as z:m={n:z.read(n) for n in z.namelist()}
  m[name]=mut(m[name]);m['RESULT_MANIFEST.json']=(json.dumps({n:hashlib.sha256(b).hexdigest() for n,b in m.items() if n!='RESULT_MANIFEST.json'})+'\n').encode();p=self.root/(self._testMethodName+'.zip')
  with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
   for n,b in m.items():z.writestr(n,b)
  return p
 def j(self,f):
  def mutate(b):a=clean_json(b);f(a);return(json.dumps(a)+'\n').encode()
  return mutate
 def test_complete_original_C197_source_summary_and_artifacts_bound(self):
  r=self.bind();self.assertEqual(r['decision'],parent.DECISION);self.assertGreaterEqual(len(r['artifacts_bound']),42);self.assertFalse(r['model_outputs_independently_recomputed']);self.assertEqual(r['new_model_forwards'],0)
 def test_original_C197_source_change_blocks_rebound_manifest(self):
  p=self.altered('source/core197.py',lambda b:b+b'\n# forged\n')
  with self.assertRaises(ValueError):self.bind(p)
 def test_C197_parent_decision_change_blocks(self):
  p=self.altered('SUMMARY.json',self.j(lambda s:s.__setitem__('parent_decision','LC2_CLOSED')))
  with self.assertRaises(ValueError):self.bind(p)
 def test_C197_native_count_change_blocks(self):
  p=self.altered('SUMMARY.json',self.j(lambda s:s['native_summaries']['native_final_round0'].__setitem__('passed',84)))
  # The synthetic fixture is not required to have all84 acceptance.
  if self.s['native_summaries']['native_final_round0']['passed']==84:
   p=self.altered('SUMMARY.json',self.j(lambda s:s['native_summaries']['native_final_round0'].__setitem__('passed',0)))
  with self.assertRaises(ValueError):self.bind(p)
 def test_C197_diagnostic_artifact_change_blocks(self):
  p=self.altered('diagnostics/DIAGNOSTIC_SUMMARY.json',self.j(lambda s:s.__setitem__('forged',True)))
  with self.assertRaises(ValueError):self.bind(p)
 def test_C197_replay_artifact_change_blocks(self):
  p=self.altered('replay/REPLAY_SUMMARY.json',self.j(lambda s:s.__setitem__('maximum_native_KL_gap',1.)))
  with self.assertRaises(ValueError):self.bind(p)
 def test_wrong_whole_hash_stops_before_upstream_loader(self):
  c=copy.deepcopy(self.c);c['pins']['C197']['sha256']='0'*64
  with self.assertRaises(ValueError):load(self.paths,self.root/'wrong_hash',c)
 def test_unsafe_archive_path_rejected(self):
  p=self.root/'unsafe.zip'
  with zipfile.ZipFile(p,'w') as z:z.writestr('../escape',b'{}')
  with self.assertRaises(ValueError):
   with contextlib.closing(Archive(p)) as z:z.verify_manifest()
if __name__=='__main__':unittest.main(verbosity=2)
