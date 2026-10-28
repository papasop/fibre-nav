"""Original C169 summary/source binding on a NumPy-generated synthetic archive."""
import copy,contextlib,io,tempfile,unittest,zipfile,shutil
from core170 import *
from inputs170 import bind169
from fixture169 import fixture,pack
from worker169 import run as run169
from audit169 import audit as audit168
from diagnostics169 import diagnostics as diag168
class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);cls.paths,cp,cls.ctx,src=fixture(cls.root/'C168_fixture');cls.c=contract();cls.c.update(production=False,pins=copy.deepcopy(cp['pins']))
  with contextlib.redirect_stdout(io.StringIO()):s=run169(cls.paths,cls.root/'C169',cp,lambda *a:copy.deepcopy(cls.ctx))
  need(s['complete'],'Synthetic original C169: '+str(s['errors']));shutil.copytree(ROOT/'vendor169',cls.root/'C169/source',ignore=shutil.ignore_patterns('__pycache__'));cls.paths['C169']=pack(cls.root/'C169',cls.root/'C169.zip');cls.c['pins']['C169']=dict(path=str(cls.paths['C169']),sha256=sha(cls.paths['C169']))
  cls.a=audit168(cls.paths['C168'],cls.ctx,cls.root/'independent_C168',cp);cls.d=diag168(cls.a,cls.root/'independent_diag')
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup()
 def changed(self,name,fn):
  with zipfile.ZipFile(self.paths['C169']) as z:m={n:z.read(n) for n in z.namelist()}
  m[name]=fn(m[name]);m['RESULT_MANIFEST.json']=(json.dumps({n:hashlib.sha256(v).hexdigest() for n,v in m.items() if n!='RESULT_MANIFEST.json'})+'\n').encode();p=self.root/(self._testMethodName+'.zip')
  with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
   for n,v in m.items():z.writestr(n,v)
  return p
 def j(self,fn):
  def edit(b):x=clean_json(b);fn(x);return (json.dumps(x)+'\n').encode()
  return edit
 def reject(self,n,f):
  p=self.changed(n,f)
  with self.assertRaises((ValueError,KeyError)):bind169(p,self.a,self.d,self.root/(self._testMethodName+'_out'),self.c)
 def test_complete_C169_summary_reconstructed_without_model(self):
  with archive155.prior.no_models(),safe.no_solvers():r=bind169(self.paths['C169'],self.a,self.d,self.root/'success',self.c)
  self.assertEqual(r['decision'],prior.DECISION);self.assertEqual(r['new_model_forwards'],0);self.assertEqual(r['new_LP_calls'],0)
 def test_C169_source_change_even_rebound_manifest_blocks(self):self.reject('source/core169.py',lambda b:b+b'\n# changed\n')
 def test_C169_summary_change_even_rebound_manifest_blocks(self):self.reject('SUMMARY.json',self.j(lambda s:s['diagnostics'].__setitem__('native_vs_FP64_response_gate_disagreements',999)))
 def test_C169_parent_input_link_change_blocks(self):self.reject('ORIGINAL_INPUTS.json',self.j(lambda s:s['C168'].__setitem__('sha256','0'*64)))
 def test_C169_incomplete_blocks(self):self.reject('SUMMARY.json',self.j(lambda s:s.__setitem__('complete',False)))
 def test_C169_duplicate_diagnostic_forgery_blocks(self):self.reject('diagnostics/DIAGNOSTIC_SUMMARY.json',self.j(lambda s:s.__setitem__('no_new_head_or_step_selected',False)))
if __name__=='__main__':unittest.main(verbosity=2)
