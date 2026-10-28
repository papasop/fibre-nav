"""Original C175 summary/source binding on synthetic whole scientific archives."""
import copy,contextlib,io,tempfile,unittest,zipfile,shutil
from core176 import *
from inputs176 import bind175
from fixture175 import fixture,pack
from worker175 import run as run175
from audit175 import audit as audit174
from recompute175 import supplemental
class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);cls.paths,cp,cls.ctx,src=fixture(cls.root/'C174_fixture');cls.c=contract();cls.c.update(production=False,parent=cp,pins=copy.deepcopy(cp['pins']))
  with contextlib.redirect_stdout(io.StringIO()):s=run175(cls.paths,cls.root/'C175',cp,lambda *a:copy.deepcopy(cls.ctx))
  need(s['complete'],'Synthetic C175 audit generation: '+str(s['errors']));shutil.copytree(ROOT/'vendor175',cls.root/'C175/source',ignore=shutil.ignore_patterns('__pycache__'));cls.paths['C175']=pack(cls.root/'C175',cls.root/'C175.zip');cls.c['pins']['C175']=dict(path=str(cls.paths['C175']),sha256=sha(cls.paths['C175']))
  with contextlib.redirect_stdout(io.StringIO()):cls.a=audit174(cls.paths['C174'],cls.ctx,cls.root/'independent_C174',cp)
  cls.d=supplemental(cls.a,cls.root/'independent_C175_coordinates')
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup()
 def changed(self,name,fn):
  with zipfile.ZipFile(self.paths['C175']) as z:m={n:z.read(n) for n in z.namelist()}
  m[name]=fn(m[name]);m['RESULT_MANIFEST.json']=(json.dumps({n:hashlib.sha256(v).hexdigest() for n,v in m.items() if n!='RESULT_MANIFEST.json'})+'\n').encode();p=self.root/(self._testMethodName+'.zip')
  with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
   for n,v in m.items():z.writestr(n,v)
  return p
 def j(self,fn):
  def edit(b):v=clean_json(b);fn(v);return (json.dumps(v)+'\n').encode()
  return edit
 def reject(self,n,fn):
  p=self.changed(n,fn)
  with self.assertRaises((ValueError,KeyError)):bind175(p,self.a,self.d,self.root/(self._testMethodName+'_out'),self.c)
 def test_complete_original_C175_source_and_summary_reconstructed(self):
  with archive155.prior.no_models(),safe.no_solvers():r=bind175(self.paths['C175'],self.a,self.d,self.root/'complete',self.c)
  self.assertEqual(r['decision'],prior.DECISION);self.assertTrue(r['C175_scientific_summary_reconstructed']);self.assertEqual(r['new_LP_calls'],0)
 def test_original_C175_source_change_rebound_manifest_blocks(self):self.reject('source/core175.py',lambda b:b+b'\n# changed\n')
 def test_original_C170_anchor_change_blocks(self):self.reject('SUMMARY.json',self.j(lambda v:v.__setitem__('original_C170_fixed_final_head_digest','0'*64)))
 def test_coordinate_summary_forgery_blocks(self):self.reject('SUMMARY.json',self.j(lambda v:v['coordinate_diagnostics'].__setitem__('derivative_accuracy_or_curvature_bound_established',True)))
 def test_incomplete_original_C175_blocks(self):self.reject('SUMMARY.json',self.j(lambda v:v.__setitem__('complete',False)))
 def test_original_C174_ancestry_forgery_blocks(self):self.reject('ORIGINAL_INPUTS.json',self.j(lambda v:v['C174'].__setitem__('sha256','0'*64)))
 def test_coordinate_duplicate_forgery_blocks(self):self.reject('diagnostics/SUPPLEMENTAL_SUMMARY.json',self.j(lambda v:v.__setitem__('LC2_closed',True)))
 def test_parent_diagnostic_duplicate_forgery_blocks(self):self.reject('C174_arithmetic/PARENT_DIAGNOSTIC_SUMMARY_RECOMPUTED.json',self.j(lambda v:v.__setitem__('step_or_head_selected',True)))
 def test_original_recorded_Jacobian_count_cannot_be_hidden(self):self.reject('SUMMARY.json',self.j(lambda v:v.__setitem__('archived_response_Jacobian_blocks_checked',0)))
if __name__=='__main__':unittest.main(verbosity=2)
