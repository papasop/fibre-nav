"""C173 original source/summary linkage on NumPy synthetic protocol archives."""
import copy,contextlib,io,tempfile,unittest,zipfile,shutil
from core174 import *
from inputs174 import bind173
from fixture173 import fixture,pack
from worker173 import run as run173
from audit173 import audit as audit172
from diagnostics173 import diagnostics as diag173
class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);cls.paths,cp,cls.ctx,src=fixture(cls.root/'C172_fixture');cls.c=contract();cls.c.update(production=False,parent=cp,pins=copy.deepcopy(cp['pins']))
  with contextlib.redirect_stdout(io.StringIO()):s=run173(cls.paths,cls.root/'C173',cp,lambda *a:copy.deepcopy(cls.ctx))
  need(s['complete'],'Synthetic C173 audit generation: '+str(s['errors']));shutil.copytree(ROOT/'vendor173',cls.root/'C173/source',ignore=shutil.ignore_patterns('__pycache__'));cls.paths['C173']=pack(cls.root/'C173',cls.root/'C173.zip');cls.c['pins']['C173']=dict(path=str(cls.paths['C173']),sha256=sha(cls.paths['C173']))
  with contextlib.redirect_stdout(io.StringIO()):cls.a=audit172(cls.paths['C172'],cls.ctx,cls.root/'independent_C172',cp)
  cls.d=diag173(cls.a,cls.root/'independent_coordinates')
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup()
 def changed(self,name,fn):
  with zipfile.ZipFile(self.paths['C173']) as z:m={n:z.read(n) for n in z.namelist()}
  m[name]=fn(m[name]);m['RESULT_MANIFEST.json']=(json.dumps({n:hashlib.sha256(v).hexdigest() for n,v in m.items() if n!='RESULT_MANIFEST.json'})+'\n').encode();p=self.root/(self._testMethodName+'.zip')
  with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
   for n,v in m.items():z.writestr(n,v)
  return p
 def j(self,fn):
  def edit(b):v=clean_json(b);fn(v);return (json.dumps(v)+'\n').encode()
  return edit
 def reject(self,n,fn):
  p=self.changed(n,fn)
  with self.assertRaises((ValueError,KeyError)):bind173(p,self.a,self.d,self.root/(self._testMethodName+'_out'),self.c)
 def test_complete_C173_summary_bound_without_model_or_solver(self):
  with archive155.prior.no_models(),safe.no_solvers():r=bind173(self.paths['C173'],self.a,self.d,self.root/'success',self.c)
  self.assertEqual(r['decision'],prior.DECISION);self.assertTrue(r['C173_scientific_summary_reconstructed'])
 def test_source_change_even_rebound_manifest_blocks(self):self.reject('source/core173.py',lambda b:b+b'\n# changed\n')
 def test_C170_anchor_digest_change_blocks(self):self.reject('SUMMARY.json',self.j(lambda v:v.__setitem__('original_C170_fixed_final_head_digest','0'*64)))
 def test_coordinate_summary_change_blocks(self):self.reject('SUMMARY.json',self.j(lambda v:v['coordinate_diagnostics'].__setitem__('coordinate_switch_establishes_causality',True)))
 def test_incomplete_C173_blocks(self):self.reject('SUMMARY.json',self.j(lambda v:v.__setitem__('complete',False)))
 def test_original_C172_ancestry_change_blocks(self):self.reject('ORIGINAL_INPUTS.json',self.j(lambda v:v['C172'].__setitem__('sha256','0'*64)))
 def test_coordinate_duplicate_forgery_blocks(self):self.reject('diagnostics/DIAGNOSTIC_SUMMARY.json',self.j(lambda v:v.__setitem__('LC2_closed',True)))
 def test_parent_diagnostic_duplicate_forgery_blocks(self):self.reject('C172_arithmetic/PARENT_DIAGNOSTIC_SUMMARY_RECOMPUTED.json',self.j(lambda v:v.__setitem__('step_or_head_selected',True)))
if __name__=='__main__':unittest.main(verbosity=2)
