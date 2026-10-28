"""Original C171 schema/summary/source bindings; synthetic ancestry injected."""
import copy,contextlib,io,tempfile,unittest,zipfile,shutil
from core172 import *
from inputs172 import bind171
from fixture171 import fixture,pack
from worker171 import run as run171
from audit171 import audit as audit170
from diagnostics171 import diagnostics as diag170
class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);cls.paths,cp,cls.ctx,src=fixture(cls.root/'C170_fixture');cls.c=contract();cls.c.update(production=False,parent=cp,pins=copy.deepcopy(cp['pins']))
  with contextlib.redirect_stdout(io.StringIO()):s=run171(cls.paths,cls.root/'C171',cp,lambda *a:copy.deepcopy(cls.ctx))
  need(s['complete'],'Synthetic C171 parent: '+str(s['errors']));shutil.copytree(ROOT/'vendor171',cls.root/'C171/source',ignore=shutil.ignore_patterns('__pycache__'));cls.paths['C171']=pack(cls.root/'C171',cls.root/'C171.zip');cls.c['pins']['C171']=dict(path=str(cls.paths['C171']),sha256=sha(cls.paths['C171']))
  with contextlib.redirect_stdout(io.StringIO()):cls.a=audit170(cls.paths['C170'],cls.ctx,cls.root/'independent_C170',cp)
  cls.d=diag170(cls.a,cls.root/'independent_diag')
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup()
 def changed(self,name,fn):
  with zipfile.ZipFile(self.paths['C171']) as z:m={n:z.read(n) for n in z.namelist()}
  m[name]=fn(m[name]);m['RESULT_MANIFEST.json']=(json.dumps({n:hashlib.sha256(v).hexdigest() for n,v in m.items() if n!='RESULT_MANIFEST.json'})+'\n').encode();p=self.root/(self._testMethodName+'.zip')
  with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
   for n,v in m.items():z.writestr(n,v)
  return p
 def j(self,fn):
  def edit(b):x=clean_json(b);fn(x);return (json.dumps(x)+'\n').encode()
  return edit
 def reject(self,n,f):
  p=self.changed(n,f)
  with self.assertRaises((ValueError,KeyError)):bind171(p,self.a,self.d,self.root/(self._testMethodName+'_out'),self.c)
 def test_complete_C171_original_summary_independently_bound(self):
  with archive155.prior.no_models(),safe.no_solvers():r=bind171(self.paths['C171'],self.a,self.d,self.root/'success',self.c)
  self.assertEqual(r['decision'],prior.DECISION);self.assertTrue(r['C171_scientific_summary_reconstructed'])
 def test_source_change_even_rebound_manifest_blocks(self):self.reject('source/core171.py',lambda b:b+b'\n# changed\n')
 def test_C170_final_head_digest_forgery_blocks(self):self.reject('SUMMARY.json',self.j(lambda s:s.__setitem__('fixed_final_step8_head_digest','0'*64)))
 def test_C171_summary_change_blocks(self):self.reject('SUMMARY.json',self.j(lambda s:s['diagnostics'].__setitem__('native_vs_FP64_response_gate_disagreements',999)))
 def test_C171_parent_input_link_change_blocks(self):self.reject('ORIGINAL_INPUTS.json',self.j(lambda s:s['C170'].__setitem__('sha256','0'*64)))
 def test_C171_incomplete_blocks(self):self.reject('SUMMARY.json',self.j(lambda s:s.__setitem__('complete',False)))
 def test_C171_diagnostic_duplicate_change_blocks(self):self.reject('diagnostics/DIAGNOSTIC_SUMMARY.json',self.j(lambda s:s.__setitem__('LC2_closed',True)))
 def test_exact_certificate_duplicate_change_blocks(self):self.reject('C170_arithmetic/ALL8_EXACT_DIRECTION_CHECKS_RECOMPUTED.json',self.j(lambda s:s[0].__setitem__('strict_recorded_common_descent',False)))
if __name__=='__main__':unittest.main(verbosity=2)
