"""Exact direction/source binding and fixed prewrite step controls."""
import copy,tempfile,unittest,io,zipfile,contextlib
from pathlib import Path
from unittest.mock import patch
from core168 import *
from inputs168 import bind167
from fixture168 import fixture,pack
class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);cls.paths,cls.c,cls.ctx,cls.a,cls.src=fixture(cls.root/'fixture')
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup()
 def changed(self,name,fn):
  out=self.root/(self._testMethodName+'.zip')
  with zipfile.ZipFile(self.paths['C167']) as z:m={n:z.read(n) for n in z.namelist()}
  m[name]=fn(m[name]);m['RESULT_MANIFEST.json']=(json.dumps({n:hashlib.sha256(v).hexdigest() for n,v in m.items() if n!='RESULT_MANIFEST.json'})+'\n').encode()
  with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
   for n,v in m.items():z.writestr(n,v)
  return out
 def reject(self,name,fn):
  p=self.changed(name,fn)
  with self.assertRaises((ValueError,KeyError)):bind167(p,self.ctx,self.a,self.root/(self._testMethodName+'_out'),self.c)
 def j(self,fn):
  def edit(b):x=clean_json(b);fn(x);return (json.dumps(x)+'\n').encode()
  return edit
 def arr(self,fn):
  def edit(b):
   with np.load(io.BytesIO(b),allow_pickle=False) as z:x={k:z[k] for k in z.files}
   fn(x);o=io.BytesIO();np.savez_compressed(o,**x);return o.getvalue()
  return edit
 def test_complete_positive_exact_certificate_bound_without_solver(self):
  with archive155.prior.no_models(),safe.no_solvers():d,r=bind167(self.paths['C167'],self.ctx,self.a,self.root/'positive',self.c)
  self.assertEqual(d.shape,(3,2,36));self.assertEqual(r['new_LP_calls'],0);self.assertEqual(r['classification'],'RECORDED_GRADIENT_COMMON_DESCENT_CERTIFIED');self.assertNotIn('scipy',sys.modules)
 def test_original_source_changed_even_rebound_manifest_blocks(self):self.reject('source/core167.py',lambda b:b+b'\n# changed\n')
 def test_exact_fraction_forgery_blocks(self):self.reject('EXACT_RATIONAL_CERTIFICATE.json',self.j(lambda c:c['optimal_margin_lower'].__setitem__('numerator','999')))
 def test_direction_changed_even_rebound_manifest_blocks(self):self.reject('DIRECTION_CERTIFICATE_ARRAYS.npz',self.arr(lambda x:x['direction'].__setitem__(0,.1)))
 def test_gradient_row_changed_blocks(self):self.reject('DIRECTION_CERTIFICATE_ARRAYS.npz',self.arr(lambda x:x['raw'].__setitem__((0,0),2.)))
 def test_full_problem_tail_changed_blocks(self):self.reject('RECORDED_LINEAR_PROBLEM.npz',self.arr(lambda x:x['raw'].__setitem__((83,215),.001)))
 def test_dual_negative_blocks(self):self.reject('DIRECTION_CERTIFICATE_ARRAYS.npz',self.arr(lambda x:x['dual'].__setitem__(0,-.1)))
 def test_anchor_changed_blocks(self):self.reject('RECORDED_LINEAR_PROBLEM.npz',self.arr(lambda x:x['anchor'].__setitem__((0,0,0),.001)))
 def test_parent_decision_reclassified_blocks(self):self.reject('SUMMARY.json',self.j(lambda s:s.__setitem__('parent_C165_decision','passed')))
 def test_empty_or_unresolved_summary_never_used_as_positive_direction(self):self.reject('SUMMARY.json',self.j(lambda s:s.__setitem__('recorded_linear_problem_classification','RECORDED_GRADIENT_FEASIBILITY_UNRESOLVED')))
 def test_missing_requested_state_in_binding_blocks(self):self.reject('RECORDED_PROBLEM_BINDING.json',self.j(lambda s:s['ordered_state_ids'].pop()))
 def test_certificate_hash_forgery_blocks(self):self.reject('SUMMARY.json',self.j(lambda s:s.__setitem__('direction_witness_sha256','0'*64)))
 def test_wrong_original_link_blocks(self):self.reject('ORIGINAL_INPUTS.json',self.j(lambda s:s['C165'].__setitem__('sha256','0'*64)))
 def test_changed_original_step_design_blocks(self):self.reject('DESIGN_FROZEN_BEFORE_SOLVE.json',self.j(lambda s:s.__setitem__('anchor','another checkpoint')))
 def test_incomplete_archive_blocks(self):self.reject('SUMMARY.json',self.j(lambda s:s.__setitem__('complete',False)))
 def test_requested_registry_exact_all1848_no_duplicates(self):
  r=requests();self.assertEqual(len(r),1848);self.assertEqual(len({tuple(x.items()) for x in r}),1848);self.assertEqual([x['arm'] for x in r if x['stage']=='FP64_probe'][::84],ARM_NAMES)
 def test_default_three_steps_both_signs_no_clipping(self):
  p=np.zeros((3,2,36));d=np.ones_like(p);heads,re=step_panel(p,d);self.assertEqual(re['positive_steps'],[1e-5,1e-4,.001]);self.assertEqual(list(heads),ARM_NAMES)
  for arm in re['arms']:np.testing.assert_array_equal(heads[arm['name']],p+arm['alpha']*d)
 def test_near_boundary_scales_only_from_prewrite_values(self):
  p=np.full((3,2,36),.125-1e-6);d=np.ones_like(p);heads,r=step_panel(p,d);self.assertLessEqual(r['positive_steps'][-1],(.125-float(p.flat[0]))/2);self.assertTrue(all(np.max(abs(x))<.125 for x in heads.values()))
 def test_zero_direction_blocks_before_model(self):
  with self.assertRaises(ValueError):step_panel(np.zeros((3,2,36)),np.zeros((3,2,36)))
 def test_direction_outside_unit_box_blocks(self):
  with self.assertRaises(ValueError):step_panel(np.zeros((3,2,36)),np.full((3,2,36),1.01))
 def test_nonfinite_direction_blocks(self):
  d=np.ones((3,2,36));d.flat[0]=np.nan
  with self.assertRaises(ValueError):step_panel(np.zeros_like(d),d)
 def test_coefficient_boundary_with_nonzero_direction_blocks(self):
  with self.assertRaises(ValueError):step_panel(np.full((3,2,36),.125),np.ones((3,2,36)))
 def test_old_and_new_direction_kept_distinct(self):self.reject('RECORDED_LINEAR_PROBLEM.npz',self.arr(lambda x:x['old_C165_direction'].__setitem__((0,0,0),.5)))
 def test_wrong_whole_input_hash_blocks_before_any_model(self):
  from worker168 import run
  c=copy.deepcopy(self.c);c['pins']['C167']['sha256']='0'*64
  with contextlib.redirect_stdout(io.StringIO()):s=run(self.paths,self.root/'wrongwhole',c,lambda *a:(_ for _ in ()).throw(AssertionError('No model before binding')))
  self.assertFalse(s['complete']);self.assertEqual(s['new_model_forwards'],0);self.assertFalse(s['LC2_closed'])
if __name__=='__main__':unittest.main(verbosity=2)
