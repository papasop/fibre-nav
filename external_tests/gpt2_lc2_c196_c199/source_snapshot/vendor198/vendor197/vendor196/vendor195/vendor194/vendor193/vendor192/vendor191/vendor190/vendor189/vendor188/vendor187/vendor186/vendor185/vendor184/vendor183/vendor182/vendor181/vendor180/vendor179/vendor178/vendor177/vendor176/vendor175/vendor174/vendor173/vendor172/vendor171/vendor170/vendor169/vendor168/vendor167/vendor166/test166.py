"""Complete CPU audit, original iteration replay and rebound corruption controls."""
import copy,io,tempfile,unittest,zipfile
from unittest.mock import patch
from core166 import *
from audit166 import audit
from worker166 import run
from fixture166 import fixture,changed,json_change,array_change
class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);cls.paths,cls.c,cls.ctx=fixture(cls.root/'fixture')
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup()
 def reject(self,name,fn):
  z=changed(self.paths['C165'],self.root/(self._testMethodName+'.zip'),name,fn,True)
  with contextlib.redirect_stdout(io.StringIO()),self.assertRaises((ValueError,KeyError,RuntimeError)):audit(z,copy.deepcopy(self.ctx),self.root/(self._testMethodName+'_out'),self.c)
 def rows(self,fn):
  def edit(raw):rr=[clean_json(x) for x in raw.splitlines()];fn(rr);return ('\n'.join(json.dumps(r,allow_nan=True) for r in rr)+'\n').encode()
  return edit
 def test_full1848_worker_no_model_or_new_optimization(self):
  with contextlib.redirect_stdout(io.StringIO()),archive155.prior.no_models(),safe.no_solvers():s=run(self.paths,self.root/'complete',self.c,lambda *a:copy.deepcopy(self.ctx))
  self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['actual_FP64_point_arrays_rehashed'],672);self.assertEqual(s['native_FP32_point_arrays_rehashed'],588);self.assertEqual(s['native_endpoint_records_audited'],1176);self.assertEqual(s['scalar_central_difference_rows_audited'],252);self.assertFalse(s['LC2_closed']);self.assertFalse(s['all_eleven_originals_independently_bound']);self.assertFalse(s['gradients_independently_recomputed']);self.assertEqual(s['new_direction_optimization_iterations'],0);self.assertFalse('torch' in sys.modules);self.assertTrue(s['diagnostics']['all1512_response_coordinate_FD_rows_retained'])
 def test_wrong_whole_hash_stops_before_payload(self):
  cc=copy.deepcopy(self.c);cc['pins']['C165']['sha256']='0'*64
  with contextlib.redirect_stdout(io.StringIO()),patch('worker166.audit',side_effect=AssertionError('No payload')):s=run(self.paths,self.root/'wrongwhole',cc,lambda *a:(_ for _ in ()).throw(AssertionError('No parent loader')))
  self.assertFalse(s['complete'])
 def test_changed_original_source_even_rebound_manifest(self):self.reject('source/core165.py',lambda b:b+b'\n# changed\n')
 def test_step_registry_threshold_change_blocks(self):self.reject('DESIGN_FROZEN_BEFORE_MODEL.json',json_change(lambda a:a['arms'][0].__setitem__('alpha',-.001)))
 def test_anchor_changed_blocks(self):self.reject('ANCHOR_C163_FIXED_FINAL_HEAD.npz',array_change(lambda a:a['P'].__setitem__((0,0,0),.1)))
 def test_saved_direction_changed_blocks(self):self.reject('ONE_FIXED_DIRECTION.npz',array_change(lambda a:a['direction'].__setitem__((0,0,0),.123)))
 def test_saved_direction_anchor_substitution_blocks(self):self.reject('ONE_FIXED_DIRECTION.npz',array_change(lambda a:a['anchor'].__setitem__((0,0,0),.1)))
 def test_original_iteration_omitted_blocks(self):self.reject('ALL4096_DIRECTION_ITERATIONS.json',json_change(lambda a:a.pop()))
 def test_original_iteration_line_step_forged_blocks(self):self.reject('ALL4096_DIRECTION_ITERATIONS.json',json_change(lambda a:a[0].__setitem__('line_step',.991)))
 def test_original_iteration_vertex_forged_blocks(self):self.reject('ALL4096_DIRECTION_ITERATIONS.json',json_change(lambda a:a[0].__setitem__('vertex_state_id','xor_10__forbidden')))
 def test_original_iteration_gap_forged_blocks(self):self.reject('ALL4096_DIRECTION_ITERATIONS.json',json_change(lambda a:a[-1].__setitem__('FW_gap_before',.5)))
 def test_direction_common_flag_recomputed(self):self.reject('DIRECTION_DIAGNOSTICS.json',json_change(lambda a:a.__setitem__('computed_strict_common_response_descent',not a['computed_strict_common_response_descent'])))
 def test_direction_weights_forged_blocks(self):self.reject('DIRECTION_DIAGNOSTICS.json',json_change(lambda a:a['normalization_weights'].__setitem__(0,.99)))
 def test_actual_FP64_point_changed_rebound_receipts_blocks(self):self.reject('plus_1e-6/ACTUAL_TORCH_FP64_POINTS.npz',array_change(lambda a:a[IDS[0]].__setitem__(0,.1)))
 def test_actual_point_hash_forged_blocks(self):self.reject('plus_1e-6/ACTUAL_TORCH_FP64_POINT_BINDINGS.json',json_change(lambda a:a[0].__setitem__('actual_Torch_FP64_sha256','0'*64)))
 def test_actual_point_counter_changed_blocks(self):self.reject('plus_1e-6/ACTUAL_FP64_POINTS_SEALED_BEFORE_MEASUREMENTS.json',json_change(lambda a:a.__setitem__('model_forwards_so_far',340)))
 def test_probe_head_changed_blocks(self):self.reject('plus_5e-6/HEAD.npz',array_change(lambda a:a['P'].__setitem__((0,0,0),.1)))
 def test_native_point_changed_blocks(self):self.reject('plus_1e-5/PHYSICAL_FP32_POINTS.npz',array_change(lambda a:a[IDS[0]].__setitem__(0,.1)))
 def test_all6_before_probe_receipt_forgery_blocks(self):self.reject('ALL6_PROBES_SEALED_BEFORE_FIRST_PROBE.json',json_change(lambda a:a.__setitem__('all6_before_first_probe',False)))
 def test_missing_or_permuted_state_records_block(self):
  self.reject('ALL1848_RECORDS.jsonl',self.rows(lambda a:a.pop(84)));self.reject('ALL1848_RECORDS.jsonl',self.rows(lambda a:a.__setitem__(slice(84,86),a[84:86][::-1])))
 def test_gradient_nonfinite_blocks(self):self.reject('ALL1848_RECORDS.jsonl',self.rows(lambda a:a[84]['response_gradient'][0][0].__setitem__(0,float('nan'))))
 def test_recorded_loss_recomputed(self):self.reject('ALL1848_RECORDS.jsonl',self.rows(lambda a:a[84]['objective_parts'].__setitem__('read_loss',.5)))
 def test_native_gate_forgery_recomputed(self):self.reject('ALL1848_RECORDS.jsonl',self.rows(lambda a:a[0]['qualification'].__setitem__('complete_pass',not a[0]['qualification']['complete_pass'])))
 def test_FP64_replay_forgery_recomputed(self):self.reject('ALL1848_RECORDS.jsonl',self.rows(lambda a:a[84]['C163_final_FP64_replay'].__setitem__('observed_Linf',.1)))
 def test_reported_map_error_recomputed(self):self.reject('ALL1848_RECORDS.jsonl',self.rows(lambda a:a[84].__setitem__('map_Linf',1e-13)))
 def test_scalar_FD_sign_and_scale_forgery_recomputed(self):self.reject('ALL252_CENTRAL_DIFFERENCE_CHECKS.json',json_change(lambda a:a[0]['checks']['response'].__setitem__('central_difference',123.)))
 def test_native_repeat_omission_blocks(self):self.reject('ALL588_NATIVE_REPEAT_CHECKS.json',json_change(lambda a:a.pop()))
 def test_parent_pass_count_forgery_blocks(self):self.reject('SUMMARY.json',json_change(lambda a:a['diagnostics']['native_summaries']['anchor']['0'].__setitem__('passed',84)))
 def test_no_models_or_solvers_guard(self):
  with archive155.prior.no_models(),safe.no_solvers():
   with self.assertRaises(ValueError):__import__('torch')
   with self.assertRaises(RuntimeError):np.linalg.solve(np.eye(2),np.ones(2))
 def test_unsafe_duplicate_paths_rejected(self):
  for name,names in [('unsafe',['../bad']),('duplicate',['same','same'])]:
   p=self.root/(name+'.zip')
   with zipfile.ZipFile(p,'w') as z:
    for n in names:z.writestr(n,b'x')
   with self.assertRaises(ValueError):Archive(p)
 def test_independent_replay_near_opposing_and_zero_gradients(self):
  for mode in ('opposed','nearly_opposed','zero_row','empty'):
   p=np.zeros((3,2,3));rr=[]
   for i,sid in enumerate(IDS):
    g=np.ones_like(p) if i==0 else -np.ones_like(p)
    if mode=='nearly_opposed' and i==1:g[0,0,0]+=1e-8
    if mode=='zero_row' and i==1:g*=0
    rr.append(dict(state_id=sid,objective_parts=dict(response_loss=1. if i<2 and mode!='empty' else 0.),response_gradient=g.tolist(),guard_gradient=np.zeros_like(p).tolist()))
   d,ds,h=parent.direction(rr,p);expected,diag,trace,contracts=replay_direction(rr,p,d,h);compare(expected,ds);self.assertFalse(diag['optimality_or_infeasibility_certified']);self.assertEqual(len(contracts),84)
if __name__=='__main__':unittest.main(verbosity=2)
