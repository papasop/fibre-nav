"""Pure NumPy synthetic archive/Adam/gate/boundary tests; no model imports."""
import copy,io,tempfile,unittest,zipfile
from unittest.mock import patch
from core162 import *
from audit162 import audit
from worker162 import run
from fixture162 import fixture,changed,json_change,array_change
class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);cls.paths,cls.c,cls.ctx=fixture(cls.root/'fixture')
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup()
 def reject(self,name,edit,rebind=True):
  z=changed(self.paths['C161'],self.root/(self._testMethodName+'.zip'),name,edit,rebind)
  with contextlib.redirect_stdout(io.StringIO()),self.assertRaises((ValueError,RuntimeError,KeyError)):audit(z,copy.deepcopy(self.ctx),self.root/(self._testMethodName+'_out'),self.c)
 def edit_rows(self,fn):
  def edit(raw):rr=[clean_json(line) for line in raw.splitlines()];fn(rr);return ('\n'.join(json.dumps(r,allow_nan=True) for r in rr)+'\n').encode()
  return edit
 def test_full1092_archive_worker_no_Torch_fit_solver_or_model(self):
  with contextlib.redirect_stdout(io.StringIO()),archive155.prior.no_models(),safe.no_solvers():s=run(self.paths,self.root/'complete',self.c,lambda *a:copy.deepcopy(self.ctx))
  self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['training_records_audited'],672);self.assertEqual(s['training_updates_audited'],8);self.assertEqual(s['native_endpoint_records_audited'],336);self.assertEqual(s['actual_FP32_proposals_rehashed'],168);self.assertEqual(s['NumPy_continuous_map_bindings_reconstructed'],840);self.assertFalse(s['LC2_closed']);self.assertFalse(s['gradients_independently_recomputed']);self.assertFalse(s['Torch_FP64_forward_point_bytes_independently_rehashed']);self.assertFalse(s['endpoint_KL_independently_recomputed']);self.assertFalse(s['all_original_parent_archives_independently_bound']);self.assertFalse(any(n in sys.modules for n in ('torch','transformers','peft')));self.assertIn('CONSTRAINT_FAILURE',s['parent_decision']);self.assertEqual(s['new_model_forwards'],0);self.assertTrue(s['diagnostics']['all672_transitions_retained'])
 def test_wrong_whole_hash_blocks_before_parent_loader(self):
  cc=copy.deepcopy(self.c);cc['pins']['C161']['sha256']='0'*64
  with contextlib.redirect_stdout(io.StringIO()),patch('worker162.audit',side_effect=AssertionError('Must not inspect payload')):s=run(self.paths,self.root/'wrongwhole',cc,lambda *a:(_ for _ in ()).throw(AssertionError('Must not load parents')))
  self.assertFalse(s['complete']);self.assertEqual(s['new_model_forwards'],0)
 def test_source_change_blocks_even_rebound_manifest(self):self.reject('source/core161.py',lambda b:b+b'\n# changed\n')
 def test_fixed_schedule_threshold_change_blocks(self):self.reject('DESIGN_FROZEN_BEFORE_MODEL.json',json_change(lambda a:a['optimizer'].__setitem__('learning_rate',1e-4)))
 def test_actual_fixed_final_head_change_blocks(self):self.reject('FIXED_FINAL_HEAD_STEP8.npz',array_change(lambda a:a['P'].__setitem__((-1,-1,-1),a['P'][-1,-1,-1]+.001)))
 def test_actual_before_head_change_even_rebound_seal(self):self.reject('step_01_before/HEAD.npz',array_change(lambda a:a['P'].__setitem__((-1,-1,-1),a['P'][-1,-1,-1]+.001)))
 def test_actual_FP32_point_change_even_rebound_seal(self):self.reject('final_step8/PHYSICAL_FP32_POINTS.npz',array_change(lambda a:a[IDS[0]].__setitem__(0,a[IDS[0]][0]+.001)))
 def test_NumPy_point_binding_change_even_rebound_seal(self):self.reject('step_02_before/POINT_BINDINGS.json',json_change(lambda a:a[0].__setitem__('continuous_FP64_sha256','0'*64)))
 def test_aggregate_gradient_change_blocks(self):self.reject('UPDATE_STEP_03.npz',array_change(lambda a:a['gradient'].__setitem__((0,0,0),a['gradient'][0,0,0]+.1)))
 def test_moment_change_blocks(self):self.reject('UPDATE_STEP_05.npz',array_change(lambda a:a['m_after'].__setitem__((0,0,0),a['m_after'][0,0,0]+.01)))
 def test_after_update_change_blocks(self):self.reject('UPDATE_STEP_08.npz',array_change(lambda a:a['after'].__setitem__((0,0,0),a['after'][0,0,0]+.001)))
 def test_forged_history_summary_recomputed(self):self.reject('ALL8_UPDATE_HISTORY.json',json_change(lambda a:a[3]['aggregation'].__setitem__('objective',1.)))
 def test_missing_state_record_blocks(self):self.reject('ALL1092_RECORDS.jsonl',self.edit_rows(lambda a:a.pop(84)))
 def test_permuted_state_records_block(self):self.reject('ALL1092_RECORDS.jsonl',self.edit_rows(lambda a:a.__setitem__(slice(84,86),a[84:86][::-1])))
 def test_nonfinite_recorded_gradient_blocks(self):self.reject('ALL1092_RECORDS.jsonl',self.edit_rows(lambda a:a[84]['response_gradient'][0][0].__setitem__(0,float('nan'))))
 def test_forged_loss_parts_recomputed(self):self.reject('ALL1092_RECORDS.jsonl',self.edit_rows(lambda a:a[84]['objective_parts'].__setitem__('read_loss',.25)))
 def test_forged_native_gate_recomputed(self):self.reject('ALL1092_RECORDS.jsonl',self.edit_rows(lambda a:a[0]['qualification'].__setitem__('complete_pass',True)))
 def test_native_observable_change_recomputed(self):self.reject('ALL1092_RECORDS.jsonl',self.edit_rows(lambda a:a[0]['metrics']['observed'].__setitem__(0,.0001)))
 def test_parent_pass_count_forgery_blocks(self):self.reject('SUMMARY.json',json_change(lambda a:a['native_final_round0'].__setitem__('passed',84)))
 def test_final_freeze_receipt_cannot_choose_intermediate(self):self.reject('FINAL_HEAD_FROZEN_BEFORE_EVALUATION.json',json_change(lambda a:a.__setitem__('step',3)))
 def test_reference_substitution_blocks(self):self.reject('FIXED_ARCHIVED_OBJECTIVE_REFERENCES.npz',array_change(lambda a:a['response'].__setitem__(0,.0001)))
 def test_native_repeat_record_cannot_be_omitted(self):self.reject('ALL168_NATIVE_REPEAT_CHECKS.json',json_change(lambda a:a.pop()))
 def test_independent_first_Adam_sign_scale_and_zero(self):
  p=np.zeros((3,2,6));g=np.linspace(-2,2,p.size).reshape(p.shape);z=p.copy();a,_,_,d=adam(p,g,z,z,1);clipped=g/max(1.,np.linalg.norm(g));np.testing.assert_allclose(a,-1e-5*clipped/(abs(clipped)+1e-8),rtol=1e-14,atol=1e-18);self.assertLessEqual(d['clipped_gradient_L2'],1.+1e-14);np.testing.assert_array_equal(adam(p,z,z,z,1)[0],p)
 def test_independent_Adam_projection_nonfinite_and_steps(self):
  p=np.full((3,2,6),.125);z=np.zeros_like(p);new,_,_,d=adam(p,-np.ones_like(p),z,z,1);np.testing.assert_array_equal(new,p);self.assertEqual(d['projected_coordinates'],36)
  for step in (0,9,True):
   with self.assertRaises(ValueError):adam(p,z,z,z,step)
  with self.assertRaises(ValueError):adam(p,z+np.nan,z,z,1)
 def test_full84_tie_weight_and_missing_denominator(self):
  rr=[dict(state_id=sid,objective_parts=dict(response_loss=1. if i<2 else 0.,read_loss=0.,KL_loss=0.,radius_loss=0.,regularizer=0.),response_gradient=np.ones((3,2,6)).tolist(),guard_gradient=np.zeros((3,2,6)).tolist()) for i,sid in enumerate(IDS)];g,a=aggregate(rr,(3,2,6));self.assertEqual(a['exact_max_tie_count'],2);self.assertAlmostEqual(a['response_weights'][0],.45+.1/84);np.testing.assert_allclose(g,1.)
  with self.assertRaises(ValueError):objective(rr[:-1])
 def test_response_classification_and_signed_slack_boundary(self):
  a=np.full(22,.0019);b=a.copy();self.assertEqual(response_class(a,b),'BOTH_RESPONSE_PASS');b[0]=.0021;self.assertEqual(response_class(a,b),'NATIVE_RESPONSE_ONLY_FAIL');a[0]=.0022;self.assertEqual(response_class(a,b),'BOTH_RESPONSE_FAIL');b[0]=.002;self.assertEqual(response_class(a,b),'FP64_RESPONSE_ONLY_FAIL')
 def test_cosine_zero_is_not_evidence_of_no_conflict(self):self.assertIsNone(cosine(np.zeros(4),np.ones(4)));self.assertAlmostEqual(cosine(np.ones(4),-np.ones(4)),-1.)
 def test_guards_block_model_import_and_linear_solvers(self):
  with archive155.prior.no_models(),safe.no_solvers():
   with self.assertRaises(ValueError):__import__('torch')
   with self.assertRaises(RuntimeError):np.linalg.solve(np.eye(2),np.ones(2))
 def test_unsafe_and_duplicate_archive_paths_block(self):
  for name,names in [('unsafe',['../bad']),('duplicate',['same','same'])]:
   p=self.root/(name+'.zip')
   with zipfile.ZipFile(p,'w') as z:
    for n in names:z.writestr(n,b'x')
   with self.assertRaises(ValueError):Archive(p)
if __name__=='__main__':unittest.main(verbosity=2)
