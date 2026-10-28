"""Synthetic archive/update/certificate/trajectory tests; no model or solver import."""
import copy,io,tempfile,unittest,zipfile,contextlib
from core189 import *
from fixture189 import fixture
from audit189 import audit
from diagnostics189 import diagnostics
from worker189 import run
class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);cls.paths,cls.c,cls.ctx,cls.src=fixture(cls.root/'fixture')
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup()
 def j(self,fn):
  def edit(b):x=clean_json(b);fn(x);return (json.dumps(x,allow_nan=False)+'\n').encode()
  return edit
 def arr(self,fn):
  def edit(b):
   with np.load(io.BytesIO(b),allow_pickle=False) as z:x={n:z[n] for n in z.files}
   fn(x);buf=io.BytesIO();np.savez_compressed(buf,**x);return buf.getvalue()
  return edit
 def changed(self,name,fn,extra=None):
  with zipfile.ZipFile(self.paths['C188']) as z:m={n:z.read(n) for n in z.namelist()}
  m[name]=fn(m[name])
  if extra:extra(m)
  m['RESULT_MANIFEST.json']=(json.dumps({n:hashlib.sha256(v).hexdigest() for n,v in m.items() if n!='RESULT_MANIFEST.json'})+'\n').encode();p=self.root/(self._testMethodName+'.zip')
  with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
   for n,v in m.items():z.writestr(n,v)
  return p
 def reject(self,name,fn,extra=None):
  p=self.changed(name,fn,extra)
  with contextlib.redirect_stdout(io.StringIO()),self.assertRaises((ValueError,KeyError)):audit(p,self.ctx,self.root/(self._testMethodName+'_out'),self.c)
 def test_complete_worker_all8_updates_840_points_without_model_or_solver(self):
  out=self.root/'full_worker'
  with archive155.prior.no_models(),safe.no_solvers(),contextlib.redirect_stdout(io.StringIO()):s=run(self.paths,out,self.c,lambda *a:copy.deepcopy(self.ctx))
  self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['decision'],DECISION);self.assertFalse(s['all_thirty_four_originals_independently_bound']);self.assertEqual(s['training_updates_audited'],8);self.assertEqual(s['ordered_records_audited'],1176);self.assertEqual(s['recorded_LP_calls_reconstructed'],8);self.assertEqual(s['actual_FP64_point_arrays_rehashed'],840);self.assertEqual(s['exact_actual_head_checks_reconstructed'],8);self.assertEqual(s['exact_signed_certificate_rows_reconstructed'],16128);self.assertEqual(s['LP_signed_constraint_rows_reconstructed'],8064);self.assertEqual(s['response_coordinate_affine_rows_recomputed'],4032);self.assertIsNone(s['original_solver_internal_linear_system_count']);self.assertEqual(s['diagnostics']['paired_native_state_outcomes_recomputed'],168);parent=clean_json((self.src/'SUMMARY.json').read_bytes());self.assertEqual(s['diagnostics']['full_native_outcomes_by_round']['0']['rescued'],parent['paired_by_round']['0']['rescued']);self.assertEqual(s['diagnostics']['full_native_outcomes_by_round']['0']['destroyed'],parent['paired_by_round']['0']['destroyed']);self.assertEqual(s['diagnostics']['remaining_failure_trajectories_exported'],len(parent['native_final_round0']['failed_ids']));self.assertEqual(s['cross_run_anchor_replay']['native_pairs'],168);self.assertEqual(s['cross_run_anchor_replay']['FP64_pairs'],84);self.assertEqual(s['endpoint_precision_reconstruction']['original_precision_rows_independently_reconstructed'],336);self.assertEqual(s['diagnostics']['activity_set_dynamics']['activity_transition_rows'],672)
  for k,v in no_compute().items():self.assertEqual(s[k],v)
  self.assertEqual(len(clean_json((out/'diagnostics/ALL672_STATE_TRANSITION_ROWS.json').read_bytes())),672);self.assertEqual(len(clean_json((out/'diagnostics/ALL84_STATE_TRAJECTORIES.json').read_bytes())),84);self.assertFalse(any(n in sys.modules for n in ('torch','scipy','transformers')))
 def test_incomplete_parent_blocks(self):self.reject('SUMMARY.json',self.j(lambda s:s.__setitem__('complete',False)))
 def test_parent_decision_reclassification_blocks(self):self.reject('SUMMARY.json',self.j(lambda s:s.__setitem__('decision','LC2_CLOSED')))
 def test_cumulative_continuation_steps_cannot_be_hidden(self):self.reject('SUMMARY.json',self.j(lambda s:s.__setitem__('cumulative_C176_C178_C180_C182_C184_C186_C188_training_steps',8)))
 def test_original_C187_audit_decision_tamper_blocks(self):self.reject('SUMMARY.json',self.j(lambda s:s.__setitem__('C187_parent_decision','forged')))
 def test_forged_rescued_count_in_paired_summary_blocks(self):self.reject('SUMMARY.json',self.j(lambda s:s['paired_by_round']['0'].__setitem__('rescued',s['paired_by_round']['0']['rescued']+1)))
 def test_older_C170_head_cannot_replace_C186_continuation_anchor(self):
  ctx=copy.deepcopy(self.ctx);ctx['C186_final_head']=ctx['C178_final_head'].copy()
  with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(ValueError):audit(self.paths['C188'],ctx,self.root/'old_anchor_rejected',self.c)
 def test_source_tamper_even_rebound_manifest_blocks(self):self.reject('source/core188.py',lambda b:b+b'\n# changed\n')
 def test_fixed_initial_head_change_blocks(self):self.reject('INITIAL_C186_FIXED_FINAL_HEAD.npz',self.arr(lambda x:x['P'].__setitem__((0,0,0),.001)))
 def test_fixed_final_head_not_step8_blocks(self):self.reject('FIXED_FINAL_HEAD_STEP8.npz',self.arr(lambda x:x['P'].__setitem__((0,0,0),0.)))
 def test_update_sign_scale_change_blocks(self):self.reject('UPDATE_STEP_01.npz',self.arr(lambda x:x['after'].__setitem__((0,0,0),.001)))
 def test_missing_training_request_blocks(self):self.reject('ALL1176_REQUESTS_FROZEN.json',self.j(lambda x:x.pop()))
 def test_reordered_state_request_blocks(self):self.reject('ALL1176_REQUESTS_FROZEN.json',self.j(lambda x:x.reverse()))
 def test_wrong_source_ancestry_blocks(self):self.reject('ORIGINAL_INPUTS.json',self.j(lambda x:x['C175'].__setitem__('sha256','0'*64)))
 def test_gradient_duplicate_changed_blocks(self):self.reject('step_01_before/ALL84_CURRENT_GRADIENT_ROWS.json',self.j(lambda x:x[0]['response_gradient'][0][0].__setitem__(0,.5)))
 def test_current_active_registry_changed_blocks(self):self.reject('step_01_before/CURRENT_RECORDED_LINEAR_PROBLEM.npz',self.arr(lambda x:x.__setitem__('active',x['active'][1:])))
 def test_current_gradient_tail_changed_blocks(self):self.reject('step_01_before/CURRENT_RECORDED_LINEAR_PROBLEM.npz',self.arr(lambda x:x['guards'].__setitem__((83,35),1.)))
 def test_direction_changed_blocks(self):self.reject('step_01_before/FIXED_STEP_DIRECTION.npz',self.arr(lambda x:x['direction'].__setitem__(0,.5)))
 def test_exact_fraction_forgery_blocks(self):self.reject('step_01_before/EXACT_RECORDED_DIRECTION_CHECK.json',self.j(lambda x:x['minimum_normalized_recorded_cap_slack'].__setitem__('numerator','999')))
 def test_exact_raw_contraction_forgery_blocks(self):self.reject('step_01_before/EXACT_RECORDED_DIRECTION_CHECK.json',self.j(lambda x:x['all1008_signed_rows'][0]['normalized_recorded_cap_slack'].__setitem__('numerator','999')))
 def test_saved_solver_vector_mapping_change_blocks(self):self.reject('step_01_before/SOLVER_DIAGNOSTIC.json',self.j(lambda x:x.__setitem__('raw_claimed_normalized_margin',9.)))
 def test_original_history_update_change_blocks(self):self.reject('ALL8_UPDATE_HISTORY.json',self.j(lambda x:x[0]['update'].__setitem__('step_size',.002)))
 def test_native_gate_forgery_blocks(self):self.reject('ALL336_NATIVE_ENDPOINTS.json',self.j(lambda x:x['native_final0'][0]['qualification'].__setitem__('complete_pass',False)))
 def test_native_repeat_forgery_blocks(self):self.reject('ALL168_NATIVE_REPEAT_CHECKS.json',self.j(lambda x:x[0].__setitem__('within_replay_tolerance',False)))
 def test_reference_recenter_blocks(self):self.reject('model_runtime/REFERENCES_FROZEN_BEFORE_FOCUS.json',self.j(lambda x:x['references']['archived_FP32'].__setitem__(0,.001)))
 def test_actual_point_tamper_even_rebound_local_seal_blocks(self):
  def extra(m):
   n='step_01_before/ACTUAL_FP64_POINTS_SEALED_BEFORE_MEASUREMENTS.json';x=clean_json(m[n]);x['artifact_seals']['ACTUAL_TORCH_FP64_POINTS.npz']=hashlib.sha256(m['step_01_before/ACTUAL_TORCH_FP64_POINTS.npz']).hexdigest();m[n]=(json.dumps(x)+'\n').encode()
  self.reject('step_01_before/ACTUAL_TORCH_FP64_POINTS.npz',self.arr(lambda x:x[IDS[0]].__setitem__(0,.1)),extra)
 def test_nonfinite_actual_point_blocks(self):self.reject('step_01_before/ACTUAL_TORCH_FP64_POINTS.npz',self.arr(lambda x:x[IDS[0]].__setitem__(0,np.nan)))
 def test_saved_actual_forward_count_changed_blocks(self):self.reject('ALL10_ACTUAL_FP64_POINT_SEALS.json',self.j(lambda x:x[0].__setitem__('model_forwards_so_far',3)))
 def test_new_LP_in_parent_count_cannot_be_hidden(self):self.reject('SUMMARY.json',self.j(lambda x:x.__setitem__('new_linear_program_calls',0)))
 def test_unknown_parent_internal_solve_count_retained_null(self):self.reject('SUMMARY.json',self.j(lambda x:x.__setitem__('linear_systems_solved',0)))
 def test_wrong_whole_hash_blocks_before_context_loader(self):
  c=copy.deepcopy(self.c);c['pins']['C188']['sha256']='0'*64
  with contextlib.redirect_stdout(io.StringIO()):s=run(self.paths,self.root/'wronghash',c,lambda *a:(_ for _ in ()).throw(AssertionError('No context read before whole hash')))
  self.assertFalse(s['complete']);self.assertEqual(s['new_model_forwards'],0);self.assertFalse(s['LC2_closed'])
 def test_unsafe_archive_path_blocks(self):
  p=self.root/'unsafe.zip'
  with zipfile.ZipFile(p,'w') as z:z.writestr('../escape','{}')
  with self.assertRaises(ValueError):
   with contextlib.closing(Archive(p)) as z:z.verify_manifest()
 def test_forged_training_loss_in_primary_records_blocks(self):
  def edit(b):
   rows=[clean_json(x) for x in b.splitlines()];r=next(x for x in rows if x['stage']=='training');r['objective_parts']['response_loss']+=1.;return ('\n'.join(json.dumps(x) for x in rows)+'\n').encode()
  self.reject('ALL1176_RECORDS.jsonl',edit)
 def test_actual_head_difference_certificate_forgery_blocks(self):self.reject('step_01_before/EXACT_ACTUAL_HEAD_DIFFERENCE_BEFORE_COMMIT.json',self.j(lambda x:x['all1008_signed_rows'][0]['normalized_recorded_cap_slack'].__setitem__('numerator','999')))
 def test_proposed_head_differs_from_committed_update_blocks(self):self.reject('step_01_before/PROPOSED_NEXT_HEAD_BEFORE_COMMIT.npz',self.arr(lambda x:x['P'].__setitem__((0,0,0),.01)))
 def test_final_FP64_repeat_record_removed_blocks(self):self.reject('ALL84_FINAL_FP64_REPEATS.json',self.j(lambda x:x.pop()))
 def test_coordinate_Jacobian_duplicate_change_blocks(self):self.reject('step_01_before/ALL84_CURRENT_GRADIENT_ROWS.json',self.j(lambda x:x[0]['response_coordinate_gradients'][0][0][0].__setitem__(0,123.)))
 def test_precision_JSON_row_tamper_blocks(self):self.reject('ALL336_ENDPOINT_PRECISION_MARGINS.json',self.j(lambda x:x[0].__setitem__('native_response_slack',.5)))
 def test_precision_summary_tamper_blocks(self):self.reject('ENDPOINT_PRECISION_SUMMARY.json',self.j(lambda x:x['by_head']['final']['0'].__setitem__('FP64_response_only_passed',x['by_head']['final']['0']['FP64_response_only_passed']-1)))
 def test_precision_main_summary_duplicate_tamper_blocks(self):self.reject('SUMMARY.json',self.j(lambda x:x['endpoint_precision_margins'].__setitem__('FP64_full_acceptance_evaluated',True)))
 def test_precision_CSV_tamper_blocks(self):self.reject('ALL336_ENDPOINT_PRECISION_MARGINS.csv',lambda b:b.replace(b'False',b'True',1))
 def test_precision_missing_row_blocks(self):self.reject('ALL336_ENDPOINT_PRECISION_MARGINS.json',self.j(lambda x:x.pop()))
 def test_rational_head_difference_keeps_exact_binary_endpoints(self):
  from diagnostics189 import true_head_delta_dot
  b=np.array([1.]);a=np.array([np.nextafter(1.,2.)]);q=true_head_delta_dot(np.array([3.]),b,a);self.assertEqual(q,Fraction(3,2**52))
if __name__=='__main__':unittest.main(verbosity=2)
