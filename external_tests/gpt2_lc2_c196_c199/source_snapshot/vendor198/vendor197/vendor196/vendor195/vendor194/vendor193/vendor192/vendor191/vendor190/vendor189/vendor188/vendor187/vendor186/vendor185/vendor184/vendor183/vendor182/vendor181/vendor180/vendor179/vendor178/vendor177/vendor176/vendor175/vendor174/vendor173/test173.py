"""Synthetic archive, exact arithmetic, coordinate descriptions and stop gates."""
import copy,contextlib,io,tempfile,unittest,zipfile
from core173 import *
from fixture173 import fixture,pack
from audit173 import audit
from diagnostics173 import diagnostics,pair_description,maxima
from worker173 import run
class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);cls.paths,cls.c,cls.ctx,cls.src=fixture(cls.root/'fixture')
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup()
 def changed(self,fn):
  with zipfile.ZipFile(self.paths['C172']) as z:m={n:z.read(n) for n in z.namelist()}
  fn(m);m['RESULT_MANIFEST.json']=(json.dumps({n:hashlib.sha256(v).hexdigest() for n,v in m.items() if n!='RESULT_MANIFEST.json'})+'\n').encode();p=self.root/(self._testMethodName+'.zip')
  with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
   for n,v in m.items():z.writestr(n,v)
  return p
 def j(self,m,n,fn):v=clean_json(m[n]);fn(v);m[n]=(json.dumps(v,ensure_ascii=False,allow_nan=False)+'\n').encode()
 def arr(self,m,n,fn):
  b=io.BytesIO(m[n])
  with np.load(b,allow_pickle=False) as z:v={k:z[k].copy() for k in z.files}
  fn(v);b=io.BytesIO();np.savez_compressed(b,**v);m[n]=b.getvalue()
 def reject(self,fn):
  p=self.changed(fn)
  with archive155.prior.no_models(),safe.no_solvers():
   with self.assertRaises((ValueError,KeyError,AssertionError)):audit(p,copy.deepcopy(self.ctx),self.root/(self._testMethodName+'_out'),self.c)
 def test_full_worker_all18_1848_records_no_model_or_solver(self):
  with contextlib.redirect_stdout(io.StringIO()):s=run(self.paths,self.root/'complete',self.c,lambda *a:copy.deepcopy(self.ctx))
  self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['ordered_records_audited'],1848);self.assertEqual(s['actual_FP64_point_arrays_rehashed'],672);self.assertEqual(s['native_endpoint_records_audited'],1176);self.assertEqual(s['recorded_LP_calls_reconstructed'],1);self.assertIsNone(s['original_solver_internal_linear_system_count']);self.assertEqual(s['new_linear_program_calls'],0);self.assertEqual(s['linear_systems_solved'],0);self.assertEqual(s['new_model_forwards'],0);self.assertFalse(s['all_eighteen_originals_independently_bound']);self.assertFalse(s['LC2_closed']);self.assertFalse(s['candidate_selected']);self.assertEqual(s['coordinate_diagnostics']['coordinate_transition_rows'],3024);self.assertEqual(s['coordinate_diagnostics']['coordinate_central_secant_rows'],1512);self.assertEqual(s['coordinate_diagnostics']['paired_state_rows'],504);self.assertTrue((self.root/'complete/diagnostics/ALL3024_RESPONSE_COORDINATE_TRANSITIONS.csv').is_file());self.assertFalse(any(n in sys.modules for n in ('torch','scipy','transformers','peft')))
 def test_wrong_whole_hash_stops_before_context_loader(self):
  c=copy.deepcopy(self.c);c['pins']['C172']['sha256']='0'*64
  with contextlib.redirect_stdout(io.StringIO()):s=run(self.paths,self.root/'wrong_hash',c,lambda *a:(_ for _ in ()).throw(AssertionError('Loader must not run')))
  self.assertFalse(s['complete']);self.assertEqual(s['new_model_forwards'],0)
 def test_source_change_even_rebound_manifest_blocks(self):self.reject(lambda m:m.__setitem__('source/core172.py',m['source/core172.py']+b'\n# changed\n'))
 def test_parent_decision_not_reclassified(self):self.reject(lambda m:self.j(m,'SUMMARY.json',lambda v:v.__setitem__('decision','LC2_PASSED')))
 def test_incomplete_parent_blocks(self):self.reject(lambda m:self.j(m,'SUMMARY.json',lambda v:v.__setitem__('complete',False)))
 def test_original_LP_call_cannot_be_hidden(self):self.reject(lambda m:self.j(m,'SUMMARY.json',lambda v:v.__setitem__('new_linear_program_calls',0)))
 def test_unknown_parent_internal_solves_must_remain_null(self):self.reject(lambda m:self.j(m,'SUMMARY.json',lambda v:v.__setitem__('linear_systems_solved',0)))
 def test_frozen_anchor_head_change_blocks(self):self.reject(lambda m:self.arr(m,'ANCHOR_C170_FIXED_FINAL_HEAD.npz',lambda v:v['P'].__setitem__((0,0,0),.01)))
 def test_reference_recenter_blocks(self):self.reject(lambda m:self.arr(m,'FIXED_ARCHIVED_OBJECTIVE_REFERENCES.npz',lambda v:v['response'].__setitem__(0,.001)))
 def test_step_grid_change_blocks(self):self.reject(lambda m:self.j(m,'FIXED6_STEP_PANEL_BEFORE_PROBES.json',lambda v:v['positive_steps'].__setitem__(1,.0002)))
 def test_direction_change_blocks(self):self.reject(lambda m:self.arr(m,'ONE_FRESH_FIXED_DIRECTION.npz',lambda v:v['direction'].__setitem__((0,0,0),.1)))
 def test_gradient_duplicate_change_blocks(self):self.reject(lambda m:self.j(m,'ALL84_FRESH_GRADIENT_RECORDS.json',lambda v:v[0]['response_gradient'][0][0].__setitem__(0,123.)))
 def test_current_free_mask_change_blocks(self):self.reject(lambda m:self.arr(m,'CURRENT_ANCHOR_LINEAR_PROBLEM.npz',lambda v:v['free'].__setitem__(0,False)))
 def test_solver_raw_vector_mapping_forgery_blocks(self):self.reject(lambda m:self.j(m,'SOLVER_DIAGNOSTIC.json',lambda v:v.__setitem__('raw_claimed_margin',99.)))
 def test_exact_witness_fraction_change_blocks(self):self.reject(lambda m:self.j(m,'EXACT_RECORDED_DIRECTION_CHECK.json',lambda v:v['normalized_margin_lower'].__setitem__('numerator','999')))
 def test_primary_record_missing_blocks(self):
  def edit(m):rr=m['ALL1848_RECORDS.jsonl'].splitlines();m['ALL1848_RECORDS.jsonl']=b'\n'.join(rr[:-1])+b'\n'
  self.reject(edit)
 def test_primary_record_order_change_blocks(self):
  def edit(m):rr=m['ALL1848_RECORDS.jsonl'].splitlines();rr[0],rr[1]=rr[1],rr[0];m['ALL1848_RECORDS.jsonl']=b'\n'.join(rr)+b'\n'
  self.reject(edit)
 def test_native_gate_forgery_blocks(self):
  def edit(m):rr=[clean_json(b) for b in m['ALL1848_RECORDS.jsonl'].splitlines()];rr[0]['qualification']['complete_pass']=not rr[0]['qualification']['complete_pass'];m['ALL1848_RECORDS.jsonl']=b'\n'.join(json.dumps(v).encode() for v in rr)+b'\n'
  self.reject(edit)
 def test_nonfinite_gradient_primary_record_blocks(self):
  def edit(m):rr=[clean_json(b) for b in m['ALL1848_RECORDS.jsonl'].splitlines()];rr[84]['response_gradient'][0][0][0]=float('nan');m['ALL1848_RECORDS.jsonl']=b'\n'.join(json.dumps(v).encode() for v in rr)+b'\n'
  self.reject(edit)
 def test_primary_loss_forgery_blocks(self):
  def edit(m):rr=[clean_json(b) for b in m['ALL1848_RECORDS.jsonl'].splitlines()];rr[168]['objective_parts']['response_loss']+=.1;m['ALL1848_RECORDS.jsonl']=b'\n'.join(json.dumps(v).encode() for v in rr)+b'\n'
  self.reject(edit)
 def test_actual_point_changed_even_rebound_local_seal_blocks(self):
  def edit(m):
   n='plus_s3/ACTUAL_TORCH_FP64_POINTS.npz';self.arr(m,n,lambda v:v[IDS[0]].__setitem__(0,.1));self.j(m,'plus_s3/ACTUAL_FP64_POINTS_SEALED_BEFORE_MEASUREMENTS.json',lambda v:v['artifact_seals'].__setitem__('ACTUAL_TORCH_FP64_POINTS.npz',hashlib.sha256(m[n]).hexdigest()))
  self.reject(edit)
 def test_probe_receipt_cannot_claim_zero_backwards(self):self.reject(lambda m:self.j(m,'ALL6_PROBES_SEALED_BEFORE_FIRST_PROBE.json',lambda v:v.__setitem__('backward_calls_so_far',0)))
 def test_missing_native_repeat_blocks(self):self.reject(lambda m:self.j(m,'ALL588_NATIVE_REPEAT_CHECKS.json',lambda v:v.pop()))
 def test_forged_central_difference_blocks(self):self.reject(lambda m:self.j(m,'ALL252_CENTRAL_DIFFERENCES_VS_FRESH_AD.json',lambda v:v[0]['checks']['response'].__setitem__('current_C172_central_difference',999.)))
 def test_summary_statistic_forgery_blocks(self):self.reject(lambda m:self.j(m,'SUMMARY.json',lambda v:v['diagnostics']['all3_positive_arms_active_loss_changes']['plus_s1'].__setitem__('strict_response_loss_decreases',84)))
 def test_signed_actual_head_dot_forgery_blocks(self):self.reject(lambda m:self.j(m,'ALL504_SIGNED_LOSS_CHANGES.json',lambda v:v[0]['actual_binary64_head_change_diagnostic']['response']['exact_recorded_gradient_dot_actual_binary64_head_difference'].__setitem__('numerator','999')))
 def test_switch_with_loss_increase_has_positive_frozen_max_gap(self):
  r=pair_description([.002,.0019,0,0,0,0],[.0018,.0021,0,0,0,0]);self.assertTrue(r['max_coordinate_set_changed']);self.assertTrue(r['disjoint_max_coordinate_sets']);self.assertGreater(r['response_Linf_change'],0);self.assertAlmostEqual(r['actual_max_minus_anchor_max_coordinate_envelope'],.0003);self.assertTrue(r['switch_is_not_causal_explanation'])
 def test_loss_increase_without_switch_is_retained(self):
  r=pair_description([.002,.001,0,0,0,0],[.0021,.001,0,0,0,0]);self.assertFalse(r['max_coordinate_set_changed']);self.assertGreater(r['response_Linf_change'],0);self.assertEqual(r['actual_max_minus_anchor_max_coordinate_envelope'],0)
 def test_sign_flip_without_coordinate_switch_is_separate(self):
  r=pair_description([.002,0,0,0,0,0],[-.0021,0,0,0,0,0]);self.assertFalse(r['max_coordinate_set_changed']);self.assertEqual(r['strict_sign_flip_coordinate_indices'],[0])
 def test_exact_ties_all_coordinates_retained(self):
  r=pair_description([.002,-.002,0,0,0,0],[.001,-.001,0,0,0,0]);self.assertEqual(r['anchor_max_coordinate_indices'],[0,1]);self.assertEqual(r['probe_max_coordinate_indices'],[0,1]);self.assertFalse(r['max_coordinate_set_changed'])
 def test_zero_response_does_not_drop_maximum_ties(self):self.assertEqual(maxima([0.]*6),(0.,list(range(6))))
 def test_tiny_sign_flip_not_lost_to_underflow(self):self.assertEqual(pair_description([1e-300,0,0,0,0,0],[-1e-300,0,0,0,0,0])['strict_sign_flip_coordinate_indices'],[0])
 def test_nonfinite_and_wrong_coordinate_shape_block(self):
  for x in ([0.]*5,[float('nan')]*6):
   with self.assertRaises(ValueError):pair_description(x,[0.]*6)
 def test_empty_active_no_arbitrary_descent_certificate(self):
  p=np.zeros((3,2,6));rr=[dict(state_id=s,objective_parts=dict(response_loss=0.),response_gradient=np.zeros_like(p).tolist(),guard_gradient=np.zeros_like(p).tolist()) for s in IDS];pr=problem(rr,p);c=certificate(pr,np.zeros_like(p));self.assertFalse(c['strict_recorded_common_descent']);self.assertIsNone(c['normalized_margin_lower'])
  with self.assertRaises(ValueError):certificate(pr,np.ones_like(p))
 def test_unsafe_archive_paths_rejected(self):
  p=self.root/'unsafe.zip'
  with zipfile.ZipFile(p,'w') as z:z.writestr('../unsafe',b'x')
  with self.assertRaises(ValueError):Archive(p)
if __name__=='__main__':unittest.main(verbosity=2)
