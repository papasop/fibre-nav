"""Synthetic NumPy archive tests. Every fixture is explicitly non-scientific."""
import copy,io,tempfile,unittest,zipfile,contextlib
from core169 import *
from fixture169 import fixture
from audit169 import audit
from diagnostics169 import diagnostics
from worker169 import run
class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);cls.paths,cls.c,cls.ctx,cls.src=fixture(cls.root/'fixture')
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup()
 def changed(self,name,fn,extra=None):
  with zipfile.ZipFile(self.paths['C168']) as z:m={n:z.read(n) for n in z.namelist()}
  m[name]=fn(m[name])
  if extra:extra(m)
  m['RESULT_MANIFEST.json']=(json.dumps({n:hashlib.sha256(v).hexdigest() for n,v in m.items() if n!='RESULT_MANIFEST.json'})+'\n').encode();p=self.root/(self._testMethodName+'.zip')
  with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
   for n,v in m.items():z.writestr(n,v)
  return p
 def j(self,fn):
  def edit(b):x=clean_json(b);fn(x);return (json.dumps(x,allow_nan=False)+'\n').encode()
  return edit
 def arr(self,fn):
  def edit(b):
   with np.load(io.BytesIO(b),allow_pickle=False) as z:x={n:z[n] for n in z.files}
   fn(x);buf=io.BytesIO();np.savez_compressed(buf,**x);return buf.getvalue()
  return edit
 def reject(self,name,fn,extra=None):
  p=self.changed(name,fn,extra)
  with self.assertRaises((ValueError,KeyError)):audit(p,self.ctx,self.root/(self._testMethodName+'_out'),self.c)
 def test_complete_full_worker_audit_without_model_or_solver(self):
  out=self.root/'full_worker'
  with archive155.prior.no_models(),safe.no_solvers(),contextlib.redirect_stdout(io.StringIO()):s=run(self.paths,out,self.c,lambda *a:copy.deepcopy(self.ctx))
  self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['decision'],DECISION);self.assertFalse(s['all_fourteen_originals_independently_bound']);self.assertFalse(s['exact_C167_recorded_gradient_certificate_reverified']);self.assertEqual(s['ordered_records_audited'],1848);self.assertEqual(s['actual_FP64_point_arrays_rehashed'],672)
  for k,v in no_compute().items():self.assertEqual(s[k],v)
  self.assertEqual(len(clean_json((out/'diagnostics/ALL1512_RESPONSE_COORDINATE_SECANTS.json').read_bytes())),1512);self.assertFalse(any(k in sys.modules for k in ('torch','transformers','scipy')))
 def test_parent_summary_forgery_blocks(self):self.reject('SUMMARY.json',self.j(lambda s:s['diagnostics']['native_summaries']['anchor']['0'].__setitem__('passed',84)))
 def test_incomplete_parent_blocks(self):self.reject('SUMMARY.json',self.j(lambda s:s.__setitem__('complete',False)))
 def test_parent_decision_reclassification_blocks(self):self.reject('SUMMARY.json',self.j(lambda s:s.__setitem__('decision','LC2_CLOSED')))
 def test_actual_anchor_changed_blocks(self):self.reject('ANCHOR_C163_FIXED_FINAL_HEAD.npz',self.arr(lambda x:x['P'].__setitem__((0,0,0),.01)))
 def test_actual_direction_changed_blocks(self):self.reject('ONE_FIXED_C167_DIRECTION.npz',self.arr(lambda x:x['direction'].__setitem__((0,0,0),-.5)))
 def test_fixed_steps_changed_blocks(self):self.reject('FIXED6_STEP_PANEL_BEFORE_MODEL.json',self.j(lambda s:s['positive_steps'].__setitem__(2,.002)))
 def test_source_change_even_rebound_manifest_blocks(self):self.reject('source/core168.py',lambda b:b+b'\n# changed\n')
 def test_original_input_link_changed_blocks(self):self.reject('ORIGINAL_INPUTS.json',self.j(lambda s:s['C167'].__setitem__('sha256','0'*64)))
 def test_missing_state_blocks(self):self.reject('ALL1848_REQUESTS_FROZEN.json',self.j(lambda s:s.pop()))
 def test_request_order_changed_blocks(self):self.reject('ALL1848_REQUESTS_FROZEN.json',self.j(lambda s:s.reverse()))
 def test_native_gate_forgery_blocks(self):self.reject('ALL1176_NATIVE_ENDPOINTS.json',self.j(lambda s:s['anchor__round0'][0]['qualification'].__setitem__('complete_pass',True)))
 def test_duplicate_probe_loss_forgery_blocks(self):self.reject('ALL6_FP64_PROBE_RECORDS.json',self.j(lambda s:s['plus_s1'][0]['objective_parts'].__setitem__('response_loss',0.)))
 def test_central_difference_forgery_blocks(self):self.reject('ALL252_CENTRAL_DIFFERENCES_VS_ARCHIVED_AD.json',self.j(lambda s:s[0]['checks']['response'].__setitem__('difference',999.)))
 def test_one_sided_change_forgery_blocks(self):self.reject('ALL504_SIGNED_LOSS_CHANGES.json',self.j(lambda s:s[0].__setitem__('observed_response_loss_change',999.)))
 def test_repeat_check_forgery_blocks(self):self.reject('ALL588_NATIVE_REPEAT_CHECKS.json',self.j(lambda s:s[0].__setitem__('within_replay_tolerance',False)))
 def test_pre_first_state_forward_count_blocks(self):self.reject('ALL6_PROBES_SEALED_BEFORE_FIRST_STATE.json',self.j(lambda s:s.__setitem__('model_forwards_so_far',4)))
 def test_actual_point_changed_with_rebound_local_seals_blocks(self):
  def extra(m):
   key='plus_s1/ACTUAL_FP64_POINTS_SEALED_BEFORE_MEASUREMENTS.json';r=clean_json(m[key]);r['artifact_seals']['ACTUAL_TORCH_FP64_POINTS.npz']=hashlib.sha256(m['plus_s1/ACTUAL_TORCH_FP64_POINTS.npz']).hexdigest();m[key]=(json.dumps(r)+'\n').encode()
  self.reject('plus_s1/ACTUAL_TORCH_FP64_POINTS.npz',self.arr(lambda x:x[IDS[0]].__setitem__(0,.1)),extra)
 def test_nonfinite_actual_point_blocks(self):self.reject('plus_s1/ACTUAL_TORCH_FP64_POINTS.npz',self.arr(lambda x:x[IDS[0]].__setitem__(0,np.nan)))
 def test_primary_reference_changed_blocks(self):self.reject('model_runtime/REFERENCES_FROZEN_BEFORE_FOCUS.json',self.j(lambda s:s['references']['archived_FP32'].__setitem__(0,.001)))
 def test_wrong_whole_hash_blocks_before_parent_loader(self):
  c=copy.deepcopy(self.c);c['pins']['C168']['sha256']='0'*64
  with contextlib.redirect_stdout(io.StringIO()):s=run(self.paths,self.root/'wronghash',c,lambda *a:(_ for _ in ()).throw(AssertionError('No parent read before hash')))
  self.assertFalse(s['complete']);self.assertEqual(s['new_model_forwards'],0);self.assertFalse(s['LC2_closed'])
 def test_failure_rows_retained_and_no_step_selected(self):
  a=audit(self.paths['C168'],self.ctx,self.root/'retained_audit',self.c);d=diagnostics(a,self.root/'retained_diag');self.assertTrue(d['no_new_head_or_step_selected']);self.assertEqual(len(d['all6_arm_statistics']),6);self.assertEqual(d['all6_arm_statistics']['plus_s1']['native_by_round']['0']['passed'],79);self.assertEqual(len(d['all6_arm_statistics']['plus_s1']['native_by_round']['0']['failed_ids']),5)
 def test_nonzero_secant_disagreement_does_not_claim_AD_error(self):
  a=audit(self.paths['C168'],self.ctx,self.root/'secant_audit',self.c);d=diagnostics(a,self.root/'secant_diag');self.assertFalse(d['causal_explanation_established']);self.assertFalse(d['curvature_or_systematic_error_bound_certified']);self.assertTrue(d['all_secants_and_linear_residuals_descriptive_only'])
 def test_wrong_recorded_loss_blocks_after_rebinding_result_manifest(self):
  def edit(b):
   rows=[clean_json(x) for x in b.splitlines()];row=next(x for x in rows if x['stage']=='FP64_probe');row['objective_parts']['response_loss']+=1.;return ('\n'.join(json.dumps(x) for x in rows)+'\n').encode()
  self.reject('ALL1848_RECORDS.jsonl',edit)
 def test_wrong_committed_head_blocks_even_with_result_manifest(self):
  self.reject('plus_s3/HEAD.npz',self.arr(lambda x:x['P'].__setitem__((0,0,0),.02)))
 def test_new_backward_count_cannot_be_hidden_in_parent(self):
  self.reject('SUMMARY.json',self.j(lambda s:s.__setitem__('new_backward_calls',1)))
 def test_cross_run_baseline_difference_is_not_a_recenter_permission(self):
  r=clean_json((self.src/'model_runtime/REFERENCES_FROZEN_BEFORE_FOCUS.json').read_bytes());self.assertEqual(r['references']['archived_FP32'],[0.]*6);self.assertEqual(r['references']['archived_FP64'],[0.]*6)
 def test_unsafe_zip_path_rejected(self):
  p=self.root/'unsafe.zip'
  with zipfile.ZipFile(p,'w') as z:z.writestr('../escape.json','{}')
  with self.assertRaises(ValueError):
   with contextlib.closing(Archive(p)) as z:z.verify_manifest()
if __name__=='__main__':unittest.main(verbosity=2)
