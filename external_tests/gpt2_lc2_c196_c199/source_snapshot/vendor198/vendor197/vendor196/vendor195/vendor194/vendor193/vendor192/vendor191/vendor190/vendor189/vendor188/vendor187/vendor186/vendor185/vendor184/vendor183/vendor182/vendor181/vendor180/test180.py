"""Fixed schedule, counters, exact certificates, no recovery and all-state retention."""
import copy,tempfile,unittest,contextlib,io
from unittest.mock import patch
from core180 import *
from fixture180 import fixture,Analytic,synthetic_solver
from worker180 import run
class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);cls.paths,cls.c,cls.ctx=fixture(cls.root/'inputs')
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup()
 def scenario(self,edit=None,factory=Analytic,backend=synthetic_solver):
  paths=dict(self.paths);c=copy.deepcopy(self.c);ctx=copy.deepcopy(self.ctx);out=self.root/self._testMethodName
  if edit:edit(paths,c,ctx)
  with contextlib.redirect_stdout(io.StringIO()):s=run(paths,out,c,factory,lambda *a:ctx,backend)
  return s,out
 def test_complete_fixed8_all84_all1008_rows_and_native_repeats(self):
  s,out=self.scenario();self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['new_training_steps'],8);self.assertEqual(s['cumulative_C176_C178_C180_training_steps'],24);self.assertEqual(s['new_backward_calls'],5376);self.assertEqual(s['new_Jacobians'],672);self.assertEqual(s['new_response_coordinate_Jacobian_rows'],4032);self.assertEqual(s['new_model_parameter_Jacobians'],0);self.assertEqual(s['new_model_forwards'],2355);self.assertEqual(s['functional_FP64_forwards'],1682);self.assertEqual(s['records_completed'],1176);self.assertEqual(s['actual_FP64_point_tensors_saved'],840);self.assertEqual(s['new_linear_program_calls'],8);self.assertFalse(s['LC2_closed']);self.assertFalse(s['candidate_selected']);self.assertFalse(s['all_twenty_five_originals_independently_bound']);self.assertIsNone(s['linear_systems_solved']);self.assertTrue(s['FP64_final_repeat_consistent']);self.assertEqual(s['endpoint_precision_margins']['rows'],336);self.assertFalse(s['endpoint_precision_margins']['FP64_full_acceptance_evaluated']);self.assertEqual(s['endpoint_precision_margins']['additional_model_forwards'],0)
  hist=clean_json((out/'ALL8_UPDATE_HISTORY.json').read_bytes());self.assertEqual(len(hist),8)
  with np.load(out/'INITIAL_C178_FIXED_FINAL_HEAD.npz',allow_pickle=False) as z:np.testing.assert_array_equal(z['P'],self.ctx['C178_final_head']);self.assertFalse(np.array_equal(z['P'],self.ctx['C176_final_head']))
  for i,h in enumerate(hist,1):
   self.assertEqual(h['step'],i);self.assertEqual(h['update']['step_size'],.001);self.assertTrue(h['all1008_signed_rows_retained']);self.assertGreater(h['exact_actual_head_minimum_cap_slack']['approximate'],0);folder=out/('step_%02d_before'%i);cert=clean_json((folder/'EXACT_ACTUAL_HEAD_DIFFERENCE_BEFORE_COMMIT.json').read_bytes());self.assertEqual(cert['signed_rows'],1008);self.assertEqual(cert['head_difference_mode'],'exact difference of actual binary64 head endpoints');self.assertTrue(cert['strict_recorded_active_margin'])
  with np.load(out/'UPDATE_STEP_08.npz') as z:last=z['after'].copy()
  with np.load(out/'FIXED_FINAL_HEAD_STEP8.npz') as z:np.testing.assert_array_equal(last,z['P'])
  for name in ('native_initial_round0','native_initial_round1','native_final_round0','native_final_round1'):self.assertEqual(s[name]['n_evaluated'],84)
  freeze=clean_json((out/'FINAL_HEAD_FROZEN_BEFORE_EVALUATION.json').read_bytes());self.assertFalse(freeze['selected_from_candidates']);self.assertEqual(freeze['step'],8);self.assertEqual(freeze['record_count_before_final_test'],756)
 def test_registry_fixed_order_complete(self):
  r=requests();self.assertEqual(len(r),1176);self.assertEqual(r[84],dict(stage='training',step=0,state_id=IDS[0]));self.assertEqual(r[-1],dict(stage='native_final1',step=8,state_id=IDS[-1]));self.assertEqual(sum(x['stage']=='training' for x in r),672)
 def test_wrong_C179_whole_hash_stops_before_model(self):
  s,out=self.scenario(lambda p,c,x:c['pins']['C179'].__setitem__('sha256','0'*64));self.assertFalse(s['complete']);self.assertEqual(s['new_model_forwards'],0);self.assertEqual(s['new_training_steps'],0)
 def test_wrong_parent_decision_stops_before_model(self):
  s,out=self.scenario(lambda p,c,x:x['C179_receipt'].__setitem__('decision','forged'));self.assertFalse(s['complete']);self.assertEqual(s['new_model_forwards'],0)
 def test_native_replay_stops_before_any_AD(self):
  s,out=self.scenario(lambda p,c,x:x['C178_final_native']['0'][0]['metrics']['observed'].__setitem__(0,.01));self.assertFalse(s['complete']);self.assertEqual(s['new_backward_calls'],0);self.assertEqual(s['new_training_steps'],0)
 def test_FP64_replay_stops_before_LP_or_update(self):
  s,out=self.scenario(lambda p,c,x:x['C178_final_FP64'][0]['observed'].__setitem__(0,.01));self.assertFalse(s['complete']);self.assertEqual(s['new_backward_calls'],8);self.assertEqual(s['new_Jacobians'],1);self.assertEqual(s['new_linear_program_calls'],0);self.assertEqual(s['new_training_steps'],0)
 def test_nonpositive_witness_stops_no_retry_or_final_head(self):
  class R(dict):
   def __getattr__(self,k):return self[k]
  calls=[]
  def backend(cost,**kw):calls.append(1);return R(x=np.zeros(len(cost)),success=False,status=1)
  s,out=self.scenario(backend=backend);self.assertFalse(s['complete']);self.assertEqual(len(calls),1);self.assertEqual(s['new_training_steps'],0);self.assertEqual(s['new_Jacobians'],84);self.assertEqual(s['new_backward_calls'],672);self.assertFalse(s['final_head_frozen']);self.assertFalse((out/'FIXED_FINAL_HEAD_STEP8.npz').exists());self.assertFalse(s['solver_optimality_or_infeasibility_certified'])
 def test_partial_second_batch_preserves_updates_and_gradient_counts(self):
  def factory(*a):
   rt=Analytic(*a);old=rt.coefficient_record;count=[0]
   def rec(*args,**kw):
    count[0]+=1
    if count[0]==97:raise RuntimeError('injected second-batch partial AD')
    return old(*args,**kw)
   rt.coefficient_record=rec;return rt
  s,out=self.scenario(factory=factory);self.assertFalse(s['complete']);self.assertEqual(s['new_training_steps'],1);self.assertEqual(s['new_backward_calls'],96*8);self.assertEqual(s['new_Jacobians'],96);self.assertEqual(s['new_linear_program_calls'],1);self.assertFalse(s['final_head_frozen']);self.assertFalse((out/'FIXED_FINAL_HEAD_STEP8.npz').exists())
 def test_final_repeat_mismatch_never_qualifies(self):
  def factory(*a):
   rt=Analytic(*a);old=rt.coefficient_record
   def rec(*args,**kw):
    r=old(*args,**kw)
    if rt.batch['folder'].name=='final_step8_repeat' and r['state_id']==IDS[0]:r['observed'][0]+=.01
    return r
   rt.coefficient_record=rec;return rt
  s,out=self.scenario(factory=factory);self.assertFalse(s['complete']);self.assertEqual(s['new_training_steps'],8);self.assertTrue(s['final_head_frozen']);self.assertNotIn('final_training_all84_passed_both_rounds',s)
 def test_committed_update_artifact_mutation_blocks(self):
  def factory(*a):
   rt=Analytic(*a);old=rt.coefficient_record
   def rec(*args,**kw):
    r=old(*args,**kw)
    if rt.batch['folder'].name=='step_02_before' and r['state_id']==IDS[0]:f=rt.out.parent/'UPDATE_STEP_01.npz';f.write_bytes(f.read_bytes()+b'tamper')
    return r
   rt.coefficient_record=rec;return rt
  s,out=self.scenario(factory=factory);self.assertFalse(s['complete']);self.assertFalse(s['final_head_frozen']);self.assertIn('Fixed initial artifact/UPDATE_STEP_01.npz',''.join(s['errors']))
 def test_empty_active_all8_slots_zero_direction_no_solver(self):
  from fixture171 import observed
  def edit(paths,c,ctx):
   for sid in IDS:ctx['data']['d']['points'][sid]['raw'][:6]=.001
   chain=control.source168.old.parent.parent.previous.constants(ctx['data']);p=ctx['C178_final_head'];frames=[];native=[]
   for sid in IDS:
    pt,e=mapping(chain,p,sid);case=sid.split('__')[0];o=observed(pt,sid);met=archive155.full_metric(observed(pt.astype(np.float32),sid),0.,pt.astype(np.float32),case);q=archive155.endpoint_gates(met,case,pt.astype(np.float32),chain[sid+'__prepared'].astype(np.float32),True);native.append(dict(state_id=sid,case=case,metrics=met,qualification=q));frames.append(dict(state_id=sid,observed=o.tolist(),KL=0.))
   ctx.update(C178_final_FP64=frames,C178_final_native={'0':copy.deepcopy(native),'1':copy.deepcopy(native)})
  s,out=self.scenario(edit,backend=lambda *a,**k:(_ for _ in ()).throw(AssertionError('No LP for empty active')));self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['new_training_steps'],8);self.assertEqual(s['new_linear_program_calls'],0);self.assertEqual(s['nonzero_head_updates'],0);self.assertFalse(s['LC2_closed']);hist=clean_json((out/'ALL8_UPDATE_HISTORY.json').read_bytes());self.assertTrue(all(h['empty_active_noop'] and h['active_state_ids']==[] for h in hist))
 def test_actual_head_certificate_failure_blocks_commit(self):
  import worker180
  old=worker180.exact_certificate
  def cert(pr,d,head=None):
   r=old(pr,d,head)
   if head is not None:r['strict_recorded_active_margin']=False
   return r
  with patch.object(worker180,'exact_certificate',cert):s,out=self.scenario()
  self.assertFalse(s['complete']);self.assertEqual(s['new_training_steps'],0);self.assertEqual(s['new_linear_program_calls'],1);self.assertFalse((out/'UPDATE_STEP_01.npz').exists());self.assertFalse(s['final_head_frozen'])
 def test_update_sign_scale_and_full_step_free_mask(self):
  p=np.full((3,2,6),.01);d=np.full_like(p,-.5);q,r=update(p,d);np.testing.assert_array_equal(q,p+.001*d);self.assertEqual(r['step_size'],.001);self.assertFalse(r['head_projection_performed']);p.ravel()[0]=.1245;d.ravel()[0]=1
  with self.assertRaises(ValueError):update(p,d)
 def test_update_rejects_wrong_dtype_nonfinite_and_bounds(self):
  p=np.zeros((3,2,6))
  for d in (p.astype(np.float32),np.full_like(p,np.nan),np.full_like(p,1.1)):
   with self.assertRaises(ValueError):update(p,d)
if __name__=='__main__':unittest.main(verbosity=2)
