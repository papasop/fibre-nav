"""All-coordinate LP algebra, exact affine margins, complete protocol and stops."""
import copy,tempfile,unittest,contextlib,io,types
from core174 import *
from solver174 import solve
from fixture174 import fixture,Analytic,synthetic_solver
from worker174 import run
def rows(active=True):
 p=np.zeros((3,2,6));rr=[]
 for i,sid in enumerate(IDS):
  r=np.full(6,.0022 if active and i<5 else .001);J=np.zeros((6,)+p.shape)
  for j in range(6):J[j].ravel()[j]=1.
  m=float(max(abs(r)));R=max(m/SAFETY-1,0)**2;top=abs(r)==m;g=2*max(m/SAFETY-1,0)/SAFETY*np.mean(J[top],axis=0);rr.append(dict(state_id=sid,observed=np.r_[r,np.zeros(16)].tolist(),objective_parts=dict(response_loss=R,read_loss=0.,KL_loss=0.,radius_loss=0.,regularizer=0.),response_coordinate_gradients=J.tolist(),response_gradient=g.tolist(),guard_gradient=np.zeros_like(p).tolist()))
 return p,rr
class Checks(unittest.TestCase):
 def test_all84_all6_both_signs_in_LP(self):
  p,rr=rows();pr=problem(rr,p);self.assertEqual(pr['J'].shape,(84,6,36));self.assertEqual(pr['A'].shape,(1008,36));self.assertEqual(len(set(map(tuple,pr['registry']))),1008);self.assertEqual(pr['active'].tolist(),list(range(5)));self.assertEqual(pr['registry'][-1].tolist(),[83,5,1])
 def test_problem_rhs_sign_scale_and_caps(self):
  p,rr=rows();pr=problem(rr,p);self.assertAlmostEqual(pr['A'][1,0],ETA/SAFETY);self.assertAlmostEqual(pr['A'][0,0],-ETA/SAFETY);self.assertAlmostEqual(pr['rhs'][1],0);self.assertAlmostEqual(pr['rhs'][0],2*.0022/SAFETY);self.assertEqual(pr['caps'][83],SAFETY)
 def test_actual_SciPy_one_primal_all1008_rows(self):
  from scipy.optimize import linprog
  p,rr=rows();pr=problem(rr,p);seen=[]
  def backend(cost,**kw):seen.append(kw);return linprog(cost,**kw)
  sol=solve(pr,backend);cert=exact_certificate(pr,sol['direction']);self.assertEqual(len(seen),1);self.assertEqual(seen[0]['A_ub'].shape,(1008,37));self.assertEqual(seen[0]['bounds'][-1],(0.,None));self.assertTrue(cert['strict_recorded_active_margin']);self.assertEqual(cert['signed_rows'],1008)
 def test_inactive_coordinate_can_block_old_direction(self):
  p,rr=rows();J=np.asarray(rr[-1]['response_coordinate_gradients']);J[5].ravel()[0]=-100.;rr[-1]['response_coordinate_gradients']=J.tolist();r=np.asarray(rr[-1]['observed']);r[5]=.00179;rr[-1]['observed']=r.tolist();pr=problem(rr,p);d=-np.ones_like(p);cert=exact_certificate(pr,d);self.assertFalse(cert['strict_recorded_active_margin']);bad=[r for r in cert['all1008_signed_rows'] if int(r['normalized_recorded_cap_slack']['numerator'])<0];self.assertTrue(any(r['state_id']==IDS[-1] and r['coordinate']==5 for r in bad))
 def test_other_coordinate_of_active_state_is_not_dropped(self):
  p,rr=rows();J=np.asarray(rr[0]['response_coordinate_gradients']);J[5].ravel()[0]=-100.;rr[0]['response_coordinate_gradients']=J.tolist();r=np.asarray(rr[0]['observed']);r[5]=.00219;rr[0]['observed']=r.tolist();rr[0]['response_gradient']=(2*(.0022/SAFETY-1)/SAFETY*np.mean(J[:5],axis=0)).tolist();pr=problem(rr,p);self.assertFalse(exact_certificate(pr,-np.ones_like(p))['strict_recorded_active_margin'])
 def test_empty_active_has_no_solver_no_arbitrary_direction(self):
  p,rr=rows(False);pr=problem(rr,p);sol=solve(pr,lambda *a,**k:(_ for _ in ()).throw(AssertionError('No LP')));cert=exact_certificate(pr,sol['direction']);self.assertFalse(cert['strict_recorded_active_margin']);self.assertEqual(sol['calls'],0)
  with self.assertRaises(ValueError):exact_certificate(pr,np.ones_like(p))
 def test_wrong_gradient_units_fail_scalar_chain(self):
  p,rr=rows();rr[0]['response_coordinate_gradients']=(np.asarray(rr[0]['response_coordinate_gradients'])/SAFETY).tolist()
  with self.assertRaises(ValueError):problem(rr,p)
 def test_nonfinite_or_wrong_shape_J_blocks(self):
  p,rr=rows();rr[0]['response_coordinate_gradients'][0][0][0][0]=float('nan')
  with self.assertRaises(ValueError):problem(rr,p)
  p,rr=rows();rr[0]['response_coordinate_gradients']=rr[0]['response_coordinate_gradients'][:5]
  with self.assertRaises(ValueError):problem(rr,p)
 def test_state_order_and_missing_rows_block(self):
  p,rr=rows()
  for bad in (rr[::-1],rr[:-1]):
   with self.assertRaises(ValueError):problem(bad,p)
 def test_full_panel_safe_free_mask(self):
  p,rr=rows();p.ravel()[0]=.1245;pr=problem(rr,p);self.assertFalse(pr['free'][0]);d=np.zeros_like(p);d.ravel()[0]=1
  with self.assertRaises(ValueError):exact_certificate(pr,d)
 def test_actual_head_endpoint_fraction_uses_unrounded_difference(self):
  p,rr=rows();p[:]=.01;pr=problem(rr,p);d=-np.ones_like(p);head=p+ETA*d;cert=exact_certificate(pr,d,head);first=cert['all1008_signed_rows'][1]['recorded_linear_signed_response'];want=rational(.0022)+rational(head.ravel()[0])-rational(p.ravel()[0]);self.assertEqual(first,encoded(want))
 def test_wrong_direction_dtype_or_bound_blocks(self):
  p,rr=rows();pr=problem(rr,p)
  for d in (p.astype(np.float32),np.full_like(p,1.1),np.full_like(p,np.nan)):
   with self.assertRaises(ValueError):exact_certificate(pr,d)
 def run_fixture(self,edit=None,rf=Analytic,backend=synthetic_solver):
  t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);root=Path(t.name);paths,c,ctx=fixture(root/'inputs')
  if edit:edit(paths,c,ctx)
  with contextlib.redirect_stdout(io.StringIO()):s=run(paths,root/'out',c,rf,lambda *a:copy.deepcopy(ctx),backend)
  return s,root/'out'
 def test_complete_NumPy_protocol_all6_all84_counts_and_no_selection(self):
  s,out=self.run_fixture();self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['new_model_forwards'],3699);self.assertEqual(s['functional_FP64_forwards'],1346);self.assertEqual(s['new_backward_calls'],672);self.assertEqual(s['new_response_coordinate_Jacobian_rows'],504);self.assertEqual(s['new_Jacobians'],84);self.assertEqual(s['new_model_parameter_Jacobians'],0);self.assertEqual(s['new_linear_program_calls'],1);self.assertEqual(s['records_completed'],1848);self.assertEqual(s['LP_signed_rows'],1008);self.assertEqual(s['new_training_steps'],0);self.assertFalse(s['candidate_selected']);self.assertFalse(s['LC2_closed']);self.assertFalse(s['all_nineteen_originals_independently_bound']);self.assertIsNone(s['linear_systems_solved']);self.assertEqual(s['diagnostics']['response_coordinate_affine_rows_saved'],3024)
  seal=clean_json((out/'ALL6_PROBES_SEALED_BEFORE_FIRST_PROBE.json').read_bytes());self.assertEqual(seal['model_forwards_so_far'],339);self.assertEqual(seal['backward_calls_so_far'],672);self.assertEqual(seal['probe_records_completed'],0);self.assertIn('ALL6_ACTUAL_HEAD_AFFINE_CERTIFICATES.json',seal['artifact_seals'])
  certs=clean_json((out/'ALL6_ACTUAL_HEAD_AFFINE_CERTIFICATES.json').read_bytes())
  for name in ('plus_s1','plus_s2','plus_s3'):self.assertTrue(certs[name]['strict_recorded_active_margin'])
  for name in ['anchor']+ARM_NAMES:
   for rnd in ('0','1'):self.assertEqual(s['diagnostics']['native_summaries'][name][rnd]['n_evaluated'],84)
 def test_wrong_C173_whole_hash_stops_before_model(self):
  s,out=self.run_fixture(lambda p,c,x:c['pins']['C173'].__setitem__('sha256','0'*64));self.assertFalse(s['complete']);self.assertEqual(s['new_model_forwards'],0)
 def test_wrong_parent_decision_stops_before_model(self):
  s,out=self.run_fixture(lambda p,c,x:x['C173_receipt'].__setitem__('decision','forged'));self.assertFalse(s['complete']);self.assertEqual(s['new_model_forwards'],0)
 def test_native_replay_failure_stops_before_Jacobians(self):
  s,out=self.run_fixture(lambda p,c,x:x['C172_anchor_native']['0'][0]['metrics']['observed'].__setitem__(0,.01));self.assertFalse(s['complete']);self.assertEqual(s['new_backward_calls'],0);self.assertEqual(s['new_Jacobians'],0)
 def test_FP64_replay_failure_stops_before_LP(self):
  s,out=self.run_fixture(lambda p,c,x:x['C172_anchor_FP64'][0]['observed'].__setitem__(0,.01));self.assertFalse(s['complete']);self.assertEqual(s['new_backward_calls'],8);self.assertEqual(s['new_Jacobians'],1);self.assertEqual(s['new_linear_program_calls'],0)
 def test_no_positive_exact_margin_stops_without_probes_retry_or_infeasibility_claim(self):
  class R(dict):
   def __getattr__(self,k):return self[k]
  calls=[]
  def backend(cost,**kw):calls.append(1);return R(x=np.zeros(len(cost)),success=False,status=1)
  s,out=self.run_fixture(backend=backend);self.assertFalse(s['complete']);self.assertEqual(len(calls),1);self.assertEqual(s['new_backward_calls'],672);self.assertEqual(s['new_linear_program_calls'],1);self.assertEqual(s['temporary_probe_heads'],0);self.assertFalse(s['solver_optimality_or_infeasibility_certified'])
 def test_partial_gradient_failure_preserves_Jacobian_counts(self):
  def rf(*a):
   rt=Analytic(*a);original=rt.coefficient_record;counter=[0]
   def record(*args,**kwargs):
    counter[0]+=1
    if counter[0]==13:raise RuntimeError('synthetic partial batch')
    return original(*args,**kwargs)
   rt.coefficient_record=record;return rt
  s,out=self.run_fixture(rf=rf);self.assertFalse(s['complete']);self.assertEqual(s['new_backward_calls'],96);self.assertEqual(s['new_Jacobians'],12);self.assertEqual(s['new_response_coordinate_Jacobian_rows'],72);self.assertEqual(s['new_linear_program_calls'],0)
if __name__=='__main__':unittest.main(verbosity=2)
