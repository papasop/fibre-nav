"""Fixed panel, solver/witness, complete NumPy-only protocol and stop tests."""
import copy,tempfile,unittest,contextlib,io,types
from core172 import *
from worker172 import run
from fixture172 import context_fixture,Analytic,synthetic_solver
from solver170 import solve
def rows(nactive=9):
 p=np.zeros((3,2,6));rr=[]
 for i,sid in enumerate(IDS):
  g=np.zeros_like(p)
  if i<nactive:g.ravel()[i%p.size]=1.
  rr.append(dict(state_id=sid,objective_parts=dict(response_loss=1. if i<nactive else 0.,read_loss=0.,KL_loss=0.,radius_loss=0.,regularizer=0.),response_gradient=g.tolist(),guard_gradient=np.zeros_like(p).tolist()))
 return p,rr
class Checks(unittest.TestCase):
 def test_registry_all1848_all84_and_both_signs(self):
  rr=requests();self.assertEqual(len(rr),1848);self.assertEqual(sum(r['stage']=='FP64_anchor' for r in rr),84);self.assertEqual(sum(r['stage']=='FP64_probe' for r in rr),504);self.assertEqual(sum(r['stage'].startswith('native') for r in rr),1176);self.assertEqual(len({tuple(r.items()) for r in rr}),1848);self.assertEqual([a['alpha'] for a in design()['arms']],[-1e-4,1e-4,-3e-4,3e-4,-1e-3,1e-3])
 def test_panel_actual_binary64_formula_no_projection(self):
  p=np.full((3,2,6),.01);d=np.linspace(-1,1,p.size).reshape(p.shape);hs,re=step_panel(p,d)
  for a in re['arms']:np.testing.assert_array_equal(hs[a['name']],np.add(p,np.multiply(a['alpha'],d)))
  self.assertFalse(re['head_projection_performed']);self.assertFalse(re['step_shrinking_performed'])
 def test_panel_free_mask_reserves_largest_step(self):
  p,rr=rows();p.ravel()[0]=.1245;pr=problem(rr,p);self.assertFalse(pr['free'][0]);d=np.zeros_like(p);d.ravel()[0]=1
  with self.assertRaises(ValueError):step_panel(p,d)
 def test_nonfinite_head_direction_blocks(self):
  p,_=rows()
  for q,d in [(p+np.nan,p),(p,p+np.inf)]:
   with self.assertRaises(ValueError):step_panel(q,d)
 def test_wrong_dtype_and_shape_blocks(self):
  p,_=rows()
  for q,d in [(p.astype(np.float32),p),(p,p.astype(np.float32)),(p.ravel(),p.ravel())]:
   with self.assertRaises(ValueError):step_panel(q,d)
 def test_direction_bound_blocks(self):
  p,_=rows()
  with self.assertRaises(ValueError):step_panel(p,np.full_like(p,1.01))
 def test_full84_gradient_order_and_zero_active_rows_retained(self):
  p,rr=rows();pr=problem(rr,p);self.assertEqual(pr['raw'].shape,(84,36));self.assertEqual(pr['active'].tolist(),list(range(9)))
  for bad in (rr[:-1],rr[::-1]):
   with self.assertRaises(ValueError):problem(bad,p)
 def test_nonfinite_gradient_blocks(self):
  p,rr=rows();rr[0]['response_gradient'][0][0][0]=float('nan')
  with self.assertRaises(ValueError):problem(rr,p)
 def test_actual_SciPy_single_primal_on_synthetic_gradient(self):
  from scipy.optimize import linprog
  p,rr=rows();pr=problem(rr,p);calls=[]
  def backend(cost,**kw):calls.append(kw);return linprog(cost,**kw)
  r=solve(pr,backend);cert=control.verify_direction(pr,r['direction']);self.assertEqual(len(calls),1);self.assertEqual(calls[0]['method'],'highs-ds');self.assertTrue(cert['strict_recorded_common_descent'])
 def test_empty_active_zero_direction_no_solver(self):
  p,rr=rows(0);pr=problem(rr,p);r=solve(pr,lambda *a,**k:(_ for _ in ()).throw(AssertionError('No LP')));cert=control.verify_direction(pr,r['direction']);heads,panel=step_panel(p,r['direction'].reshape(p.shape));self.assertFalse(cert['strict_recorded_common_descent']);self.assertIsNone(cert['normalized_margin_lower']);self.assertTrue(panel['zero_direction_noop']);self.assertTrue(all(np.array_equal(q,p) for q in heads.values()))
 def test_active_zero_gradient_not_certified(self):
  p,rr=rows();rr[0]['response_gradient']=np.zeros_like(p).tolist();cert=control.verify_direction(problem(rr,p),-np.ones(p.size));self.assertFalse(cert['strict_recorded_common_descent'])
 def test_opposing_rows_not_falsely_certified(self):
  p,rr=rows(2);rr[1]['response_gradient']=(-np.asarray(rr[0]['response_gradient'])).tolist();self.assertFalse(control.verify_direction(problem(rr,p),-np.ones(p.size))['strict_recorded_common_descent'])
 def test_guard_ascent_not_mislabeled_protected(self):
  p,rr=rows();rr[0]['guard_gradient']=(-np.ones_like(p)).tolist();cert=control.verify_direction(problem(rr,p),-np.ones(p.size));self.assertGreater(cert['all84_exact_contractions'][0]['guard_dot']['approximate'],0)
 def test_solver_without_primal_no_fallback(self):
  p,rr=rows();calls=[]
  def backend(cost,**kw):calls.append(1);return types.SimpleNamespace(x=None)
  with self.assertRaises((ValueError,TypeError)):solve(problem(rr,p),backend)
  self.assertEqual(len(calls),1)
 def run_fixture(self,edit=None,rf=Analytic,solver=synthetic_solver):
  t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);root=Path(t.name);paths,c,ctx=context_fixture(root/'inputs')
  if edit:edit(paths,c,ctx)
  with contextlib.redirect_stdout(io.StringIO()):s=run(paths,root/'out',c,rf,lambda *a:copy.deepcopy(ctx),solver)
  return s,root/'out'
 def test_complete_worker_all6_full84_no_selection_or_training(self):
  s,out=self.run_fixture();self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['new_model_forwards'],3699);self.assertEqual(s['functional_FP64_forwards'],1346);self.assertEqual(s['new_backward_calls'],168);self.assertEqual(s['new_linear_program_calls'],1);self.assertEqual(s['records_completed'],1848);self.assertEqual(s['actual_FP64_point_tensors_saved'],672);self.assertEqual(s['temporary_probe_heads'],6);self.assertEqual(s['new_training_steps'],0);self.assertFalse(s['candidate_selected']);self.assertFalse(s['LC2_closed']);self.assertFalse(s['all_seventeen_originals_independently_bound'])
  seal=clean_json((out/'ALL6_PROBES_SEALED_BEFORE_FIRST_PROBE.json').read_bytes());self.assertEqual(seal['model_forwards_so_far'],339);self.assertEqual(seal['backward_calls_so_far'],168);self.assertEqual(seal['probe_records_completed'],0);self.assertEqual(len(seal['all6_head_digests']),6);self.assertEqual(len(clean_json((out/'ALL252_CENTRAL_DIFFERENCES_VS_FRESH_AD.json').read_bytes())),252);self.assertEqual(len(clean_json((out/'ALL504_SIGNED_LOSS_CHANGES.json').read_bytes())),504)
  for name in ['anchor']+ARM_NAMES:
   for rnd in ('0','1'):self.assertEqual(s['diagnostics']['native_summaries'][name][rnd]['n_evaluated'],84)
  self.assertIsNone(s['solver_internal_linear_system_count']);self.assertIsNone(s['linear_systems_solved']);self.assertTrue(s['diagnostics']['fresh_gradients_at_C170_final_anchor'])
 def test_wrong_input_hash_stops_before_loader_and_model(self):
  s,out=self.run_fixture(lambda p,c,x:c['pins']['C171'].__setitem__('sha256','0'*64));self.assertFalse(s['complete']);self.assertEqual(s['new_model_forwards'],0);self.assertEqual(s['new_linear_program_calls'],0)
 def test_wrong_parent_decision_stops_before_model(self):
  s,out=self.run_fixture(lambda p,c,x:x['C171_receipt'].__setitem__('decision','forged'));self.assertFalse(s['complete']);self.assertEqual(s['new_model_forwards'],0)
 def test_native_replay_drift_stops_before_AD(self):
  s,out=self.run_fixture(lambda p,c,x:x['C170_final_native']['0'][0]['metrics']['observed'].__setitem__(0,.01));self.assertFalse(s['complete']);self.assertEqual(s['records_completed'],1);self.assertEqual(s['new_backward_calls'],0);self.assertEqual(s['new_linear_program_calls'],0)
 def test_FP64_replay_drift_stops_before_LP_or_probe(self):
  s,out=self.run_fixture(lambda p,c,x:x['C170_final_FP64'][0]['observed'].__setitem__(0,.01));self.assertFalse(s['complete']);self.assertEqual(s['records_completed'],85);self.assertEqual(s['new_backward_calls'],2);self.assertEqual(s['new_linear_program_calls'],0);self.assertFalse((out/'FIXED6_STEP_PANEL_BEFORE_PROBES.json').exists())
 def test_nonpositive_direction_stops_without_probe_or_fallback(self):
  calls=[]
  class Result(dict):
   def __getattr__(self,k):return self[k]
  def backend(cost,**kw):calls.append(1);return Result(x=np.zeros(len(cost)),success=False,status=1)
  s,out=self.run_fixture(solver=backend);self.assertFalse(s['complete']);self.assertEqual(s['new_backward_calls'],168);self.assertEqual(s['new_linear_program_calls'],1);self.assertEqual(len(calls),1);self.assertEqual(s['temporary_probe_heads'],0);self.assertFalse((out/'ALL6_PROBES_SEALED_BEFORE_FIRST_PROBE.json').exists())
 def test_partial_gradient_failure_preserves_counts(self):
  def factory(*a):
   rt=Analytic(*a);original=rt.coefficient_record;counter=[0]
   def record(*a,**k):
    counter[0]+=1
    if counter[0]==13:raise RuntimeError('synthetic partial AD')
    return original(*a,**k)
   rt.coefficient_record=record;return rt
  s,out=self.run_fixture(rf=factory);self.assertFalse(s['complete']);self.assertEqual(s['new_backward_calls'],24);self.assertEqual(s['records_completed'],96);self.assertEqual(s['new_linear_program_calls'],0)
 def test_probe_mutation_blocks_and_keeps_all_heads_sealed(self):
  def factory(*a):
   rt=Analytic(*a);original=rt.coefficient_record;counter=[0]
   def record(*args,**kw):
    r=original(*args,**kw);counter[0]+=1
    if counter[0]==168:(rt.out.parent/'plus_s3/HEAD.npz').write_bytes(b'changed after first probe')
    return r
   rt.coefficient_record=record;return rt
  s,out=self.run_fixture(rf=factory);self.assertFalse(s['complete']);self.assertEqual(s['temporary_probe_heads'],6);self.assertFalse(s['candidate_selected']);self.assertEqual(s['new_backward_calls'],168)
if __name__=='__main__':unittest.main(verbosity=2)
